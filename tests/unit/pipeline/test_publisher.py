import random

import pytest

from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    Backoff,
    BrokerUnavailableError,
    PublishCounters,
    PublishRejectedError,
    PublishTimeoutError,
    PublishUnroutableError,
    publish_with_retry,
)
from edu_graphrag.pipeline.retry_after import MAX_RETRY_AFTER, retry_after_seconds
from tests.fakes import FakeTaskPublisher, seeded


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_backoff_ceiling_doubles_up_to_the_cap() -> None:
    backoff = Backoff(base=0.02, cap=2.0)
    ceilings = []
    for _ in range(10):
        ceilings.append(backoff.ceiling())
        backoff.next_delay()

    assert ceilings[:7] == pytest.approx([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
    assert ceilings[7:] == [2.0, 2.0, 2.0]


def test_backoff_full_jitter_stays_within_bounds() -> None:
    backoff = Backoff(base=0.02, cap=2.0, rng=seeded(42))
    for attempt in range(40):
        ceiling = backoff.ceiling()
        delay = backoff.next_delay()
        assert backoff.min_delay <= delay <= ceiling, attempt
        assert delay > 0


def test_backoff_jitter_actually_spreads_delays() -> None:
    delays = set()
    for seed in range(20):
        backoff = Backoff(base=1.0, cap=1.0, rng=seeded(seed))
        delays.add(round(backoff.next_delay(), 3))

    assert len(delays) > 10  # not a fixed delay


def test_backoff_never_returns_zero() -> None:
    class ZeroRandom(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return 0.0

    assert Backoff(rng=ZeroRandom()).next_delay() == Backoff().min_delay > 0


def test_backoff_reset() -> None:
    backoff = Backoff(base=0.02, cap=2.0)
    for _ in range(20):
        backoff.next_delay()
    assert backoff.ceiling() == 2.0

    backoff.reset()

    assert backoff.attempt == 0
    assert backoff.last_delay == 0
    assert backoff.ceiling() == pytest.approx(0.02)


def test_backoff_survives_a_very_long_outage() -> None:
    backoff = Backoff(attempt=10_000)
    assert backoff.ceiling() == backoff.cap


async def test_retry_on_nack_publishes_the_same_message_without_loss() -> None:
    publisher = FakeTaskPublisher(
        outcomes=[PublishRejectedError("nack"), PublishTimeoutError("t"), BrokerUnavailableError()]
    )
    message = TaskMessage.new("x", source="batch")
    sleep = RecordingSleep()
    counters = PublishCounters()
    backoff = Backoff(rng=seeded(0))

    attempts = await publish_with_retry(
        publisher, message, backoff=backoff, confirm_timeout=1.5, counters=counters, sleep=sleep
    )

    assert attempts == 4
    assert publisher.attempts == [message.task_id] * 4  # the same task every time
    assert publisher.published == [message]
    assert publisher.timeouts == [1.5] * 4
    assert (counters.accepted, counters.rejected, counters.timeouts, counters.unavailable) == (
        1,
        1,
        1,
        1,
    )
    assert counters.refused == 3
    assert len(sleep.delays) == 3
    assert all(d > 0 for d in sleep.delays), "a retry must never busy-loop"
    assert backoff.attempt == 0, "success resets the backoff"


async def test_acked_publish_does_not_sleep() -> None:
    sleep = RecordingSleep()

    await publish_with_retry(
        FakeTaskPublisher(),
        TaskMessage.new("x", source="batch"),
        backoff=Backoff(),
        confirm_timeout=1,
        counters=PublishCounters(),
        sleep=sleep,
    )

    assert sleep.delays == []


async def test_unroutable_is_not_retried() -> None:
    publisher = FakeTaskPublisher(outcomes=[PublishUnroutableError("no binding")])
    sleep = RecordingSleep()

    with pytest.raises(PublishUnroutableError):
        await publish_with_retry(
            publisher,
            TaskMessage.new("x", source="batch"),
            backoff=Backoff(),
            confirm_timeout=1,
            counters=PublishCounters(),
            sleep=sleep,
        )

    assert len(publisher.attempts) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    ("consumers", "concurrency", "task_seconds", "expected"),
    [
        (1, 4, 0.2, 2),  # 0.2 * 200 * 0.2 / 4 = 2 s
        (2, 4, 0.2, 1),
        (4, 4, 0.2, 1),  # 0.5 s -> at least 1 s
        (1, 1, 0.2, 8),
        (1, 1, 10.0, MAX_RETRY_AFTER),  # clamped
        (0, 4, 0.2, MAX_RETRY_AFTER),  # nobody drains the queue
        (1, 4, 0.0, 1),
    ],
)
def test_retry_after_formula(
    consumers: int, concurrency: int, task_seconds: float, expected: int
) -> None:
    assert (
        retry_after_seconds(
            queue_max_length=200,
            task_seconds=task_seconds,
            consumers=consumers,
            concurrency=concurrency,
        )
        == expected
    )


def test_retry_after_grows_with_queue_size() -> None:
    small = retry_after_seconds(queue_max_length=100, task_seconds=0.2, consumers=1, concurrency=1)
    large = retry_after_seconds(queue_max_length=200, task_seconds=0.2, consumers=1, concurrency=1)

    assert (small, large) == (4, 8)
