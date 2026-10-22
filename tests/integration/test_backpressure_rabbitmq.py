"""Queue overflow against a real RabbitMQ started by Testcontainers (needs Docker).

The `tasks` topology is declared here with queue x-arguments (production uses a policy with the
same max-length / reject-publish) and a small limit, so the queue fills in milliseconds.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import aio_pika
import pytest
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from aiormq.exceptions import DeliveryError
from pydantic import SecretStr
from testcontainers.community.rabbitmq import RabbitMqContainer

from edu_graphrag.components.rabbit import RabbitComponent
from edu_graphrag.config import RabbitSettings
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.source import Source
from edu_graphrag.pipeline.topology import TASKS_QUEUE, TASKS_ROUTING_KEY

pytestmark = pytest.mark.integration

IMAGE = "rabbitmq:4.2-management-alpine"
LIMIT = 20
# Quorum queues check the limit before enqueueing, so they accept one message over it.
MAX_DEPTH = LIMIT + 1


@pytest.fixture(scope="module")
def rabbit_settings() -> Iterator[RabbitSettings]:
    container = RabbitMqContainer(IMAGE)
    try:
        container.start()
    except Exception as exc:  # docker missing / daemon down / image cannot be pulled
        pytest.skip(f"Docker is unavailable, cannot start {IMAGE}: {type(exc).__name__}: {exc}")
    try:
        yield RabbitSettings(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5672)),
            user=container.username,
            password=SecretStr(container.password),
        )
    finally:
        container.stop()


@dataclass
class Broker:
    settings: RabbitSettings
    connection: AbstractRobustConnection
    exchange: Any
    queue: Any

    async def depth(self) -> int:
        """Ready messages. A passive declare on a RAW channel: robust channels cache declarations
        and would return stale counters."""
        probe = await self.connection.channel()
        try:
            raw = await probe.get_underlay_channel()
            declare_ok = await raw.queue_declare(TASKS_QUEUE.name, passive=True)
            return declare_ok.message_count or 0
        finally:
            await probe.close()

    async def publish(self, message: TaskMessage) -> None:
        """One publish with a confirm; raises DeliveryError on basic.nack."""
        await self.exchange.publish(
            aio_pika.Message(message.model_dump_json().encode(), message_id=str(message.task_id)),
            TASKS_ROUTING_KEY,
            mandatory=True,
            timeout=5,
        )


@pytest.fixture
async def broker(rabbit_settings: RabbitSettings) -> AsyncIterator[Broker]:
    connection = await aio_pika.connect_robust(rabbit_settings.url, timeout=10)
    try:
        channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
        exchange = await channel.declare_exchange(
            "tasks", aio_pika.ExchangeType.DIRECT, durable=True
        )
        queue = await channel.declare_queue(
            TASKS_QUEUE.name,
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-max-length": LIMIT,
                "x-overflow": "reject-publish",
            },
        )
        await queue.bind(exchange, routing_key=TASKS_ROUTING_KEY)
        yield Broker(rabbit_settings, connection, exchange, queue)
        with contextlib.suppress(Exception):
            await queue.delete(if_unused=False, if_empty=False)
    finally:
        await connection.close()


def new_task() -> TaskMessage:
    return TaskMessage.new("overflow test", "batch")


async def fill_until_nack(broker: Broker, attempts: int) -> tuple[int, int, int]:
    """Returns (accepted, nacked, max queue depth seen)."""
    accepted = nacked = max_depth = 0
    for _ in range(attempts):
        try:
            await broker.publish(new_task())
        except DeliveryError:
            nacked += 1
        else:
            accepted += 1
        max_depth = max(max_depth, await broker.depth())
    return accepted, nacked, max_depth


async def test_broker_nacks_publishes_to_a_full_queue(broker: Broker) -> None:
    accepted, nacked, max_depth = await fill_until_nack(broker, LIMIT + 5)

    assert nacked >= 4, "the broker must refuse what does not fit"
    assert LIMIT <= accepted <= MAX_DEPTH
    assert max_depth <= MAX_DEPTH
    print(f"\n[overflow] accepted={accepted} nacked={nacked} max_depth={max_depth}")


@dataclass
class SlowConsumer:
    """Prefetch 1, ack after ~20 ms."""

    broker: Broker
    consumed: list[UUID] = field(default_factory=list)

    async def run(self) -> None:
        channel = await self.broker.connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.get_queue(TASKS_QUEUE.name, ensure=False)

        async def handle(message: AbstractIncomingMessage) -> None:
            await asyncio.sleep(0.02)
            self.consumed.append(TaskMessage.model_validate_json(message.body).task_id)
            await message.ack()

        await queue.consume(handle)
        try:
            await asyncio.Event().wait()
        finally:
            await channel.close()


async def test_publisher_is_slowed_down_without_losing_tasks(broker: Broker) -> None:
    component = RabbitComponent(broker.settings, connect_timeout=10)
    await component.startup()
    assert component.is_connected

    sleeps: list[float] = []

    async def recording_sleep(delay: float) -> None:
        sleeps.append(delay)
        await asyncio.sleep(delay)

    source = Source(publisher=component, max_in_flight=8, confirm_timeout=5, sleep=recording_sleep)
    tasks = [new_task() for _ in range(3 * LIMIT)]
    drain = SlowConsumer(broker)
    depths: list[int] = []
    consumer = asyncio.create_task(drain.run())

    async def sample_depth() -> None:
        while True:
            depths.append(await broker.depth())
            await asyncio.sleep(0.05)

    sampler = asyncio.create_task(sample_depth())
    started = time.monotonic()
    try:
        counters = await asyncio.wait_for(source.run(tasks), timeout=40)
        deadline = time.monotonic() + 15
        while len(drain.consumed) < len(tasks) and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.05)
    finally:
        elapsed = time.monotonic() - started
        for task in (sampler, consumer):
            task.cancel()
        await asyncio.gather(sampler, consumer, return_exceptions=True)
        await component.shutdown()

    assert len(drain.consumed) == len(tasks), "every task is delivered exactly once"
    assert set(drain.consumed) == {t.task_id for t in tasks}
    assert counters.accepted == len(tasks)
    # Concurrent in-flight publishes may each slip past the limit check: bound is LIMIT + window.
    assert max(depths) <= LIMIT + source.max_in_flight
    # The queue (20) is smaller than the batch (60): the producer must have been refused.
    assert counters.rejected > 0
    # Every refusal was followed by a real sleep (no busy loop), and refusals/s are bounded.
    assert len(sleeps) == counters.refused
    assert all(delay >= source.backoff.min_delay for delay in sleeps)
    assert counters.rejected / elapsed < 1000
    # The batch cannot finish faster than the consumer drains what did not fit in the queue.
    assert elapsed >= (len(tasks) - MAX_DEPTH - 1) * 0.02
    print(
        f"\n[backpressure] rejected={counters.rejected} max_depth={max(depths)} "
        f"samples={len(depths)} elapsed={elapsed:.1f}s"
    )


async def test_publishing_recovers_after_the_queue_drains(broker: Broker) -> None:
    _, nacked, _ = await fill_until_nack(broker, LIMIT + 5)
    assert nacked > 0
    with pytest.raises(DeliveryError):
        await broker.publish(new_task())

    drained = 0
    while await broker.queue.get(fail=False, no_ack=True) is not None:
        drained += 1
    assert drained >= LIMIT
    assert await broker.depth() == 0

    for _ in range(5):  # accepted immediately, no stuck "full" state
        await broker.publish(new_task())
    assert await broker.depth() == 5
