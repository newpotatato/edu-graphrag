"""Publishing tasks with broker-driven backpressure.

How the producer learns that the system is overloaded
-----------------------------------------------------
The `tasks` queue is bounded by a RabbitMQ policy: `max-length` + `overflow: reject-publish`.
Every publish goes through a channel in *publisher confirms* mode, so the broker answers each
message individually:

* `basic.ack`  - the message is in the (quorum, replicated) queue;
* `basic.nack` - the queue is full and refused it (reject-publish);
* `basic.return` (we publish with `mandatory=True`) - nothing is bound to the routing key,
  i.e. the topology is missing: a configuration error, not overload.

aio-pika turns a nack into `DeliveryError` and a return into its subclass `PublishError`
(the channel is opened with `on_return_raises=True`, otherwise a returned message is
silently "acked"). This was verified against RabbitMQ 4.2.9: a quorum queue with
`x-max-length=10` acked 11 publishes (quorum queues check the limit before enqueueing, so
they overshoot by one) and nacked the next ones; after the queue was drained back to 8
(quorum queues resume at 80 % of max-length, a built-in hysteresis) publishes were acked
again. FastStream's `RabbitBroker.publish` surfaces the same `DeliveryError`, but it is used
by the worker; the producer talks to aio-pika directly to keep explicit control over the
confirm channel, the per-message timeout and the return handling.

So the signal is per message, synchronous with the publish, produced by the component that
actually holds the bounded resource, and it needs no extra protocol. Rejected alternatives:

* polling queue depth via the Management API - the number is stale by the time it is read
  (stats are emitted every few seconds), every producer adds HTTP load to the broker, many
  producers race for the same "free" slots (check-then-act), the producer needs management
  credentials, and nothing guarantees that the *next* message fits;
* `connection.blocked` / memory and disk alarms - broker-wide and coarse: they trigger only
  at the global watermarks, block *every* publisher on the connection (including the API),
  and say nothing about this particular queue;
* `overflow: drop-head` - the queue never refuses, it silently throws away the oldest tasks;
* a fixed publish rate - rate limiting, not backpressure: it does not adapt when workers are
  added, removed or slowed down, and is either too slow or overflows;
* consumer -> producer credits (an RPC channel) or Kafka-style lag tracking - an extra
  protocol and moving parts for something the broker already tells us.

What the producer does with the signal lives here (`Backoff`, `publish_with_retry`):
a refused message is retried after an exponentially growing delay with full jitter, and the
number of unconfirmed messages is bounded by the caller (see `pipeline.source`).
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from edu_graphrag.pipeline.messages import TaskMessage

logger = structlog.get_logger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class PublishError(Exception):
    """The broker did not take responsibility for the message."""


class PublishRejectedError(PublishError):
    """basic.nack: the queue is full (overflow=reject-publish). Retry later."""


class PublishTimeoutError(PublishError):
    """No confirm in time. The message may or may not be in the queue (at-least-once)."""


class BrokerUnavailableError(PublishError):
    """No connection/channel to the broker right now."""


class PublishUnroutableError(PublishError):
    """basic.return: nothing is bound to the routing key. Topology missing: not retryable."""


class TaskPublisher(Protocol):
    async def publish(self, message: TaskMessage, *, confirm_timeout: float) -> None:
        """Publish one message and wait for the broker's confirm.

        Raises a `PublishError` subclass when the broker refused it or did not confirm.
        """

    async def consumer_count(self) -> int | None:
        """Consumers of the tasks queue right now (a round trip), None if unknown."""


@dataclass
class Backoff:
    """Exponential backoff with full jitter (AWS Architecture Blog, "Exponential Backoff And
    Jitter"): delay ~ U(0, min(cap, base * 2**n)).

    One instance is shared by all in-flight publishes of a producer: every refusal raises the
    ceiling, every accepted publish resets it. Jitter spreads the retries of the window over
    time instead of hitting the broker in synchronized waves. `min_delay` keeps the loop from
    ever spinning with a zero sleep.

    Defaults: base 20 ms is about one broker round trip with a quorum write, so the first
    retry is almost immediate (the queue might have had a single slot freed). The cap of 2 s
    keeps the workers busy: a quorum queue resumes accepting at 80 % of max-length, and with
    the default policy (200) the remaining 160 tasks take 160 / 20 tasks/s = 8 s for one worker
    (4 threads x 200 ms) to drain, so the producer is back long before the queue runs dry.
    """

    base: float = 0.02
    cap: float = 2.0
    min_delay: float = 0.001
    rng: random.Random = field(default_factory=random.Random)
    attempt: int = 0
    last_delay: float = 0.0

    def ceiling(self) -> float:
        # Exponent is clamped so 2**attempt cannot overflow on a very long outage.
        exponent: float = 2.0 ** min(self.attempt, 32)
        return min(self.cap, self.base * exponent)

    def next_delay(self) -> float:
        delay = max(self.min_delay, self.rng.uniform(0, self.ceiling()))
        self.attempt += 1
        self.last_delay = delay
        return delay

    def reset(self) -> None:
        self.attempt = 0
        self.last_delay = 0.0


@dataclass
class PublishCounters:
    accepted: int = 0
    rejected: int = 0
    timeouts: int = 0
    unavailable: int = 0

    @property
    def refused(self) -> int:
        return self.rejected + self.timeouts + self.unavailable


async def publish_with_retry(
    publisher: TaskPublisher,
    message: TaskMessage,
    *,
    backoff: Backoff,
    confirm_timeout: float,
    counters: PublishCounters,
    sleep: Sleep = asyncio.sleep,
) -> int:
    """Publish until the broker confirms; returns the number of attempts.

    The *same* message (same task_id) is retried, so a timed-out publish that did reach the
    queue only produces a redelivery that the idempotent store absorbs. There is no attempt
    limit: a batch source must not lose tasks, and while it waits it holds a window slot, so
    it stops reading the input. Unroutable messages are a configuration error and propagate.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            await publisher.publish(message, confirm_timeout=confirm_timeout)
        except PublishUnroutableError:
            raise
        except PublishError as exc:
            if isinstance(exc, PublishRejectedError):
                counters.rejected += 1
            elif isinstance(exc, PublishTimeoutError):
                counters.timeouts += 1
            else:
                counters.unavailable += 1
            delay = backoff.next_delay()
            logger.debug(
                "publish_refused",
                task_id=str(message.task_id),
                reason=type(exc).__name__,
                attempt=attempts,
                retry_in_ms=round(delay * 1000, 1),
            )
            await sleep(delay)
        else:
            counters.accepted += 1
            backoff.reset()
            return attempts


__all__ = [
    "Backoff",
    "BrokerUnavailableError",
    "PublishCounters",
    "PublishError",
    "PublishRejectedError",
    "PublishTimeoutError",
    "PublishUnroutableError",
    "TaskPublisher",
    "publish_with_retry",
]
