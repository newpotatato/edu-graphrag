"""RabbitMQ connection used to publish tasks (API and batch source).

One robust connection (aio-pika reconnects it and restores channels) with one channel in
publisher-confirms mode. Publishing goes through `publish()`, which maps broker answers to the
pipeline's `PublishError`s (see `pipeline.publisher` for the backpressure reasoning).

Degraded start: if RabbitMQ is down when the app starts, the app still starts (/healthz and
/api/v1/version answer, the CD smoke test runs the image alone), `rabbitmq_connect_failed` is
logged, the component keeps reconnecting in the background every `reconnect_interval`
seconds, POST /api/v1/tasks answers 503 meanwhile and /api/v1/health shows it as down.

The component does not declare the topology: `pipeline.admin` owns exchanges, queues and the
limits policy. A missing binding shows up as an unroutable publish (`mandatory=True`).
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import aio_pika
import structlog
from aio_pika.abc import AbstractExchange, AbstractRobustConnection
from aio_pika.exceptions import (
    AMQPError,
    ChannelInvalidStateError,
    DeliveryError,
)
from aio_pika.exceptions import (
    PublishError as ReturnedError,
)

from edu_graphrag.config import RabbitSettings
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    BrokerUnavailableError,
    PublishRejectedError,
    PublishTimeoutError,
    PublishUnroutableError,
)
from edu_graphrag.pipeline.topology import TASKS_EXCHANGE, TASKS_QUEUE, TASKS_ROUTING_KEY

logger = structlog.get_logger(__name__)

Connect = Callable[..., Awaitable[AbstractRobustConnection]]


class RabbitComponent:
    name = "rabbitmq"

    def __init__(
        self,
        settings: RabbitSettings,
        *,
        connect_timeout: float,
        publisher_confirms: bool = True,
        reconnect_interval: float = 5.0,
        connect: Connect = aio_pika.connect_robust,
    ) -> None:
        self._url = settings.url
        self._connect_timeout = connect_timeout
        self._confirms = publisher_confirms
        self.reconnect_interval = reconnect_interval
        self._connect_fn = connect
        self._connection: AbstractRobustConnection | None = None
        self._exchange: AbstractExchange | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._exchange is not None

    async def startup(self) -> None:
        try:
            await self._connect()
        except Exception as exc:  # degraded start: the app must come up without the broker
            logger.error("rabbitmq_connect_failed", error=f"{type(exc).__name__}: {exc}")
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _connect(self) -> None:
        connection = await self._connect_fn(self._url, timeout=self._connect_timeout)
        try:
            # on_return_raises (confirms only): an unroutable mandatory message raises instead of being "acked".
            channel = await connection.channel(
                publisher_confirms=self._confirms, on_return_raises=self._confirms
            )
            exchange = await channel.get_exchange(TASKS_EXCHANGE.name, ensure=False)
        except BaseException:
            await connection.close()
            raise
        self._connection, self._exchange = connection, exchange
        logger.info("rabbitmq_connected", publisher_confirms=self._confirms)

    async def _reconnect_loop(self) -> None:
        # Only the *first* connection needs this: once connected, connect_robust reconnects.
        while self._exchange is None:
            await asyncio.sleep(self.reconnect_interval)
            try:
                await self._connect()
            except Exception as exc:
                logger.warning("rabbitmq_reconnect_failed", error=f"{type(exc).__name__}: {exc}")

    async def shutdown(self) -> None:
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None
        if self._connection is not None:
            await self._connection.close()
        self._connection = self._exchange = None

    async def publish(self, message: TaskMessage, *, confirm_timeout: float) -> None:
        """One attempt. With confirms: returns on basic.ack, raises a `PublishError` otherwise.
        Without confirms (anti-pattern baseline) it returns once the frame is written."""
        if self._exchange is None:
            raise BrokerUnavailableError("not connected to RabbitMQ")
        amqp_message = aio_pika.Message(
            message.model_dump_json().encode(),
            message_id=str(message.task_id),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        try:
            # The outer timeout also bounds waiting for a channel that is being restored.
            async with asyncio.timeout(confirm_timeout):
                await self._exchange.publish(
                    amqp_message, TASKS_ROUTING_KEY, mandatory=True, timeout=confirm_timeout
                )
        except ReturnedError as exc:
            raise PublishUnroutableError(str(exc)) from exc
        except DeliveryError as exc:
            raise PublishRejectedError(str(exc)) from exc
        except TimeoutError as exc:
            raise PublishTimeoutError(f"no confirm within {confirm_timeout}s") from exc
        except (AMQPError, ChannelInvalidStateError, ConnectionError) as exc:
            raise BrokerUnavailableError(f"{type(exc).__name__}: {exc}") from exc

    async def _declare_tasks_passive(self) -> Any:
        """Passive declare of `tasks` on a throwaway channel: a 404 closes that channel, not
        the publishing one. The raw channel is used because aio-pika's robust channel caches
        queues it declared and would return stale counters."""
        if self._connection is None:
            raise BrokerUnavailableError("not connected to RabbitMQ")
        channel = await self._connection.channel()
        try:
            raw = await channel.get_underlay_channel()
            return await raw.queue_declare(TASKS_QUEUE.name, passive=True)
        finally:
            with contextlib.suppress(Exception):
                await channel.close()

    async def consumer_count(self) -> int | None:
        try:
            declare_ok = await self._declare_tasks_passive()
        except Exception as exc:  # used on the 503 path only: never fail it
            logger.warning("rabbitmq_consumer_count_failed", error=type(exc).__name__)
            return None
        return int(declare_ok.consumer_count)

    def server_version(self) -> str:
        if self._connection is None or self._connection.transport is None:
            raise BrokerUnavailableError("not connected to RabbitMQ")
        version = self._connection.transport.connection.server_properties.get("version", "?")
        return version.decode() if isinstance(version, bytes) else str(version)

    async def fetch_version(self) -> str:
        """Server version plus a real round trip: the state of the `tasks` queue."""
        declare_ok = await self._declare_tasks_passive()
        return (
            f"{self.server_version()} (queue {TASKS_QUEUE.name}: "
            f"{declare_ok.message_count} ready, {declare_ok.consumer_count} consumers)"
        )
