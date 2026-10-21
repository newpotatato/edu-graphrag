import asyncio
from typing import Any, cast

import aio_pika
import pytest
from aio_pika.exceptions import ChannelInvalidStateError, DeliveryError
from aio_pika.exceptions import PublishError as ReturnedError
from aiormq import spec
from aiormq.abc import DeliveredMessage
from pamqp.header import ContentHeader
from pydantic import SecretStr

from edu_graphrag.components import build_components, run_components
from edu_graphrag.components.neo4j import Neo4jComponent
from edu_graphrag.components.postgres import PostgresComponent, build_postgres_url
from edu_graphrag.components.rabbit import Connect, RabbitComponent
from edu_graphrag.config import Neo4jSettings, PostgresSettings, RabbitSettings, Settings
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    BrokerUnavailableError,
    PublishRejectedError,
    PublishTimeoutError,
    PublishUnroutableError,
)
from tests.fakes import FakeComponent, FakeConnect, FakeConnection, FakeDeclareOk


async def test_components_start_in_order_and_stop_in_reverse() -> None:
    events: list[str] = []
    components = [FakeComponent("a", events=events), FakeComponent("b", events=events)]

    async with run_components(components):
        assert events == ["start:a", "start:b"]

    assert events == ["start:a", "start:b", "stop:b", "stop:a"]


async def test_started_components_are_stopped_if_later_startup_fails() -> None:
    events: list[str] = []
    components = [
        FakeComponent("a", events=events),
        FakeComponent("b", events=events, startup_error=ValueError("bad config")),
    ]

    with pytest.raises(ValueError, match="bad config"):
        async with run_components(components):
            pytest.fail("must not get here")

    assert events == ["start:a", "stop:a"]


def test_build_components_registers_postgres_neo4j_and_rabbitmq(settings: Settings) -> None:
    # The model is served by the workers now, not by the API process.
    assert [c.name for c in build_components(settings)] == ["postgres", "neo4j", "rabbitmq"]


def test_postgres_url_escapes_password_and_uses_async_driver() -> None:
    url = build_postgres_url(
        PostgresSettings(host="db", port=5433, user="u", password=SecretStr("p@ss/word"), db="d")
    )

    assert url.drivername == "postgresql+asyncpg"
    assert url.render_as_string(hide_password=False) == (
        "postgresql+asyncpg://u:p%40ss%2Fword@db:5433/d"
    )
    assert "p@ss" not in str(url)


async def test_postgres_component_lifecycle_without_database() -> None:
    component = PostgresComponent(PostgresSettings(password=SecretStr("x")), connect_timeout=1)
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.engine

    await component.startup()  # lazy: must succeed even though no database is running
    assert component.engine is not None

    await component.shutdown()
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.engine


async def test_neo4j_component_lifecycle_without_database() -> None:
    component = Neo4jComponent(Neo4jSettings(password=SecretStr("x")), connect_timeout=1)
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.driver

    await component.startup()
    assert component.driver is not None

    await component.shutdown()
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.driver


# --- RabbitMQ -----------------------------------------------------------------------------


def make_rabbit(
    connect: FakeConnect | None = None, **kwargs: Any
) -> tuple[RabbitComponent, FakeConnect]:
    connect = connect or FakeConnect()
    settings = RabbitSettings(host="mq", user="u", password=SecretStr("p@ss"))
    component = RabbitComponent(
        settings, connect_timeout=1, connect=cast(Connect, connect), **kwargs
    )
    return component, connect


async def test_rabbit_startup_opens_a_confirm_channel_without_declaring() -> None:
    rabbit, connect = make_rabbit()

    await rabbit.startup()

    assert rabbit.is_connected
    (call,) = connect.calls
    assert call == {"url": "amqp://u:p%40ss@mq:5672/%2F", "timeout": 1}
    (channel,) = connect.connection.channels
    assert channel.kwargs == {"publisher_confirms": True, "on_return_raises": True}
    assert connect.connection.exchange is not None
    assert connect.connection.exchange.name == "tasks"

    await rabbit.shutdown()
    assert connect.connection.closed
    assert not rabbit.is_connected


async def test_rabbit_publish_sends_a_persistent_json_message() -> None:
    rabbit, connect = make_rabbit()
    await rabbit.startup()
    message = TaskMessage.new("hello", source="api")

    await rabbit.publish(message, confirm_timeout=0.5)

    assert connect.connection.exchange is not None
    ((amqp, routing_key, kwargs),) = connect.connection.exchange.published
    assert routing_key == "tasks"
    assert kwargs == {"mandatory": True, "timeout": 0.5}
    assert amqp.message_id == str(message.task_id)
    assert amqp.content_type == "application/json"
    assert amqp.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert TaskMessage.model_validate_json(amqp.body) == message


def _nack() -> DeliveryError:
    return DeliveryError(None, spec.Basic.Nack(delivery_tag=1))


def _returned() -> ReturnedError:
    frame = spec.Basic.Return(reply_code=312, reply_text="NO_ROUTE")
    delivered = DeliveredMessage(
        delivery=frame,
        header=ContentHeader(),
        body=b"",
        channel=None,  # type: ignore[arg-type]
    )
    return ReturnedError(delivered, frame)


@pytest.mark.parametrize(
    ("broker_error", "expected"),
    [
        (_nack(), PublishRejectedError),
        (_returned(), PublishUnroutableError),
        (TimeoutError(), PublishTimeoutError),
        (ChannelInvalidStateError("closed"), BrokerUnavailableError),
        (ConnectionResetError("reset"), BrokerUnavailableError),
    ],
    ids=["nack", "return", "timeout", "channel-closed", "connection-reset"],
)
async def test_rabbit_publish_maps_broker_answers(
    broker_error: BaseException, expected: type[Exception]
) -> None:
    rabbit, connect = make_rabbit()
    await rabbit.startup()
    assert connect.connection.exchange is not None
    connect.connection.exchange.error = broker_error

    with pytest.raises(expected):
        await rabbit.publish(TaskMessage.new("x", source="api"), confirm_timeout=0.5)


async def test_rabbit_publish_is_bounded_by_the_timeout() -> None:
    rabbit, connect = make_rabbit()
    await rabbit.startup()
    assert connect.connection.exchange is not None
    connect.connection.exchange.delay = 5

    with pytest.raises(PublishTimeoutError):
        await rabbit.publish(TaskMessage.new("x", source="api"), confirm_timeout=0.05)


async def wait_connected(rabbit: RabbitComponent) -> None:
    for _ in range(200):
        if rabbit.is_connected:
            return
        await asyncio.sleep(0.01)
    pytest.fail("did not reconnect")


async def test_rabbit_degraded_start_and_background_reconnect() -> None:
    rabbit, connect = make_rabbit(FakeConnect(failures=2), reconnect_interval=0.01)

    await rabbit.startup()  # must not raise: the app starts without the broker

    assert not rabbit.is_connected
    with pytest.raises(BrokerUnavailableError):
        await rabbit.publish(TaskMessage.new("x", source="api"), confirm_timeout=0.5)
    assert await rabbit.consumer_count() is None
    with pytest.raises(BrokerUnavailableError):
        await rabbit.fetch_version()
    with pytest.raises(BrokerUnavailableError):
        rabbit.server_version()

    await wait_connected(rabbit)
    assert len(connect.calls) == 3
    await rabbit.shutdown()


async def test_rabbit_shutdown_stops_reconnecting() -> None:
    rabbit, connect = make_rabbit(FakeConnect(failures=10**6), reconnect_interval=0.01)
    await rabbit.startup()
    await asyncio.sleep(0.05)

    await rabbit.shutdown()
    calls = len(connect.calls)
    await asyncio.sleep(0.05)

    assert len(connect.calls) == calls


async def test_rabbit_closes_the_connection_if_the_channel_fails() -> None:
    connect = FakeConnect(FakeConnection(channel_error=ConnectionError("boom")))
    rabbit, _ = make_rabbit(connect, reconnect_interval=10)

    await rabbit.startup()

    assert not rabbit.is_connected
    assert connect.connection.closed
    await rabbit.shutdown()


async def test_rabbit_fire_and_forget_channel_has_no_confirms() -> None:
    rabbit, connect = make_rabbit(publisher_confirms=False)

    await rabbit.startup()

    assert connect.connection.channels[0].kwargs["publisher_confirms"] is False


@pytest.mark.parametrize("version", ["4.2.9", b"4.2.9"])
async def test_rabbit_fetch_version_does_a_passive_declare(version: object) -> None:
    connect = FakeConnect(FakeConnection(version=version))
    connect.connection.raw.declare_ok = FakeDeclareOk(message_count=12, consumer_count=2)
    rabbit, _ = make_rabbit(connect)
    await rabbit.startup()

    reported = await rabbit.fetch_version()

    assert reported == "4.2.9 (queue tasks: 12 ready, 2 consumers)"
    assert connect.connection.raw.declared == [("tasks", True)]
    # The declare runs on a throwaway channel, which is closed afterwards.
    assert connect.connection.channels[-1].closed
    assert not connect.connection.channels[0].closed


async def test_rabbit_consumer_count() -> None:
    connect = FakeConnect()
    connect.connection.raw.declare_ok = FakeDeclareOk(consumer_count=3)
    rabbit, _ = make_rabbit(connect)
    await rabbit.startup()

    assert await rabbit.consumer_count() == 3

    connect.connection.raw.error = ConnectionError("404 NOT_FOUND")
    assert await rabbit.consumer_count() is None
