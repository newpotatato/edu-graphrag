from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from pydantic import SecretStr

from edu_graphrag.api.task_service import (
    API_CONFIRM_TIMEOUT,
    STORE_RETRY_AFTER,
    PostgresTaskStore,
    StoreUnavailableError,
    TaskService,
)
from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.components.rabbit import RabbitComponent
from edu_graphrag.config import PipelineSettings, PostgresSettings, Settings
from edu_graphrag.main import build_task_service
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    BrokerUnavailableError,
    PublishRejectedError,
    PublishTimeoutError,
)
from tests.conftest import ClientFactory
from tests.fakes import FakeComponent, FakeTaskPublisher, InMemoryTaskStore


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def task_store() -> InMemoryTaskStore:
    return InMemoryTaskStore()


@pytest.fixture
def publisher() -> FakeTaskPublisher:
    return FakeTaskPublisher(consumers=1)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(
    task_store: InMemoryTaskStore, publisher: FakeTaskPublisher, clock: Clock
) -> TaskService:
    # Defaults: max-length 200, 4 threads per worker, 200 ms per task.
    return TaskService(
        task_store, publisher, pipeline=PipelineSettings(), broker_retry_after=5, clock=clock
    )


@pytest.fixture
async def api(make_client: ClientFactory, service: TaskService) -> AsyncIterator[AsyncClient]:
    async with make_client([FakeComponent("postgres")], service) as client:
        yield client


# --- POST /api/v1/tasks: accepted ---------------------------------------------------------


async def test_accepted_task_is_committed_published_and_located(
    api: AsyncClient, task_store: InMemoryTaskStore, publisher: FakeTaskPublisher
) -> None:
    response = await api.post("/api/v1/tasks", json={"text": "квадратное уравнение"})

    assert response.status_code == 202
    body = response.json()
    task_id = UUID(body["task_id"])
    assert body == {
        "task_id": str(task_id),
        "status": "pending",
        "status_url": f"/api/v1/tasks/{task_id}",
    }
    assert response.headers["Location"] == f"/api/v1/tasks/{task_id}"
    assert task_store.commits == 1
    assert task_store.rows[task_id].status == "pending"
    (message,) = publisher.published
    assert message.task_id == task_id
    assert message.source == "api"
    assert publisher.timeouts == [API_CONFIRM_TIMEOUT], "a single short attempt"


async def test_location_can_be_polled(api: AsyncClient) -> None:
    created = await api.post("/api/v1/tasks", json={"text": "x"})

    response = await api.get(created.headers["Location"])

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert response.json()["finished_at"] is None
    assert response.json()["processing_ms"] is None


async def test_confirm_timeout_never_exceeds_the_configured_one(
    task_store: InMemoryTaskStore, publisher: FakeTaskPublisher
) -> None:
    service = TaskService(
        task_store, publisher, pipeline=PipelineSettings(publish_confirm_timeout=0.1)
    )

    await service.submit("x")

    assert publisher.timeouts == [0.1]


# --- POST /api/v1/tasks: load shedding ----------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [PublishRejectedError("nack"), PublishTimeoutError("no confirm")],
    ids=["nack", "timeout"],
)
async def test_refused_publish_gives_503_with_retry_after_and_rolls_back(
    api: AsyncClient,
    task_store: InMemoryTaskStore,
    publisher: FakeTaskPublisher,
    error: Exception,
) -> None:
    publisher.outcomes = [error]

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.status_code == 503
    # 0.2 * 200 * 0.2 s / (1 consumer * 4 threads) = 2 s
    assert response.headers["Retry-After"] == "2"
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["status"] == 503
    assert body["title"] == "Task queue is full"
    assert body["retry_after"] == 2
    assert task_store.rows == {}, "no PENDING row after a 503"
    assert (task_store.commits, task_store.rollbacks) == (0, 1)
    assert len(publisher.attempts) == 1, "the request never retries or waits for space"


async def test_broker_down_gives_503_with_reconnect_interval(
    api: AsyncClient, task_store: InMemoryTaskStore, publisher: FakeTaskPublisher
) -> None:
    publisher.outcomes = [BrokerUnavailableError("not connected")]

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["title"] == "Task queue is unavailable"
    assert task_store.rows == {}
    assert publisher.consumer_queries == 0


async def test_retry_after_scales_with_workers(
    api: AsyncClient, publisher: FakeTaskPublisher
) -> None:
    publisher.consumers = 4
    publisher.outcomes = [PublishRejectedError("nack")]

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.headers["Retry-After"] == "1"


async def test_retry_after_without_consumers_is_long(
    api: AsyncClient, publisher: FakeTaskPublisher
) -> None:
    publisher.consumers = 0
    publisher.outcomes = [PublishRejectedError("nack")]

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.headers["Retry-After"] == "60"


async def test_consumer_count_is_cached_on_the_503_path(
    api: AsyncClient, publisher: FakeTaskPublisher, clock: Clock
) -> None:
    publisher.outcomes = [PublishRejectedError("nack")] * 3

    await api.post("/api/v1/tasks", json={"text": "x"})
    await api.post("/api/v1/tasks", json={"text": "x"})
    assert publisher.consumer_queries == 1

    clock.now += 1.5
    await api.post("/api/v1/tasks", json={"text": "x"})
    assert publisher.consumer_queries == 2


@pytest.mark.parametrize(("consumers", "delay"), [(None, 0.0), (4, 1.0)], ids=["unknown", "slow"])
async def test_unknown_consumer_count_assumes_one_worker(
    service: TaskService, publisher: FakeTaskPublisher, consumers: int | None, delay: float
) -> None:
    publisher.consumers = consumers
    publisher.consumer_delay = delay

    assert await service.retry_after(PublishRejectedError("nack")) == 2


async def test_database_down_gives_503_without_publishing(
    api: AsyncClient, task_store: InMemoryTaskStore, publisher: FakeTaskPublisher
) -> None:
    task_store.insert_error = StoreUnavailableError("refused")

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(STORE_RETRY_AFTER)
    assert response.json()["title"] == "Task store is unavailable"
    assert publisher.attempts == []


async def test_commit_failure_after_confirm_still_accepts(
    api: AsyncClient, task_store: InMemoryTaskStore, publisher: FakeTaskPublisher
) -> None:
    # The broker owns the task now; the worker's upsert will create the row.
    task_store.commit_error = StoreUnavailableError("connection lost during COMMIT")

    response = await api.post("/api/v1/tasks", json={"text": "x"})

    assert response.status_code == 202
    assert len(publisher.published) == 1
    assert task_store.rows == {}


# --- validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 20_001}, {"text": 1}, {"text": "x", "a": 1}],
    ids=["missing", "empty", "blank", "too-long", "not-str", "extra"],
)
async def test_invalid_body_is_422(
    api: AsyncClient, publisher: FakeTaskPublisher, body: dict[str, object]
) -> None:
    response = await api.post("/api/v1/tasks", json=body)

    assert response.status_code == 422
    assert publisher.attempts == []


async def test_text_limit_is_inclusive(api: AsyncClient) -> None:
    response = await api.post("/api/v1/tasks", json={"text": "x" * 20_000})

    assert response.status_code == 202


# --- GET /api/v1/tasks/{id} ---------------------------------------------------------------


async def test_done_task_shows_result_and_timings(
    api: AsyncClient, task_store: InMemoryTaskStore, service: TaskService
) -> None:
    message = await service.submit("x")
    pending = task_store.rows[message.task_id]
    task_store.rows[message.task_id] = replace(
        pending,
        status="done",
        result={"label": "math", "confidence": 0.9},
        model_name="subject-classifier",
        model_version="3",
        attempts=1,
        finished_at=pending.submitted_at + timedelta(milliseconds=250),
    )

    response = await api.get(f"/api/v1/tasks/{message.task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["result"] == {"label": "math", "confidence": 0.9}
    assert body["model_version"] == "3"
    assert body["source"] == "api"
    assert body["processing_ms"] == 250.0


async def test_unknown_task_is_404(api: AsyncClient) -> None:
    response = await api.get(f"/api/v1/tasks/{uuid4()}")

    assert response.status_code == 404


async def test_malformed_task_id_is_422(api: AsyncClient) -> None:
    assert (await api.get("/api/v1/tasks/not-a-uuid")).status_code == 422


async def test_get_with_database_down_is_503(
    api: AsyncClient, task_store: InMemoryTaskStore
) -> None:
    task_store.get_error = StoreUnavailableError("refused")

    response = await api.get(f"/api/v1/tasks/{uuid4()}")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(STORE_RETRY_AFTER)


# --- wiring -------------------------------------------------------------------------------


async def test_tasks_are_503_when_pipeline_is_not_configured(client: AsyncClient) -> None:
    # The default `components` fixture has neither Postgres nor RabbitMQ.
    response = await client.post("/api/v1/tasks", json={"text": "x"})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


async def test_old_synchronous_endpoints_are_gone(api: AsyncClient) -> None:
    assert (await api.post("/api/v1/process", json={"texts": ["x"]})).status_code == 404
    assert (await api.get("/api/v1/model")).status_code == 404


async def test_openapi_documents_202_and_503(api: AsyncClient) -> None:
    spec = (await api.get("/openapi.json")).json()

    post = spec["paths"]["/api/v1/tasks"]["post"]
    assert {"202", "422", "503"} <= set(post["responses"])
    assert "Retry-After" in post["responses"]["503"]["headers"]
    assert "Location" in post["responses"]["202"]["headers"]
    assert "/api/v1/tasks/{task_id}" in spec["paths"]


def test_build_task_service_needs_postgres_and_rabbit(settings: Settings) -> None:
    postgres = PostgresComponent(settings.postgres, connect_timeout=1)
    rabbit = RabbitComponent(settings.rabbit, connect_timeout=1)

    assert build_task_service(settings, [postgres, rabbit]) is not None
    assert build_task_service(settings, [postgres]) is None
    assert build_task_service(settings, [rabbit]) is None


async def test_postgres_store_wraps_database_errors() -> None:
    postgres = PostgresComponent(
        PostgresSettings(
            host="127.0.0.1", port=5432, user="nobody", password=SecretStr("wrong"), db="x"
        ),
        connect_timeout=2,
    )
    await postgres.startup()
    store = PostgresTaskStore(postgres)
    try:
        with pytest.raises(StoreUnavailableError):
            async with store.transaction() as tx:
                await tx.insert_pending(TaskMessage.new("x", source="api"))
        with pytest.raises(StoreUnavailableError):
            await store.get(uuid4())
    finally:
        await postgres.shutdown()
