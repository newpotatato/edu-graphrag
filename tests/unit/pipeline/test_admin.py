"""RabbitMQ admin helper: request definitions are pure; HTTP is replaced by a fake transport."""

import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest
from pydantic import SecretStr

from edu_graphrag.config import PipelineSettings, RabbitSettings
from edu_graphrag.pipeline import admin
from edu_graphrag.pipeline.admin import ApiRequest, ManagementClient
from edu_graphrag.pipeline.topology import tasks_policy


class FakeApi:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.requests: list[ApiRequest] = []
        self.responses = responses or {}

    def __call__(self, request: ApiRequest) -> Any:
        self.requests.append(request)
        return self.responses.get(request.path)


def rabbit(vhost: str = "/") -> RabbitSettings:
    return RabbitSettings(password=SecretStr("pw"), user="edu", host="rmq", vhost=vhost)


def test_declare_requests_cover_the_topology() -> None:
    requests = admin.declare_requests("/")
    paths = [(r.method, r.path) for r in requests]
    assert paths == [
        ("PUT", "/api/exchanges/%2F/tasks"),
        ("PUT", "/api/exchanges/%2F/tasks.dlx"),
        ("PUT", "/api/queues/%2F/tasks"),
        ("PUT", "/api/queues/%2F/tasks.dlq"),
        ("POST", "/api/bindings/%2F/e/tasks/q/tasks"),
        ("POST", "/api/bindings/%2F/e/tasks.dlx/q/tasks.dlq"),
    ]
    assert requests[0].body is not None
    assert requests[0].body["type"] == "direct"
    assert requests[1].body is not None
    assert requests[1].body["type"] == "fanout"
    # Queues carry only their type: limits live in the policy, x-arguments are immutable.
    assert requests[2].body == {
        "durable": True,
        "auto_delete": False,
        "arguments": {"x-queue-type": "quorum"},
    }
    assert requests[4].body == {"routing_key": "tasks", "arguments": {}}


def test_policy_bounded() -> None:
    request = admin.policy_request("/", max_length=50, delivery_limit=3)
    assert request.method == "PUT"
    assert request.path == "/api/policies/%2F/tasks-limits"
    assert request.body is not None
    assert request.body["pattern"] == "^tasks$"
    assert request.body["apply-to"] == "queues"
    assert request.body["definition"] == {
        "delivery-limit": 3,
        "dead-letter-exchange": "tasks.dlx",
        "max-length": 50,
        "overflow": "reject-publish",
        "dead-letter-strategy": "at-least-once",
    }


def test_policy_unbounded_has_no_max_length() -> None:
    definition = tasks_policy(max_length=0, delivery_limit=5)["definition"]
    assert isinstance(definition, dict)
    assert "max-length" not in definition
    assert definition["delivery-limit"] == 5


def test_summarize_queue_and_nodes() -> None:
    raw = {
        "messages": 7,
        "messages_ready": 3,
        "messages_unacknowledged": 4,
        "consumers": 2,
        "policy": "tasks-limits",
        "effective_policy_definition": {"max-length": 200},
        "message_stats": {"publish_details": {"rate": 12.5}, "ack_details": {"rate": 10.0}},
    }
    summary = admin.summarize_queue(raw)
    assert summary["messages_unacknowledged"] == 4
    assert summary["publish_rate"] == 12.5
    assert summary["deliver_rate"] == 0.0
    assert summary["effective_policy_definition"] == {"max-length": 200}
    assert admin.summarize_queue({})["messages"] == 0
    nodes = admin.summarize_nodes([{"name": "rabbit@x", "mem_used": 1, "mem_limit": 2}])
    assert nodes == [{"name": "rabbit@x", "mem_used": 1, "mem_limit": 2, "mem_alarm": None}]


def test_client_declare_and_stats() -> None:
    api = FakeApi(
        {
            "/api/queues/%2F/tasks": {"messages": 5},
            "/api/queues/%2F/tasks.dlq": {"messages": 1},
            "/api/nodes": [{"name": "rabbit@rabbitmq", "mem_used": 100}],
        }
    )
    client = ManagementClient(rabbit(), transport=api)
    assert len(client.declare()) == 6
    stats = client.stats()
    assert stats["queues"]["tasks"]["messages"] == 5
    assert stats["queues"]["tasks.dlq"]["messages"] == 1
    assert stats["nodes"][0]["mem_used"] == 100


def test_cli_apply_policy_defaults_from_settings(capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi()
    client = ManagementClient(rabbit(), transport=api)
    pipeline = PipelineSettings(queue_max_length=42, delivery_limit=2)
    assert admin.main(["apply-policy"], client=client, pipeline=pipeline) == 0
    assert api.requests[-1].body is not None
    assert api.requests[-1].body["definition"]["max-length"] == 42
    assert api.requests[-1].body["definition"]["delivery-limit"] == 2

    admin.main(["apply-policy", "--max-length", "0"], client=client, pipeline=pipeline)
    assert api.requests[-1].body is not None
    assert "max-length" not in api.requests[-1].body["definition"]
    printed = capsys.readouterr().out.strip().splitlines()
    assert json.loads(printed[-1])["definition"]["delivery-limit"] == 2


def test_cli_rejects_bad_values() -> None:
    client = ManagementClient(rabbit(), transport=FakeApi())
    with pytest.raises(SystemExit):
        admin.main(["apply-policy", "--max-length", "-1"], client=client)


def test_cli_declare_and_stats(capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi({"/api/nodes": [], "/api/queues/%2F/tasks": {}, "/api/queues/%2F/tasks.dlq": {}})
    client = ManagementClient(rabbit(), transport=api)
    admin.main(["declare"], client=client)
    admin.main(["stats"], client=client)
    out = capsys.readouterr().out
    assert "PUT /api/queues/%2F/tasks" in out
    assert '"queues"' in out


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_urllib_transport_sends_auth_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = request.data
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ManagementClient(rabbit(vhost="edu"))
    assert client.send(ApiRequest("PUT", "/api/x", {"a": 1})) == {"ok": True}
    assert seen["url"] == "http://rmq:15672/api/x"
    assert seen["method"] == "PUT"
    assert seen["auth"].startswith("Basic ")
    assert json.loads(seen["body"]) == {"a": 1}

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse(b""))
    assert client.send(ApiRequest("POST", "/api/y", {})) is None


def test_urllib_transport_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            Message(),
            io.BytesIO(b"not_authorised"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="401 not_authorised"):
        ManagementClient(rabbit()).send(ApiRequest("GET", "/api/overview"))
