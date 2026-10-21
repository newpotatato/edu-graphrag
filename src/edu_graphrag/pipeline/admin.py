"""RabbitMQ administration through the Management HTTP API.

    python -m edu_graphrag.pipeline.admin declare
    python -m edu_graphrag.pipeline.admin apply-policy [--max-length N] [--delivery-limit N]
    python -m edu_graphrag.pipeline.admin stats

* `declare` creates the exchanges, queues and bindings of `pipeline.topology` (idempotent:
  PUT of an identical object and POST of an existing binding are no-ops).
* `apply-policy` (re)applies the `tasks` queue limits. A policy, unlike queue x-arguments,
  can be changed on a live queue: this is how the queue capacity is changed at runtime.
  Defaults come from QUEUE_MAX_LENGTH / DELIVERY_LIMIT. The worker reads DELIVERY_LIMIT too
  (to know its last attempt), so change it in `.env`, not only here.
* `stats` prints queue depths, consumer counts, rates and broker memory as JSON. It is for
  OBSERVATION by experiments and humans; no component uses it as a control signal (the
  publishers' backpressure signal is the broker's publisher-confirm nack).

Only the standard library is used (urllib): the runtime image has no httpx.
"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from faststream.rabbit import RabbitExchange, RabbitQueue

from edu_graphrag.config import PipelineSettings, RabbitSettings
from edu_graphrag.pipeline.topology import (
    DEAD_LETTER_EXCHANGE,
    DEAD_LETTER_QUEUE,
    POLICY_NAME,
    TASKS_EXCHANGE,
    TASKS_QUEUE,
    TASKS_ROUTING_KEY,
    tasks_policy,
)

OBSERVED_QUEUES = (TASKS_QUEUE.name, DEAD_LETTER_QUEUE.name)


@dataclass(frozen=True, slots=True)
class ApiRequest:
    method: str
    path: str
    body: dict[str, Any] | None = None


Transport = Callable[[ApiRequest], Any]


def _vhost(vhost: str) -> str:
    return quote(vhost, safe="")


def exchange_request(vhost: str, exchange: RabbitExchange) -> ApiRequest:
    return ApiRequest(
        "PUT",
        f"/api/exchanges/{_vhost(vhost)}/{quote(exchange.name, safe='')}",
        {
            "type": exchange.type.value,
            "durable": exchange.durable,
            "auto_delete": exchange.auto_delete,
            "internal": False,
            "arguments": {},
        },
    )


def queue_request(vhost: str, queue: RabbitQueue) -> ApiRequest:
    # Same arguments as FastStream declares (type only), otherwise redeclaring would fail
    # with PRECONDITION_FAILED. Limits come from the policy.
    return ApiRequest(
        "PUT",
        f"/api/queues/{_vhost(vhost)}/{quote(queue.name, safe='')}",
        {"durable": queue.durable, "auto_delete": False, "arguments": dict(queue.arguments or {})},
    )


def binding_request(
    vhost: str, exchange: RabbitExchange, queue: RabbitQueue, routing_key: str
) -> ApiRequest:
    return ApiRequest(
        "POST",
        f"/api/bindings/{_vhost(vhost)}/e/{quote(exchange.name, safe='')}"
        f"/q/{quote(queue.name, safe='')}",
        {"routing_key": routing_key, "arguments": {}},
    )


def declare_requests(vhost: str) -> list[ApiRequest]:
    return [
        exchange_request(vhost, TASKS_EXCHANGE),
        exchange_request(vhost, DEAD_LETTER_EXCHANGE),
        queue_request(vhost, TASKS_QUEUE),
        queue_request(vhost, DEAD_LETTER_QUEUE),
        binding_request(vhost, TASKS_EXCHANGE, TASKS_QUEUE, TASKS_ROUTING_KEY),
        # Fanout: the routing key is ignored, dead letters keep their original one.
        binding_request(vhost, DEAD_LETTER_EXCHANGE, DEAD_LETTER_QUEUE, ""),
    ]


def policy_request(vhost: str, *, max_length: int, delivery_limit: int) -> ApiRequest:
    return ApiRequest(
        "PUT",
        f"/api/policies/{_vhost(vhost)}/{quote(POLICY_NAME, safe='')}",
        tasks_policy(max_length=max_length, delivery_limit=delivery_limit),
    )


def _rate(stats: dict[str, Any], key: str) -> float:
    details = stats.get(f"{key}_details")
    return float(details.get("rate", 0.0)) if isinstance(details, dict) else 0.0


def summarize_queue(raw: dict[str, Any]) -> dict[str, Any]:
    message_stats = raw.get("message_stats") or {}
    return {
        "messages": raw.get("messages", 0),
        "messages_ready": raw.get("messages_ready", 0),
        "messages_unacknowledged": raw.get("messages_unacknowledged", 0),
        "consumers": raw.get("consumers", 0),
        "publish_rate": _rate(message_stats, "publish"),
        "deliver_rate": _rate(message_stats, "deliver_get"),
        "ack_rate": _rate(message_stats, "ack"),
        "redeliver_rate": _rate(message_stats, "redeliver"),
        "policy": raw.get("policy"),
        "effective_policy_definition": raw.get("effective_policy_definition") or {},
    }


def summarize_nodes(raw: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": node.get("name"),
            "mem_used": node.get("mem_used"),
            "mem_limit": node.get("mem_limit"),
            "mem_alarm": node.get("mem_alarm"),
        }
        for node in raw
    ]


class ManagementClient:
    def __init__(self, settings: RabbitSettings, *, transport: Transport | None = None) -> None:
        self._settings = settings
        self._base_url = f"http://{settings.host}:{settings.management_port}"
        self._transport = transport or self._urllib_transport

    def _urllib_transport(self, request: ApiRequest) -> Any:
        credentials = f"{self._settings.user}:{self._settings.password.get_secret_value()}"
        data = None if request.body is None else json.dumps(request.body).encode()
        http_request = urllib.request.Request(  # noqa: S310 - fixed http scheme
            self._base_url + request.path,
            data=data,
            method=request.method,
            headers={
                "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode(),
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=10) as response:  # noqa: S310
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{request.method} {request.path}: {exc.code} {detail}") from exc
        return json.loads(payload) if payload else None

    def send(self, request: ApiRequest) -> Any:
        return self._transport(request)

    def declare(self) -> list[str]:
        done = []
        for request in declare_requests(self._settings.vhost):
            self.send(request)
            done.append(f"{request.method} {request.path}")
        return done

    def apply_policy(self, *, max_length: int, delivery_limit: int) -> dict[str, Any]:
        request = policy_request(
            self._settings.vhost, max_length=max_length, delivery_limit=delivery_limit
        )
        self.send(request)
        return request.body or {}

    def stats(self) -> dict[str, Any]:
        vhost = _vhost(self._settings.vhost)
        queues = {
            name: summarize_queue(
                self.send(ApiRequest("GET", f"/api/queues/{vhost}/{quote(name, safe='')}"))
            )
            for name in OBSERVED_QUEUES
        }
        nodes = summarize_nodes(self.send(ApiRequest("GET", "/api/nodes")))
        return {"queues": queues, "nodes": nodes}


def main(
    argv: Sequence[str] | None = None,
    *,
    client: ManagementClient | None = None,
    pipeline: PipelineSettings | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m edu_graphrag.pipeline.admin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("declare", help="declare exchanges, queues and bindings")
    policy = commands.add_parser("apply-policy", help="apply the tasks queue policy")
    policy.add_argument("--max-length", type=int, default=None, help="0 = unbounded")
    policy.add_argument("--delivery-limit", type=int, default=None)
    commands.add_parser("stats", help="queue and broker statistics as JSON")
    args = parser.parse_args(argv)

    client = client or ManagementClient(RabbitSettings())
    if args.command == "declare":
        for line in client.declare():
            print(line)
    elif args.command == "apply-policy":
        pipeline = pipeline or PipelineSettings()
        max_length = pipeline.queue_max_length if args.max_length is None else args.max_length
        delivery_limit = (
            pipeline.delivery_limit if args.delivery_limit is None else args.delivery_limit
        )
        if max_length < 0 or delivery_limit < 1:
            parser.error("--max-length must be >= 0 and --delivery-limit >= 1")
        print(json.dumps(client.apply_policy(max_length=max_length, delivery_limit=delivery_limit)))
    else:
        print(json.dumps(client.stats(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
