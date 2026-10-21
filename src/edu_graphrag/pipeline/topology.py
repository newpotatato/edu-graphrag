"""RabbitMQ topology shared by publishers and workers.

Queues are declared with their type only. Limits (max-length, overflow, delivery-limit, dead
lettering) come from a RabbitMQ policy (see `pipeline.admin`): queue x-arguments are immutable
and redeclaring with different values fails, while a policy can be changed at runtime.

    publisher --(routing key "tasks")--> exchange `tasks` (direct) --> quorum queue `tasks`
                                                                         | rejected / over
                                                                         | delivery-limit
                                                                         v
                                         exchange `tasks.dlx` (fanout) --> quorum queue `tasks.dlq`
"""

from faststream.rabbit import ExchangeType, QueueType, RabbitExchange, RabbitQueue

TASKS_ROUTING_KEY = "tasks"

TASKS_EXCHANGE = RabbitExchange("tasks", type=ExchangeType.DIRECT, durable=True)
DEAD_LETTER_EXCHANGE = RabbitExchange("tasks.dlx", type=ExchangeType.FANOUT, durable=True)

TASKS_QUEUE = RabbitQueue(
    "tasks",
    queue_type=QueueType.QUORUM,
    durable=True,
    routing_key=TASKS_ROUTING_KEY,
)
DEAD_LETTER_QUEUE = RabbitQueue("tasks.dlq", queue_type=QueueType.QUORUM, durable=True)

POLICY_NAME = "tasks-limits"


def tasks_policy(max_length: int, delivery_limit: int) -> dict[str, object]:
    """Policy definition for the `tasks` queue. `max_length=0` means unbounded (experiment
    baseline without backpressure)."""
    definition: dict[str, object] = {
        "delivery-limit": delivery_limit,
        "dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
    }
    if max_length > 0:
        # reject-publish: the broker nacks publishes to a full queue (the publisher's signal).
        # at-least-once dead lettering requires exactly this overflow mode.
        definition |= {
            "max-length": max_length,
            "overflow": "reject-publish",
            "dead-letter-strategy": "at-least-once",
        }
    return {
        "pattern": f"^{TASKS_QUEUE.name}$",
        "apply-to": "queues",
        "priority": 10,
        "definition": definition,
    }
