"""`Retry-After` for a 503 from POST /api/v1/tasks when the queue refused the task.

    retry_after = clamp( ceil( FRACTION * queue_max_length * t_task / (consumers * concurrency) ),
                         1, MAX_RETRY_AFTER )

* `queue_max_length * FRACTION` - how many tasks must be finished before the queue accepts
  again. A quorum queue with `overflow: reject-publish` does not reopen on the first free slot:
  it keeps refusing until it is drained to 80 % of max-length (measured on RabbitMQ 4.2.9:
  max-length 10, publishes accepted again only after the depth fell from 11 to 8). So
  FRACTION = 0.2 is not a guess: any retry earlier than that is guaranteed another 503.
* `consumers * concurrency / t_task` - aggregate worker throughput, tasks per second.
  `consumers` is read from the broker (passive declare of `tasks`, only on the 503 path),
  `concurrency` is WORKER_CONCURRENCY, `t_task` is about INFERENCE_DELAY_MS (the dominant,
  deliberately fixed cost; real inference adds a few ms, so the estimate is a lower bound).

Example with the defaults (max-length 200, 4 threads, 200 ms) and one worker:
0.2 * 200 * 0.2 / (1 * 4) = 2 s; with two workers 1 s.

Special cases: no consumers -> nothing drains, the client is told to come back much later
(MAX_RETRY_AFTER); broker unreachable -> the API's reconnect interval (it cannot accept
anything before it has reconnected).
"""

import math

# Quorum queues resume accepting publishes at 80 % of max-length.
QUORUM_RESUME_FRACTION = 0.2
MAX_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1


def retry_after_seconds(
    *,
    queue_max_length: int,
    task_seconds: float,
    consumers: int,
    concurrency: int,
    fraction: float = QUORUM_RESUME_FRACTION,
) -> int:
    if consumers <= 0:
        return MAX_RETRY_AFTER
    throughput = consumers * max(concurrency, 1) / max(task_seconds, 1e-3)
    # round() first: float noise must not turn an exact 2.0 into 3 s.
    wait = math.ceil(round(fraction * queue_max_length / throughput, 6))
    return max(MIN_RETRY_AFTER, min(MAX_RETRY_AFTER, wait))
