# ruff: noqa: RUF001
"""Generate the checked-in Grafana dashboard. Run from the repository root."""

import json
from pathlib import Path
from typing import Any

PROM = {"type": "prometheus", "uid": "edu-prometheus"}
LOKI = {"type": "loki", "uid": "edu-loki"}
COLORS = ("#3987e5", "#d95926", "#199e70")
TASKS = '{queue="tasks"}'
WORKER = '{instance=~"${worker:regex}"}'
DONE = '{outcome="ok",instance=~"${worker:regex}"}'


def timeseries(
    panel_id: int,
    title: str,
    unit: str,
    targets: list[tuple[str, str]],
    description: str,
    *,
    dashed: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "description": description,
        "gridPos": {
            "x": 12 * ((panel_id - 1) % 2),
            "y": 8 * ((panel_id - 1) // 2),
            "w": 12,
            "h": 8,
        },
        "datasource": PROM,
        "targets": [
            {"refId": chr(65 + i), "expr": expr, "legendFormat": legend, "datasource": PROM}
            for i, (expr, legend) in enumerate(targets)
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": 0,
                "color": {"mode": "palette-classic-by-name"},
                "custom": {
                    "lineWidth": 2,
                    "fillOpacity": 0,
                    "showPoints": "never",
                    "spanNulls": False,
                },
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": label},
                    "properties": [
                        {"id": "color", "value": {"mode": "fixed", "fixedColor": COLORS[i]}},
                        *(
                            [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [7, 5]}}]
                            if label in dashed
                            else []
                        ),
                    ],
                }
                for i, (_, label) in enumerate(targets)
                if "{{" not in label
            ],
        },
        "options": {
            "legend": {
                "showLegend": len(targets) > 1 or "{{" in targets[0][1],
                "displayMode": "table",
                "placement": "bottom",
                "calcs": ["lastNotNull"],
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def quantiles(metric: str, labels: str = "") -> list[tuple[str, str]]:
    return [
        (
            f"histogram_quantile({q}, sum by (le) (rate({metric}_bucket{labels}[$__rate_interval])))",
            name,
        )
        for q, name in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99"))
    ]


def dashboard() -> dict[str, Any]:
    panels = [
        timeseries(
            1,
            "Очередь: готовые, неподтверждённые и лимит",
            "short",
            [
                (f"max(rabbitmq_queue_messages_ready{TASKS})", "Ready"),
                (f"max(rabbitmq_queue_messages_unacked{TASKS})", "Unacked"),
                (f"max(edu_queue_limit{TASKS})", "Лимит ready"),
            ],
            "Лимит относится к ready; unacked считаются отдельно.",
            dashed=("Лимит ready",),
        ),
        timeseries(
            2,
            "Публикации и завершения",
            "suffix: задач/с",
            [
                (
                    "sum(rate(edu_source_published_total[$__rate_interval]))",
                    "Источник: подтверждено",
                ),
                (
                    'sum(rate(edu_worker_tasks_total{outcome="ok"}[$__rate_interval]))',
                    "Воркеры: завершено",
                ),
            ],
            "Сравнивайте в устойчивом режиме, когда глубина очереди не растёт.",
        ),
        timeseries(
            3,
            "Завершения по воркерам",
            "suffix: задач/с",
            [
                (
                    f"sum by (instance) (rate(edu_worker_tasks_total{DONE}[$__rate_interval]))",
                    "{{instance}}",
                ),
            ],
            "Только первая успешно сохранённая запись задачи.",
        ),
        timeseries(
            4,
            "Задачи в работе по воркерам",
            "short",
            [
                (f"edu_worker_in_flight{WORKER}", "{{instance}}"),
            ],
            "Gauge, не счётчик. На реплике ограничен prefetch/concurrency.",
        ),
        timeseries(
            5,
            "Длительность inference",
            "s",
            quantiles("edu_worker_inference_duration_seconds", WORKER),
            "Перцентили считаются по объединённым бакетам выбранных воркеров.",
        ),
        timeseries(
            6,
            "Длительность POST /tasks",
            "s",
            quantiles("edu_api_request_duration_seconds", '{route="/api/v1/tasks",method="POST"}'),
            "Время HTTP-ответа, включая отказы 503; это не время выполнения задачи.",
        ),
        timeseries(
            7,
            "Память процессов (RSS)",
            "bytes",
            [
                (
                    'edu_process_rss_bytes{job=~"edu-source|edu-worker",instance=~"${worker:regex}"} or edu_process_rss_bytes{job="edu-source"}',
                    "{{job}} · {{instance}}",
                ),
            ],
            "Последнее наблюдение источника исчезает по lookback, когда процесс завершился.",
        ),
        timeseries(
            8,
            "Доля 503 на POST /tasks",
            "percentunit",
            [
                (
                    '(sum(rate(edu_api_requests_total{route="/api/v1/tasks",method="POST",status_code="503"}[$__rate_interval])) or vector(0)) / sum(rate(edu_api_requests_total{route="/api/v1/tasks",method="POST"}[$__rate_interval]))',
                    "503",
                ),
            ],
            "Без запросов знаменателя нет; 503 возможен и по причинам, отличным от переполнения.",
        ),
        timeseries(
            9,
            "NACK источника",
            "suffix: отказов/с",
            [
                (
                    'sum(rate(edu_source_publish_refusals_total{reason="PublishRejectedError"}[$__rate_interval]))',
                    "Broker NACK",
                ),
            ],
            "Не включает таймаут publisher confirm и отключение брокера.",
        ),
        timeseries(
            10,
            "Необработанные сообщения в DLQ",
            "short",
            [
                ('max(rabbitmq_queue_messages{queue="tasks.dlq"})', "DLQ"),
            ],
            "Gauge брокера. Диагностический подписчик не опустошает tasks.dlq.",
        ),
        timeseries(
            11,
            "Время задачи до сохранения",
            "s",
            quantiles("edu_worker_task_duration_seconds", WORKER),
            "От submitted_at, включает backoff/очередь. Только успешные результаты; не SLI полной когорты.",
        ),
        timeseries(
            12,
            "Сколько задач подтвердил источник",
            "short",
            [
                ("edu_source_published_total", "{{instance}}"),
            ],
            "Кумулятивный счётчик на запуск; виден и для пачки с одной точкой после flush.",
        ),
    ]
    panels.append(
        {
            "id": 13,
            "type": "logs",
            "title": "Последние ошибки и события DLQ",
            "gridPos": {"x": 0, "y": 48, "w": 24, "h": 9},
            "datasource": LOKI,
            "targets": [
                {
                    "refId": "A",
                    "expr": '{service=~"edu-(api|source|worker)",level=~"error|critical|warning"} | json | __error__="" | event=~"task_failed|poison_message|dead_letter_.*|task_shed|model_load_failed|.*unavailable"',
                    "datasource": LOKI,
                }
            ],
            "options": {
                "showTime": True,
                "showLabels": False,
                "enableLogDetails": True,
                "wrapLogMessage": True,
            },
        }
    )
    return {
        "uid": "edu-pipeline",
        "title": "Конвейер",
        "tags": ["edu-graphrag", "hw5"],
        "timezone": "browser",
        "schemaVersion": 41,
        "version": 1,
        "editable": False,
        "time": {"from": "now-15m", "to": "now"},
        "refresh": "5s",
        "panels": panels,
        "templating": {
            "list": [
                {
                    "name": "worker",
                    "label": "Воркер",
                    "type": "query",
                    "datasource": PROM,
                    "definition": "label_values(edu_worker_in_flight, instance)",
                    "query": {
                        "query": "label_values(edu_worker_in_flight, instance)",
                        "refId": "Worker",
                    },
                    "refresh": 2,
                    "includeAll": True,
                    "allValue": ".*",
                    "multi": True,
                    "current": {"text": "All", "value": "$__all"},
                }
            ]
        },
    }


def main() -> None:
    path = Path("observability/grafana/dashboards/pipeline.json")
    path.write_text(json.dumps(dashboard(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
