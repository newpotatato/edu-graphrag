# ДЗ 5. Наблюдаемость конвейера

| Сигнал | Путь | Где смотреть |
|---|---|---|
| Логи | API / source / worker → JSON stdout → Alloy → Loki | Grafana → Explore → Loki |
| Трейсы | OTel SDK → OTLP/HTTP → Collector → Tempo | Grafana → Explore → Tempo |
| Метрики приложения | OTel SDK → Collector → OTLP receiver Prometheus | Grafana → Dashboards → Конвейер |
| Метрики очередей | RabbitMQ Prometheus plugin → `/metrics/per-object` :15692 → Prometheus | Ready / Unacked / DLQ на дашборде |

Collector применяет `memory_limiter` **до** `batch`. Экспорт идёт в фоне с ограниченными очередями и таймаутами. Потеря телеметрии не отменяет подтверждения брокера или коммита PostgreSQL. При обычном `just up` SDK выключен по умолчанию. Настройки дашборда, источников данных и правил алертов хранятся в `observability/`; импортировать их вручную не нужно.

## Запуск из PowerShell

Команды ниже предназначены для локального Docker Desktop. Из корня репозитория выполните `just env`; если `.env` создан впервые, заполните поля локальных паролей и S3-ключей. `docker info --format '{{.OSType}}'` должен вернуть `linux`.

| Команда | Ожидаемый результат |
|---|---|
| `just up` | Бизнес-стек без мониторинга. Для воркера нужен настроенный алиас в Registry. |
| `just obs up logs` | Alloy + Loki, без SDK-экспорта. |
| `just obs up traces` | Collector + Tempo; только экспорт трейсов. |
| `just obs up metrics` | Collector + Prometheus; экспорт метрик и скрейп RabbitMQ. |
| `just obs up observability` | Все сигналы и Grafana (`127.0.0.1:3000` по умолчанию). Используется существующий Registry. |
| `just obs demo-up` | Отдельный проект `edu-graphrag-obs-demo`, со своими очередью и томами; API `18000`, Grafana `13000`, MLflow `15000`, воркеры `18101–18104`. Заполняет Registry синтетической моделью `demo-subject-classifier@champion`. |

`just obs config observability` проверяет конфигурацию без запуска. Для ручных команд используйте `just obs compose observability …` или `just obs demo …`. Эти обёртки задают и профиль Compose, и параметры SDK: один `--profile` экспорт не включает. Если первый запуск требует установки зависимостей для генератора пачки и тестов, выполните `uv sync --locked`.

После `just obs demo-up` откройте Grafana на <http://127.0.0.1:13000/d/edu-pipeline> или найдите «Конвейер» в Dashboards. Вход: `admin` и `GRAFANA_ADMIN_PASSWORD` из `.env` (локальный дефолт в `.env.example`). `just obs demo ps` покажет контейнеры. `Invoke-RestMethod http://127.0.0.1:18101/ready` должен показать загруженную модель и подключённый брокер; если порт первой реплики другой, возьмите его из `ps`. `Invoke-RestMethod http://127.0.0.1:18000/healthz` проверяет API без обращения к БД.

Остановка демо: `just obs demo down`, без удаления томов. Чтобы вернуться с мониторинга на базовый режим основного проекта: `just obs compose observability down`, затем `just up`. Если переключаете профили на одном проекте, сначала остановите прежний профиль. `stop` одного Collector симулирует отказ, а не выключает SDK. Не используйте `down -v`, `just clean` или `scripts/experiment.py reset` на рабочем наборе данных.

## 1. Одна задача — один трейс

```powershell
$api = 'http://127.0.0.1:18000'
$task = Invoke-RestMethod -Method Post -Uri "$api/api/v1/tasks" -ContentType 'application/json' -Body '{"text":"linear combination of basis vectors"}'
$task
Invoke-RestMethod "$api$($task.status_url)"
```

`POST` должен вернуть `202` с ID и `Location`. Повторите `GET`, пока статус не станет `done`. В Loki Explore найдите задачу (подставьте `$task.task_id`), раскройте запись и нажмите **Open trace**:

```logql
{service=~"edu-(api|source|worker)",env="demo"} | json | task_id="ТУТ_UUID"
```

В Tempo ожидается `POST /api/v1/tasks` → PRODUCER `publish` → CONSUMER `process` → `inference` и `db.save_result`. У `process` родитель — `publish` из W3C `traceparent`. `inference` создаётся в потоке с копией контекста; у него видны `model.alias` и фактически загруженная `model.version`. `db.save_result` закрывается после выхода из транзакции, ACK — только после возврата из обработчика. Из выбранного спана откройте **Logs for this span**: запрос ищет логи по активному `trace_id`. Для поиска в Tempo по ID задачи: `{ span.task.id = "ТУТ_UUID" }`. `GET` статуса — отдельный HTTP-запрос со своим трейcом, но тем же `task_id`.

- `task_id` идентифицирует работу; `trace_id` связывает спаны; `span_id` указывает один этап; `batch_id` группирует логи одного запуска источника. Это поля, не индексные метки Loki.
- `publish` длится до confirm/NACK, `source.task` охватывает все попытки публикации. События `publish.backoff` содержат причину, номер попытки и паузу.
- `messaging.publish_to_delivery.seconds` — оценка от начала принятой попытки отправки до получения воркером. Она зависит от синхронизации часов, включает транспорт, а при повторе — и время предыдущей обработки. Это **не** чистое время в ready-очереди.
- `inference` показывает работу модели, включая задержку демо, а `db.save_result` — запись и commit. `edu_worker_task_duration_seconds` считает время от `submitted_at` до успешного завершения, не считает потерянные задачи.

## 2. Пачка 6000: масштабирование и память

В окне A создайте новый вход и запустите источник. Если настроили `SOURCE_INPUT_DIR`, положите файл в эту папку вместо `data/tasks`.

```powershell
$run = [guid]::NewGuid().ToString('N')
$file = "demo-$run.jsonl"
uv run --locked python -m edu_graphrag.pipeline.observability_demo fixture "data/tasks/$file" --count 6000
just obs demo run --rm --no-deps source --file "/input/$file" --max-in-flight 16
```

Файл не перезаписывается; ID уникальны для каждого запуска. Источник завершает работу после получения подтверждений; в `source_finished` проверьте `published_total=6000`, `lost=0`, `skipped=0`. Пока пачка идёт, в окне B дождитесь устойчивой нагрузки на один воркер и добавьте второй:

```powershell
just obs demo up -d --wait --scale worker=2 worker
just obs demo ps worker
$ids = @(just obs demo ps -q alloy loki otel-collector tempo prometheus grafana)
if ($ids.Count -gt 0) { docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' $ids }
```

В Grafana сравните установившиеся участки до и после: общий throughput, завершения по `instance`, ready/unacked, RSS, p50/p95/p99. В первом приближении `4 / 0,2 с = 20 задач/с` на реплику — расчёт, не результат замера. Снимите `docker stats` несколько раз во время нагрузки; сложите **фактическое** использование шести контейнеров мониторинга отдельно от бизнеса. Их настроенные лимиты суммарно **1504 МиБ** (160 + 256 + 192 + 256 + 384 + 256), это не измерение. Loki и Tempo рассчитаны на примерно двое суток хранения, Prometheus — два дня или до лимита диска. Grafana и Alloy сохраняют состояние в томах.

Для CSV и сравнения с выключенным SDK используйте режимы на одинаковых модели, входном размере, числе реплик и исходном состоянии очереди. Новый файл должен иметь новые ID; иначе upsert засчитает повтор как уже выполненную задачу. Не переносите числа из ДЗ 4 на стенд ДЗ 5 без нового замера.

## 3. Очередь заполнена: 503, NACK и алерт

Выполняйте на отдельном демостенде после завершения предыдущей пачки. При остановленных воркерах источник с backpressure ждёт свободного места; после проверки **обязательно** запустите воркеры вновь. В окне A:

```powershell
just obs demo stop worker
$sat = "saturation-$([guid]::NewGuid().ToString('N')).jsonl"
uv run --locked python -m edu_graphrag.pipeline.observability_demo fixture "data/tasks/$sat" --count 240
just obs demo run --rm --no-deps source --file "/input/$sat" --max-in-flight 16
```

В окне B, когда ready держится у лимита, отправьте HTTP-запрос — он должен быстро получить `503` и `Retry-After`, а в трейсе появится ERROR с ошибкой публикации. Для PowerShell 5.1 удобно вывести статус и заголовки через curl:

```powershell
'{"text":"saturation fixture"}' | curl.exe -i -H 'Content-Type: application/json' --data-binary '@-' http://127.0.0.1:18000/api/v1/tasks
```

Состояние брокера можно получить так (команда в контейнере API ничего не меняет):

```powershell
just obs demo exec -T app python -m edu_graphrag.pipeline.admin stats
```

Покажите `source_backpressure` в Loki, `publish.backoff` и `PublishRejectedError` в трейсе, ready и 503/NACK на графиках. После `ready > 90%` непрерывно в течение минуты правило проходит Pending → Firing; интервал оценки 10 с. При работающих воркерах quorum hysteresis может опускать ready ниже 90%, поэтому постоянное состояние проще показать с паузой воркеров. После показа восстановите обработку:

```powershell
just obs demo up -d --wait --scale worker=2 worker
```

NACK по переполнению откатывает вставку `pending` в API. Таймаут confirm — другой случай: ответ неизвестен, сообщение могло дойти до брокера, поэтому по одному статусу `503` нельзя доказывать, что обработки не было. Лимит относится к **ready**, а не к `ready + unacked`; небольшое превышение при параллельных публикациях возможно. Дашборд и алерт читают лимит из `QUEUE_MAX_LENGTH`, тот же параметр использует `rabbitmq-init` при установке политики. Для другого лимита поменяйте `.env`, заново примените политику через `just obs demo run --rm rabbitmq-init` и пересоздайте API; изменение только через RabbitMQ UI оставит график со старым лимитом. Аналогично держите `DELIVERY_LIMIT` согласованным с настройкой воркеров.

## 4. Ошибка, повторные доставки и настоящий DLQ

```powershell
just obs demo run --rm --no-deps --entrypoint python source -m edu_graphrag.pipeline.observability_demo poison
just obs demo run --rm --no-deps --entrypoint python source -m edu_graphrag.pipeline.observability_demo poison --malformed
just obs demo exec -T app python -m edu_graphrag.pipeline.admin stats
```

Для валидной демозадачи ищите `task_id="a036b3f2-ec4e-44f0-bfe4-852478b78f3a"`. При `DELIVERY_LIMIT=3` ожидаются попытки 1–4, каждая со своим `process` в том же трейсе. В логах — `task_failed` с `attempt` и безопасным стеком. `dead_letter.requested` на последнем NACK ещё не означает доставку в DLQ. Брокер отправляет dead letter в `tasks.dlq`, а подписчик на диагностическую копию добавляет спан `dead_letter` и лог `dead_letter_arrived` с `reason=delivery_limit`. Для malformed retry не нужен: один reject, затем `reason=rejected`. В Grafana подождите 20 с непрерывного `tasks.dlq > 0` плюс цикл оценки; алерт должен стать Firing. Проверяйте очередь в метриках или admin stats, а не только по спану диагностической копии. Оставьте `DLQ_RECORDER_ENABLED=false` на демо: recorder потребляет реальную DLQ, а tap — только копию с TTL и ограничением размера.

```logql
{service="edu-worker",env="demo"} | json | event=~"task_failed|poison_message|dead_letter_arrived"
```

## 5. Graceful shutdown и отказ мониторинга

На фоне активной **демо**-пачки откройте `just obs demo logs -f worker` в отдельном окне. В другом остановите одну реплику, чтобы вторая продолжала разбирать очередь:

```powershell
$id = @(just obs demo ps -q worker)[0]
if ($id) { docker stop --time 30 $id }
just obs demo up -d --wait --scale worker=2 worker
```

Покажите `worker_shutdown_started` с ненулевым in-flight, завершение обработки или повторную доставку незавершённого сообщения и `worker_stopped`. Если in-flight был ноль, повторите под активной нагрузкой. Проверяйте итоговый `done` и отсутствие дублирующих строк по одному ID. SIGTERM даёт время обработчикам; при превышении grace period или SIGKILL это не гарантия нулевой повторной доставки.

Затем остановите Collector, выполните обычную отправку из шага 1, дождитесь `done`, восстановите Collector. Повторите с Loki по отдельности:

```powershell
just obs demo stop otel-collector
# Отправьте новую задачу и убедитесь в done.
just obs demo up -d --wait otel-collector
just obs demo stop loki
# Отправьте ещё одну задачу и убедитесь в done.
just obs demo up -d --wait loki alloy
```

Ожидайте провалы/задержку графиков, а не гарантированный буфер всего интервала. После восстановления отправьте свежую задачу и найдите её сигналы. Для проверки короткой пачки создайте новый файл с `--count 1`, запустите source, проверьте его итоговый счётчик в Prometheus/Grafana. `rate()` из единственной точки не определён. Периодический reader выполняет финальную коллекцию при штатном завершении; `SIGKILL` и недоступная сеть не гарантируют flush.

## 6. SLO: доля принятых задач, завершённых не позднее 30 с

После полностью подтверждённого запуска шага 2 и минимум 30 с с момента окончания публикаций в окне A:

```powershell
just obs demo run --rm --no-deps --entrypoint python -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 source -m edu_graphrag.pipeline.observability_demo slo "/input/$file" --threshold 30
```

`demo_slo` покажет `good`, `late`, `failed`, `unfinished`, `missing` и SLI (`good / expected`). Цель для демонстрации: ≥ 99%. Берём все уникальные ID подтверждённой пачки; пропавшие и незавершённые входят в знаменатель. `submitted_at` проставляется источником до публикации, поэтому задержка включает ожидание окна и backoff; SQL `finished_at` проставляется перед коммитом и виден только после него. Это практическая оценка end-to-end по timestamps, а не измерение точного момента последнего fsync. `503` отражает отдельный SLI приёма HTTP, а не неуспешное завершение уже принятой работы. Если пачка прервана и подтверждены не все ID, не используйте её как когорту принятых задач.

## Запросы и ответы на защите

| Вопрос | Запрос / ответ |
|---|---|
| Сколько процентов лимита занято? | `max(rabbitmq_queue_messages_ready{queue="tasks"}) / max(edu_queue_limit{queue="tasks"} > 0)` |
| Темп подтверждений источника? | `sum(rate(edu_source_published_total[1m]))`; для короткого завершившегося source смотрите сам `edu_source_published_total`. |
| Что с p95 inference? | `histogram_quantile(0.95, sum by (le) (rate(edu_worker_inference_duration_seconds_bucket[1m])))`. Сначала складываем rate бакетов, не усредняем p95 реплик. |
| Зачем бакеты? | p95 интерполируется между соседними границами. При задержке 200 мс бакеты сгущены около 200 мс: 150/200/250/350 мс. При смене задержки/модели сверяйте бакеты с наблюдаемой латентностью; для общей агрегации у всех реплик должны быть одинаковые границы. |
| Что в cardinality? | Лейблы Loki: `service`, `level`, `env`: 3 сервиса × 5 уровней × 1 env дают до 15 комбинаций при этих условиях. У inference 9 конечных бакетов + `+Inf`, `_sum`, `_count`: 12 рядов на воркер, примерно 24 при двух (без служебного `target_info`). HTTP использует шаблон маршрута, ограниченный метод и код. UUID в labels создал бы до 6000 новых комбинаций за пачку вместо переиспользования рядов. `instance` различает реплики, но частые пересоздания увеличивают число исторических рядов. |
| Почему не Pushgateway? | Источник живёт недолго, скрейп может его пропустить. Он отправляет кумулятивные метрики по OTLP, в том числе на выходе; не нужно отдельно очищать группы Pushgateway. RabbitMQ, напротив, живёт постоянно, и его удобно скрейпить. |
| Что такое head/tail sampling? | Head принимает решение у первого спана; дёшево, но будущая ошибка неизвестна. Tail принимает решение после получения спанов: можно сохранить ошибки и медленные запросы и семплировать остальные. Это требует буферов в Collector и маршрутизации всех спанов трейса в один обработчик. Tail не восстановит спаны, отброшенные head sampling. Локально `always_on`; для production можно начать с parent-based trace ID ratio по бюджету или использовать совместимую tail-схему. |
| Как защищены логи? | JSON в stdout, уровень `LOG_LEVEL`. В traces/logs нет текста задачи, локальных переменных и сообщений исключений; стек содержит имена файлов, функций и номера строк. DSN userinfo маскируется. Не добавляйте секреты в `OTEL_RESOURCE_ATTRIBUTES` или пользовательские поля логов. Docker socket для Alloy монтируется `:ro`, но доступ к API Docker остаётся привилегированным: только доверенный локальный стенд. |

## Проверка

```powershell
uv sync --locked
just lint
just test
uv run --locked pytest --integration tests/integration/test_observability_rabbitmq.py --no-cov -q -rs
docker compose config --quiet
'logs','traces','metrics','observability' | ForEach-Object { just obs config $_ }
```

Unit-тесты проверяют W3C-восстановление после завершения HTTP-запроса, два параллельных потока с разными трейcами, ошибку на коммите с NACK, `503`, redaction, отсутствие UUID в метриках и тихую health-пробу. Тест с настоящим RabbitMQ требует запущенного Docker Desktop. Проверка синтаксиса Compose не проверяет экспорт OTLP, наличие метрик, работу ссылок Grafana, алерты, 6000 задач или память: это подтверждается только прогоном стенда.
