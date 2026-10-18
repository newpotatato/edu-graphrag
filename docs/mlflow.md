# MLflow: от датасета до модели в приложении

```
архив учебных материалов
   ↓ ingest (коннектор → конвертер → чанки)          edu_graphrag.ml.ingest
датасет study-chunks (parquet, версия = digest)
   ↓ EDA                                               edu_graphrag.ml.eda
Experiment / Runs  (params, metrics, artifacts, inputs)
   ↓                                                   edu_graphrag.ml.classifier / ml.rag
Model Registry: subject-classifier, study-retriever (версии + alias champion/challenger)
   ↓ загрузка ОДИН раз при старте                      components/mlflow_model.py
FastAPI-приложение → POST /api/v1/process
```

## 1. Развёртывание

`docker compose up -d --wait mlflow` (или `just mlflow-up`), UI: <http://localhost:5000>.

| Сервис | Роль | Почему так |
|---|---|---|
| `mlflow` | Tracking Server + UI + REST API + Model Registry | свой образ: официальный `ghcr.io/mlflow/mlflow` без драйвера Postgres и boto3 |
| `mlflow-db` (Postgres 17) | **Backend Store**: эксперименты, runs, params, metrics, tags, registry | отдельный инстанс: ML-метаданные не смешиваются с продуктовой БД |
| `seaweedfs` (S3 API) | **Artifact Store**: модели, графики, таблицы, bucket `mlflow` | MinIO больше нет (репозиторий в архиве, образов нет); SeaweedFS — живой S3-совместимый аналог |
| `seaweedfs-init` | одноразовая job: создать bucket (идемпотентно) | |

Ключевое решение — **проксирование артефактов** (`--serve-artifacts --artifacts-destination s3://mlflow`):
клиенты (скрипты обучения и приложение) знают только `MLFLOW_TRACKING_URI`, а S3-ключи есть
только у сервера. Клиентам выставлено `MLFLOW_ENABLE_PROXY_MULTIPART_{UPLOAD,DOWNLOAD}=false`, чтобы
MLflow 3 не пытался отдавать presigned-ссылки прямо на внутренний S3.

## 2. Датасет и EDA

`just dataset C:/obsidian_workspace/university` → `data/processed/` (в git не попадает: это учебные
материалы). 223 документа (PDF, ipynb, md, tex, docx, pptx) → 3036 чанков.

- **Коннектор** (`Connector` Protocol) — точка расширения для других источников (YouTube, LMS,
  очередь). Сейчас один: файловая система; метки `grade/semester/subject` берутся из пути.
- **Инкрементальность**: у каждого файла `content_hash` (sha256); неизменённые не конвертируются.
- **Версия датасета** = sha256 по отсортированным `(chunk_id, sha256(text))` → `8a3156547e45`.

`just eda` логирует run `eda-<version>` в эксперимент `edu-graphrag-eda`: 8 графиков, таблицы,
`eda_summary.md`. Главные выводы, повлиявшие на модели:

- дисбаланс классов 254:1 (`linear_algebra` 33 % чанков) → метрика macro-F1 и `class_weight`;
- 9 групп побайтно одинаковых файлов → дедупликация **до** разбиения, иначе утечка train/test;
- 3 пустых документа (сканы без текстового слоя — кандидаты на OCR), 2 битых файла.

## 3. Dataset tracking: source, digest, lineage

Датасет логируется как `mlflow.data.from_pandas(df, source=..., name="study-chunks", targets="subject")`
и привязывается к run через `mlflow.log_input(..., context=...)`.

- **source** — откуда данные (путь к parquet), чтобы их можно было найти и перечитать;
- **digest** — хеш содержимого, считает MLflow: одинаковые данные → одинаковый digest, поэтому
  видно, на одних ли данных обучались два run;
- **context** — роль датасета в run: `training`, `evaluation`, `index`, `eda`;
- **lineage** — цепочка `dataset (digest) → run (inputs) → logged model → registered version`.
  В UI: версия модели → ссылка на run → вкладка Inputs/Datasets. Плюс тег `dataset_version` у
  run и у версии модели.

## 4. Эксперименты

### `subject-classifier` — к какому предмету относится фрагмент текста

Разбиение **по документам** (чанки одного файла не попадают и в train, и в test), seed 42,
14 классов (предметы с < 3 документами отброшены), train 155 док./2298 чанков, test 39/483.

| Run | Что меняем | test macro-F1 (осн.) | accuracy | doc accuracy |
|---|---|---|---|---|
| `baseline` | TF-IDF слова 1–2 + LogReg | 0.629 | 0.824 | 0.846 |
| `char_ngrams` | char_wb 2–5 (устойчиво к морфологии и формулам) | 0.679 | 0.851 | 0.846 |
| **`balanced`** | + `class_weight="balanced"` | **0.728** | **0.890** | 0.872 |
| `word_balanced` | слова + balanced | 0.706 | 0.870 | 0.897 |
| `svm` | LinearSVC + калибровка | 0.691 | 0.878 | 0.846 |

Вспомогательные метрики: weighted-F1, balanced accuracy, train macro-F1 (разрыв = переобучение),
время обучения, мс на чанк. Артефакты: confusion matrix, отчёт по классам, ошибочные примеры,
топ-признаки по классам.

### `study-retriever` — поиск фрагментов под вопрос (R в RAG)

Оценка на синтетическом QA-наборе: 165 вопросов по всем 18 предметам, золото — документ-источник.
Основная метрика **recall@5** (есть ли нужный документ в топ-5), вспомогательные: recall@1/10,
MRR@10, nDCG@10, задержка p50/p95, recall@5 по предметам.

| Run | recall@1 | **recall@5** | MRR@10 | p50 |
|---|---|---|---|---|
| `bm25` (baseline) | 0.624 | 0.855 | 0.722 | 10 мс |
| **`bm25_prefix5`** (токены обрезаны до 5 букв ≈ стемминг) | **0.661** | **0.897** | **0.756** | 11 мс |
| `dense` (multilingual MiniLM) | 0.479 | 0.709 | 0.579 | 32 мс |
| `hybrid_rrf` (BM25 + dense, RRF) | 0.606 | 0.800 | 0.689 | 43 мс |

Индекс хранится **внутри артефактов модели**: версия модели = версия индекса, у `champion` нет
вопроса «а с каким индексом она работает». Оговорка: вопросы писались по тексту чанков, это
завышает лексический поиск; маленькая мультиязычная модель плохо знает математическую лексику.

## 5. Model Registry

| Модель | v1 | v2 | `champion` | `challenger` |
|---|---|---|---|---|
| `subject-classifier` | baseline | balanced | **v2** | v1 |
| `study-retriever` | bm25 | bm25_prefix5 | **v2** | v1 |

Лучший run выбирается кодом (`search_runs` по основной метрике), не руками. Регистрация
идемпотентна. У версий — теги (variant, метрика, dataset_version) и ссылка на run.
**Alias** — именованный изменяемый указатель на версию: код просит `@champion`, а какая это
версия, решается в реестре.

## 6. Приложение: `POST /api/v1/process`

При старте компонент `MlflowModelComponent`:
1. резолвит `models:/{MODEL_NAME}@{MODEL_ALIAS}` → **конкретную версию**;
2. загружает **именно её** (в отдельном потоке, event loop не блокируется) и держит в памяти;
3. `/process` только вызывает `predict` уже загруженной модели (тоже в потоке).

```bash
curl -X POST localhost:8000/api/v1/process -H "Content-Type: application/json" \
     -d '{"texts": ["Собственные значения линейного оператора"]}'
# {"model": {"name": "subject-classifier", "alias": "champion", "version": "2", ...},
#  "predictions": [{"label": "linear_algebra", "confidence": 0.46, "top3": [...]}]}
```

Какая версия используется: поле `model` в ответе, заголовок `X-Model-Version`, `GET /api/v1/model`
(`just model`) и компонент `mlflow_model` в `/api/v1/health`. Тот же код обслуживает и ретривер:
`MODEL_NAME=study-retriever`. Если MLflow недоступен при старте — приложение всё равно стартует,
`/process` отвечает 503, `/health` показывает компонент down.

### Сценарий защиты: переключить alias

```bash
just champion 1     # alias champion → v1 + рестарт приложения
just model          # "version": "1"
just champion 2     # обратно
```

До рестарта приложение продолжает обслуживать ту версию, которую загрузило: это осознанно.

### Почему «горячее обновление» — плохая идея и как лучше

Если приложение само периодически перечитывает alias и подменяет модель в памяти:

- **нет атомарности и воспроизводимости**: часть запросов обработана старой версией, часть
  новой; по логам не восстановить, что ответило пользователю;
- **реплики расходятся**: у каждой свой момент перечитывания — две версии в проде одновременно;
- **двойная память и паузы** на время загрузки второй модели; ошибка загрузки — в рантайме, а не
  на деплое;
- **нет контроля выкатки**: переключение alias мгновенно уходит на 100 % трафика без canary и
  без быстрого отката процесса;
- **деплой мимо CI/CD**: изменение поведения сервиса без сборки, проверки и следа в истории.

Как лучше:

- **неизменяемые деплои**: при деплое резолвить alias → версию и закреплять её (env
  `MODEL_VERSION` / тег образа / запечь модель в образ); новая версия = новый деплой через CI/CD;
- **blue-green / canary**: поднять реплики с новой версией рядом, перевести часть трафика,
  сравнить метрики, затем переключить или откатить;
- **отдельный model serving** (MLflow serve, KServe, Seldon, BentoML) с версионированными
  эндпоинтами и управлением трафиком;
- если всё же нужен reload — атомарная подмена ссылки после успешной загрузки и прогрева,
  версия модели в каждом ответе и логе, единая координация реплик.

## Команды

| | |
|---|---|
| `just mlflow-up` | поднять MLflow + хранилища |
| `just dataset <DATA_ROOT>` | собрать датасет |
| `just eda` | EDA → MLflow |
| `just train-classifier` | все варианты классификатора + регистрация + alias |
| `just train-retriever` | варианты ретривера (по умолчанию BM25) + регистрация + alias |
| `just champion <version> [model]` | переключить champion и перезапустить приложение |
| `just model` | какую версию обслуживает приложение |
