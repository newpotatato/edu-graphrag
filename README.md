# edu-graphrag

[![CI](https://github.com/newpotatato/edu-graphrag/actions/workflows/ci.yml/badge.svg)](https://github.com/newpotatato/edu-graphrag/actions/workflows/ci.yml)
[![CD](https://github.com/newpotatato/edu-graphrag/actions/workflows/cd.yml/badge.svg)](https://github.com/newpotatato/edu-graphrag/actions/workflows/cd.yml)

Мультимодальный GraphRAG-агрегатор учебных материалов (PDF, презентации, видеолекции, код)
с общим графом курса (EduKG) и персональным графом знаний студента (PKG).

Сейчас в репозитории — каркас сервиса: асинхронное API, подключение к PostgreSQL и Neo4j,
тесты, Docker-образ и CI/CD, публикующий версионированный образ в GitHub Container Registry.

**Стек:** Python 3.12 · FastAPI · SQLAlchemy 2 (asyncpg) + Alembic · Neo4j async driver ·
structlog · uv · ruff · mypy · pytest · Docker · GitHub Actions · GHCR

---

## Быстрый старт

Нужны [uv](https://docs.astral.sh/uv/), Docker и [just](https://github.com/casey/just)
(`winget install Casey.Just` / `brew install just`).

```bash
just setup        # uv sync + установка pre-commit хуков
just env          # .env из .env.example -> впишите POSTGRES_PASSWORD и NEO4J_PASSWORD
just up           # собрать образ и поднять app + postgres + neo4j, дождаться healthy
just smoke        # дёрнуть все эндпоинты
```

Без just: `uv sync && cp .env.example .env && docker compose up --build -d --wait`.

Все команды — `just` (без аргументов):

| Команда | Что делает |
|---|---|
| `just lint` | ruff check + ruff format --check + mypy (как в CI) |
| `just fmt` | отформатировать и применить безопасные автофиксы |
| `just test` | unit-тесты с coverage |
| `just test-integration` | + интеграционные с реальными БД (нужен `just up`) |
| `just up` / `down` / `clean` | поднять / остановить / остановить и удалить данные |
| `just logs [service]` | логи сервиса |
| `just migrate` | `alembic upgrade head` в контейнере |
| `just release [patch\|minor\|major]` | поднять версию, закоммитить, тегнуть, запушить → CD |

Swagger UI: <http://localhost:8000/docs>, Neo4j Browser: <http://localhost:7474>.

---

## Эндпоинты

| Метод и путь | Назначение |
|---|---|
| `GET /healthz` | liveness: процесс жив и отвечает. **Не ходит в БД** |
| `GET /api/v1/version` | версия приложения и коммит, из которого собран образ |
| `GET /api/v1/health` | end-to-end проверка: версия и время ответа каждого компонента |

```jsonc
// GET /api/v1/health -> 200, или 503 если хоть что-то лежит (тело то же, с причиной)
{
  "status": "ok",
  "components": [
    {"name": "postgres", "status": "up", "version": "17.11",   "latency_ms": 11.8, "error": null},
    {"name": "neo4j",    "status": "up", "version": "5.26.31", "latency_ms": 53.9, "error": null}
  ]
}
```

**Почему `/healthz` без `/api/v1`.** Это инфраструктурный контракт, а не публичное API: его
дёргают Docker healthcheck, Kubernetes-пробы и балансировщики, и путь не должен меняться при
выходе `/api/v2`. Он не проверяет БД намеренно: при падении БД оркестратор не должен
перезапускать здоровое приложение. Зависимости проверяет `/api/v1/health`.

**Версия приложения ≠ версия API.** `v1` — версия контракта API. Версия приложения берётся из
**одного места** — `project.version` в `pyproject.toml` — через метаданные установленного пакета
(`importlib.metadata`). Её же видят OpenAPI (`/docs`) и метки образа. CD не опубликует образ,
если git-тег не совпадает с `pyproject.toml` или запущенный образ отдаёт другую версию.

**`/api/v1/health`**: проверки идут параллельно (`asyncio.gather`), у каждой свой таймаут
(`HEALTH_CHECK_TIMEOUT`), время меряется `time.perf_counter()`. Ошибка одного компонента
не роняет ответ — она попадает в поле `error`. Используются пулы, созданные при старте, а не
новое подключение на каждый запрос.

---

## Структура

```
src/edu_graphrag/
  __init__.py          # __version__ из метаданных пакета
  main.py              # create_app(): lifespan, middleware, роутеры
  config.py            # Settings (pydantic-settings), всё из env / .env
  logging_config.py    # structlog: JSON в stdout, stdlib-логгеры через тот же формат
  middleware.py        # X-Request-ID + одна access-строка лога на запрос
  schemas.py           # pydantic-модели ответов
  api/
    system.py          # /healthz
    deps.py            # зависимости FastAPI (settings, components)
    v1/                # /api/v1/version, /api/v1/health
  components/          # внешние зависимости за общим протоколом Component
    base.py            #   startup / shutdown / fetch_version
    postgres.py        #   SQLAlchemy async engine (asyncpg)
    neo4j.py           #   neo4j AsyncDriver
    registry.py        #   какие компоненты есть + их запуск/остановка
  services/health.py   # параллельная проверка компонентов с таймаутами
migrations/            # Alembic (async env; URL из тех же настроек, что у приложения)
tests/unit/            # без внешних сервисов, компоненты подменяются фейками
tests/integration/     # с настоящими Postgres и Neo4j (--integration)
scripts/smoke.py       # проверка эндпоинтов запущенного приложения
```

**Как добавить новую БД** (Redis, MinIO, векторное хранилище…): модуль в `components/`
с методами `startup/shutdown/fetch_version` + одна строка в `registry.build_components` +
сервис в `docker-compose.yml`. Эндпоинты и их тесты менять не нужно.

**Асинхронность.** Весь I/O асинхронный: asyncpg, neo4j AsyncDriver, uvicorn с uvloop.
Блокирующие вызовы в `async`-коде ловит правило ruff `ASYNC`, забытый `await` — mypy.
Движок SQLAlchemy и драйвер Neo4j ленивые, поэтому приложение стартует и при лежащей БД,
а `/api/v1/health` честно сообщает, что именно недоступно.

---

## Зависимости

`pyproject.toml` + `uv.lock` (точные версии, `uv sync --locked` воспроизводит окружение).

| Где | Что | Куда попадает |
|---|---|---|
| `[project.dependencies]` | pydantic, pydantic-settings, structlog — общее ядро | везде |
| группа `web` | fastapi, uvicorn, sqlalchemy+asyncpg, alembic, neo4j | образ API |
| группа `dev` | ruff, mypy, pytest(+asyncio, cov), httpx, pre-commit | только локально и в CI |

Группы позволят позже собирать отдельные образы (например, индексатор с ML-зависимостями),
не затаскивая их в образ API. Docker ставит `uv sync --no-dev`.

---

## Качество кода

**ruff** (линтер + форматтер, `pyproject.toml`):

| Правила | Зачем |
|---|---|
| `E`, `W`, `F` | базовые ошибки: неиспользуемые импорты, неопределённые имена |
| `I` | порядок импортов |
| `UP` | современный синтаксис под Python 3.12 |
| `B` | bugbear: вероятные баги (мутабельные дефолты и т.п.) |
| `ASYNC` | блокирующие вызовы внутри `async def` — ключевое для асинхронного сервиса |
| `S` | bandit: захардкоженные пароли, небезопасные вызовы |
| `N`, `C4`, `SIM`, `RET`, `ARG`, `PTH` | именование, упрощения, неиспользуемые аргументы, pathlib |
| `LOG`, `G` | корректное использование логгеров |
| `PT` | стиль pytest |
| `RUF` | правила самого ruff |

`E501` выключен — длину строк держит форматтер. В тестах разрешены `assert` и фейковые пароли.

**mypy --strict** с плагином pydantic: аннотации везде, без неявного `Any`.

**pre-commit** (`.pre-commit-config.yaml`): trailing whitespace, EOF, LF-окончания строк,
большие файлы (>500 КБ), валидность YAML/TOML/JSON, merge-конфликты, приватные ключи,
забытые `breakpoint()`, ruff, ruff-format, актуальность `uv.lock`, mypy.

---

## Логирование

structlog, одна JSON-строка на событие в stdout (12-factor: сбор логов — забота инфраструктуры).
Логи uvicorn и драйверов идут через тот же форматтер. Каждый запрос получает `request_id`
(берётся из `X-Request-ID` или генерируется, возвращается в ответе) — он автоматически
попадает во все логи этого запроса.

```json
{"component": "neo4j", "status": "up", "version": "5.26.31", "latency_ms": 53.93, "event": "component_checked", "request_id": "659806cd...", "level": "info", "logger": "edu_graphrag.services.health", "timestamp": "2026-09-24T20:45:37.10Z"}
{"method": "GET", "path": "/api/v1/health", "status_code": 200, "duration_ms": 61.2, "event": "request_finished", "request_id": "659806cd...", "level": "info", "logger": "edu_graphrag.middleware", "timestamp": "2026-09-24T20:45:37.11Z"}
```

`LOG_JSON=false` — цветной человекочитаемый вывод для локальной отладки.

---

## Тесты

```bash
just test               # unit: несколько секунд, без Docker
just test-integration   # + реальные Postgres и Neo4j из docker compose
```

- **Unit**: компоненты подменяются `FakeComponent` — так проверяются сценарии, которые трудно
  воспроизвести с настоящей БД: компонент упал, ответил дольше таймаута, проверки идут
  параллельно, остановка уже запущенных компонентов при ошибке старта, формат логов,
  валидация настроек, соответствие версии `pyproject.toml`.
- **Integration**: настоящие версии БД и end-to-end `/api/v1/health`.
- Coverage с порогом **90 %** (`fail_under`), branch coverage включён.

---

## Docker

- **Multi-stage**: builder с uv ставит зависимости в `/app/.venv`, runtime — `python:3.12-slim`
  только с venv (без uv, исходников, тестов, dev-зависимостей).
- **Кэш слоёв**: сначала только `pyproject.toml` + `uv.lock` → зависимости, потом код.
  Правка кода не переустанавливает пакеты.
- **Не root**: пользователь с числовым UID 10001; файлы приложения принадлежат root и
  недоступны процессу на запись.
- `HEALTHCHECK` на `/healthz`, `.dockerignore` по принципу allowlist.
- Коммит передаётся через `--build-arg GIT_SHA` и виден в `/api/v1/version`.

**docker-compose.yml**: app + postgres + neo4j в сети `backend`; именованные volumes для данных;
healthcheck у всех сервисов и `depends_on: service_healthy`; наружу опубликовано только
приложение (БД — лишь на `127.0.0.1` для локальных тестов); лимиты CPU/памяти (heap Neo4j
подогнан под лимит); ротация логов `json-file` 10 МБ × 3; секреты — только из `.env`.

---

## CI/CD

**CI** (`.github/workflows/ci.yml`) — на push в любую ветку, кроме `main`, на pull request и
как первый этап CD. Отдельные jobs, чтобы по красному крестику сразу было видно, что сломалось:

| Job | Проверка |
|---|---|
| Lint & format | `ruff check` (аннотации прямо в diff PR), `ruff format --check` |
| Type check | `mypy --strict` |
| Pre-commit hygiene | все хуки pre-commit на всех файлах — ловит коммиты с `--no-verify` |
| Dockerfile lint | hadolint |
| Tests & coverage | unit + integration с Postgres и Neo4j в service-контейнерах, порог coverage, отчёт в summary и артефактах |

**CD** (`.github/workflows/cd.yml`) — публикует образ в `ghcr.io/newpotatato/edu-graphrag`:

| Событие | Теги образа | Зачем |
|---|---|---|
| push в `main` | `main`, `sha-<short>` | каждое принятое изменение — готовый образ, однозначно связанный с коммитом |
| push тега `vX.Y.Z` | `X.Y.Z`, `X.Y`, `X`*, `latest` | осознанный релиз по SemVer; `latest` = последний релиз |
| ручной запуск | как у выбранной ветки/тега | повторить публикацию без нового коммита |

\* мажорный тег не ставится для `0.x` — по SemVer эта линейка нестабильна.

Почему так:
- **Сначала CI, потом публикация** (`needs: ci`): в реестр не попадает образ, не прошедший тесты.
- **Ветки и PR не публикуют**: непроверенный код не должен становиться образом, а PR из форков
  не имеют доступа к токену записи.
- **Неизменяемый `sha-*`** — по любому образу понятно, из какого он коммита; `main` и `latest` —
  удобные «плавающие» указатели.
- **GHCR + `GITHUB_TOKEN`**: токен выдаётся на время job и истекает, никаких хранимых секретов;
  права ограничены `packages: write` только для job публикации.
- **Защита версии**: тег `vX.Y.Z` обязан совпадать с `pyproject.toml`; перед push образ
  запускается и `/api/v1/version` сверяется с ожидаемой версией.

Релиз: `just release patch` (→ `uv version --bump patch`, коммит, тег `vX.Y.Z`, push) —
дальше pre-commit → CI → CD отрабатывают автоматически.

```bash
docker pull ghcr.io/newpotatato/edu-graphrag:latest
```

---

## Конфигурация

Все параметры — переменные окружения (или `.env`, см. `.env.example`). Паролей по умолчанию нет:
без них приложение не стартует.

| Переменная | По умолчанию | |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_JSON` | `true` | `false` — человекочитаемые логи |
| `HEALTH_CHECK_TIMEOUT` | `2.0` | таймаут одной проверки в `/api/v1/health`, с |
| `POSTGRES_HOST` / `_PORT` / `_USER` / `_DB` | `localhost` / `5432` / `postgres` / `edu_graphrag` | |
| `POSTGRES_PASSWORD` | — | обязательно |
| `NEO4J_URI` / `NEO4J_USER` | `bolt://localhost:7687` / `neo4j` | |
| `NEO4J_PASSWORD` | — | обязательно, ≥ 8 символов |
| `APP_COMMIT_SHA` | `unknown` | выставляется при сборке образа |
