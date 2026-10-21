# Developer commands. Run `just` to list them.
# Recipes use syntax that works both in sh (Linux/macOS/CI) and PowerShell (Windows).

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
# ML recipes read MLFLOW_TRACKING_URI etc. from .env
set dotenv-load := true

export PYTHONUTF8 := "1"
export MLFLOW_DISABLE_AGENT_HINT := "1"

# List available recipes
default:
    @just --list

# Install all dependency groups and git hooks
setup:
    uv sync
    uv run pre-commit install

# Create .env from .env.example (never overwrites)
env:
    {{ if path_exists(".env") == "true" { "echo '.env already exists'" } else { "cp .env.example .env" } }}

# Format code and apply safe lint fixes
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint, format check and type check (same as CI)
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy

# Unit tests with coverage; extra pytest args are passed through
test *args:
    uv run pytest {{ args }}

# All tests including integration ones (needs `just up`)
test-integration:
    uv run pytest --integration

# Run every pre-commit hook on the whole repository
hooks:
    uv run pre-commit run --all-files

# Everything CI checks
check: lint test

# Build the image and start the stack, waiting until all services are healthy
up $GIT_SHA=`git rev-parse --short HEAD`:
    docker compose up --build -d --wait

# Stop the stack (data volumes are kept)
down:
    docker compose down

# Stop the stack and delete database volumes
clean:
    docker compose down -v

# Follow logs of a service
logs service="app":
    docker compose logs -f {{ service }}

# Show service status
ps:
    docker compose ps

# Start only MLflow and its stores (backend DB, SeaweedFS, bucket init)
mlflow-up:
    docker compose up --build -d --wait mlflow

# Print the MLflow UI address
mlflow-ui port="5000":
    @echo "MLflow UI: http://localhost:{{ port }}"

# Apply database migrations inside the running app container
migrate:
    docker compose exec app alembic upgrade head

# Call all endpoints of the running app
smoke url="http://localhost:8000":
    uv run python scripts/smoke.py {{ url }}

# ---------- ML pipeline (needs `just mlflow-up`) ----------

# Build documents/chunks parquet from the study archive (incremental by content hash)
dataset data_root:
    uv run python -m edu_graphrag.ml.ingest.build_dataset --data-root {{ data_root }} --out data/processed

# EDA plots and tables, logged to MLflow together with the dataset
eda:
    uv run python -m edu_graphrag.ml.eda --data data/processed --log-to-mlflow

# Train all subject-classifier variants, register baseline + best, set aliases
train-classifier:
    uv run python -m edu_graphrag.ml.classifier.train --all
    uv run python -m edu_graphrag.ml.classifier.register

# Evaluate retriever variants on the QA set, register baseline + best, set aliases
train-retriever *variants="bm25 bm25_prefix5":
    uv run python -m edu_graphrag.ml.rag.experiment --data data/processed --qa data/qa/qa.jsonl --variants {{ variants }}

# Point the champion alias to a version, then restart the app to serve it
champion version model="subject-classifier":
    uv run python -c "import mlflow; mlflow.MlflowClient().set_registered_model_alias('{{ model }}', 'champion', '{{ version }}'); print('{{ model }}@champion -> v{{ version }}')"
    docker compose restart app

# Show which model version the running app serves
model url="http://localhost:8000":
    uv run python -c "import json, urllib.request; print(json.dumps(json.load(urllib.request.urlopen('{{ url }}/api/v1/model')), indent=2))"

# ---------- task pipeline (RabbitMQ + inference workers) ----------

# Start RabbitMQ, declare topology + queue policy, migrate the DB, start one worker
pipeline-up $GIT_SHA=`git rev-parse --short HEAD`:
    docker compose up --build -d --wait rabbitmq worker

# Scale the inference workers (existing ones keep running)
workers n="1":
    docker compose up -d --wait --no-recreate --scale worker={{ n }} worker

# Change the tasks queue capacity at runtime (RabbitMQ policy; 0 = unbounded)
queue-limit n:
    uv run python -m edu_graphrag.pipeline.admin apply-policy --max-length {{ n }}

# Queue depths, consumers, rates and broker memory (observation only)
queue-stats:
    uv run python -m edu_graphrag.pipeline.admin stats

# Follow the logs of all worker replicas
worker-logs:
    docker compose logs -f worker

# Send SIGTERM to the workers: they finish in-flight tasks, then exit (graceful shutdown demo)
sigterm-worker:
    docker compose kill -s SIGTERM worker

# Bump version (patch|minor|major), commit, tag and push: CD publishes the image
release bump="patch":
    uv version --bump {{ bump }}
    git add pyproject.toml uv.lock
    git commit -m "chore(release): v$(uv version --short)"
    git tag -a "v$(uv version --short)" -m "v$(uv version --short)"
    git push --follow-tags
