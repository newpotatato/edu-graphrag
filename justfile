# Developer commands. Run `just` to list them.
# Recipes use syntax that works both in sh (Linux/macOS/CI) and PowerShell (Windows).

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

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

# Apply database migrations inside the running app container
migrate:
    docker compose exec app alembic upgrade head

# Call all endpoints of the running app
smoke url="http://localhost:8000":
    uv run python scripts/smoke.py {{ url }}

# Bump version (patch|minor|major), commit, tag and push: CD publishes the image
release bump="patch":
    uv version --bump {{ bump }}
    git add pyproject.toml uv.lock
    git commit -m "chore(release): v$(uv version --short)"
    git tag -a "v$(uv version --short)" -m "v$(uv version --short)"
    git push --follow-tags
