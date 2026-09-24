from pathlib import Path

import pytest
from pydantic import ValidationError

from edu_graphrag.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("POSTGRES_PASSWORD", "NEO4J_PASSWORD", "HEALTH_CHECK_TIMEOUT", "POSTGRES_HOST"):
        monkeypatch.delenv(name, raising=False)
    # `.env` is resolved relative to the working directory: ignore a developer's local one.
    monkeypatch.chdir(tmp_path)


def test_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-secret")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo-secret")
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "3.5")

    settings = Settings()

    assert settings.postgres.host == "postgres"
    assert settings.postgres.password.get_secret_value() == "pg-secret"
    assert settings.neo4j.password.get_secret_value() == "neo-secret"
    assert settings.health_check_timeout == 3.5


def test_secrets_are_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-secret")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo-secret")

    assert "pg-secret" not in repr(Settings())


def test_passwords_have_no_defaults() -> None:
    with pytest.raises(ValidationError, match="password"):
        Settings()


def test_health_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "0")

    with pytest.raises(ValidationError, match="health_check_timeout"):
        Settings()
