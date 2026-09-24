"""Application settings. Everything comes from environment variables or `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"


class PostgresSettings(BaseSettings):
    """Uses the same variable names as the official postgres image."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=_ENV_FILE, extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr
    db: str = "edu_graphrag"


class Neo4jSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=_ENV_FILE, extra="ignore")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    app_name: str = "edu-graphrag"
    # Commit the image was built from; injected by the Docker build.
    app_commit_sha: str = "unknown"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # Per-component timeout for /api/v1/health, seconds.
    health_check_timeout: float = Field(default=2.0, gt=0)

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
