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


class MlflowSettings(BaseSettings):
    """Model served by /api/v1/process. No prefix: the names match MLflow's own env variables."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "subject-classifier"
    # The app serves whatever version this alias points to at startup.
    model_alias: str = "champion"
    # Off for deployments that do not serve a model at all.
    model_enabled: bool = True


class RabbitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RABBITMQ_", env_file=_ENV_FILE, extra="ignore")

    host: str = "localhost"
    port: int = 5672
    management_port: int = 15672
    user: str = "edu"
    password: SecretStr
    vhost: str = "/"

    @property
    def url(self) -> str:
        from urllib.parse import quote

        password = quote(self.password.get_secret_value(), safe="")
        vhost = quote(self.vhost, safe="")
        return f"amqp://{quote(self.user, safe='')}:{password}@{self.host}:{self.port}/{vhost}"


class PipelineSettings(BaseSettings):
    """Task pipeline: queue limits, worker concurrency and the simulated inference cost."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # Capacity of the `tasks` queue; applied as a RabbitMQ policy (changeable at runtime).
    queue_max_length: int = Field(default=200, ge=0)
    # Deliveries before a failing message is dead-lettered to `tasks.dlq`.
    delivery_limit: int = Field(default=3, ge=1)
    # Size of the worker's inference thread pool; prefetch_count is derived from it.
    worker_concurrency: int = Field(default=4, ge=1)
    # Extra time spent per inference: makes the worker's throughput reproducible.
    inference_delay_ms: int = Field(default=200, ge=0)
    # How long a publisher waits for the broker's confirm before giving up on a message.
    publish_confirm_timeout: float = Field(default=2.0, gt=0)


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
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)
    rabbit: RabbitSettings = Field(default_factory=RabbitSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
