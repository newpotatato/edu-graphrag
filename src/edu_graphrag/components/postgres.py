from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from edu_graphrag.config import PostgresSettings


def build_postgres_url(settings: PostgresSettings) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username=settings.user,
        password=settings.password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=settings.db,
    )


class PostgresComponent:
    name = "postgres"

    def __init__(self, settings: PostgresSettings, *, connect_timeout: float) -> None:
        self._url = build_postgres_url(settings)
        self._connect_timeout = connect_timeout
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("PostgresComponent is not started")
        return self._engine

    async def startup(self) -> None:
        # The engine is lazy: no connection is opened here, so the app starts
        # even when Postgres is down and /api/v1/health can report it.
        self._engine = create_async_engine(
            self._url,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            connect_args={"timeout": self._connect_timeout},
        )

    async def shutdown(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    async def fetch_version(self) -> str:
        async with self.engine.connect() as conn:
            version = await conn.scalar(text("SHOW server_version"))
        return str(version)
