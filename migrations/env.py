"""Alembic environment: async engine, URL from application settings."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from edu_graphrag.components.postgres import build_postgres_url
from edu_graphrag.config import PostgresSettings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ORM models do not exist yet; set this to `Base.metadata` once they do,
# so that `alembic revision --autogenerate` can diff against it.
target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`)."""
    url = build_postgres_url(PostgresSettings()).render_as_string(hide_password=False)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(build_postgres_url(PostgresSettings()))
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
