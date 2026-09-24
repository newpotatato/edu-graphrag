import tomllib
from pathlib import Path

from httpx import AsyncClient

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


async def test_version_comes_from_pyproject(client: AsyncClient) -> None:
    """The endpoint must report exactly the version declared in pyproject.toml."""
    expected = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json()["version"] == expected


async def test_version_contains_name_and_commit(client: AsyncClient) -> None:
    response = await client.get("/api/v1/version")

    body = response.json()
    assert body["name"] == "edu-graphrag"
    assert body["commit"] == "test-sha"


async def test_openapi_reports_same_version(client: AsyncClient) -> None:
    version = (await client.get("/api/v1/version")).json()["version"]

    openapi = (await client.get("/openapi.json")).json()

    assert openapi["info"]["version"] == version
