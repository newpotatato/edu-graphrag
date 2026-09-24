from neo4j import AsyncDriver, AsyncGraphDatabase, RoutingControl

from edu_graphrag.config import Neo4jSettings


class Neo4jComponent:
    name = "neo4j"

    def __init__(self, settings: Neo4jSettings, *, connect_timeout: float) -> None:
        self._settings = settings
        self._connect_timeout = connect_timeout
        self._driver: AsyncDriver | None = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jComponent is not started")
        return self._driver

    async def startup(self) -> None:
        # The driver connects lazily on the first query.
        self._driver = AsyncGraphDatabase.driver(
            self._settings.uri,
            auth=(self._settings.user, self._settings.password.get_secret_value()),
            connection_timeout=self._connect_timeout,
        )

    async def shutdown(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def fetch_version(self) -> str:
        records, _, _ = await self.driver.execute_query(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS version",
            routing_=RoutingControl.READ,
        )
        return str(records[0]["version"])
