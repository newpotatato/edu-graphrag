"""Contract for an external dependency (database, cache, object storage, ...).

To add a new one: implement this protocol in a new module and register it in
`components.registry.build_components`. The health endpoint picks it up automatically.
"""

from typing import Protocol


class Component(Protocol):
    name: str

    async def startup(self) -> None:
        """Create clients/pools. Must not require the dependency to be reachable."""

    async def shutdown(self) -> None:
        """Release clients/pools."""

    async def fetch_version(self) -> str:
        """Do a real round trip to the dependency and return its version."""
