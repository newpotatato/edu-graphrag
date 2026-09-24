import asyncio
from dataclasses import dataclass, field


@dataclass
class FakeComponent:
    """In-memory `Component` whose behaviour (version, latency, failure) is set by the test."""

    name: str
    version: str = "1.0.0"
    delay: float = 0.0
    error: Exception | None = None
    startup_error: Exception | None = None
    events: list[str] = field(default_factory=list)
    checks: int = 0

    async def startup(self) -> None:
        if self.startup_error is not None:
            raise self.startup_error
        self.events.append(f"start:{self.name}")

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")

    async def fetch_version(self) -> str:
        self.checks += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.version
