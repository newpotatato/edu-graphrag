"""Call every endpoint of a running app and print status + body.

Usage: `just smoke` or `uv run python scripts/smoke.py http://localhost:8000`.
Exits with 1 if liveness or version checks fail (health may legitimately be 503).
"""

import json
import sys
import urllib.error
import urllib.request

ENDPOINTS = ("/healthz", "/api/v1/version", "/api/v1/health")


def call(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - URL is from the CLI
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    ok = True
    for path in ENDPOINTS:
        status, body = call(base.rstrip("/") + path)
        print(f"{status} {path}\n{json.dumps(body, indent=2, ensure_ascii=False)}\n")
        if path != "/api/v1/health" and status != 200:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
