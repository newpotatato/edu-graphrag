"""Call every endpoint of a running app and print status + body.

Usage: `just smoke` or `uv run python scripts/smoke.py http://localhost:8000`.
Exits with 1 if liveness or version checks fail. Health may legitimately be 503, and so may
POST /api/v1/tasks (queue full or broker down: load shedding with Retry-After).
"""

import json
import sys
import urllib.error
import urllib.request

GET_ENDPOINTS = ("/healthz", "/api/v1/version", "/api/v1/health")
MUST_BE_200 = ("/healthz", "/api/v1/version")


def call(url: str, body: object | None = None) -> tuple[int, dict[str, str], object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(  # noqa: S310 - URL is from the CLI
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.load(exc)


def show(method: str, path: str, status: int, body: object, extra: str = "") -> None:
    print(f"{status} {method} {path}{extra}\n{json.dumps(body, indent=2, ensure_ascii=False)}\n")


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    ok = True
    for path in GET_ENDPOINTS:
        status, _, body = call(base + path)
        show("GET", path, status, body)
        if path in MUST_BE_200 and status != 200:
            ok = False

    status, headers, body = call(base + "/api/v1/tasks", {"text": "smoke test: x^2 = 4"})
    retry_after = headers.get("Retry-After")
    show(
        "POST", "/api/v1/tasks", status, body, f" (Retry-After: {retry_after})" * bool(retry_after)
    )
    if status == 202:
        location = headers["Location"]
        status, _, body = call(base + location)
        show("GET", location, status, body)
        ok = ok and status == 200
    elif status != 503:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
