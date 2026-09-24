import json
import logging

import pytest
import structlog

from edu_graphrag.logging_config import configure_logging


def test_json_logs_include_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)
    structlog.contextvars.bind_contextvars(request_id="r-1")
    try:
        structlog.get_logger("test").info("something_happened", answer=42)
    finally:
        structlog.contextvars.clear_contextvars()

    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "something_happened"
    assert record["answer"] == 42
    assert record["request_id"] == "r-1"
    assert record["level"] == "info"
    assert "timestamp" in record


def test_stdlib_loggers_are_rendered_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)

    logging.getLogger("uvicorn.error").warning("server says %s", "hi")

    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "server says hi"
    assert record["logger"] == "uvicorn.error"
    assert record["level"] == "warning"


def test_level_filters_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_logs=True)

    logging.getLogger("x").info("hidden")

    assert capsys.readouterr().out == ""


def test_console_renderer_for_local_development(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=False)

    logging.getLogger("x").info("readable")

    out = capsys.readouterr().out
    assert "readable" in out
    assert not out.lstrip().startswith("{")
