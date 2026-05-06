"""Tracing hooks.

Emits spans to LangSmith when LANGSMITH_API_KEY is configured, and always records
wall-clock duration in the returned span dict for local use.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# LangSmith client is initialised lazily at import time so that the module is
# still importable in environments without the API key set.
_ls_client: Any = None

def _get_langsmith_client() -> Any:
    global _ls_client  # noqa: PLW0603
    if _ls_client is not None:
        return _ls_client
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return None
    try:
        from langsmith import Client
        _ls_client = Client(api_key=api_key)
    except Exception:
        logger.debug("LangSmith client initialisation failed; falling back to local tracing only.")
        _ls_client = None
    return _ls_client


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that times a span and, when LangSmith is configured, records
    it as a LangSmith run of type 'chain'.

    The yielded dict always contains:
      - ``name`` – span name
      - ``attributes`` – copy of input attributes
      - ``duration_seconds`` – wall-clock duration (populated on exit)
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    client = _get_langsmith_client()
    run_id: Any = None

    if client is not None:
        try:
            import uuid
            from datetime import datetime, timezone

            run_id = uuid.uuid4()
            client.create_run(
                id=run_id,
                name=name,
                run_type="chain",
                inputs=attributes or {},
                start_time=datetime.now(tz=timezone.utc),
                project_name=os.environ.get("LANGSMITH_PROJECT", "multi-agent-research-lab"),
            )
        except Exception:
            logger.debug("LangSmith create_run failed for span=%s", name, exc_info=True)
            run_id = None

    try:
        yield span
    except Exception as exc:
        if client is not None and run_id is not None:
            try:
                from datetime import datetime, timezone

                client.update_run(
                    run_id,
                    end_time=datetime.now(tz=timezone.utc),
                    error=str(exc),
                )
            except Exception:
                logger.debug("LangSmith update_run (error) failed for span=%s", name, exc_info=True)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started

        if client is not None and run_id is not None:
            try:
                from datetime import datetime, timezone

                client.update_run(
                    run_id,
                    end_time=datetime.now(tz=timezone.utc),
                    outputs={"duration_seconds": span["duration_seconds"]},
                )
            except Exception:
                logger.debug("LangSmith update_run failed for span=%s", name, exc_info=True)
