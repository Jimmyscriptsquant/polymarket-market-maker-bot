from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog


def _console_renderer(logger: Any, name: str, event_dict: dict[str, Any]) -> str:
    """Human-readable console log format.

    Output: HH:MM:SS | LEVEL | message | key=val key=val ...
    """
    ts = event_dict.pop("timestamp", "")
    # Extract just the time portion (HH:MM:SS) from ISO timestamp
    if "T" in str(ts):
        ts = str(ts).split("T")[1][:8]

    level = event_dict.pop("level", "info").upper()
    msg = event_dict.pop("message", event_dict.pop("event", ""))
    logger_name = event_dict.pop("logger", "")

    # Color codes for terminal
    colors = {
        "DEBUG": "\033[90m",     # gray
        "INFO": "\033[36m",      # cyan
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    reset = "\033[0m"
    color = colors.get(level, "")

    # Format key=value pairs, skip verbose/internal fields
    skip_keys = {"_record", "_from_structlog"}
    extras = []
    for k, v in sorted(event_dict.items()):
        if k in skip_keys:
            continue
        # Truncate long values (token IDs, error messages)
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "..."
        extras.append(f"{k}={sv}")
    extra_str = "  ".join(extras)

    # Build the log line
    parts = [f"{ts}", f"{color}{level:<7}{reset}", f"{msg}"]
    if extra_str:
        parts.append(f"\033[90m{extra_str}{reset}")

    return " | ".join(parts)


def configure_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO") -> None:
    # Suppress noisy HTTP request logs from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            _console_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
