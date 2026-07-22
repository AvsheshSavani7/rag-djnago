"""
Logging setup for SEC feed poller test scripts (collector + processor).

Writes to console and a daily log file under sec_daily_feed/logs/ (or SEC_FEED_LOG_DIR).
"""

import logging
import os
import time
from typing import Optional

from sec_rss_parser.sec_feed_daily_store import feed_now


def default_log_dir(feed_dir: str) -> str:
    return os.environ.get("SEC_FEED_LOG_DIR", os.path.join(feed_dir, "logs"))


def configure_test_logger(role: str, feed_dir: str, level: int = logging.INFO) -> str:
    """
    Configure root logger with console + daily file handler.

    role: e.g. 'collector' or 'processor'
    Returns absolute path to today's log file.
    """
    log_dir = default_log_dir(feed_dir)
    os.makedirs(log_dir, exist_ok=True)
    day = feed_now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{role}_{day}.log")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def log_elapsed_ms(
    logger: logging.Logger,
    prefix: str,
    step: str,
    started: float,
    **fields,
) -> float:
    """Log elapsed milliseconds since started (monotonic). Returns elapsed ms."""
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if fields:
        extra = " | ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(
            "%s timing | %s | %.1f ms | %s",
            prefix,
            step,
            elapsed_ms,
            extra,
        )
    else:
        logger.info("%s timing | %s | %.1f ms", prefix, step, elapsed_ms)
    return elapsed_ms
