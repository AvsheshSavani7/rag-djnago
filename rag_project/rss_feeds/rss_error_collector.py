"""
Batch RSS pipeline errors per article and send one email at the end.

Use RSSArticleErrorRegistry in services.py for webhook processing, and
record_rss_error() from LLM/summary helpers to append errors without sending immediately.
"""
from __future__ import annotations

import traceback
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.exception_email import send_exception_email
from core.pipeline_logger import RSS

_active_collector: ContextVar[Optional["RSSArticleErrorCollector"]] = ContextVar(
    "rss_active_error_collector",
    default=None,
)


@dataclass
class RSSErrorEntry:
    step: str
    message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class RSSArticleErrorCollector:
    """Collects errors for one RSS article; sends a single digest email when flushed."""

    def __init__(
        self,
        article_url: Optional[str],
        feed_title: str,
        article_title: Optional[str],
        flow: str,
    ):
        self.article_url = article_url or ""
        self.feed_title = feed_title or ""
        self.article_title = article_title or ""
        self.flow = flow
        self.errors: List[RSSErrorEntry] = []

    def add(
        self,
        step: str,
        message: str,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        tb = None
        if exception is not None:
            tb = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        self.errors.append(
            RSSErrorEntry(
                step=step,
                message=message,
                traceback=tb,
                context=context,
            )
        )

    def has_errors(self) -> bool:
        return bool(self.errors)

    def send_if_errors(self) -> bool:
        if not self.errors:
            return False

        log_lines: List[str] = []
        structured: List[Dict[str, Any]] = []
        for index, entry in enumerate(self.errors, start=1):
            block = f"[{index}] {entry.step}: {entry.message}"
            if entry.traceback:
                block = f"{block}\n{entry.traceback.rstrip()}"
            if entry.context:
                block = f"{block}\ncontext: {entry.context}"
            log_lines.append(block)
            structured.append(
                {
                    "step": entry.step,
                    "message": entry.message,
                    "traceback": entry.traceback,
                    "context": entry.context,
                }
            )

        error_count = len(self.errors)
        summary = f"{error_count} error(s) during RSS processing for article"

        return send_exception_email(
            pipeline=RSS,
            error_message=summary,
            context={
                "module": "rss_feeds.rss_error_collector",
                "feed_title": self.feed_title,
                "article_url": self.article_url,
                "article_title": self.article_title,
                "flow": self.flow,
                "error_count": error_count,
            },
            log_records=log_lines,
            extra_data=structured,
            email_type="rss_article_error",
        )


class RSSArticleErrorRegistry:
    """
    Per-webhook registry keyed by article URL.

    Merger flow runs resolve_rss_item_flow and route_and_summarize in separate loops;
    this registry merges errors for the same article before flush_all().
    """

    def __init__(self, flow: str, feed_title: str):
        self.flow = flow
        self.feed_title = feed_title
        self._collectors: Dict[str, RSSArticleErrorCollector] = {}

    def _collector_key(self, article_url: Optional[str]) -> str:
        key = (article_url or "").strip()
        return key or f"unknown-{len(self._collectors)}"

    def get_collector(
        self,
        article_url: Optional[str],
        article_title: Optional[str] = None,
    ) -> RSSArticleErrorCollector:
        key = self._collector_key(article_url)
        if key not in self._collectors:
            self._collectors[key] = RSSArticleErrorCollector(
                article_url=article_url,
                feed_title=self.feed_title,
                article_title=article_title,
                flow=self.flow,
            )
        elif article_title and not self._collectors[key].article_title:
            self._collectors[key].article_title = article_title
        return self._collectors[key]

    def activate(self, collector: RSSArticleErrorCollector) -> Token:
        return _active_collector.set(collector)

    def deactivate(self, token: Token) -> None:
        _active_collector.reset(token)

    def flush_all(self) -> None:
        for collector in self._collectors.values():
            collector.send_if_errors()


def record_rss_error(
    step: str,
    message: str,
    exception: Optional[BaseException] = None,
    email_type: str = "rss_article_error",
    **context: Any,
) -> None:
    """
    Append an error to the active per-article collector, or send immediately if none.
    """
    collector = _active_collector.get()
    if collector is not None:
        collector.add(step, message, exception=exception, **context)
        return

    send_exception_email(
        pipeline=RSS,
        error_message=message,
        context={
            "module": "rss_feeds.rss_error_collector",
            "step": step,
            **context,
        },
        exception=exception,
        email_type=email_type,
    )
