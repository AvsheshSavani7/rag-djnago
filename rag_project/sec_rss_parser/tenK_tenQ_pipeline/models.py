"""Data classes: DealContext, ParsedParagraph, FilingInfo."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DealContext:
    ticker: str
    target_company: str
    target_aliases: list
    acquirer_company: str
    acquirer_aliases: list
    merger_sub: Optional[str]
    deal_value: Optional[str]
    announcement_date: Optional[str]
    expected_close: Optional[str]
    deal_type: str
    key_terms: list
    raw_response: str = ""

    def save(self, path: Path) -> None:
        from dataclasses import asdict
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DealContext":
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("raw_response", None)
        if "raw_response" not in data:
            data["raw_response"] = ""
        return cls(**data)


@dataclass
class ParsedParagraph:
    index: int
    text: str
    section: str
    word_count: int
    is_header: bool = False
    relevance_score: int = 0
    rationale: str = ""
    key_info: list = field(default_factory=list)
    category: str = ""
    timing_assessment: str = ""
    timing_flag: bool = False
    regulatory_assessment: str = ""
    regulatory_flag: bool = False


@dataclass
class FilingInfo:
    """Metadata about a processed filing."""
    url: str
    period_date: str          # e.g., "2024-12-31"
    filing_type: str          # "10-K" or "10-Q"
    excerpts_path: Path
    excerpts: list = field(default_factory=list)
    label: str = ""           # e.g., "FY24 10-K" or "Q1 2025 10-Q"
