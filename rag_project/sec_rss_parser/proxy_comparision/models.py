"""
models.py — Dataclasses for the Proxy Comp Pipeline.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Block:
    """A single semantic element from the filing HTML."""
    type: str           # "heading" | "paragraph" | "list_item" | "table"
    text: str           # cleaned text content
    meta: Dict[str, Any] = field(default_factory=dict)  # level, row_count, etc.
    index: int = 0      # position in document
    topic: str = ""     # content-based topic tag from classify_blocks()


@dataclass
class CanonicalSection:
    """A section of the filing mapped to a canonical ID."""
    section_id: str     # canonical ID: "background", "consideration_summary", etc.
    raw_title: str      # original title from the document
    blocks: List[Block] = field(default_factory=list)
    start_block_idx: int = 0
    end_block_idx: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\b[\w']+\b", self.text))


@dataclass
class PriorityFacts:
    """Structured facts extracted from a filing."""
    dates: Dict[str, Any] = field(default_factory=dict)
    consideration: Dict[str, Any] = field(default_factory=dict)
    financing: Dict[str, Any] = field(default_factory=dict)
    sh_votes: Dict[str, Any] = field(default_factory=dict)
    regulatory: List[Dict[str, Any]] = field(default_factory=list)
    closing_guidance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalDocument:
    """Fully parsed and enriched filing."""
    form_type: str
    filing_date: str
    source_url: str
    doc_type_family: str    # "proxy_like" | "registration_like" | "tender_like"
    blocks: List[Block] = field(default_factory=list)
    sections: List[CanonicalSection] = field(default_factory=list)
    priority_facts: PriorityFacts = field(default_factory=PriorityFacts)


@dataclass
class ChangeEvent:
    """A detected change between two filings."""
    tier: int                       # 1 or 2
    category: str                   # "background" | "dates" | "consideration" | etc.
    change_type: str                # "newly_disclosed" | "updated" | "removed" | "insert" | "other"
    field: Optional[str] = None     # specific field name
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    summary: Optional[str] = None   # LLM-generated description
    confidence: float = 1.0
    diff_detail: Optional[str] = None  # raw diff markup for appendix
