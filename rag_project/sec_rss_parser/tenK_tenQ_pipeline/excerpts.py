"""Generate and load excerpts JSON files."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .models import DealContext, ParsedParagraph


def generate_excerpts_json(
    paragraphs: List[ParsedParagraph],
    deal: DealContext,
    threshold: int,
    output_path: Path,
    source_url: Optional[str] = None,
    period_date: str = "",
    filing_type: str = "",
) -> None:
    relevant = sorted(
        [p for p in paragraphs if p.relevance_score >= threshold and not p.is_header],
        key=lambda x: x.index,
    )

    excerpts = []
    for para in relevant:
        highlight = "none"
        if para.timing_flag and para.regulatory_flag:
            highlight = "both"
        elif para.timing_flag:
            highlight = "timing"
        elif para.regulatory_flag:
            highlight = "regulatory"

        excerpts.append({
            "paragraph_index": para.index,
            "section": para.section,
            "relevance_score": para.relevance_score,
            "category": para.category,
            "timing_flag": para.timing_flag,
            "regulatory_flag": para.regulatory_flag,
            "highlight": highlight,
            "timing_assessment": para.timing_assessment,
            "regulatory_assessment": para.regulatory_assessment,
            "key_info": para.key_info,
            "text": para.text,
            "word_count": para.word_count,
        })

    output = {
        "metadata": {
            "ticker": deal.ticker,
            "target": deal.target_company,
            "acquirer": deal.acquirer_company,
            "deal_value": deal.deal_value,
            "deal_type": deal.deal_type,
            "expected_close": deal.expected_close,
            "source_url": source_url,
            "period_date": period_date,
            "filing_type": filing_type,
            "generated": datetime.now().isoformat(),
            "threshold": threshold,
        },
        "summary": {
            "total_paragraphs_scored": len([p for p in paragraphs if not p.is_header]),
            "relevant_excerpts": len(excerpts),
            "critical_excerpts": sum(1 for e in excerpts if e["relevance_score"] >= 8),
            "timing_flagged": sum(1 for e in excerpts if e["timing_flag"]),
            "regulatory_flagged": sum(1 for e in excerpts if e["regulatory_flag"]),
        },
        "excerpts": excerpts,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Saved: {output_path} ({len(excerpts)} excerpts)")


def _parse_excerpts_data(data: dict, threshold: int, source_label: str) -> Tuple[list, dict]:
    """Shared parsing logic for excerpts JSON (from file or URL)."""
    if "excerpts" in data:
        paragraphs = data["excerpts"]
        meta = data.get("metadata", {})
    elif "paragraphs" in data:
        paragraphs = [
            p for p in data["paragraphs"]
            if p.get("relevance_score", 0) >= threshold and not p.get("is_header", False)
        ]
        meta = {
            "ticker": data.get("ticker"),
            "target": data.get("target"),
            "acquirer": data.get("acquirer"),
        }
    else:
        raise ValueError("Unrecognized JSON format: missing 'excerpts' or 'paragraphs'")

    for p in paragraphs:
        p["_filing_source"] = source_label

    return paragraphs, meta


def load_excerpts(filepath: Path, threshold: int = 6) -> Tuple[list, dict]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    source_label = filepath.stem
    return _parse_excerpts_data(data, threshold, source_label)


def load_excerpts_from_url(url: str, threshold: int = 6, source_label: str = None) -> Tuple[list, dict]:
    """Fetch excerpts JSON from URL (e.g. S3) and return (paragraphs, meta)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    label = source_label or url.split("/")[-1].replace(".json", "") if url else "unknown"
    return _parse_excerpts_data(data, threshold, label)
