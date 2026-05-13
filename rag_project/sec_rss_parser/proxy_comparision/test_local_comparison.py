"""
test_local_comparison.py — Local-only pipeline test (no MongoDB, no S3).

Processes both filings from scratch (ingest -> classify -> section-map -> extract facts),
then runs the comparison and saves all output locally.

Usage:
    cd rag_project
    python -m sec_rss_parser.proxy_comparision.test_local_comparison
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from anthropic import Anthropic

from sec_rss_parser.proxy_comparision.models import CanonicalDocument
from sec_rss_parser.proxy_comparision.html_parser import ingest_filing
from sec_rss_parser.proxy_comparision.classifier import classify_blocks
from sec_rss_parser.proxy_comparision.section_mapper import build_sections
from sec_rss_parser.proxy_comparision.extractor import extract_priority_facts
from sec_rss_parser.proxy_comparision.differ import compute_pairwise_diff
from sec_rss_parser.proxy_comparision.report_writer import (
    generate_change_report, format_txt_header,
)
from sec_rss_parser.proxy_comparision.docx_builder import create_changes_docx
from sec_rss_parser.proxy_comparision.config import get_form_label

# ─────────────────────────────────────────────────────────────────────────────
# Records (from your Untitled-1)
# ─────────────────────────────────────────────────────────────────────────────

OLD_RECORD = {
    "_id": "308f58d0-786e-484a-900f-8a1271a8bdb1",
    "accession_number": "0001193125-26-117933",
    "cik_number": "0001824920",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/1824920/000119312526117933/d88629ds4.htm",
    "filing_date": "2026-03-20",
    "deal_id": "6977e4505fa114e4288a1f50",
    "form_type": "S-4",
}

NEW_RECORD = {
    "_id": "eccd50bb-8e3e-44d4-a9dc-54ed5646fb51",
    "accession_number": "0001193125-26-129336",
    "cik_number": "0001824920",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/1824920/000119312526129336/d88629ds4a.htm",
    "filing_date": "2026-03-27",
    "deal_id": "6977e4505fa114e4288a1f50",
    "form_type": "S-4/A",
}

# ─────────────────────────────────────────────────────────────────────────────
# Local output directory
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent / "local_test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _save_cache_json(doc: CanonicalDocument, label: str):
    """Save cache artifacts locally (priority_facts, topic_blocks, sections)."""
    sub = OUTPUT_DIR / label
    sub.mkdir(exist_ok=True)

    # Priority facts
    pf_path = sub / "priority_facts.json"
    pf_path.write_text(json.dumps(asdict(doc.priority_facts), indent=2, default=str))
    print(f"  [Local] Saved: {pf_path}")

    # Topic blocks
    topic_blocks = {}
    for block in doc.blocks:
        if block.topic and block.text.strip():
            topic_blocks.setdefault(block.topic, []).append(block.text)
    tb_path = sub / "topic_blocks.json"
    tb_path.write_text(json.dumps(topic_blocks, indent=2, ensure_ascii=False))
    print(f"  [Local] Saved: {tb_path}")

    # Sections
    sections_data = []
    for s in doc.sections:
        sections_data.append({
            "section_id": s.section_id,
            "raw_title": s.raw_title,
            "start_block_idx": s.start_block_idx,
            "end_block_idx": s.end_block_idx,
            "block_texts": [b.text for b in s.blocks],
            "block_topics": [b.topic for b in s.blocks],
            "block_types": [b.type for b in s.blocks],
        })
    sec_path = sub / "sections.json"
    sec_path.write_text(json.dumps(sections_data, indent=2, ensure_ascii=False))
    print(f"  [Local] Saved: {sec_path}")

    # Block-level dump for debugging
    blocks_data = [
        {"index": b.index, "type": b.type, "topic": b.topic, "text": b.text[:300]}
        for b in doc.blocks
    ]
    blocks_path = sub / "blocks_debug.json"
    blocks_path.write_text(json.dumps(blocks_data, indent=2, ensure_ascii=False))
    print(f"  [Local] Saved: {blocks_path}")


def build_cache_local(record: dict, client: Anthropic, label: str) -> CanonicalDocument:
    """Full pipeline: ingest -> classify -> section-map -> extract facts. All local."""
    url = record["sec_document_url"]
    form_type = record.get("form_type", "PROXY")

    print(f"\n{'='*72}")
    print(f"  Building cache: {label}  ({form_type})")
    print(f"  URL: {url}")
    print(f"{'='*72}")

    # Phase 1: Ingest HTML
    print(f"\n  [Phase 1] Ingesting filing...")
    doc = ingest_filing(url)
    # Override form_type from record (more reliable than URL guess)
    doc.form_type = form_type
    print(f"  Blocks: {len(doc.blocks)}")

    # Phase 2: Classify blocks
    print(f"\n  [Phase 2] Classifying blocks...")
    classify_blocks(doc, client)

    # Phase 3: Build sections
    print(f"\n  [Phase 3] Building sections...")
    build_sections(doc, client)

    # Phase 4: Extract priority facts
    print(f"\n  [Phase 4] Extracting priority facts...")
    extract_priority_facts(doc, client)

    # Save cache locally
    print(f"\n  [Phase 5] Saving cache locally...")
    _save_cache_json(doc, label)

    return doc


def run_local_comparison(old_doc: CanonicalDocument, new_doc: CanonicalDocument,
                         ticker: str, target: str, acquirer: str):
    """Run diff + report generation, save everything locally."""
    old_label = get_form_label(old_doc.form_type)
    new_label = get_form_label(new_doc.form_type)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    diag_dir = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    # Compute diff
    print(f"\n{'='*72}")
    print(f"  Running comparison: {old_label} -> {new_label}")
    print(f"{'='*72}")
    events = compute_pairwise_diff(
        client, old_doc, new_doc,
        deal_output_dir=str(diag_dir),
        filing_id="local_test",
    )

    # Save raw change events
    changes_data = [
        {
            "tier": e.tier,
            "category": e.category,
            "change_type": e.change_type,
            "field": e.field,
            "old_value": e.old_value,
            "new_value": e.new_value,
            "summary": e.summary,
            "confidence": e.confidence,
        }
        for e in events
    ]
    changes_path = OUTPUT_DIR / "changes.json"
    changes_path.write_text(json.dumps(changes_data, indent=2, ensure_ascii=False))
    print(f"\n  [Local] Saved changes JSON: {changes_path}")

    # Generate change report text
    print(f"\n  Generating change report text...")
    change_text = generate_change_report(client, events, ticker, old_label, new_label)
    timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")

    txt_content = format_txt_header(
        ticker, target,
        f"Changes: {old_label} -> {new_label}",
        "", timestamp,
    )
    txt_content += change_text + f"\n\n{'='*72}\n"

    txt_path = OUTPUT_DIR / "change_report.txt"
    txt_path.write_text(txt_content)
    print(f"  [Local] Saved change report TXT: {txt_path}")

    # Generate DOCX
    docx_path = OUTPUT_DIR / "change_report.docx"
    create_changes_docx(
        change_text, str(docx_path),
        ticker, target, acquirer,
        old_label, new_label, timestamp,
    )
    print(f"  [Local] Saved change report DOCX: {docx_path}")

    # Summary
    tier1 = sum(1 for e in events if e.tier == 1)
    tier2 = sum(1 for e in events if e.tier == 2)
    print(f"\n{'='*72}")
    print(f"  RESULTS")
    print(f"{'='*72}")
    print(f"  Tier 1 changes: {tier1}")
    print(f"  Tier 2 changes: {tier2}")
    print(f"  Total events:   {len(events)}")
    print(f"")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.relative_to(OUTPUT_DIR)}  ({size_kb:.1f} KB)")

    # Print other_material events specifically
    other_mat = [e for e in events if e.category == "other_material"]
    print(f"\n  OTHER MATERIAL events: {len(other_mat)}")
    for e in other_mat:
        print(f"    - [{e.change_type}] {e.field}: {(e.new_value or e.summary or '')[:100]}")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check your .env file.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Deal metadata (hardcoded for this test — avoids MongoDB lookup)
    TICKER = "SKYT"
    TARGET = "SkyWater Technology"
    ACQUIRER = "IonQ, Inc."

    # Step 1: Build cache for OLD record (S-4)
    old_doc = build_cache_local(OLD_RECORD, client, label="old_s4")

    # Step 2: Build cache for NEW record (S-4/A)
    new_doc = build_cache_local(NEW_RECORD, client, label="new_s4a")

    # Step 3: Run comparison
    events = run_local_comparison(old_doc, new_doc, TICKER, TARGET, ACQUIRER)
