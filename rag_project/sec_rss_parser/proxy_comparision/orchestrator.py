"""
orchestrator.py — Main entry point for the Proxy Comp Pipeline.
S3 + MongoDB only (no local paths). Compatible with plain Python (no Django).
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional

from .db import ProxyDB
from .s3_utils import (
    upload_json,
    upload_docx_bytes,
    upload_text,
    download_json_from_url,
    proxy_comp_key_suffix,
)
from .models import CanonicalDocument, PriorityFacts, Block
from .html_parser import ingest_filing
from .classifier import classify_blocks
from .section_mapper import build_sections
from .extractor import extract_priority_facts
from .differ import compute_pairwise_diff
from .report_writer import generate_change_report, format_txt_header
from .docx_builder import create_changes_docx
from .config import get_form_label, get_form_family


def _build_cache(record: dict, client, db: ProxyDB, deal_id: str) -> dict | None:
    """Ingest, classify, section-map, and extract facts for a single filing record.
    Uploads cached artifacts to S3 and updates MongoDB.
    Returns the cache dict on success (so caller can apply it if record is not in DB).
    """
    record_id = record["_id"] if record else None
    if not record_id:
        return None
    url = record["sec_document_url"]

    db.set_cache_status(record_id, status="building")

    try:
        # Phase 1: Ingest
        print(f"\n  [Cache] Ingesting: {url.split('/')[-1][:60]}")
        doc = ingest_filing(url)

        # Phase 1B: Classify blocks
        print(f"  [Cache] Classifying blocks...")
        classify_blocks(doc, client)

        # Phase 2: Build sections
        print(f"  [Cache] Building sections...")
        build_sections(doc, client)

        # Phase 3: Extract priority facts
        print(f"  [Cache] Extracting priority facts...")
        extract_priority_facts(doc, client)

        # Upload priority_facts JSON to S3
        priority_facts_data = asdict(doc.priority_facts)
        pf_key_suffix = proxy_comp_key_suffix(deal_id, record_id, "priority_facts.json")
        _, priority_facts_url = upload_json(priority_facts_data, pf_key_suffix)
        print(f"  [Cache] Uploaded priority facts: {priority_facts_url}")

        # Topic blocks JSON
        topic_blocks = {}
        for block in doc.blocks:
            if block.topic and block.text.strip():
                topic_blocks.setdefault(block.topic, []).append(block.text)
        tb_key_suffix = proxy_comp_key_suffix(deal_id, record_id, "topic_blocks.json")
        _, topic_blocks_url = upload_json(topic_blocks, tb_key_suffix)
        print(f"  [Cache] Uploaded topic blocks: {topic_blocks_url}")

        # Sections JSON
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
        sec_key_suffix = proxy_comp_key_suffix(deal_id, record_id, "sections.json")
        _, sections_url = upload_json(sections_data, sec_key_suffix)
        print(f"  [Cache] Uploaded sections: {sections_url}")

        filing_date_str = doc.filing_date if isinstance(doc.filing_date, str) else str(doc.filing_date) if doc.filing_date else "unknown"
        cache = {
            "status": "ready",
            "form_type": doc.form_type,
            "form_family": doc.doc_type_family,
            "filing_date": filing_date_str,
            "priority_facts_url": priority_facts_url,
            "topic_blocks_url": topic_blocks_url,
            "sections_url": sections_url,
        }
        db.set_cache_status(record_id, **cache)
        print(f"  [Cache] Record {record_id} cache status: ready")
        return cache

    except Exception as e:
        db.set_cache_status(record_id, status="error", error=str(e))
        raise


def _load_can_doc_from_cache(cache_node: dict) -> CanonicalDocument:
    """Reconstruct a CanonicalDocument from cached artifacts (S3 URLs)."""
    from .models import CanonicalSection

    form_type = cache_node.get("form_type", "PROXY")
    doc_type_family = cache_node.get("form_family", get_form_family(form_type))
    filing_date = cache_node.get("filing_date", "unknown")

    # Load priority_facts from S3 URL
    priority_facts = PriorityFacts()
    priority_facts_url = cache_node.get("priority_facts_url", "")
    if priority_facts_url:
        try:
            pf_data = download_json_from_url(priority_facts_url)
            priority_facts = PriorityFacts(
                dates=pf_data.get("dates", {}),
                consideration=pf_data.get("consideration", {}),
                financing=pf_data.get("financing", {}),
                sh_votes=pf_data.get("sh_votes", {}),
                regulatory=pf_data.get("regulatory", []),
                closing_guidance=pf_data.get("closing_guidance", {}),
            )
        except Exception as e:
            print(f"  [Cache] Warning: could not load priority_facts from URL: {e}")

    # Load topic_blocks from S3 URL
    blocks = []
    topic_blocks_url = cache_node.get("topic_blocks_url", "")
    if topic_blocks_url:
        try:
            topic_blocks_data = download_json_from_url(topic_blocks_url)
            idx = 0
            for topic, texts in topic_blocks_data.items():
                for text in texts:
                    blocks.append(Block(
                        type="paragraph",
                        text=text,
                        index=idx,
                        topic=topic,
                    ))
                    idx += 1
        except Exception as e:
            print(f"  [Cache] Warning: could not load topic_blocks from URL: {e}")

    # Load sections from S3 URL
    sections = []
    sections_url = cache_node.get("sections_url", "")
    if sections_url:
        try:
            sections_data = download_json_from_url(sections_url)
            for s_data in sections_data:
                s_blocks = []
                block_texts = s_data.get("block_texts", [])
                block_topics = s_data.get("block_topics", [])
                block_types = s_data.get("block_types", [])
                for i, text in enumerate(block_texts):
                    topic = block_topics[i] if i < len(block_topics) else ""
                    btype = block_types[i] if i < len(block_types) else "paragraph"
                    s_blocks.append(Block(
                        type=btype,
                        text=text,
                        index=s_data.get("start_block_idx", 0) + i,
                        topic=topic,
                    ))
                sections.append(CanonicalSection(
                    section_id=s_data["section_id"],
                    raw_title=s_data["raw_title"],
                    blocks=s_blocks,
                    start_block_idx=s_data.get("start_block_idx", 0),
                    end_block_idx=s_data.get("end_block_idx", 0),
                ))
        except Exception as e:
            print(f"  [Cache] Warning: could not load sections from URL: {e}")

    return CanonicalDocument(
        form_type=form_type,
        filing_date=filing_date,
        source_url="",
        doc_type_family=doc_type_family,
        blocks=blocks,
        sections=sections,
        priority_facts=priority_facts,
    )


def run_comparison(
    latest_doc_record: dict,
    past_doc_record: dict,
    env_path: Optional[Path] = None,
) -> dict:
    """Main entry point: compare two proxy filing records (S3 + MongoDB only).

    Args:
        latest_doc_record: dict with _id, sec_document_url, deal_id, form_type, proxy.*
        past_doc_record:   dict with _id, sec_document_url, deal_id, form_type, proxy.*
        env_path:          optional path to .env file

    Returns:
        dict with status, change counts, and S3 URLs for outputs.
    """
    # 1. Load .env if provided
    if env_path is not None:
        env_path = Path(env_path)
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

    # 2. Check ANTHROPIC_API_KEY
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Check your .env file.")

    # 3. Create Anthropic client
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # 4. Initialize MongoDB (sec_filing_summary)
    db = ProxyDB()
    print(f"[Pipeline] MongoDB: {db.db_and_collection} (comparison node will be written here)")

    # 5. Ensure both records exist in DB (upsert minimal doc if missing) so comparison node is persisted
    db.ensure_record_exists(past_doc_record)
    db.ensure_record_exists(latest_doc_record)

    # 6. Normalize _id to string for both records
    deal_id = latest_doc_record.get("deal_id", "unknown_deal")
    past_id = str(past_doc_record.get("_id", ""))
    latest_id = str(latest_doc_record.get("_id", ""))

    # 7. Build cache for past_doc if needed
    def _cache_node(record):
        if record is None:
            return {}
        return ((record.get("proxy") or {}).get("comparison") or {}).get("cache", {})

    def _apply_cache_to_record(record: dict, cache: dict) -> None:
        """Set proxy.comparison.cache on record when record is not in DB (e.g. test runner)."""
        if not record or not cache:
            return
        if record.get("proxy") is None:
            record["proxy"] = {}
        if record["proxy"].get("comparison") is None:
            record["proxy"]["comparison"] = {}
        record["proxy"]["comparison"]["cache"] = cache

    past_cache = _cache_node(past_doc_record)
    if past_cache.get("status") != "ready":
        print(f"\n[Pipeline] Building cache for past doc: {past_id}")
        past_cache_built = _build_cache(past_doc_record, client, db, deal_id)
        refetched = db.get_by_id(past_id)
        if refetched is not None:
            past_doc_record = refetched
        elif past_cache_built is not None:
            _apply_cache_to_record(past_doc_record, past_cache_built)
        past_cache = _cache_node(past_doc_record)

    # 8. Build cache for latest_doc if needed
    latest_cache = _cache_node(latest_doc_record)
    if latest_cache.get("status") != "ready":
        print(f"\n[Pipeline] Building cache for latest doc: {latest_id}")
        latest_cache_built = _build_cache(latest_doc_record, client, db, deal_id)
        refetched = db.get_by_id(latest_id)
        if refetched is not None:
            latest_doc_record = refetched
        elif latest_cache_built is not None:
            _apply_cache_to_record(latest_doc_record, latest_cache_built)
        latest_cache = _cache_node(latest_doc_record)

    # 9. Load both CanonicalDocuments from cache (S3 URLs)
    print(f"\n[Pipeline] Loading cached documents...")
    past_can_doc = _load_can_doc_from_cache(past_cache)
    latest_can_doc = _load_can_doc_from_cache(latest_cache)

    old_label = get_form_label(past_can_doc.form_type)
    new_label = get_form_label(latest_can_doc.form_type)
    ticker = latest_doc_record.get("ticker", deal_id)

    # 10. Temp dir for diagnostic files only (not persisted)
    temp_dir = tempfile.mkdtemp(prefix="proxy_comp_")

    # 11. Compute pairwise diff (diagnostics go to temp_dir)
    print(f"\n[Pipeline] Computing diff: {old_label} -> {new_label}")
    events = compute_pairwise_diff(
        client,
        past_can_doc,
        latest_can_doc,
        deal_output_dir=temp_dir,
        filing_id=latest_id,
    )

    # 12. Generate change report
    print(f"\n[Pipeline] Generating change report...")
    change_text = generate_change_report(client, events, ticker, old_label, new_label)
    file_timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")

    # 13. Build outputs and upload to S3
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
    changes_key_suffix = proxy_comp_key_suffix(deal_id, latest_id, "changes.json")
    _, changes_json_url = upload_json(changes_data, changes_key_suffix)

    txt_content = format_txt_header(
        ticker, deal_id,
        f"Changes: {old_label} -> {new_label}",
        "", file_timestamp,
    )
    txt_content += change_text + f"\n\n{'='*72}\n"
    txt_key_suffix = proxy_comp_key_suffix(deal_id, latest_id, "change_report.txt")
    _, change_txt_url = upload_text(txt_key_suffix, txt_content)

    target = latest_doc_record.get("target", deal_id)
    acquirer = latest_doc_record.get("acquirer", "TBD")
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_changes_docx(
            change_text, tmp_path,
            ticker, target, acquirer,
            old_label, new_label, file_timestamp,
        )
        with open(tmp_path, "rb") as f:
            docx_bytes = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    docx_key_suffix = proxy_comp_key_suffix(deal_id, latest_id, "change_report.docx")
    _, change_docx_url = upload_docx_bytes(docx_bytes, docx_key_suffix)

    # 14. Update MongoDB result status (S3 URLs)
    tier1_count = sum(1 for e in events if e.tier == 1)
    tier2_count = sum(1 for e in events if e.tier == 2)
    db.set_result_status(
        latest_id,
        status="complete",
        changes_json_url=changes_json_url,
        change_txt_url=change_txt_url,
        change_docx_url=change_docx_url,
        tier1_changes=tier1_count,
        tier2_changes=tier2_count,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n[Pipeline] Done.")
    print(f"  Tier 1 changes: {tier1_count}")
    print(f"  Tier 2 changes: {tier2_count}")
    print(f"  Changes JSON: {changes_json_url}")
    print(f"  Change report TXT: {change_txt_url}")
    print(f"  Change report DOCX: {change_docx_url}")

    return {
        "status": "complete",
        "tier1_changes": tier1_count,
        "tier2_changes": tier2_count,
        "changes_json_url": changes_json_url,
        "change_txt_url": change_txt_url,
        "change_docx_url": change_docx_url,
        "deal_id": deal_id,
    }
