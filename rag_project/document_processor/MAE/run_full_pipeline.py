#!/usr/bin/env python3
"""
Full MAE Pipeline: Single run → single final JSON.
Django-compatible version

All data flows in memory. Only two reads from disk:
  1. Deal input (static variable or load from JSON once)
  2. Benchmark: final_results/benchmark_20260211_151459.json

Only one write: final_analysis_{deal}_{timestamp}.json

Steps:
  Step 1: 6_prep_new_MAE.process_deals_data(DEAL_INPUT) → clauses
  Step 2: 6_classify (reads benchmark only).process_clauses(clauses) → classification
  Step 3: 7_risk.analyze_new_deal_from_data(classification) → risk_assessment
  Step 4: 8_compliance.analyze_deal_from_data(classification) → compliance
"""

import json
import math
import os
import re
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

# Django compatibility
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass


# ================================
# DEAL INPUT: set in memory (or load from file once)
# ================================
# Option A: Single deal from static variables (no file read). Set both:
DEAL_NAME = "United Homes Group"
DEAL_MAE_TEXT = """\"Company Material Adverse Effect\" means any state of facts, circumstance, condition, event, change, development, occurrence, result or effect (each, an "Effect") that, individually or in the aggregate with any one or more other Effects, (i) would prevent the ability of the Company to consummate the Transactions by the End Date, or (ii) has had, or would reasonably be expected to have, a material adverse effect on the business, financial condition, properties, assets, liabilities or results of operations of the Acquired Companies, taken as a whole. However, solely for purposes of a Company Material Adverse Effect under subclause (ii), no Effect relating to, or resulting or arising from any of the following matters shall be deemed to constitute a Company Material Adverse Effect or shall be considered in determining whether there has been, or would reasonably be expected to be, a Company Material Adverse Effect:

(A) any general economic, regulatory, political, business, financial or market conditions in the United States or elsewhere in the world;

(B) any changes in credit, debt, financial or capital markets or in interest or exchange rates, in each case, in the United States or elsewhere in the world;

(C) any conditions generally affecting the industries in which the Acquired Companies operate;

(D) any geopolitical conditions, any outbreak, continuation or escalation of any military conflict, declared or undeclared war, armed hostilities or acts of foreign or domestic terrorism (including cyberterrorism);

(E) any epidemic, pandemic (including COVID-19), plague or other outbreak of illness or public health event (or COVID-19 Measures or other restrictions that relate to, or arise out of, an epidemic, pandemic, plague or outbreak of illness or public health event);

(F) any hurricane, flood, tornado, earthquake or other natural disaster or act of God or Effect resulting from weather conditions;

(G) any failure by the Company or any of the Company Subsidiaries to meet any internal or external projections or forecasts or any decline in the price of Company Common Stock or other Company Securities (but excluding, in each case, the underlying causes of such failure or decline, as applicable, unless such underlying causes would otherwise be excepted from this definition);

(H) the public announcement or pendency of the Transactions, including, in any such case, the impact thereof on relationships, contractual or otherwise, with customers, suppliers, vendors, lenders, investors, licensors, licensees or venture partners or employees (provided that this clause (H) shall not apply to representations and warranties that specifically address the consequences of entry into this Agreement of the consummation of the transactions contemplated thereby);

(I) any changes resulting or arising from the identity of, or any facts or circumstances relating to, Parent, Merger Sub or any of their respective Affiliates;

(J) changes in Applicable Laws or the interpretation thereof;

(K) changes in GAAP or any other applicable accounting standards or the interpretation thereof;

(L) any action required to be taken by the Company pursuant to the terms of this Agreement or taken at the written direction of Parent or Merger Sub or the failure of the Company to take any action that requires consent of Parent to the extent Parent fails to give its consent thereto after a written request therefor;

(M) any breach of this Agreement by Parent or Merger Sub; or

(N) changes in the market price or trading volume of the Class A Common Stock (but excluding, in each case, the underlying causes of such changes, unless such underlying causes would otherwise be excepted from this definition).

Notwithstanding the foregoing, any Effect relating to or arising out of or resulting from any matter referred to in clause (A), (B), (C), (D), (E), (F), (J) or (K) above may constitute, and be taken into account in determining the occurrence of, a Company Material Adverse Effect if and only to the extent that such matter has a materially disproportionate adverse effect on the Acquired Companies, taken as a whole, as compared generally to other participants that operate in the industries in which the Acquired Companies operate."""

# Option B: Or set full DEAL_INPUT dict (overrides DEAL_NAME + DEAL_MAE_TEXT if set)
# DEAL_INPUT = {"mae_clauses": [{"dealName": "My Deal", "text": "..."}]}
#
# Option C: If neither set, load from UHG.json (only JSON read besides benchmark).


def _load_deal_input():
    path = _SCRIPT_DIR / "UHG.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"Deal input not found: {path}. Set DEAL_INPUT or create UHG.json")


def _sanitize_for_json(obj, non_finite_float=999.0):
    """Recursively replace inf/nan floats so final JSON is valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v, non_finite_float) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v, non_finite_float) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return non_finite_float
    return obj


_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent
# Use AWS S3 URL for benchmark (can also use local path)
BENCHMARK_FILE = "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/MAE_BenchMark/benchmark_MAE.json"


def _load_module(name: str, script_name: str):
    path = _SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup_django():
    """Setup Django environment for imports"""
    if str(_BASE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(_BASE_DIR.parent))
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
    try:
        django.setup()
    except Exception:
        pass  # Already setup


def fetch_mae_text_from_pinecone(deal_id: str) -> tuple[str, str] | None:
    """
    Fetch MAE clause text from Pinecone for a given deal_id.

    Args:
        deal_id: The deal ID to fetch chunks for

    Returns:
        Tuple of (deal_name, mae_text) or None if not found
    """
    try:
        # Import Django models
        _setup_django()
        from document_processor.models import ProcessingJob
        from document_processor.pinecone_utils import PineconeSectionFetcher

        # Get deal information
        try:
            deal = ProcessingJob.objects.get(id=deal_id)
            deal_name = deal.target_name or deal.acquire_name or f"Deal_{deal_id}"
            print(f"📄 Found deal: {deal_name}")
        except Exception as e:
            print(f"⚠️  Could not fetch deal name: {e}")
            deal_name = f"Deal_{deal_id}"

        # Fetch chunks from Pinecone
        print(f"🔍 Fetching chunks from Pinecone for deal_id: {deal_id}")
        fetcher = PineconeSectionFetcher()
        all_chunks = fetcher.get_all_chunks_for_deal(deal_id)

        if not all_chunks:
            print(f"❌ No chunks found in Pinecone for deal_id: {deal_id}")
            return None

        print(f"   Found {len(all_chunks)} total chunks")

        # Filter chunks by MAE-related labels
        mae_chunks = []
        target_labels = [
            "Definition > Company Material Adverse Effect",
            "Company Material Adverse Effect",
            "Definition > Material Adverse Effect"
        ]

        for chunk in all_chunks:
            label = chunk.get('label', '')
            # Check if label contains any of the target labels (case-insensitive)
            if any(target.lower() in label.lower() for target in target_labels):

                mae_chunks.append(chunk)
                print(f"   ✓ Found MAE chunk: {label}")

        if not mae_chunks:
            print(
                f"❌ No MAE chunks found with labels: {', '.join(target_labels)}")
            return None

        # Combine text from all MAE chunks
        mae_text = "\n\n".join([chunk.get('text', '') for chunk in mae_chunks])

        print(
            f"✅ Extracted MAE text: {len(mae_text)} characters from {len(mae_chunks)} chunks")

        return deal_name, mae_text

    except Exception as e:
        print(f"❌ Error fetching MAE text from Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_step1(deal_input: dict):
    """Step 1: Extract clauses from in-memory deal data. No file write."""
    from dotenv import load_dotenv
    load_dotenv()
    prep = _load_module("prep", "6_prep_new_MAE.py")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    deal_name, clauses = prep.process_deals_data(deal_input, api_key)
    return deal_name, clauses


def _run_step2(clauses: list, deal_name: str):
    """Step 2: Classify clauses. Reads benchmark JSON only. No other file I/O."""
    from dotenv import load_dotenv
    load_dotenv()
    classify_mod = _load_module("classify", "6_classify_new_clauses.py")
    ClauseClassifier = classify_mod.ClauseClassifier
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise RuntimeError("COHERE_API_KEY not set")
    # Check if benchmark exists (local file) or is a URL
    if not BENCHMARK_FILE.startswith("http"):
        if not Path(BENCHMARK_FILE).exists():
            raise FileNotFoundError(f"Benchmark not found: {BENCHMARK_FILE}")
    classifier = ClauseClassifier(BENCHMARK_FILE, cohere_key)
    classification = classifier.process_clauses(clauses, deal_name)
    return classification


def _run_step3(classification: dict):
    """Step 3: Risk assessment from in-memory classification. No file I/O."""
    from dotenv import load_dotenv
    load_dotenv()
    risk_mod = _load_module("risk", "7_new_MAE_risk.py")
    NewDealRiskAnalyzer = risk_mod.NewDealRiskAnalyzer
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    # Pass benchmark if it's a URL or if local file exists
    benchmark = BENCHMARK_FILE if (BENCHMARK_FILE.startswith(
        "http") or Path(BENCHMARK_FILE).exists()) else None
    analyzer = NewDealRiskAnalyzer(api_key, benchmark)
    risk_results = analyzer.analyze_new_deal_from_data(
        classification, analyze_all=False, auto_confirm=True
    )
    return risk_results or {}


def _run_step4(classification: dict):
    """Step 4: Compliance from in-memory classification. No file I/O."""
    from dotenv import load_dotenv
    load_dotenv()
    compliance_mod = _load_module("compliance", "8_LLM_Check_against_risks.py")
    ComplianceChecker = compliance_mod.ComplianceChecker
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    checker = ComplianceChecker(api_key)
    return checker.analyze_deal_from_data(classification)


def run_pipeline_for_deal_id(deal_id: str) -> dict | None:
    """
    Run MAE pipeline for a specific deal_id by fetching MAE text from Pinecone.

    Args:
        deal_id: The deal ID to process

    Returns:
        Dictionary with analysis results or None if MAE text not found
    """
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print(f"MAE PIPELINE – Processing deal_id: {deal_id}")
    print("=" * 80)

    # Step 0: Fetch MAE text from Pinecone
    print("\n📌 Step 0: Fetch MAE text from Pinecone")
    print("-" * 40)
    result = fetch_mae_text_from_pinecone(deal_id)

    if result is None:
        print("\n❌ Pipeline stopped: No MAE text found in Pinecone")
        print("   Make sure chunks with labels containing:")
        print("   - 'Definition > Company Material Adverse Effect'")
        print("   - 'Material Adverse Effect'")
        print("   exist for this deal_id")
        return None

    deal_name, mae_text = result
    print(f"   → Found MAE text for: {deal_name}")

    # Prepare deal input
    deal_input = {
        "mae_clauses": [{
            "dealName": deal_name,
            "text": mae_text
        }]
    }

    # Step 1: Extract clauses
    print("\n📌 Step 1: Extract clauses (6_prep_new_MAE)")
    print("-" * 40)
    deal_name, clauses = _run_step1(deal_input)
    print(f"   → {len(clauses)} clauses extracted")

    if not clauses:
        print("\n❌ Pipeline stopped: No clauses extracted")
        return None

    # Step 2: Classify clauses
    print("\n📌 Step 2: Classify clauses (6_classify)")
    print("-" * 40)
    classification = _run_step2(clauses, deal_name)
    print(f"   → {len(classification.get('results', []))} clauses classified")

    # Step 3: Risk assessment
    print("\n📌 Step 3: Risk assessment (7_risk)")
    print("-" * 40)
    risk_assessment = _run_step3(classification)
    print("   → Risk assessment complete")

    # Step 4: Compliance checks
    print("\n📌 Step 4: Compliance checks (8_compliance)")
    print("-" * 40)
    compliance = _run_step4(classification)
    print("   → Compliance checks complete")

    # Prepare final results
    final_results = {
        "deal_id": deal_id,
        "deal_name": deal_name,
        "pipeline_timestamp": datetime.now().isoformat(),
        "classification": classification,
        "risk_assessment": risk_assessment,
        "compliance": compliance,
    }

    # Save to file
    deal_safe = re.sub(r'[^\w\-]', '_', deal_name)[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _SCRIPT_DIR / f"final_analysis_{deal_safe}_{timestamp}.json"
    final_clean = _sanitize_for_json(final_results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_clean, f, indent=2, ensure_ascii=False)

    print(f"📄 Results saved to: {out_path}")

    # Save to MongoDB
    try:
        print("\n📌 Saving results to MongoDB...")
        _setup_django()
        from document_processor.models import MAEAnalysis

        # Prepare data for MongoDB
        mongo_data = {
            'deal_name': deal_name,
            'pipeline_timestamp': final_results['pipeline_timestamp'],
            'classification': final_results.get('classification'),
            'risk_assessment': final_results.get('risk_assessment'),
            'compliance': final_results.get('compliance')
        }

        # Save or update in MongoDB
        mae_record = MAEAnalysis.save_or_update(deal_id, mongo_data)
        print(f"✅ Saved to MongoDB: Collection 'mae_analyses', Deal ID: {deal_id}")
        print(f"   MongoDB Document ID: {mae_record.id}")

    except Exception as e:
        print(f"⚠️  Warning: Failed to save to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        print("   Pipeline results are still available in JSON file")

    print("\n" + "=" * 80)
    print("✅ Pipeline complete!")
    print(f"📄 JSON: {out_path}")
    print(f"💾 MongoDB: Collection 'mae_analyses', Deal ID: {deal_id}")
    print("=" * 80)

    return final_results


def main():
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("MAE FULL PIPELINE – in-memory, single final JSON only")
    print("=" * 80)
    print("Reads: deal input (variable or UHG.json) + benchmark JSON only.")
    print("Writes: final_analysis_*.json only.\n")

    # Deal input: DEAL_INPUT dict, or build from DEAL_NAME + DEAL_MAE_TEXT, or load UHG.json
    deal_input = globals().get("DEAL_INPUT")
    if deal_input is None and globals().get("DEAL_NAME") and globals().get("DEAL_MAE_TEXT"):
        deal_input = {"mae_clauses": [
            {"dealName": DEAL_NAME, "text": DEAL_MAE_TEXT}]}
    if deal_input is None:
        deal_input = _load_deal_input()
    n_deals = len(deal_input.get("mae_clauses", [])) if isinstance(
        deal_input, dict) else len(deal_input)
    print(f"📂 Deal input: in-memory ({n_deals} deal(s))")

    # Step 1
    print("\n📌 Step 1: Extract clauses (6_prep_new_MAE) – in-memory")
    print("-" * 40)
    deal_name, clauses = _run_step1(deal_input)
    print(f"   → {len(clauses)} clauses, deal: {deal_name}")

    # Step 2 (reads benchmark only)
    print("\n📌 Step 2: Classify clauses (6_classify) – benchmark from JSON only")
    print("-" * 40)
    classification = _run_step2(clauses, deal_name)
    print(f"   → {len(classification.get('results', []))} classified")

    # Step 3
    print("\n📌 Step 3: Risk assessment (7_risk) – in-memory")
    print("-" * 40)
    risk_assessment = _run_step3(classification)
    print("   → risk assessment done")

    # Step 4
    print("\n📌 Step 4: Compliance (8_compliance) – in-memory")
    print("-" * 40)
    compliance = _run_step4(classification)
    print("   → compliance done")

    # Only save: final JSON
    deal_safe = re.sub(r'[^\w\-]', '_', deal_name)[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = {
        "deal_name": deal_name,
        "pipeline_timestamp": datetime.now().isoformat(),
        "classification": classification,
        "risk_assessment": risk_assessment,
        "compliance": compliance,
    }
    out_path = _SCRIPT_DIR / f"final_analysis_{deal_safe}_{timestamp}.json"
    final_clean = _sanitize_for_json(final)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_clean, f, indent=2, ensure_ascii=False)

    print(f"   {out_path}")

    # Save to MongoDB (use deal_name as deal_id if no specific deal_id available)
    try:
        print("\n📌 Saving results to MongoDB...")
        _setup_django()
        from document_processor.models import MAEAnalysis

        # Use deal_name as unique identifier if no deal_id available
        unique_deal_id = deal_safe  # Use sanitized deal_name as unique ID

        # Prepare data for MongoDB
        mongo_data = {
            'deal_name': deal_name,
            'pipeline_timestamp': final['pipeline_timestamp'],
            'classification': final.get('classification'),
            'risk_assessment': final.get('risk_assessment'),
            'compliance': final.get('compliance')
        }

        # Save or update in MongoDB
        mae_record = MAEAnalysis.save_or_update(unique_deal_id, mongo_data)
        print(
            f"✅ Saved to MongoDB: Collection 'mae_analyses', Deal ID: {unique_deal_id}")
        print(f"   MongoDB Document ID: {mae_record.id}")

    except Exception as e:
        print(f"⚠️  Warning: Failed to save to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        print("   Pipeline results are still available in JSON file")

    print("\n" + "=" * 80)
    print("✅ Pipeline complete. Single output (only file written):")
    print(f"   JSON: {out_path}")
    print(f"   MongoDB: Collection 'mae_analyses'")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        raise
