#!/usr/local/bin/python3
"""
Stage 10 (Termination): HTML Dashboard Generator — v2

Redesigned dashboard. Shows what this termination agreement *means*,
not how it scores vs a benchmark.

Layout:
  1. Header         — Deal ID, timestamp, deal-type label (not a risk badge)
  2. Termination Guide — top-of-page table: Trigger | Who Invokes | Fee | Direction | Notes
  3. Structural Terms — small cards: tail, specific performance, willful breach, sole remedy
  4. Clause Audit   — STANDARD / NON-STANDARD cards; missing triggers at top
  5. Provision Checks — present/absent cards (Stage 9); no risk scores
  6. Full Clause Table — reference table at bottom

Removed entirely:
  - Risk scores (0-10)
  - Benchmark comparison section
  - Party balance bar chart
  - "Outlier clauses" count
  - Investigation priority labels

Usage:
    python3 10_generate_dashboard_termination.py {deal_id}

Example:
    python3 10_generate_dashboard_termination.py d937868dex21
"""

import json
import os
import sys
import glob
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
TERMINATION_DIR = os.path.join(PROJECT_ROOT, "Termination_Embeddings_v1")
NEW_DEAL_REPORTS = os.path.join(TERMINATION_DIR, "new_deal_reports")
DASHBOARD_OUTPUT = os.path.join(TERMINATION_DIR, "dashboard_output")

STANDARD_SIMILARITY_THRESHOLD = 0.80

STANDARD_TRIGGER_TYPES = [
    "mutual_consent",
    "outside_date",
    "regulatory_block",
    "shareholder_vote_failure",
    "target_fiduciary_out",
    "target_breach",
    "acquirer_breach",
]

INVOKER_MAP = {
    "mutual_consent":            "Either party",
    "outside_date":              "Either party",
    "regulatory_block":          "Either party",
    "shareholder_vote_failure":  "Either party",
    "target_fiduciary_out":      "Company (target)",
    "acquirer_fiduciary_out":    "Parent (acquirer)",
    "target_breach":             "Parent (acquirer)",
    "acquirer_breach":           "Company (target)",
    "financing_failure":         "Company (target)",
    "other":                     "See clause",
}

# Maps each normalized trigger_type to (ctf_aliases, rtf_aliases, exp_aliases).
# See generate_termination_summary.py for full design notes.
TRIGGER_TYPE_TO_FEE_ALIASES = {
    "target_breach":            ({"company_breach", "target_breach", "company_terminable_breach"},
                                 set(), set()),
    "acquirer_breach":          (set(),
                                 {"acquirer_breach", "parent_breach", "parent_terminable_breach"},
                                 set()),
    "target_fiduciary_out":     ({"adverse_recommendation_change", "company_change_of_recommendation",
                                  "target_fiduciary_out"},
                                 set(), set()),
    "acquirer_fiduciary_out":   (set(),
                                 {"adverse_recommendation_change", "parent_change_of_recommendation",
                                  "acquirer_fiduciary_out"},
                                 set()),
    "regulatory_block":         (set(),
                                 {"regulatory_block", "regulatory_termination", "antitrust_block"},
                                 set()),
    "outside_date":             (set(), set(), set()),
    "financing_failure":        (set(),
                                 {"financing_failure", "financing_condition"},
                                 set()),
    "shareholder_vote_failure": (set(), set(), set()),
    "mutual_consent":           (set(), set(), set()),
}

PROVISION_DESCRIPTIONS = {
    "outside_date_duration":  "Sets the deadline by which the merger must close before either party can walk away.",
    "regulatory_rtf":         "Reverse termination fee that specifically triggers when a regulatory body blocks the deal.",
    "tail_provision":         "Extends the CTF obligation for a period after termination if a competing deal closes.",
    "specific_performance":   "Allows the non-breaching party to demand the deal actually close, not just collect a fee.",
    "willful_breach":         "Carveout that lets damages exceed the termination fee if a party intentionally breaches.",
    "matching_rights":        "Gives the acquirer a window to match any superior proposal before the target accepts it.",
    "financing_failure":      "Termination right if the acquirer cannot obtain the debt financing needed to close.",
    "unilateral_extension":   "Ability for one party to extend the outside date without the other's consent.",
}

PROVISION_LABELS = {
    "outside_date_duration": "Outside Date Duration",
    "regulatory_rtf":        "Regulatory / Antitrust RTF",
    "tail_provision":        "Tail Provision",
    "specific_performance":  "Specific Performance",
    "willful_breach":        "Willful Breach Carveout",
    "matching_rights":       "Matching Rights",
    "financing_failure":     "Financing Failure Trigger",
    "unilateral_extension":  "Unilateral Outside Date Extension",
}


# ================================
# FILE DISCOVERY
# ================================

def latest_file(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_json(path: Optional[str], label: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        print(f"  {label:<45}: not found")
        return None
    with open(path, "r") as fh:
        data = json.load(fh)
    print(f"  {label:<45}: {os.path.basename(path)}")
    return data


def _fmt_usd(n) -> Optional[str]:
    """Format a raw USD number as $XM or $XB."""
    if n is None:
        return None
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.0f}M"
    return f"${n:,.0f}"


def load_deal_financials(deal_id: str) -> Dict:
    """Load deal equity value and expense reimbursement from merged benchmark files."""
    result = {
        'deal_equity_value_usd': None,
        'deal_equity_value_text': None,
        'expense_reimbursement_usd': None,
        'expense_reimbursement_text': None,
    }
    dv_file = os.path.join(PROJECT_ROOT, 'MERGED_deal_values.json')
    if os.path.exists(dv_file):
        with open(dv_file) as f:
            for item in json.load(f):
                if item.get('deal_id') == deal_id:
                    result['deal_equity_value_usd'] = item.get('deal_equity_value_usd')
                    result['deal_equity_value_text'] = item.get('deal_equity_value_text')
                    break
    er_file = os.path.join(PROJECT_ROOT, 'MERGED_expense_reimbursement.json')
    if os.path.exists(er_file):
        with open(er_file) as f:
            for item in json.load(f):
                if item.get('deal_id') == deal_id and item.get('amount_usd'):
                    result['expense_reimbursement_usd'] = item.get('amount_usd')
                    result['expense_reimbursement_text'] = item.get('amount_text')
                    break
    return result


def load_company_name(deal_id: str) -> Optional[str]:
    """Load target company name from MERGED_company_names.json."""
    cn_file = os.path.join(PROJECT_ROOT, 'MERGED_company_names.json')
    if os.path.exists(cn_file):
        with open(cn_file) as f:
            for item in json.load(f):
                if item.get('deal_id') == deal_id:
                    return item.get('target_name')
    return None


def discover_files(deal_id: str) -> Dict:
    """Discover all available data files for a deal, including 8-K sources."""
    print(f"\nDiscovering files for deal: {deal_id}")

    # Merger agreement sources
    fees_path = os.path.join(PROJECT_ROOT, f"termination_response_{deal_id}_fees.json")
    triggers_path = os.path.join(PROJECT_ROOT, f"termination_response_{deal_id}_triggers.json")

    fees = load_json(fees_path if os.path.exists(fees_path) else None, "Fees (Agreement)")
    triggers = load_json(triggers_path if os.path.exists(triggers_path) else None, "Triggers (Agreement)")

    # 8-K / press release sources
    fees_8k_path = os.path.join(PROJECT_ROOT, f"termination_response_{deal_id}_fees_8k.json")
    if not os.path.exists(fees_8k_path):
        fees_8k_path = latest_file(f"{NEW_DEAL_REPORTS}/termination_8k_fees_{deal_id}_*.json")
    triggers_8k_path = os.path.join(PROJECT_ROOT, f"termination_response_{deal_id}_triggers_8k.json")

    fees_8k = load_json(
        fees_8k_path if fees_8k_path and os.path.exists(fees_8k_path) else None,
        "Fees (8-K/Press Release)"
    )
    triggers_8k = load_json(
        triggers_8k_path if os.path.exists(triggers_8k_path) else None,
        "Triggers (8-K/Press Release)"
    )

    classification = load_json(
        latest_file(f"{NEW_DEAL_REPORTS}/termination_classification_{deal_id}_*.json"),
        "Stage 6 classification"
    )
    assessment = load_json(
        latest_file(f"{NEW_DEAL_REPORTS}/termination_assessment_{deal_id}_*.json"),
        "Stage 7 assessment"
    )
    provisions = load_json(
        latest_file(f"{NEW_DEAL_REPORTS}/termination_provision_checks_{deal_id}_*.json"),
        "Stage 9 provision checks"
    )

    return {
        "fees": fees,
        "triggers": triggers,
        "fees_8k": fees_8k,
        "triggers_8k": triggers_8k,
        "classification": classification,
        "assessment": assessment,
        "provisions": provisions,
    }


# ================================
# MULTI-SOURCE MERGE
# ================================

FEE_FIELD_NAMES = [
    "company_termination_fee", "reverse_termination_fee",
    "parent_regulatory_termination_fee", "expense_reimbursement",
]
SCALAR_FIELD_NAMES = [
    "offer_price_per_share", "deal_equity_value_usd", "deal_equity_value_text",
    "specific_performance_available", "willful_breach_carveout",
    "tail_provision_months", "sole_remedy_for_acquirer", "outside_date_months",
    "go_shop_period_days",
]
META_KEYS = {"document_id", "extraction_source", "extracted_at", "token_usage",
             "source_url", "source_document"}


def merge_fees_with_provenance(
    fees_agmt: Optional[Dict], fees_8k: Optional[Dict]
) -> Tuple[Dict, Dict]:
    """Merge fee data from agreement and 8-K; agreement takes precedence.

    Returns (merged_fees, provenance) where provenance maps field -> source label.
    """
    if not fees_agmt and not fees_8k:
        return {}, {}
    if not fees_8k:
        prov = {k: "Agreement" for k in (fees_agmt or {}) if k not in META_KEYS}
        return fees_agmt or {}, prov
    if not fees_agmt:
        prov = {k: "8-K" for k in (fees_8k or {}) if k not in META_KEYS}
        return fees_8k or {}, prov

    # Both exist — agreement wins per field, 8-K fills gaps
    merged = dict(fees_agmt)
    prov = {}

    for field in FEE_FIELD_NAMES:
        agmt_val = (fees_agmt.get(field) or {})
        pr_val = (fees_8k.get(field) or {})
        agmt_amount = agmt_val.get("amount_usd") if isinstance(agmt_val, dict) else None
        pr_amount = pr_val.get("amount_usd") if isinstance(pr_val, dict) else None

        if agmt_amount is not None:
            prov[field] = "Agreement"
        elif pr_amount is not None:
            merged[field] = fees_8k.get(field)
            prov[field] = "8-K"

    for field in SCALAR_FIELD_NAMES:
        agmt_val = fees_agmt.get(field)
        pr_val = fees_8k.get(field)
        if agmt_val is not None:
            prov[field] = "Agreement"
        elif pr_val is not None:
            merged[field] = pr_val
            prov[field] = "8-K"

    return merged, prov


def build_discrepancies(
    fees_agmt: Optional[Dict], fees_8k: Optional[Dict]
) -> List[Dict]:
    """Find fields where agreement and 8-K sources disagree."""
    if not fees_agmt or not fees_8k:
        return []

    disc = []
    fee_labels = {
        "company_termination_fee": "Company Termination Fee",
        "reverse_termination_fee": "Reverse Termination Fee",
        "parent_regulatory_termination_fee": "Parent Regulatory Term. Fee",
        "expense_reimbursement": "Expense Reimbursement",
    }
    for field, label in fee_labels.items():
        agmt_val = (fees_agmt.get(field) or {})
        pr_val = (fees_8k.get(field) or {})
        agmt_amt = agmt_val.get("amount_usd") if isinstance(agmt_val, dict) else None
        pr_amt = pr_val.get("amount_usd") if isinstance(pr_val, dict) else None

        if agmt_amt is not None and pr_amt is not None and agmt_amt != pr_amt:
            disc.append({
                "field": label,
                "agreement_value": _fmt_usd(agmt_amt),
                "press_release_value": _fmt_usd(pr_amt),
                "type": "amount_mismatch",
            })
        elif agmt_amt is None and pr_amt is not None:
            disc.append({
                "field": label,
                "agreement_value": "Not found",
                "press_release_value": _fmt_usd(pr_amt),
                "type": "gap_agreement_missing",
            })

    return disc


# ================================
# DATA BUILDERS
# ================================

def _split_fee_text(amount_text: str, fallback_label: str) -> tuple:
    """Split '$200,000,000 (from definition: Parent Termination Fee)' into
    ('$200,000,000', 'Parent Termination Fee').  Returns (fallback_label, '') if empty."""
    if not amount_text:
        return (f"{fallback_label} (amount defined elsewhere)", "")
    idx = amount_text.find(" (from definition:")
    if idx != -1:
        amount = amount_text[:idx].strip()
        source = amount_text[idx:].strip().lstrip("(from definition:").rstrip(")")
        # cleaner strip
        source = amount_text[idx + len(" (from definition:"):].rstrip(")").strip()
        return (amount, source)
    return (amount_text.strip(), "")


def build_guide_rows(fees: Dict, triggers: Dict) -> List[Dict]:
    """Build one row per trigger_type for the termination guide table."""
    clauses = triggers.get("clauses", [])
    trigger_types = []
    seen = set()
    for c in clauses:
        tt = c.get("trigger_type", "")
        if tt and tt != "preamble" and tt not in seen:
            trigger_types.append(tt)
            seen.add(tt)

    ctf_info = fees.get("company_termination_fee", {}) or {}
    rtf_info = fees.get("reverse_termination_fee", {}) or {}
    # Fall back to parent_regulatory_termination_fee if standard RTF has no amount
    if not (rtf_info.get("amount_usd")):
        reg_info = fees.get("parent_regulatory_termination_fee", {}) or {}
        if reg_info.get("amount_usd"):
            rtf_info = reg_info
    exp_info = fees.get("expense_reimbursement", {}) or {}

    ctf_triggers = set(ctf_info.get("triggers", []) or [])
    rtf_triggers = set(rtf_info.get("triggers", []) or [])
    exp_triggers = set(exp_info.get("triggers", []) or [])

    ctf_amount_text = ctf_info.get("amount_text", "") or ""
    rtf_amount_text = rtf_info.get("amount_text", "") or ""

    tail_months = fees.get("tail_provision_months")
    sole_remedy = fees.get("sole_remedy_for_acquirer", False)

    ctf_triggers_lower = {s.lower() for s in ctf_triggers}
    rtf_triggers_lower = {s.lower() for s in rtf_triggers}
    exp_triggers_lower = {s.lower() for s in exp_triggers}

    rows = []
    for tt in trigger_types:
        invoker = INVOKER_MAP.get(tt, "See clause")

        ctf_al, rtf_al, exp_al = TRIGGER_TYPE_TO_FEE_ALIASES.get(tt, (set(), set(), set()))
        ctf_lower = {a.lower() for a in ctf_al}
        rtf_lower = {a.lower() for a in rtf_al}
        exp_lower = {a.lower() for a in exp_al}

        fee_amount = "None"
        fee_source = ""
        fee_type = ""        # "CTF" | "RTF" | "EXP" | ""
        direction = "none"   # none | company_pays | parent_pays
        notes = ""

        exp_usd = exp_info.get("amount_usd")
        exp_fmt = _fmt_usd(exp_usd)
        # Check if this trigger maps to expense reimbursement.
        # The fees JSON may use different naming (e.g. "company_breach") than the
        # triggers JSON (e.g. "target_breach"), so check both the raw name and
        # any CTF-side aliases which cover the same event.
        trigger_has_exp = bool(
            tt.lower() in exp_triggers_lower
            or ctf_lower & exp_triggers_lower
        )

        if ctf_lower & ctf_triggers_lower:
            ctf_display = _fmt_usd(ctf_info.get("amount_usd"))
            fee_amount = ctf_display if ctf_display else "CTF"
            fee_source = ""
            fee_type = "CTF"
            direction = "company_pays"
            note_parts = []
            if sole_remedy:
                note_parts.append("Sole remedy upon payment")
            if tail_months:
                note_parts.append(f"Tail: {tail_months} mo")
            notes = "; ".join(note_parts)

        elif rtf_lower & rtf_triggers_lower:
            rtf_display = _fmt_usd(rtf_info.get("amount_usd"))
            fee_amount = rtf_display if rtf_display else "RTF"
            fee_source = ""
            fee_type = "RTF"
            direction = "parent_pays"
            note_parts = []
            if sole_remedy:
                note_parts.append("Sole remedy upon payment")
            rtf_notes = rtf_info.get("notes", "") or ""
            if "regulatory" in rtf_notes.lower() or "antitrust" in rtf_notes.lower():
                note_parts.append("Separate regulatory fee for antitrust block")
            notes = "; ".join(note_parts)

        rows.append({
            "trigger_type": tt,
            "invoker": invoker,
            "fee_amount": fee_amount,
            "fee_source": fee_source,
            "fee_type": fee_type,
            "direction": direction,
            "notes": notes,
        })

    return rows


def build_clause_audit(classification: Dict) -> Dict:
    classified_clauses = classification.get("classified_clauses", [])
    present_trigger_types = set()
    audit_rows = []

    for clause in classified_clauses:
        tt = clause.get("trigger_type", "")
        similarity = clause.get("similarity_score", 0.0)
        cluster_theme = clause.get("cluster_theme", "")
        cluster_cat = clause.get("cluster_trigger_category", "")
        clause_cat = _infer_category(tt)

        type_mismatch = (
            cluster_cat and clause_cat
            and cluster_cat != clause_cat
            and tt not in ("preamble", "")
        )

        is_standard = (similarity >= STANDARD_SIMILARITY_THRESHOLD) and not type_mismatch

        explanation = ""
        if not is_standard:
            reasons = []
            if similarity < STANDARD_SIMILARITY_THRESHOLD:
                reasons.append(f"Similarity {similarity:.1%} is below {STANDARD_SIMILARITY_THRESHOLD:.0%} threshold")
            if type_mismatch:
                reasons.append(
                    f"Trigger category mismatch: clause is '{clause_cat}' "
                    f"but matched cluster category '{cluster_cat}'"
                )
            explanation = "; ".join(reasons)

        if tt and tt != "preamble":
            present_trigger_types.add(tt)

        audit_rows.append({
            "clause_id": clause.get("clause_id", ""),
            "trigger_type": tt,
            "cluster_theme": cluster_theme,
            "similarity_score": similarity,
            "status": "STANDARD" if is_standard else "NON-STANDARD",
            "type_mismatch": type_mismatch,
            "explanation": explanation,
        })

    missing_standard = [t for t in STANDARD_TRIGGER_TYPES if t not in present_trigger_types]

    signals = []
    if "acquirer_fiduciary_out" in present_trigger_types:
        signals.append({
            "trigger": "acquirer_fiduciary_out",
            "label": "Merger of Equals",
            "explanation": (
                "Parent requires its own stockholder approval and has a fiduciary out. "
                "This is characteristic of a stock-for-stock merger between two public companies, "
                "not a traditional acquisition where only the target votes."
            ),
        })
    if "financing_failure" in present_trigger_types:
        signals.append({
            "trigger": "financing_failure",
            "label": "PE / LBO",
            "explanation": (
                "A financing failure termination right indicates the buyer's obligation is "
                "contingent on debt financing. This structure is typical of private equity "
                "leveraged buyouts, not all-cash or all-stock strategic deals."
            ),
        })

    return {
        "audit_rows": audit_rows,
        "missing_standard_triggers": missing_standard,
        "notable_signals": signals,
        "present_trigger_types": present_trigger_types,
    }


def _infer_category(trigger_type: str) -> str:
    mapping = {
        "mutual_consent":           "mutual_termination",
        "shareholder_vote_failure": "mutual_termination",
        "outside_date":             "outside_date",
        "regulatory_block":         "regulatory_failure",
        "target_breach":            "breach_based",
        "acquirer_breach":          "breach_based",
        "target_fiduciary_out":     "fiduciary_out",
        "acquirer_fiduciary_out":   "fiduciary_out",
        "financing_failure":        "financing_failure",
    }
    return mapping.get(trigger_type, "")


def derive_deal_type_label(classification: Dict, fees: Dict) -> str:
    present = {
        c.get("trigger_type")
        for c in classification.get("classified_clauses", [])
        if c.get("trigger_type") and c.get("trigger_type") != "preamble"
    }
    if "acquirer_fiduciary_out" in present:
        return "Merger of Equals"
    if "financing_failure" in present:
        return "PE / Leveraged Buyout"
    if "regulatory_block" in present:
        ctf = (fees.get("company_termination_fee") or {}).get("amount_usd") or 0
        rtf = (fees.get("reverse_termination_fee") or {}).get("amount_usd") or 0
        if not rtf:
            rtf = (fees.get("parent_regulatory_termination_fee") or {}).get("amount_usd") or 0
        if rtf > 0 and ctf > 0 and (rtf / ctf) >= 1.5:
            return "Regulatory-Sensitive Deal"
    return "Strategic Acquisition"


# ================================
# HTML GENERATION
# ================================

class TerminationDashboardGenerator:

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_dashboard(self, deal_id: str) -> Optional[str]:
        print("\n" + "=" * 80)
        print("STAGE 10 (TERMINATION): HTML DASHBOARD GENERATOR v2")
        print("=" * 80)

        files = discover_files(deal_id)
        fees_agmt = files["fees"]
        triggers = files["triggers"]
        fees_8k = files["fees_8k"]
        triggers_8k = files["triggers_8k"]
        classification = files["classification"]
        assessment = files["assessment"]
        provisions = files["provisions"]

        if not classification:
            print(f"\nERROR: Stage 6 classification not found for deal: {deal_id}")
            print(f"  Looked in: {NEW_DEAL_REPORTS}")
            return None

        actual_deal_id = classification.get("deal_id", deal_id)
        analysis_ts = classification.get("analysis_timestamp", self.timestamp)

        print(f"\nGenerating dashboard for: {actual_deal_id}")

        # Multi-source merge: agreement takes precedence, 8-K fills gaps
        fees, provenance = merge_fees_with_provenance(fees_agmt, fees_8k)
        discrepancies = build_discrepancies(fees_agmt, fees_8k)

        sources = []
        if fees_agmt:
            sources.append("Merger Agreement")
        if fees_8k:
            sources.append("8-K / Press Release")

        if sources:
            print(f"  Sources: {', '.join(sources)}")
        if discrepancies:
            print(f"  Discrepancies: {len(discrepancies)} field(s) differ between sources")

        # Auto-extract deal value (upserts into deal_financials.json without wiping other deals)
        extract_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_deal_value.py")
        if os.path.exists(extract_script):
            print(f"  Running deal value extraction for {actual_deal_id}...")
            result = subprocess.run(
                [sys.executable, extract_script, actual_deal_id],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  Warning: deal value extraction failed: {result.stderr.strip()[:200]}")
            else:
                print(f"  Deal value extraction complete.")

        # Build data components
        guide_rows = build_guide_rows(fees, triggers) if (fees and triggers) else []
        audit = build_clause_audit(classification)
        deal_type_label = derive_deal_type_label(classification, fees) if fees else "Strategic Acquisition"
        deal_financials = load_deal_financials(actual_deal_id)
        company_name = load_company_name(actual_deal_id)

        html = self._build_html(
            actual_deal_id,
            analysis_ts,
            deal_type_label,
            guide_rows,
            audit,
            fees,
            classification,
            assessment,
            provisions,
            deal_financials,
            company_name,
            sources,
            provenance,
            discrepancies,
        )

        Path(DASHBOARD_OUTPUT).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(
            DASHBOARD_OUTPUT,
            f"termination_dashboard_{actual_deal_id}_{self.timestamp}.html"
        )
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"\nDashboard written: {output_file}")
        print("Opening in browser...")
        webbrowser.open("file://" + os.path.abspath(output_file))

        return output_file

    # ------------------------------------------------------------------
    # Top-level HTML builder
    # ------------------------------------------------------------------

    def _build_html(
        self,
        deal_id: str,
        analysis_ts: str,
        deal_type_label: str,
        guide_rows: List[Dict],
        audit: Dict,
        fees: Optional[Dict],
        classification: Dict,
        assessment: Optional[Dict],
        provisions: Optional[Dict],
        deal_financials: Optional[Dict] = None,
        company_name: Optional[str] = None,
        sources: Optional[List[str]] = None,
        provenance: Optional[Dict] = None,
        discrepancies: Optional[List[Dict]] = None,
    ) -> str:

        discrepancy_html = self._discrepancy_alert_html(discrepancies) if discrepancies else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name or deal_id} — Termination Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
{self._css()}
    </style>
</head>
<body>
<div class="container">

    <!-- 1. HEADER -->
{self._section_header(deal_id, analysis_ts, deal_type_label, company_name, sources)}

{discrepancy_html}

    <!-- 2. TERMINATION GUIDE -->
    <div class="section">
        <div class="section-header">Termination Guide &mdash; What Happens in Each Scenario</div>
        <div class="section-body">
{self._termination_guide_html(guide_rows, fees, deal_financials, provenance)}
        </div>
    </div>

    <!-- 3. STRUCTURAL TERMS -->
    <div class="section">
        <div class="section-header">Structural Terms</div>
        <div class="section-body">
{self._structural_terms_html(fees, provenance)}
        </div>
    </div>

    <!-- 4. CLAUSE AUDIT + MISSING PROVISIONS -->
    <div class="section">
        <div class="section-header">Clause Audit</div>
        <div class="section-body" style="padding:0">
{self._clause_audit_html(audit, classification, assessment, provisions)}
        </div>
    </div>


</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return """        :root {
            --bg-primary: #0a0e14;
            --bg-secondary: #0f1419;
            --bg-tertiary: #151c24;
            --bg-hover: #1a232e;
            --text-primary: #e6e6e6;
            --text-secondary: #8a919a;
            --text-muted: #5c6370;
            --accent-blue: #5ccfe6;
            --accent-green: #87d96c;
            --accent-yellow: #ffcc66;
            --accent-red: #f07178;
            --accent-orange: #ff8c42;
            --font-mono: 'JetBrains Mono', monospace;
            --font-sans: 'IBM Plex Sans', sans-serif;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: var(--font-sans); background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; }
        .container { max-width: 1600px; margin: 0 auto; padding: 16px; }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 4px;
            padding: 14px 20px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            font-family: var(--font-mono);
        }
        .deal-id { font-size: 1.15rem; font-weight: 600; color: var(--text-primary); }
        .deal-meta { font-size: 0.72rem; color: var(--text-muted); margin-top: 3px; }
        .deal-type-label {
            margin-left: auto;
            padding: 5px 14px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(92,207,230,0.12);
            color: var(--accent-blue);
            border: 1px solid rgba(92,207,230,0.3);
            white-space: nowrap;
        }

        /* Section */
        .section {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 4px;
            margin-bottom: 14px;
            overflow: hidden;
        }
        .section-header {
            padding: 10px 16px;
            border-bottom: 1px solid var(--bg-tertiary);
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-blue);
        }
        .section-body { padding: 14px 16px; }

        /* Guide table */
        .guide-table { width: 100%; border-collapse: collapse; }
        .guide-table th {
            font-family: var(--font-mono);
            text-align: left;
            padding: 8px 12px;
            font-weight: 500;
            font-size: 0.67rem;
            color: var(--text-muted);
            text-transform: uppercase;
            background: var(--bg-tertiary);
        }
        .guide-table td {
            font-family: var(--font-mono);
            padding: 9px 12px;
            border-bottom: 1px solid var(--bg-tertiary);
            font-size: 0.76rem;
            vertical-align: top;
        }
        .guide-table tbody tr:last-child td { border-bottom: none; }
        .guide-note { font-size: 0.68rem; color: var(--text-muted); margin-top: 2px; }
        .guide-notes-cell { max-width: 180px; white-space: normal; word-wrap: break-word; }
        .fee-summary {
            display: flex;
            gap: 12px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }
        .fee-summary-item {
            background: var(--bg-tertiary);
            border-radius: 4px;
            padding: 7px 12px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 120px;
        }
        .fee-summary-label {
            font-family: var(--font-mono);
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .fee-summary-amount {
            font-family: var(--font-mono);
            font-size: 0.95rem;
            font-weight: 600;
        }
        .fee-summary-pct {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            color: var(--text-muted);
            margin-top: 1px;
        }

        /* Structural terms cards */
        .terms-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .term-card {
            background: var(--bg-tertiary);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 4px;
            padding: 12px 14px;
        }
        .term-label {
            font-family: var(--font-mono);
            font-size: 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .term-value {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            font-weight: 600;
        }
        .term-sub {
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 3px;
        }

        /* Badge */
        .badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 3px;
            font-size: 0.68rem;
            font-weight: 500;
            font-family: var(--font-mono);
        }
        .badge-red    { background: rgba(240,113,120,0.15); color: var(--accent-red); }
        .badge-yellow { background: rgba(255,204,102,0.15); color: var(--accent-yellow); }
        .badge-green  { background: rgba(135,217,108,0.15); color: var(--accent-green); }
        .badge-blue   { background: rgba(92,207,230,0.15);  color: var(--accent-blue); }
        .badge-muted  { background: rgba(92,99,112,0.2);    color: var(--text-muted); }

        /* Clause audit cards */
        .audit-missing {
            background: rgba(240,113,120,0.07);
            border: 1px solid rgba(240,113,120,0.25);
            border-radius: 4px;
            padding: 10px 14px;
            margin-bottom: 12px;
        }
        .audit-missing-title {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--accent-red);
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .audit-missing-list {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .audit-missing-list li {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            padding: 2px 8px;
            background: rgba(240,113,120,0.1);
            border: 1px solid rgba(240,113,120,0.2);
            border-radius: 3px;
            color: var(--accent-red);
        }
        .audit-signals {
            background: rgba(92,207,230,0.06);
            border: 1px solid rgba(92,207,230,0.2);
            border-radius: 4px;
            padding: 10px 14px;
            margin-bottom: 12px;
        }
        .audit-signals-title {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .signal-item { margin-bottom: 8px; }
        .signal-label {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-blue);
        }
        .signal-explanation { font-size: 0.76rem; color: var(--text-secondary); margin-top: 2px; }
        /* Clause audit rows */
        .audit-row {
            display: grid;
            grid-template-columns: 160px 1fr 70px 110px 20px;
            align-items: center;
            padding: 9px 16px;
            border-bottom: 1px solid var(--bg-tertiary);
            cursor: pointer;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            gap: 12px;
        }
        .audit-row:hover { background: var(--bg-hover); }
        .audit-row:last-child { border-bottom: none; }
        .audit-row.non-standard { border-left: 3px solid var(--accent-yellow); }
        .audit-row.standard     { border-left: 3px solid transparent; }
        .audit-trigger { font-weight: 600; }
        .audit-theme   { color: var(--text-secondary); font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .audit-chevron { color: var(--text-muted); font-size: 0.65rem; text-align: right; transition: transform 0.15s; }
        .audit-chevron.open { transform: rotate(90deg); }
        .audit-detail {
            display: none;
            background: var(--bg-tertiary);
            padding: 12px 16px 12px 20px;
            border-bottom: 1px solid var(--bg-tertiary);
            border-left: 3px solid var(--accent-yellow);
        }
        .audit-detail.show { display: block; }
        .detail-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; margin-top: 8px; }
        .detail-label:first-child { margin-top: 0; }
        .detail-text { font-size: 0.74rem; color: var(--text-secondary); line-height: 1.55; font-family: var(--font-mono); }

        /* Source provenance tags */
        .source-tag {
            display: inline-block;
            padding: 1px 5px;
            border-radius: 2px;
            font-size: 0.55rem;
            font-weight: 500;
            font-family: var(--font-mono);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            background: rgba(92,207,230,0.1);
            color: var(--accent-blue);
            border: 1px solid rgba(92,207,230,0.2);
            vertical-align: middle;
            margin-left: 4px;
        }

        /* Discrepancy alert */
        .discrepancy-section {
            background: var(--bg-secondary);
            border: 1px solid rgba(255,140,66,0.35);
            border-left: 3px solid var(--accent-orange);
            border-radius: 4px;
            margin-bottom: 14px;
            overflow: hidden;
        }
        .discrepancy-header {
            padding: 10px 16px;
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-orange);
            border-bottom: 1px solid rgba(255,140,66,0.15);
        }
        .discrepancy-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            padding: 8px 16px;
            font-family: var(--font-mono);
            font-size: 0.74rem;
            border-bottom: 1px solid var(--bg-tertiary);
            align-items: center;
        }
        .discrepancy-row:last-child { border-bottom: none; }
        .discrepancy-field { color: var(--text-primary); font-weight: 500; }
        .discrepancy-val { color: var(--text-secondary); }
        .discrepancy-label {
            font-size: 0.62rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Missing provisions */
        .missing-prov-section {
            padding: 14px 16px;
            border-top: 1px solid var(--bg-tertiary);
        }
        .missing-prov-title {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            margin-bottom: 10px;
        }
        .missing-prov-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 8px;
        }
        .missing-prov-card {
            background: var(--bg-primary);
            border: 1px solid rgba(255,204,102,0.2);
            border-left: 3px solid var(--accent-yellow);
            border-radius: 4px;
            padding: 9px 12px;
        }
        .missing-prov-name {
            font-family: var(--font-mono);
            font-size: 0.74rem;
            font-weight: 600;
            color: var(--accent-yellow);
            margin-bottom: 3px;
        }
        .missing-prov-desc { font-size: 0.70rem; color: var(--text-secondary); }

"""

    # ------------------------------------------------------------------
    # HTML section builders
    # ------------------------------------------------------------------

    def _section_header(self, deal_id: str, analysis_ts: str, deal_type_label: str,
                        company_name: Optional[str] = None, sources: Optional[List[str]] = None) -> str:
        if company_name:
            name_html = f'<div class="deal-id">{company_name}</div><div class="deal-meta">{deal_id} &middot; Termination Analysis &middot; {analysis_ts}</div>'
        else:
            name_html = f'<div class="deal-id">{deal_id}</div><div class="deal-meta">Termination Analysis &middot; {analysis_ts}</div>'

        # Source badges
        source_badges = ""
        if sources and len(sources) > 0:
            badges = ""
            for src in sources:
                badge_cls = "badge-green" if "Agreement" in src else "badge-blue"
                badges += f' <span class="badge {badge_cls}" style="font-size:0.62rem">{src}</span>'
            source_badges = f'<div class="deal-meta" style="margin-top:4px">Sources:{badges}</div>'

        return f"""    <div class="header">
        <div>
            {name_html}
            {source_badges}
        </div>
        <div class="deal-type-label">{deal_type_label}</div>
    </div>"""

    def _discrepancy_alert_html(self, discrepancies: List[Dict]) -> str:
        """Render a warning section when agreement and 8-K sources disagree."""
        if not discrepancies:
            return ""
        rows = ""
        for d in discrepancies:
            type_label = "Mismatch" if d["type"] == "amount_mismatch" else "Gap"
            rows += f"""        <div class="discrepancy-row">
            <div class="discrepancy-field">{d['field']} <span class="badge badge-yellow" style="font-size:0.58rem">{type_label}</span></div>
            <div class="discrepancy-val"><span class="discrepancy-label">Agreement: </span>{d['agreement_value']}</div>
            <div class="discrepancy-val"><span class="discrepancy-label">8-K: </span>{d['press_release_value']}</div>
        </div>\n"""
        return f"""    <div class="discrepancy-section">
        <div class="discrepancy-header">Source Discrepancies &mdash; Review Recommended</div>
{rows}    </div>"""

    def _termination_guide_html(self, guide_rows: List[Dict], fees: Optional[Dict] = None,
                                deal_financials: Optional[Dict] = None, provenance: Optional[Dict] = None) -> str:
        if not guide_rows:
            return '<div style="color:var(--text-muted);font-size:0.8rem">Fee and trigger data not available.</div>'

        # Extract raw USD amounts for percentage calculation
        deal_value_usd = (deal_financials or {}).get('deal_equity_value_usd')
        deal_value_text = (deal_financials or {}).get('deal_equity_value_text')
        exp_reimb_text = (deal_financials or {}).get('expense_reimbursement_text')
        exp_reimb_usd = (deal_financials or {}).get('expense_reimbursement_usd')

        # Fall back to fees JSON for expense reimbursement if not in merged file
        if not exp_reimb_usd and fees:
            exp_info = (fees.get('expense_reimbursement') or {})
            exp_reimb_usd = exp_info.get('amount_usd')
            exp_reimb_text = exp_info.get('amount_text')

        ctf_usd = (fees.get('company_termination_fee', {}) or {}).get('amount_usd') if fees else None
        rtf_usd = (fees.get('reverse_termination_fee', {}) or {}).get('amount_usd') if fees else None
        # Fall back to parent_regulatory_termination_fee if standard RTF has no amount
        if not rtf_usd and fees:
            rtf_usd = (fees.get('parent_regulatory_termination_fee', {}) or {}).get('amount_usd')

        def _pct(fee_usd, deal_usd):
            if fee_usd and deal_usd and deal_usd > 0:
                return f"{fee_usd / deal_usd * 100:.1f}%"
            return None

        summary_items = ""

        # Deal equity value tile
        if deal_value_text:
            summary_items += f"""        <div class="fee-summary-item">
                <div class="fee-summary-label" style="color:var(--text-secondary)">Deal Equity Value</div>
                <div class="fee-summary-amount" style="color:var(--text-primary)">{deal_value_text}</div>
            </div>"""

        ctf_display = _fmt_usd(ctf_usd)
        ctf_pct = _pct(ctf_usd, deal_value_usd)
        pct_html = f'<div class="fee-summary-pct">{ctf_pct} of deal value</div>' if ctf_pct else ''
        ctf_color = "var(--text-primary)" if ctf_display else "var(--text-muted)"
        ctf_src = (provenance or {}).get("company_termination_fee")
        ctf_src_badge = f'<span class="source-tag">{ctf_src}</span>' if ctf_src else ''
        summary_items += f"""        <div class="fee-summary-item">
                <div class="fee-summary-label" style="color:var(--accent-yellow)">CTF {ctf_src_badge}</div>
                <div class="fee-summary-amount" style="color:{ctf_color}">{ctf_display or "Unknown"}</div>
                {pct_html}
            </div>"""

        rtf_display = _fmt_usd(rtf_usd)
        if rtf_display:
            rtf_pct = _pct(rtf_usd, deal_value_usd)
            pct_html = f'<div class="fee-summary-pct" style="color:var(--accent-blue)">{rtf_pct} of deal value</div>' if rtf_pct else ''
            rtf_src = (provenance or {}).get("reverse_termination_fee")
            rtf_src_badge = f'<span class="source-tag">{rtf_src}</span>' if rtf_src else ''
            summary_items += f"""        <div class="fee-summary-item">
                <div class="fee-summary-label" style="color:var(--accent-blue)">RTF {rtf_src_badge}</div>
                <div class="fee-summary-amount" style="color:var(--text-primary)">{rtf_display}</div>
                {pct_html}
            </div>"""

        summary_html = f'        <div class="fee-summary">{summary_items}</div>' if summary_items else ""

        # Expense reimbursement — two fixed boxes: one for acquirer, one for target
        exp_info_raw = (fees.get("expense_reimbursement") or {}) if fees else {}
        exp_triggers_list = exp_info_raw.get("triggers", []) or []
        exp_notes_raw = exp_info_raw.get("notes", "") or ""

        # Classify direction based on triggers
        ACQUIRER_SIDE = {"company_breach", "target_breach", "fiduciary_out",
                         "adverse_recommendation_change", "superior_proposal",
                         "tail_provision", "shareholder_vote_failure"}
        PARENT_SIDE   = {"acquirer_breach", "parent_breach", "regulatory_block",
                         "regulatory_termination", "financing_failure", "antitrust_block"}

        exp_to_acquirer_usd  = None
        exp_to_target_usd    = None
        exp_to_acquirer_trig = []
        exp_to_target_trig   = []

        if exp_reimb_usd:
            t_lower = {t.lower() for t in exp_triggers_list}
            if t_lower & ACQUIRER_SIDE:
                exp_to_acquirer_usd  = exp_reimb_usd
                exp_to_acquirer_trig = [t for t in exp_triggers_list if t.lower() in ACQUIRER_SIDE]
            if t_lower & PARENT_SIDE:
                exp_to_target_usd  = exp_reimb_usd
                exp_to_target_trig = [t for t in exp_triggers_list if t.lower() in PARENT_SIDE]
            # Fallback: if no direction matched, assume acquirer-side (most common)
            if not exp_to_acquirer_usd and not exp_to_target_usd:
                exp_to_acquirer_usd  = exp_reimb_usd
                exp_to_acquirer_trig = exp_triggers_list

        def _amt_qualifier(amount_usd, amount_text, notes):
            """Return formatted amount with qualifier prefix if capped/variable."""
            base = _fmt_usd(amount_usd)
            if not base:
                return None
            combined = ((amount_text or "") + " " + (notes or "")).lower()
            if any(q in combined for q in ("up to", "not to exceed", "not more than", "maximum of", "capped at")):
                return f"Up to {base}"
            return base

        def _reimb_box(title, direction_note, amount_usd, amount_text, notes, triggers):
            if amount_usd:
                amt_str = _amt_qualifier(amount_usd, amount_text, notes) or _fmt_usd(amount_usd)
                pct_val = _pct(amount_usd, deal_value_usd)
                pct_html = f'<span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted)"> &middot; {pct_val} of deal value</span>' if pct_val else ''
                trig_label = ", ".join(t.replace("_", " ").title() for t in triggers) if triggers else "See agreement"
                content = f"""
                    <div style="font-family:var(--font-mono);font-size:0.88rem;font-weight:600;color:var(--text-secondary);margin-bottom:3px">{amt_str}{pct_html}</div>
                    <div style="font-size:0.70rem;color:var(--text-muted)">Trigger: {trig_label}</div>"""
            else:
                content = '<div style="font-size:0.72rem;color:var(--text-muted);font-style:italic">Not present in this agreement</div>'
            return f"""
                <div style="background:var(--bg-tertiary);border-radius:4px;padding:9px 14px">
                    <div style="font-size:0.88rem;font-weight:600;color:var(--text-primary);margin-bottom:2px">{title}</div>
                    <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:6px">{direction_note}</div>
                    {content}
                </div>"""

        exp_amount_text = exp_info_raw.get("amount_text", "") or ""
        acquirer_box = _reimb_box(
            "Acquirer Reimbursement Expenses",
            "Company reimburses acquirer for documented out-of-pocket costs",
            exp_to_acquirer_usd, exp_amount_text, exp_notes_raw, exp_to_acquirer_trig
        )
        target_box = _reimb_box(
            "Target Reimbursement Expenses",
            "Acquirer reimburses target for documented out-of-pocket costs",
            exp_to_target_usd, exp_amount_text, exp_notes_raw, exp_to_target_trig
        )
        exp_reimb_html = f"""
            <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">
                {acquirer_box}
                {target_box}
            </div>"""

        dir_labels = {
            "none":         "None",
            "company_pays": "Company &rarr; Parent",
            "parent_pays":  "Parent &rarr; Company",
        }

        rows_html = ""
        for r in guide_rows:
            dir_text = dir_labels.get(r["direction"], "&mdash;")
            notes_html = f'<div class="guide-note">{r["notes"]}</div>' if r["notes"] else ""
            fee_type = r.get("fee_type", "")
            amount_colors = {"CTF": "var(--accent-yellow)", "RTF": "var(--accent-blue)", "EXP": "var(--text-secondary)"}
            amount_color = amount_colors.get(fee_type, "var(--text-muted)")
            fee_html = f'<span style="font-size:0.9rem;font-weight:600;color:{amount_color}">{r["fee_amount"]}</span>'
            trigger_label = r['trigger_type'].replace('_', ' ').title()
            rows_html += f"""                <tr>
                    <td>{trigger_label}</td>
                    <td>{r['invoker']}</td>
                    <td>{fee_html}</td>
                    <td>{dir_text}</td>
                    <td class="guide-notes-cell">{notes_html}</td>
                </tr>\n"""

        return f"""{summary_html}
            <table class="guide-table">
                <thead>
                    <tr>
                        <th>Trigger</th>
                        <th>Who Invokes</th>
                        <th>Fee</th>
                        <th>Direction</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
{rows_html}                </tbody>
            </table>{exp_reimb_html}"""

    def _structural_terms_html(self, fees: Optional[Dict], provenance: Optional[Dict] = None) -> str:
        if not fees:
            return '<div style="color:var(--text-muted);font-size:0.8rem">Fee data not available.</div>'

        tail = fees.get("tail_provision_months")
        sp = fees.get("specific_performance_available")
        wb = fees.get("willful_breach_carveout")
        sole = fees.get("sole_remedy_for_acquirer")

        if tail:
            tail_value = f"{tail} months"
            tail_color = "var(--accent-green)"
        else:
            tail_value = "None"
            tail_color = "var(--text-muted)"

        if sp is True:
            sp_value = "Available"
            sp_color = "var(--accent-green)"
            sp_sub = "Can compel closing; not limited to fee collection"
        elif sp is False:
            sp_value = "Not Available"
            sp_color = "var(--accent-red)"
            sp_sub = "RTF is exclusive remedy"
        else:
            sp_value = "Not Specified"
            sp_color = "var(--text-muted)"
            sp_sub = ""

        if wb is True:
            wb_value = "Survives Fee Payment"
            wb_color = "var(--accent-green)"
            wb_sub = "Willful breach exposes party to additional damages"
        elif wb is False:
            wb_value = "Capped by Fee"
            wb_color = "var(--accent-yellow)"
            wb_sub = "Fee payment is exclusive remedy even for willful breach"
        else:
            wb_value = "Not Specified"
            wb_color = "var(--text-muted)"
            wb_sub = ""

        if sole is True:
            sole_value = "Yes"
            sole_color = "var(--accent-yellow)"
            sole_sub = "Fee payment extinguishes all other remedies"
        elif sole is False:
            sole_value = "No"
            sole_color = "var(--accent-green)"
            sole_sub = "Additional remedies remain available"
        else:
            sole_value = "Not Specified"
            sole_color = "var(--text-muted)"
            sole_sub = ""

        prov = provenance or {}

        def card(label, value, color, sub="", field_key=""):
            sub_html = f'<div class="term-sub">{sub}</div>' if sub else ""
            src = prov.get(field_key)
            src_html = f' <span class="source-tag">{src}</span>' if src else ""
            return f"""                <div class="term-card">
                    <div class="term-label">{label}{src_html}</div>
                    <div class="term-value" style="color:{color}">{value}</div>
                    {sub_html}
                </div>"""

        return f"""            <div class="terms-row">
{card("Tail Provision", tail_value, tail_color, field_key="tail_provision_months")}
{card("Specific Performance", sp_value, sp_color, sp_sub, field_key="specific_performance_available")}
{card("Willful Breach", wb_value, wb_color, wb_sub, field_key="willful_breach_carveout")}
{card("Sole Remedy", sole_value, sole_color, sole_sub, field_key="sole_remedy_for_acquirer")}
            </div>"""

    def _clause_audit_html(self, audit: Dict, classification: Dict, assessment: Optional[Dict], provisions: Optional[Dict]) -> str:
        parts = []

        # Build lookup: clause_id -> original_text from classification
        original_texts = {
            c.get("clause_id", ""): c.get("original_text", "")
            for c in classification.get("classified_clauses", [])
        }

        # Build lookup: clause_id -> assessment detail from Stage 7
        assessment_by_clause = {}
        if assessment:
            for c in assessment.get("assessed_clauses", []):
                cid = c.get("clause_id", "")
                if cid:
                    assessment_by_clause[cid] = c

        # Notable signals banner
        signals = audit["notable_signals"]
        if signals:
            signal_items = ""
            for sig in signals:
                signal_items += f"""            <div class="signal-item" style="padding:10px 16px;border-bottom:1px solid var(--bg-tertiary)">
                    <div class="signal-label">{sig['label']} &mdash; {sig['trigger'].replace('_',' ').title()}</div>
                    <div class="signal-explanation">{sig['explanation']}</div>
                </div>"""
            parts.append(f"""        <div class="audit-signals" style="border-radius:0;border:none;border-bottom:1px solid var(--bg-tertiary);margin:0;padding:0">
                <div class="audit-signals-title" style="padding:8px 16px 0">Notable Signals</div>
{signal_items}        </div>""")

        # Missing standard triggers banner
        missing = audit["missing_standard_triggers"]
        if missing:
            chips = "".join(
                f'<li>{m.replace("_"," ").title()}</li>' for m in missing
            )
            parts.append(f"""        <div class="audit-missing" style="border-radius:0;border:none;border-bottom:1px solid var(--bg-tertiary);margin:0;padding:10px 16px">
                <div class="audit-missing-title">Missing Standard Triggers</div>
                <ul class="audit-missing-list">{chips}</ul>
            </div>""")

        # Missing provisions from Stage 9 (absent only) — placed next to missing triggers
        if provisions:
            checks = provisions.get("provisions_checked", {})
            absent_cards = ""
            for key, label in PROVISION_LABELS.items():
                info = checks.get(key, {})
                if info.get("present", False):
                    continue
                desc = PROVISION_DESCRIPTIONS.get(key, "")
                absent_cards += f"""            <div class="missing-prov-card">
                    <div class="missing-prov-name">{label}</div>
                    <div class="missing-prov-desc">{desc}</div>
                </div>"""
            if absent_cards:
                parts.append(f"""        <div class="missing-prov-section" style="border-top:none;border-bottom:1px solid var(--bg-tertiary)">
            <div class="missing-prov-title">Provisions Not Present in this Document</div>
            <div class="missing-prov-grid">
{absent_cards}            </div>
        </div>""")
            else:
                parts.append("""        <div class="missing-prov-section" style="border-top:none;border-bottom:1px solid var(--bg-tertiary)">
            <div style="color:var(--accent-green);font-family:var(--font-mono);font-size:0.78rem">All standard provisions present.</div>
        </div>""")

        # Expandable clause rows
        rows_html = ""
        detail_data = []
        idx = 0
        for row in audit["audit_rows"]:
            if row["trigger_type"] == "preamble":
                continue
            is_std = row["status"] == "STANDARD"
            row_class = "standard" if is_std else "non-standard"
            status_badge = (
                '<span class="badge badge-green">Standard</span>' if is_std
                else '<span class="badge badge-yellow">Non-Standard</span>'
            )
            sim_badge = f'<span class="badge {self._sim_badge_class(row["similarity_score"])}">{row["similarity_score"]:.1%}</span>'
            trigger_label = row["trigger_type"].replace("_", " ").title()

            rows_html += f"""        <div class="audit-row {row_class}" onclick="toggleDetail({idx})" id="row-{idx}">
            <div class="audit-trigger">{trigger_label}</div>
            <div class="audit-theme">{row['cluster_theme']}</div>
            <div>{sim_badge}</div>
            <div>{status_badge}</div>
            <div class="audit-chevron" id="chev-{idx}">&#9654;</div>
        </div>"""

            original = original_texts.get(row["clause_id"], "")

            # Build a plain-English "why flagged" from classification data
            why_flagged = ""
            if not is_std:
                trigger_label = row["trigger_type"].replace("_", " ").title()
                cluster_theme = row.get("cluster_theme", "")
                sim = row["similarity_score"]
                if row.get("type_mismatch"):
                    # Pull cluster category from classification for cleaner label
                    cl = next((c for c in classification.get("classified_clauses", [])
                               if c.get("clause_id") == row["clause_id"]), {})
                    matched_cat = cl.get("cluster_trigger_category", "").replace("_", " ")
                    why_flagged = (
                        f"Tagged as '{trigger_label}' but the closest benchmark match is "
                        f"the '{cluster_theme}' cluster — a {matched_cat} pattern — "
                        f"at {sim:.1%} similarity. The clause language resembles a "
                        f"{matched_cat} provision more than a standard {trigger_label} trigger."
                    )
                else:
                    why_flagged = (
                        f"Similarity to closest benchmark cluster is {sim:.1%}, below the "
                        f"{STANDARD_SIMILARITY_THRESHOLD:.0%} threshold. Best match was "
                        f"'{cluster_theme}'. The clause language differs enough from any "
                        f"benchmark {trigger_label} clause to fall outside standard range."
                    )

            detail_data.append({
                "idx": idx,
                "why_flagged": why_flagged,
                "original": (original[:600] + "…") if len(original) > 600 else original,
            })
            idx += 1

        parts.append(rows_html)

        # Inline JS data + toggle function (appended once)
        detail_json = json.dumps(detail_data)
        parts.append(f"""        <script>
        const auditDetails = {detail_json};
        function toggleDetail(i) {{
            const existing = document.getElementById('detail-' + i);
            const chev = document.getElementById('chev-' + i);
            if (existing) {{
                existing.classList.toggle('show');
                chev.classList.toggle('open');
                return;
            }}
            const d = auditDetails.find(x => x.idx === i);
            if (!d) return;
            const row = document.getElementById('row-' + i);
            const div = document.createElement('div');
            div.id = 'detail-' + i;
            div.className = 'audit-detail show';
            let html = '';
            if (d.why_flagged) {{
                html += '<div class="detail-label">Why Flagged</div>';
                html += '<div class="detail-text">' + d.why_flagged + '</div>';
            }}
            if (d.original) {{
                html += '<div class="detail-label">Clause Text</div>';
                html += '<div class="detail-text" style="font-style:italic">' + d.original + '</div>';
            }}
            div.innerHTML = html || '<div class="detail-text" style="color:var(--text-muted)">Standard — no flags.</div>';
            row.insertAdjacentElement('afterend', div);
            chev.classList.add('open');
        }}
        </script>""")

        return "\n".join(parts)

    def _sim_badge_class(self, sim: float) -> str:
        if sim < 0.70:
            return "badge-red"
        if sim < 0.80:
            return "badge-yellow"
        return "badge-green"

    def _prepare_clause_data(self, classification: Dict) -> List[Dict]:
        result = []
        for clause in classification.get("classified_clauses", []):
            if clause.get("trigger_type") == "preamble":
                continue
            tt = clause.get("trigger_type", "")
            similarity = clause.get("similarity_score", 0.0)
            cluster_cat = clause.get("cluster_trigger_category", "")
            clause_cat = _infer_category(tt)
            type_mismatch = (
                cluster_cat and clause_cat
                and cluster_cat != clause_cat
                and tt not in ("preamble", "")
            )
            is_std = (similarity >= STANDARD_SIMILARITY_THRESHOLD) and not type_mismatch
            explanation = ""
            if not is_std:
                reasons = []
                if similarity < STANDARD_SIMILARITY_THRESHOLD:
                    reasons.append(f"Similarity {similarity:.1%} is below {STANDARD_SIMILARITY_THRESHOLD:.0%} threshold")
                if type_mismatch:
                    reasons.append(
                        f"Trigger category mismatch: clause is '{clause_cat}' "
                        f"but mapped to cluster category '{cluster_cat}'"
                    )
                explanation = "; ".join(reasons)

            result.append({
                "clause_id": clause.get("clause_id", ""),
                "trigger_type": tt,
                "cluster_theme": clause.get("cluster_theme", ""),
                "similarity": similarity,
                "status": "STANDARD" if is_std else "NON-STANDARD",
                "explanation": explanation,
                "original_text": clause.get("original_text", ""),
            })
        return result


# ================================
# S3 FLOW ENTRY POINT
# ================================

def run_stage10(accession: str, doc_type: str,
                classification_url: str, assessment_url: str,
                provision_checks_url: str = None,
                fees_url: str = None, fees_8k_url: str = None,
                triggers_url: str = None, triggers_8k_url: str = None,
                deal_name: str = None) -> Dict:
    """
    New-flow entry point: download all data from S3, generate dashboard HTML,
    upload to S3. Returns dict with dashboard_html S3 URL.
    """
    from termination_s3_utils import download_json, upload_text

    print("\n" + "=" * 80)
    print("STAGE 10 (TERMINATION): HTML DASHBOARD GENERATOR (S3 FLOW)")
    print("=" * 80)
    print(f"  Accession: {accession}")

    def _load(url, label):
        if not url:
            print(f"  {label:<45}: not available")
            return None
        data = download_json(url)
        print(f"  {label:<45}: loaded from S3")
        return data

    classification = _load(classification_url, "Stage 6 classification")
    assessment = _load(assessment_url, "Stage 7 assessment")
    provisions = _load(provision_checks_url, "Stage 9 provision checks")
    fees_agmt = _load(fees_url, "Fees (Agreement)")
    fees_8k = _load(fees_8k_url, "Fees (8-K/Press Release)")
    triggers = _load(triggers_url, "Triggers (Agreement)")
    triggers_8k = _load(triggers_8k_url, "Triggers (8-K/Press Release)")

    if not classification:
        raise ValueError("Stage 6 classification is required for dashboard generation")

    actual_deal_id = classification.get("deal_id", accession)
    analysis_ts = classification.get("analysis_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    fees, provenance = merge_fees_with_provenance(fees_agmt, fees_8k)
    discrepancies = build_discrepancies(fees_agmt, fees_8k)

    sources = []
    if fees_agmt or triggers:
        sources.append("Merger Agreement")
    if fees_8k or triggers_8k:
        sources.append("8-K / Press Release")

    primary_triggers = triggers or triggers_8k

    guide_rows = build_guide_rows(fees, primary_triggers) if (fees and primary_triggers) else []
    audit = build_clause_audit(classification)
    deal_type_label = derive_deal_type_label(classification, fees) if fees else "Strategic Acquisition"

    gen = TerminationDashboardGenerator()
    html = gen._build_html(
        actual_deal_id,
        analysis_ts,
        deal_type_label,
        guide_rows,
        audit,
        fees,
        classification,
        assessment,
        provisions,
        {},
        deal_name or None,
        sources=sources if sources else None,
        provenance=provenance if provenance else None,
        discrepancies=discrepancies if discrepancies else None,
    )

    _, dashboard_url = upload_text(html, accession, doc_type, "dashboard_html.html",
                                   content_type="text/html; charset=utf-8")
    print(f"\n  Dashboard uploaded to S3: {dashboard_url}")

    print("\n" + "=" * 80)
    print("STAGE 10 COMPLETE!")
    print("=" * 80)

    return {
        "dashboard_html": dashboard_url,
    }


# ================================
# ENTRY POINT
# ================================

def main():
    print("=" * 80)
    print("STAGE 10 (TERMINATION): HTML DASHBOARD GENERATOR v2")
    print("=" * 80)

    if len(sys.argv) < 2:
        print("\nERROR: Please provide a deal_id")
        print("\nUsage:")
        print("  python3 10_generate_dashboard_termination.py {deal_id}")
        print("\nExample:")
        print("  python3 10_generate_dashboard_termination.py d937868dex21")
        if os.path.exists(NEW_DEAL_REPORTS):
            print(f"\nAvailable deals (from {NEW_DEAL_REPORTS}):")
            seen = set()
            for f in sorted(os.listdir(NEW_DEAL_REPORTS)):
                if f.startswith("termination_classification_"):
                    parts = f.replace("termination_classification_", "").split("_")
                    if parts:
                        did = parts[0]
                        if did not in seen:
                            print(f"  {did}")
                            seen.add(did)
        sys.exit(1)

    deal_id = sys.argv[1]
    generator = TerminationDashboardGenerator()
    output_file = generator.generate_dashboard(deal_id)

    if output_file:
        print(f"\nDashboard ready: {output_file}")
    else:
        print(f"\nERROR: Could not generate dashboard for deal '{deal_id}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
