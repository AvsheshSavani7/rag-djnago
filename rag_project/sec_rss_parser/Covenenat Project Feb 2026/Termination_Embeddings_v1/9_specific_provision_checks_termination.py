#!/usr/local/bin/python3
"""
Stage 9 (Termination): Specific Termination Provision Checks

Checks for 8 critical termination provisions that M&A lawyers care about,
independent of the clustering / benchmark analysis.

These are expert-defined structural tests based on deal dynamics and litigation risk.

Provisions checked:
  1. Outside Date Duration      — >= 18 months? shorter is seller-favorable
  2. Regulatory / Antitrust RTF — separate regulatory fee on top of standard RTF?
  3. Tail Provision             — 12-month standard; longer = buyer-favorable; absent = seller-favorable
  4. Specific Performance       — can target force closing, or is RTF the exclusive remedy?
  5. Willful Breach Carveout    — does liability survive termination for willful breach?
  6. Matching Rights            — acquirer matching rights before target can accept superior proposal?
  7. Financing Failure Trigger  — explicit financing failure termination right (typically absent = buyer-favorable)
  8. Unilateral Outside Date Extension — can either party unilaterally extend for regulatory reasons?

Input:
  termination_response_{accession}_fees.json    (from project root)
  termination_response_{accession}_triggers.json (from project root, optional enrichment)

Output:
  termination_provision_checks_{deal_id}_{ts}.json  in new_deal_reports/

Usage:
    python3 9_specific_provision_checks_termination.py path/to/termination_response_{accession}_fees.json
"""

import json
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import openai
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
TERMINATION_DIR = os.path.join(PROJECT_ROOT, "Termination_Embeddings_v1")
NEW_DEAL_REPORTS = os.path.join(TERMINATION_DIR, "new_deal_reports")

OPENAI_MODEL = "gpt-4o-mini"

# ================================
# JSON SCHEMAS FOR STRUCTURED OUTPUTS
# ================================

OUTSIDE_DATE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "initial_outside_date_found": {"type": "string"},
        "duration_months_estimate": {"type": "number"},
        "meets_18_month_standard": {"type": "boolean"},
        "extensions_available": {"type": "boolean"},
        "extension_mechanism": {
            "type": "string",
            "enum": ["automatic", "unilateral_either_party", "unilateral_buyer",
                     "unilateral_seller", "mutual", "none", "unknown"]
        },
        "max_extended_duration_months": {"type": "number"},
        "regulatory_extension_trigger": {"type": "boolean"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "initial_outside_date_found", "duration_months_estimate",
        "meets_18_month_standard", "extensions_available", "extension_mechanism",
        "max_extended_duration_months", "regulatory_extension_trigger", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

REGULATORY_RTF_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "has_standard_rtf": {"type": "boolean"},
        "has_separate_regulatory_rtf": {"type": "boolean"},
        "regulatory_rtf_triggers": {"type": "array", "items": {"type": "string"}},
        "regulatory_rtf_amount_text": {"type": "string"},
        "regulatory_rtf_relationship_to_standard": {
            "type": "string",
            "enum": ["additive", "alternative", "same", "none", "unknown"]
        },
        "antitrust_specific": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "has_standard_rtf", "has_separate_regulatory_rtf",
        "regulatory_rtf_triggers", "regulatory_rtf_amount_text",
        "regulatory_rtf_relationship_to_standard", "antitrust_specific",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

TAIL_PROVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "tail_months": {"type": "number"},
        "meets_12_month_standard": {"type": "boolean"},
        "is_longer_than_standard": {"type": "boolean"},
        "tail_fee_type": {
            "type": "string",
            "enum": ["company_termination_fee", "reverse_termination_fee",
                     "both", "none", "unknown"]
        },
        "threshold_percentage": {"type": "number"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "tail_months", "meets_12_month_standard", "is_longer_than_standard",
        "tail_fee_type", "threshold_percentage", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

SPECIFIC_PERFORMANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "target_can_force_closing": {"type": "boolean"},
        "acquirer_can_force_closing": {"type": "boolean"},
        "rtf_is_exclusive_remedy": {"type": "boolean"},
        "specific_performance_conditions": {"type": "array", "items": {"type": "string"}},
        "financing_condition_for_sp": {"type": "boolean"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "target_can_force_closing", "acquirer_can_force_closing",
        "rtf_is_exclusive_remedy", "specific_performance_conditions",
        "financing_condition_for_sp", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

WILLFUL_BREACH_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "carveout_explicit": {"type": "boolean"},
        "survives_rtf_payment": {"type": "boolean"},
        "survives_ctf_payment": {"type": "boolean"},
        "definition_provided": {"type": "boolean"},
        "fee_offset_against_damages": {"type": "boolean"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "carveout_explicit", "survives_rtf_payment", "survives_ctf_payment",
        "definition_provided", "fee_offset_against_damages", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

MATCHING_RIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "acquirer_has_matching_rights": {"type": "boolean"},
        "matching_window_days": {"type": "number"},
        "intervening_event_right": {"type": "boolean"},
        "notice_required_before_acceptance": {"type": "boolean"},
        "match_standard": {
            "type": "string",
            "enum": ["identical", "equally_favorable", "reasonably_acceptable", "not_specified", "unknown"]
        },
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "acquirer_has_matching_rights", "matching_window_days",
        "intervening_event_right", "notice_required_before_acceptance",
        "match_standard", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

FINANCING_FAILURE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "explicit_financing_failure_trigger": {"type": "boolean"},
        "financing_condition_in_agreement": {"type": "boolean"},
        "rtf_covers_financing_failure": {"type": "boolean"},
        "buyer_financing_efforts_standard": {"type": "string"},
        "target_cooperation_required": {"type": "boolean"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "explicit_financing_failure_trigger", "financing_condition_in_agreement",
        "rtf_covers_financing_failure", "buyer_financing_efforts_standard",
        "target_cooperation_required", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}

UNILATERAL_EXTENSION_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "buyer_can_extend_unilaterally": {"type": "boolean"},
        "seller_can_extend_unilaterally": {"type": "boolean"},
        "extension_trigger": {
            "type": "string",
            "enum": ["regulatory_only", "any_condition", "automatic", "none", "unknown"]
        },
        "number_of_extensions_allowed": {"type": "number"},
        "extension_period_months": {"type": "number"},
        "maximum_extension_date_text": {"type": "string"},
        "party_favored": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "unknown"]
        },
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "present", "buyer_can_extend_unilaterally", "seller_can_extend_unilaterally",
        "extension_trigger", "number_of_extensions_allowed", "extension_period_months",
        "maximum_extension_date_text", "party_favored",
        "risk_level", "risk_factors", "confidence"
    ],
    "additionalProperties": False
}


class TerminationProvisionChecker:
    """Runs 8 specific termination provision checks using gpt-4o-mini structured outputs."""

    def __init__(self, openai_key: str):
        self.client = openai.OpenAI(api_key=openai_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_outside_date_duration(self, combined_text: str) -> Dict:
        """Check 1: Outside Date Duration — is it >= 18 months?"""
        prompt = f"""Analyze this merger termination agreement for OUTSIDE DATE provisions.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Find the initial outside date / end date for the merger, plus any extension provisions.

KEY QUESTIONS:
1. What is the initial outside/end date (as text, e.g. 'September 22, 2026')?
2. Can you estimate the duration from signing to that date in months? (If signing date is unclear, use context clues or note unknown — set duration_months_estimate to 0 if truly unknown.)
3. Is this >= 18 months? (18 months is the standard for complex deals; shorter is generally seller-favorable because it reduces time for buyer regulatory issues.)
4. Are extensions available? What triggers them (regulatory delays only, or any condition)?
5. What is the extension mechanism (automatic, unilateral by buyer, unilateral by seller, mutual consent, none)?
6. What is the maximum possible extended duration in months (set to same as initial if no extensions)?
7. Is there a regulatory-delay-specific extension trigger?
8. Who does the outside date structure favor?

RISK ASSESSMENT:
- HIGH: No extensions, short initial date (< 12 months) — seller-favorable
- MEDIUM: Short initial date (12–18 months) with limited extensions
- LOW: >= 18 months initial, or automatic regulatory extensions to 24+ months"""

        return self._make_structured_call(prompt, OUTSIDE_DATE_SCHEMA, "outside_date_check")

    def check_regulatory_rtf(self, combined_text: str) -> Dict:
        """Check 2: Regulatory/Antitrust RTF separate from standard RTF."""
        prompt = f"""Analyze this merger termination agreement for REGULATORY or ANTITRUST-SPECIFIC TERMINATION FEE provisions.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether there is a SEPARATE regulatory or antitrust termination fee (sometimes called 'Regulatory Termination Fee' or 'Antitrust Termination Fee') on top of the standard reverse termination fee (RTF).

KEY QUESTIONS:
1. Is there a standard reverse termination fee (RTF)?
2. Is there a SEPARATE regulatory-specific or antitrust-specific termination fee?
3. What triggers the regulatory RTF (antitrust injunction, regulatory block, outside date on regulatory conditions)?
4. What is the regulatory RTF amount (as text)?
5. Is the regulatory RTF additive to the standard RTF, an alternative, or the same amount?
6. Is the regulatory RTF specifically tied to antitrust law (HSR, DOJ, FTC)?

RISK ASSESSMENT:
- HIGH: No separate regulatory RTF — buyer can walk away from regulatory issues by paying only standard RTF, creating limited antitrust effort incentive
- MEDIUM: Regulatory RTF exists but is small relative to deal size
- LOW: Robust regulatory RTF that incentivizes buyer antitrust effort"""

        return self._make_structured_call(prompt, REGULATORY_RTF_SCHEMA, "regulatory_rtf_check")

    def check_tail_provision(self, combined_text: str) -> Dict:
        """Check 3: Tail Provision — standard is 12 months."""
        prompt = f"""Analyze this merger termination agreement for TAIL PROVISION language.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Find and analyze the tail provision (sometimes called a 'topping fee', 'back-end fee', or provisions about termination fees payable if the target subsequently enters into or completes a competing transaction after this agreement terminates).

KEY QUESTIONS:
1. Is there a tail provision?
2. How many months is the tail period? (12 months is standard)
3. Does it meet the 12-month standard? Is it longer than 12 months?
4. Which termination fee does the tail apply to (company termination fee / CTF, reverse termination fee / RTF, or both)?
5. What acquisition proposal threshold percentage triggers the tail (often 20% initially, 50% at tail period — note the tail threshold)?
6. Who does this structure favor? (Longer tail = buyer-favorable; absent tail = seller-favorable)

RISK ASSESSMENT:
- HIGH: No tail provision — seller can terminate and immediately take a competing offer without paying the CTF
- MEDIUM: Tail present but < 12 months
- LOW: 12-month standard tail or longer"""

        return self._make_structured_call(prompt, TAIL_PROVISION_SCHEMA, "tail_provision_check")

    def check_specific_performance(self, combined_text: str) -> Dict:
        """Check 4: Specific Performance — can target force closing?"""
        prompt = f"""Analyze this merger termination agreement for SPECIFIC PERFORMANCE provisions.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether the target company (seller) can force closing via specific performance, or whether the reverse termination fee (RTF) is the buyer's exclusive remedy for walking away.

KEY QUESTIONS:
1. Is specific performance available as a remedy?
2. Can the TARGET (seller) force the ACQUIRER (buyer) to close (i.e., get a court order compelling the merger)?
3. Can the ACQUIRER force the TARGET to close?
4. Is the RTF the EXCLUSIVE remedy for the seller if the buyer walks away (meaning specific performance is NOT available)?
5. What conditions must be met to obtain specific performance (e.g., financing must be available, all conditions other than financing satisfied)?
6. Is specific performance conditioned on financing being funded?
7. Who does this structure favor? (RTF as exclusive remedy = buyer-favorable; specific performance available to target = seller-favorable)

RISK ASSESSMENT:
- HIGH: RTF is exclusive remedy — buyer can walk away by paying RTF, target cannot force closing
- MEDIUM: Specific performance available but heavily conditioned
- LOW: Target can freely seek specific performance to compel closing"""

        return self._make_structured_call(prompt, SPECIFIC_PERFORMANCE_SCHEMA, "specific_performance_check")

    def check_willful_breach(self, combined_text: str) -> Dict:
        """Check 5: Willful Breach Carveout — does liability survive termination?"""
        prompt = f"""Analyze this merger termination agreement for WILLFUL BREACH provisions.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether liability for willful breach survives termination and/or the payment of termination fees.

KEY QUESTIONS:
1. Is there an explicit willful breach carveout?
2. Does willful breach liability SURVIVE payment of the reverse termination fee (RTF)? (i.e., paying the RTF does NOT cap the buyer's liability if the buyer willfully breached)
3. Does willful breach liability survive payment of the company termination fee (CTF)?
4. Is 'willful breach' defined in the agreement?
5. If willful breach damages are awarded, is the previously paid termination fee offset against the damages?
6. Who does this structure favor? (No carveout = buyer-favorable; carveout present = seller-favorable because it preserves unlimited liability)

RISK ASSESSMENT:
- HIGH: No willful breach carveout — RTF is absolute cap on buyer liability, enabling strategic breach
- MEDIUM: Carveout present but narrowly defined or limited
- LOW: Broad willful breach carveout with clear definition, fee offset"""

        return self._make_structured_call(prompt, WILLFUL_BREACH_SCHEMA, "willful_breach_check")

    def check_matching_rights(self, combined_text: str) -> Dict:
        """Check 6: Matching Rights / Intervening Events."""
        prompt = f"""Analyze this merger termination agreement for MATCHING RIGHTS provisions.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether the acquirer (buyer/Parent) has matching rights before the target can accept a superior proposal, and whether there are intervening event provisions.

KEY QUESTIONS:
1. Do matching rights exist?
2. Does the acquirer have the right to match a superior proposal before the target can accept it?
3. How many business days does the acquirer have to match (e.g., 3, 4, 5 business days)?
4. Is there an 'intervening event' provision (board can change recommendation due to a development that arose after signing that is material and not related to an acquisition proposal)?
5. Must the target give written notice before accepting a superior proposal?
6. What standard must the match meet (identical terms, equally or more favorable, reasonably acceptable, etc.)?
7. Who does this structure favor? (Strong matching rights + long window = buyer-favorable)

RISK ASSESSMENT:
- HIGH: No matching rights — acquirer can be cut off immediately if superior proposal arrives
- MEDIUM: Matching rights exist but window is short (< 3 days) or standard is loose
- LOW: Standard 4-5 business day matching window with clear process"""

        return self._make_structured_call(prompt, MATCHING_RIGHTS_SCHEMA, "matching_rights_check")

    def check_financing_failure(self, combined_text: str) -> Dict:
        """Check 7: Financing Failure Trigger."""
        prompt = f"""Analyze this merger termination agreement for FINANCING FAILURE termination rights.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether there is an explicit termination right triggered by the acquirer's failure to obtain financing.

KEY QUESTIONS:
1. Is there an explicit financing failure termination trigger? (This is relatively unusual; most modern deals do NOT have this — its absence is typically buyer-favorable.)
2. Is there a financing condition in the agreement (i.e., closing is conditioned on obtaining financing)?
3. If the RTF is triggered, does it cover financing failure scenarios?
4. What is the buyer's financing efforts standard ('reasonable best efforts', 'commercially reasonable efforts', 'best efforts', or not specified)?
5. Is the target required to cooperate in the buyer's financing process?
6. Who does this structure favor? (Absent financing trigger = buyer-favorable because it limits the ways a buyer can exit)

RISK ASSESSMENT:
- HIGH: Explicit financing failure trigger — buyer has easy exit via financing failure + RTF payment (seller-unfavorable)
- MEDIUM: No explicit trigger but financing condition exists (ambiguous)
- LOW: No financing condition and no financing failure trigger — buyer fully committed to close"""

        return self._make_structured_call(prompt, FINANCING_FAILURE_SCHEMA, "financing_failure_check")

    def check_unilateral_extension(self, combined_text: str) -> Dict:
        """Check 8: Unilateral Outside Date Extension."""
        prompt = f"""Analyze this merger termination agreement for UNILATERAL OUTSIDE DATE EXTENSION rights.

AGREEMENT TEXT:
{combined_text[:50000]}

TASK: Determine whether either party can unilaterally extend the outside date / end date without the other party's consent.

KEY QUESTIONS:
1. Can either party extend the outside date unilaterally?
2. Can the BUYER specifically extend unilaterally?
3. Can the SELLER specifically extend unilaterally?
4. What triggers the right to extend (regulatory conditions not satisfied, antitrust conditions only, or any closing condition)?
5. How many unilateral extensions are allowed?
6. How long is each extension period in months?
7. What is the maximum extension date (as text, e.g. 'June 22, 2027')?
8. Who does this structure favor? (Buyer unilateral extension = buyer-favorable because it gives buyer time on regulatory; automatic extensions triggered by regulatory = mutual/neutral)

RISK ASSESSMENT:
- HIGH: Buyer has unilateral extension rights with no corresponding seller protection — seller locked in indefinitely
- MEDIUM: Automatic extensions triggered by regulatory conditions (neutral mechanism)
- LOW: Mutual consent required for extensions, or no extension rights"""

        return self._make_structured_call(prompt, UNILATERAL_EXTENSION_SCHEMA, "unilateral_extension_check")

    # ------------------------------------------------------------------
    # API call helper
    # ------------------------------------------------------------------
    def _make_structured_call(self, prompt: str, schema: Dict, check_name: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an M&A lawyer specializing in termination fee provisions and "
                            "deal protection mechanisms. Analyze merger agreement text for specific "
                            "termination-related provisions. Be precise and grounded in the actual text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": check_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=0.1,
            )
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            return json.loads(response.choices[0].message.content)

        except Exception as exc:
            print(f"\n  WARNING: Error in {check_name}: {exc}")
            return {
                "present": False,
                "risk_level": "unknown",
                "confidence": "low",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------
    def run_all_checks(self, combined_text: str) -> Dict:
        """Run all 8 provision checks against the combined text."""
        print("\nRunning 8 specific termination provision checks...")

        checks_to_run = [
            ("1. Outside Date Duration (>= 18 months?)",
             "outside_date_duration", self.check_outside_date_duration),
            ("2. Regulatory / Antitrust RTF",
             "regulatory_rtf", self.check_regulatory_rtf),
            ("3. Tail Provision (12-month standard)",
             "tail_provision", self.check_tail_provision),
            ("4. Specific Performance Availability",
             "specific_performance", self.check_specific_performance),
            ("5. Willful Breach Carveout",
             "willful_breach", self.check_willful_breach),
            ("6. Matching Rights / Intervening Events",
             "matching_rights", self.check_matching_rights),
            ("7. Financing Failure Trigger",
             "financing_failure", self.check_financing_failure),
            ("8. Unilateral Outside Date Extension",
             "unilateral_extension", self.check_unilateral_extension),
        ]

        checks = {}
        for display_name, key, check_func in checks_to_run:
            print(f"  Checking: {display_name}...")
            checks[key] = check_func(combined_text)

        return checks

    # ------------------------------------------------------------------
    # Risk summary
    # ------------------------------------------------------------------
    def generate_risk_summary(self, checks: Dict) -> Dict:
        high_risk = []
        medium_risk = []
        low_risk = []

        for provision, result in checks.items():
            level = result.get("risk_level", "unknown")
            if level == "high":
                high_risk.append(provision)
            elif level == "medium":
                medium_risk.append(provision)
            elif level == "low":
                low_risk.append(provision)

        risk_score = len(high_risk) * 3 + len(medium_risk) * 1

        if risk_score >= 12:
            overall = "high"
        elif risk_score >= 5:
            overall = "medium"
        else:
            overall = "low"

        # Traffic-light summary per provision
        traffic_light = {}
        for provision, result in checks.items():
            level = result.get("risk_level", "unknown")
            color = {"high": "red", "medium": "yellow",
                     "low": "green"}.get(level, "grey")
            party = result.get("party_favored", "unknown")
            confidence = result.get("confidence", "unknown")
            traffic_light[provision] = {
                "color": color,
                "risk_level": level,
                "party_favored": party,
                "confidence": confidence,
                "present": result.get("present", False),
                "key_factors": result.get("risk_factors", [])[:2],
            }

        return {
            "overall_risk_level": overall,
            "risk_score": risk_score,
            "max_risk_score": 24,  # 8 provisions * 3 points each
            "high_risk_provisions": high_risk,
            "medium_risk_provisions": medium_risk,
            "low_risk_provisions": low_risk,
            "provisions_checked": len(checks),
            "provisions_present": sum(1 for r in checks.values() if r.get("present")),
            "traffic_light": traffic_light,
            "top_concerns": self._top_concerns(checks),
        }

    def _top_concerns(self, checks: Dict) -> List[str]:
        concerns = []

        sp = checks.get("specific_performance", {})
        if sp.get("rtf_is_exclusive_remedy"):
            concerns.append(
                "RTF is exclusive remedy — target cannot compel closing via specific performance (buyer-favorable)"
            )

        wb = checks.get("willful_breach", {})
        if wb.get("present") is False or (wb.get("present") and not wb.get("carveout_explicit")):
            concerns.append(
                "No willful breach carveout — RTF caps buyer liability even for intentional breach (buyer-favorable)"
            )

        ff = checks.get("financing_failure", {})
        if ff.get("explicit_financing_failure_trigger"):
            concerns.append(
                "Explicit financing failure termination trigger present — buyer has easy exit path (seller-unfavorable)"
            )

        for provision, result in checks.items():
            if len(concerns) >= 5:
                break
            if result.get("risk_level") == "high" and result.get("confidence") in ("high", "medium"):
                label = provision.replace("_", " ").title()
                concerns.append(
                    f"{label} flagged HIGH risk: {'; '.join(result.get('risk_factors', [])[:1])}")

        return concerns[:5]


def build_combined_text(fees_data: Dict, triggers_data: Optional[Dict]) -> str:
    """Combine fee and trigger texts into a single analysis corpus."""
    parts = []

    # Fee notes provide key structural info
    parts.append("=== TERMINATION FEE PROVISIONS ===")
    for key in ["company_termination_fee", "reverse_termination_fee", "expense_reimbursement"]:
        section = fees_data.get(key) or {}
        if section.get("notes"):
            parts.append(f"\n[{key.upper()}]\n{section['notes']}")
        if section.get("amount_text"):
            parts.append(f"Amount: {section['amount_text']}")
        triggers = section.get("triggers", [])
        if triggers:
            parts.append(f"Triggers: {', '.join(triggers)}")

    notes = fees_data.get("notes", "")
    if notes:
        parts.append(f"\n[GENERAL NOTES]\n{notes}")

    # Trigger clause texts
    if triggers_data:
        parts.append("\n\n=== TERMINATION TRIGGER CLAUSES ===")
        for clause in triggers_data.get("clauses", []):
            ctype = clause.get("clause_type", "")
            if ctype in ("trigger", "preamble"):
                cid = clause.get("clause_id", "")
                sec = clause.get("section_number", "")
                text = clause.get("original_text", "")
                parts.append(f"\n[Section {sec} / {cid}]\n{text}")

    return "\n".join(parts)


def run_stage9(fees_s3_url: str, triggers_s3_url: str,
               accession: str, doc_type: str) -> Dict:
    """
    New-flow entry point: download fees/triggers from S3, run checks, upload results to S3.
    Returns dict with provision_checks_json S3 URL.
    """
    from termination_s3_utils import download_json, upload_json

    print("=" * 80)
    print("STAGE 9 (TERMINATION): SPECIFIC PROVISION CHECKS (S3 FLOW)")
    print("=" * 80)

    fees_data = download_json(fees_s3_url) if fees_s3_url else {}
    triggers_data = download_json(triggers_s3_url) if triggers_s3_url else None

    if not fees_data:
        raise ValueError(f"Invalid or empty fees data from {fees_s3_url}")

    deal_id = fees_data.get("document_id", accession)
    print(f"  Accession: {accession}")
    print(f"  Fees source: {fees_s3_url}")
    if triggers_data:
        print(
            f"  Triggers source: {triggers_s3_url} ({triggers_data.get('total_clauses', '?')} clauses)")
    else:
        print(f"  Triggers: not available (fees notes only)")

    combined_text = build_combined_text(fees_data, triggers_data)
    print(f"  Combined text length: {len(combined_text):,} chars")

    openai_key = os.getenv("OPENAI_API_KEY_SEC_FILING")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY_SEC_FILING not set")

    checker = TerminationProvisionChecker(openai_key)
    checks = checker.run_all_checks(combined_text)
    risk_summary = checker.generate_risk_summary(checks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "deal_id": deal_id,
        "check_timestamp": timestamp,
        "track": "termination_specific_provisions",
        "input_files": {
            "fees_source": fees_s3_url,
            "triggers_source": triggers_s3_url,
        },
        "provisions_checked": checks,
        "risk_summary": risk_summary,
        "token_usage": {
            "input_tokens": checker.total_input_tokens,
            "output_tokens": checker.total_output_tokens,
            "estimated_cost": (
                checker.total_input_tokens / 1_000_000 * 0.15
                + checker.total_output_tokens / 1_000_000 * 0.60
            ),
        },
    }

    _, provision_url = upload_json(
        output, accession, doc_type, "provision_checks_json.json")
    print(f"  Provision checks uploaded to S3: {provision_url}")

    print("\n" + "=" * 80)
    print("STAGE 9 COMPLETE!")
    print("=" * 80)

    return {
        "provision_checks_json": provision_url,
        "output": output,
    }


def main():
    print("=" * 80)
    print("STAGE 9 (TERMINATION): SPECIFIC PROVISION CHECKS")
    print("=" * 80)
    print("\nChecking 8 critical termination provisions lawyers care about")
    print(
        "Input: termination_response_{accession}_fees.json from project root")

    if len(sys.argv) < 2:
        print("\nERROR: Please provide path to fees JSON file")
        print("\nUsage:")
        print("  python3 9_specific_provision_checks_termination.py "
              "path/to/termination_response_{accession}_fees.json")
        print("\nExample:")
        print(f"  python3 9_specific_provision_checks_termination.py "
              f"{PROJECT_ROOT}/termination_response_d937868dex21_fees.json")
        return

    fees_file = sys.argv[1]
    if not os.path.exists(fees_file):
        print(f"\nERROR: File not found: {fees_file}")
        return

    print(f"\nLoading fees file: {os.path.basename(fees_file)}")
    with open(fees_file, "r") as fh:
        fees_data = json.load(fh)

    deal_id = fees_data.get("document_id", "unknown")
    if deal_id == "unknown":
        # Try to extract from filename
        basename = os.path.basename(fees_file)
        # termination_response_{deal_id}_fees.json
        parts = basename.replace(
            "termination_response_", "").replace("_fees.json", "")
        deal_id = parts

    print(f"  Deal ID: {deal_id}")

    # Try to load matching triggers file
    triggers_file = fees_file.replace("_fees.json", "_triggers.json")
    triggers_data = None
    if os.path.exists(triggers_file):
        with open(triggers_file, "r") as fh:
            triggers_data = json.load(fh)
        print(
            f"  Triggers file: {os.path.basename(triggers_file)} ({triggers_data.get('total_clauses', '?')} clauses)")
    else:
        print(f"  Triggers file: Not found (will use fees notes only)")

    combined_text = build_combined_text(fees_data, triggers_data)
    print(f"  Combined text length: {len(combined_text):,} chars")

    openai_key = os.getenv("OPENAI_API_KEY_SEC_FILING")
    if not openai_key:
        print("\nERROR: OPENAI_API_KEY_SEC_FILING not set in environment / .env")
        return

    try:
        checker = TerminationProvisionChecker(openai_key)
        checks = checker.run_all_checks(combined_text)
        risk_summary = checker.generate_risk_summary(checks)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "deal_id": deal_id,
            "check_timestamp": timestamp,
            "track": "termination_specific_provisions",
            "input_files": {
                "fees_file": os.path.basename(fees_file),
                "triggers_file": os.path.basename(triggers_file) if triggers_data else None,
            },
            "provisions_checked": checks,
            "risk_summary": risk_summary,
            "token_usage": {
                "input_tokens": checker.total_input_tokens,
                "output_tokens": checker.total_output_tokens,
                "estimated_cost": (
                    checker.total_input_tokens / 1_000_000 * 0.15
                    + checker.total_output_tokens / 1_000_000 * 0.60
                ),
            },
        }

        os.makedirs(NEW_DEAL_REPORTS, exist_ok=True)
        output_file = os.path.join(
            NEW_DEAL_REPORTS, f"termination_provision_checks_{deal_id}_{timestamp}.json"
        )
        with open(output_file, "w") as fh:
            json.dump(output, fh, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("TERMINATION PROVISION CHECK SUMMARY")
        print("=" * 80)
        print(f"\nDeal             : {deal_id}")
        print(f"Provisions checked: {risk_summary['provisions_checked']}")
        print(f"Provisions found : {risk_summary['provisions_present']}")
        print(
            f"Overall risk     : {risk_summary['overall_risk_level'].upper()}")
        print(
            f"Risk score       : {risk_summary['risk_score']}/{risk_summary['max_risk_score']}")

        tl = risk_summary["traffic_light"]
        print("\nTraffic Light Summary:")
        provision_labels = {
            "outside_date_duration": "1. Outside Date Duration",
            "regulatory_rtf":        "2. Regulatory / Antitrust RTF",
            "tail_provision":        "3. Tail Provision",
            "specific_performance":  "4. Specific Performance",
            "willful_breach":        "5. Willful Breach Carveout",
            "matching_rights":       "6. Matching Rights",
            "financing_failure":     "7. Financing Failure Trigger",
            "unilateral_extension":  "8. Unilateral Extension Right",
        }
        color_symbol = {"red": "[RED]", "yellow": "[YLW]",
                        "green": "[GRN]", "grey": "[---]"}
        for key, label in provision_labels.items():
            info = tl.get(key, {})
            sym = color_symbol.get(info.get("color", "grey"), "[---]")
            party = info.get("party_favored", "unknown")
            confidence = info.get("confidence", "?")
            print(f"  {sym} {label:<40} party: {party:<20} conf: {confidence}")

        if risk_summary["high_risk_provisions"]:
            print(
                f"\nHIGH RISK ({len(risk_summary['high_risk_provisions'])}):")
            for p in risk_summary["high_risk_provisions"]:
                print(f"  - {provision_labels.get(p, p)}")

        if risk_summary["medium_risk_provisions"]:
            print(
                f"\nMEDIUM RISK ({len(risk_summary['medium_risk_provisions'])}):")
            for p in risk_summary["medium_risk_provisions"]:
                print(f"  - {provision_labels.get(p, p)}")

        if risk_summary["top_concerns"]:
            print("\nTop Concerns:")
            for concern in risk_summary["top_concerns"]:
                print(f"  - {concern}")

        print(f"\nToken Usage:")
        print(f"  Input : {checker.total_input_tokens:,} tokens")
        print(f"  Output: {checker.total_output_tokens:,} tokens")
        print(f"  Cost  : ${output['token_usage']['estimated_cost']:.4f}")

        print("\n" + "=" * 80)
        print("STAGE 9 (TERMINATION) COMPLETE!")
        print("=" * 80)
        print(f"\nReport saved: {output_file}")
        print(f"\nNext step:")
        print(f"  python3 10_generate_dashboard_termination.py {deal_id}")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
