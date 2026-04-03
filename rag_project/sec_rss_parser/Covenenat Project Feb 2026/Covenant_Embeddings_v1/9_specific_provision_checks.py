#!/usr/bin/env python3
"""
Stage 9: Specific Covenant Provision Checks (Track 2)

Checks for 10 critical covenant provisions that lawyers care about,
independent of the clustering analysis (Track 1).

These are expert-defined risks based on litigation patterns and deal dynamics.

Usage:
    python3 9_specific_provision_checks.py path/to/openai_response_DEALID_individual_clauses.json
"""

import json
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import openai
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/new_deal_reports"

OPENAI_MODEL = "gpt-4o-mini"

# ================================
# PROVISION CHECK SCHEMAS
# ================================

ORDINARY_COURSE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "language_found": {"type": "string"},
        "section_reference": {"type": "string"},
        "materiality_qualifier": {"type": "boolean"},
        "past_practice_language": {"type": "boolean"},
        "carveouts": {
            "type": "object",
            "properties": {
                "required_by_law": {"type": "boolean"},
                "pandemic_flexibility": {"type": "boolean"},
                "industry_disruption": {"type": "boolean"},
                "disclosed_actions": {"type": "boolean"}
            },
            "required": ["required_by_law", "pandemic_flexibility", "industry_disruption", "disclosed_actions"],
            "additionalProperties": False
        },
        "preserve_relationships_language": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "language_found", "section_reference", "materiality_qualifier", "past_practice_language", "carveouts", "preserve_relationships_language", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

PARENT_CONSENT_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "consent_areas": {
            "type": "object",
            "properties": {
                "debt_incurrence": {"type": "boolean"},
                "senior_management": {"type": "boolean"},
                "capex": {"type": "boolean"},
                "material_contracts": {"type": "boolean"},
                "pricing_changes": {"type": "boolean"},
                "litigation_settlement": {"type": "boolean"}
            },
            "required": ["debt_incurrence", "senior_management", "capex", "material_contracts", "pricing_changes", "litigation_settlement"],
            "additionalProperties": False
        },
        "consent_standard": {"type": "string", "enum": ["sole_discretion", "not_unreasonably_withheld", "good_faith", "other", "unknown"]},
        "time_bound": {"type": "boolean"},
        "response_timeframe": {"type": "string"},
        "thresholds": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "consent_areas", "consent_standard", "time_bound", "response_timeframe", "thresholds", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

NO_SHOP_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "no_shop_language": {"type": "boolean"},
        "no_solicitation_language": {"type": "boolean"},
        "fiduciary_out_present": {"type": "boolean"},
        "fiduciary_out_scope": {"type": "string"},
        "matching_rights": {"type": "boolean"},
        "matching_timeframe": {"type": "string"},
        "go_shop_period": {"type": "boolean"},
        "go_shop_duration": {"type": "string"},
        "information_restrictions": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "no_shop_language", "no_solicitation_language", "fiduciary_out_present", "fiduciary_out_scope", "matching_rights", "matching_timeframe", "go_shop_period", "go_shop_duration", "information_restrictions", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

UNUSUALLY_RESTRICTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "restrictions_found": {
            "type": "object",
            "properties": {
                "hiring_freeze": {"type": "boolean"},
                "customer_discount_limits": {"type": "boolean"},
                "inventory_restrictions": {"type": "boolean"},
                "strategic_initiative_limits": {"type": "boolean"}
            },
            "required": ["hiring_freeze", "customer_discount_limits", "inventory_restrictions", "strategic_initiative_limits"],
            "additionalProperties": False
        },
        "industry_context_flexibility": {"type": "boolean"},
        "cyclical_business_considerations": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "restrictions_found", "industry_context_flexibility", "cyclical_business_considerations", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

MAE_LINKED_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "mae_language_in_covenants": {"type": "boolean"},
        "language_found": {"type": "string"},
        "mae_carveouts_narrow": {"type": "boolean"},
        "operating_covenants_strict": {"type": "boolean"},
        "litigation_risk_setup": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "mae_language_in_covenants", "language_found", "mae_carveouts_narrow", "operating_covenants_strict", "litigation_risk_setup", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

FINANCING_COOPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "buyer_needs_financing": {"type": "boolean"},
        "target_must_assist": {"type": "boolean"},
        "cooperation_burden": {"type": "string", "enum": ["excessive", "reasonable", "minimal", "unknown"]},
        "financing_condition_linked": {"type": "boolean"},
        "buyer_efforts_standard": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "buyer_needs_financing", "target_must_assist", "cooperation_burden", "financing_condition_linked", "buyer_efforts_standard", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

REGULATORY_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "integration_planning_restrictions": {"type": "boolean"},
        "gun_jumping_limitations": {"type": "boolean"},
        "information_sharing_protocols": {"type": "boolean"},
        "regulatory_coordination_weak": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "integration_planning_restrictions", "gun_jumping_limitations", "information_sharing_protocols", "regulatory_coordination_weak", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

WORKING_CAPITAL_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "specific_wc_levels": {"type": "boolean"},
        "inventory_standards": {"type": "boolean"},
        "capex_ranges": {"type": "boolean"},
        "seasonal_business_risk": {"type": "boolean"},
        "thresholds_found": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "specific_wc_levels", "inventory_standards", "capex_ranges", "seasonal_business_risk", "thresholds_found", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

EMPLOYEE_RESTRICTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "restrictions_found": {
            "type": "object",
            "properties": {
                "retention_bonus_limits": {"type": "boolean"},
                "raise_restrictions": {"type": "boolean"},
                "severance_limitations": {"type": "boolean"},
                "union_agreement_restrictions": {"type": "boolean"}
            },
            "required": ["retention_bonus_limits", "raise_restrictions", "severance_limitations", "union_agreement_restrictions"],
            "additionalProperties": False
        },
        "tight_labor_market_risk": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "restrictions_found", "tight_labor_market_risk", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}

ASYMMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "target_heavily_restricted": {"type": "boolean"},
        "buyer_lightly_restricted": {"type": "boolean"},
        "leverage_imbalance": {"type": "boolean"},
        "weak_buyer_commitment_signals": {"type": "boolean"},
        "specific_asymmetries": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["present", "target_heavily_restricted", "buyer_lightly_restricted", "leverage_imbalance", "weak_buyer_commitment_signals", "specific_asymmetries", "risk_level", "risk_factors", "confidence"],
    "additionalProperties": False
}


class SpecificProvisionChecker:
    """Checks for 10 specific covenant provisions"""

    def __init__(self, openai_key: str):
        self.client = openai.OpenAI(api_key=openai_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def check_ordinary_course(self, all_clauses_text: str) -> Dict:
        """Check #1: Ordinary Course Covenant (highest litigation risk)"""

        prompt = f"""Analyze these merger agreement covenants for ORDINARY COURSE language.

COVENANT TEXT:
{all_clauses_text[:50000]}  # Limit to avoid token limits

CRITICAL: This is the #1 litigation risk provision (see LVMH-Tiffany case).

Look for:
1. "Ordinary course of business" language
2. "Consistent with past practice"
3. "Use commercially reasonable efforts to preserve relationships"
4. "Except as required by law"
5. "Except with Parent's prior written consent"

CARVEOUTS TO IDENTIFY:
- Required by law exception
- Pandemic / force majeure flexibility
- Industry-wide disruption carveouts
- Disclosed actions carveouts

RISK ASSESSMENT:
- HIGH RISK: Very tight language, no carveouts, broad consent rights
- MEDIUM RISK: Some carveouts but limited
- LOW RISK: Flexible language, good carveouts, reasonable consent standard

Extract the exact language and assess risk level."""

        return self._make_structured_call(prompt, ORDINARY_COURSE_SCHEMA, "ordinary_course_check")

    def check_parent_consent(self, all_clauses_text: str) -> Dict:
        """Check #2: Parent Consent Rights"""

        prompt = f"""Analyze these covenants for PARENT CONSENT REQUIREMENTS.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for consent requirements in these areas:
1. Incur debt
2. Hire/fire senior management
3. CapEx above threshold
4. Enter material contracts
5. Change pricing
6. Settle litigation

CRITICAL DISTINCTIONS:
- "in Parent's sole discretion" → HIGH RISK (🔴)
- "not to be unreasonably withheld" → MEDIUM RISK (🟡)
- Time-bound (e.g., "3 business days to respond") → LOWER RISK (🟢)

Extract:
- Which areas require consent
- Consent standard used
- Any time bounds
- Dollar thresholds
- Overall risk level"""

        return self._make_structured_call(prompt, PARENT_CONSENT_SCHEMA, "parent_consent_check")

    def check_no_shop(self, all_clauses_text: str) -> Dict:
        """Check #3: No-Shop / Go-Shop Structure"""

        prompt = f"""Analyze for NO-SHOP / GO-SHOP provisions.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for:
1. No-shop language
2. No-solicitation restrictions
3. Fiduciary out provision (scope and limits)
4. Matching rights (and timing windows)
5. Go-shop period (duration)
6. Information restrictions

RED FLAGS:
- Very limited fiduciary out
- Broad matching rights
- Tight timing windows
- Strict information restrictions

These signal deal tightness and negotiation leverage."""

        return self._make_structured_call(prompt, NO_SHOP_SCHEMA, "no_shop_check")

    def check_unusually_restrictive(self, all_clauses_text: str) -> Dict:
        """Check #4: Unusually Restrictive Interim Covenants"""

        prompt = f"""Identify UNUSUALLY RESTRICTIVE interim covenants.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for:
1. Hiring freezes
2. Restrictions on customer discounts
3. Limits on inventory purchases
4. Restrictions on strategic initiatives

CONTEXT: If the business needs flexibility (cyclical industry, volatile revenue),
overly tight covenants increase breach risk.

Assess whether restrictions are unusually tight for normal business operations."""

        return self._make_structured_call(prompt, UNUSUALLY_RESTRICTIVE_SCHEMA, "unusually_restrictive_check")

    def check_mae_linked(self, all_clauses_text: str) -> Dict:
        """Check #5: MAE-Linked Operating Obligations"""

        prompt = f"""Check if operating covenants are linked to MAE definitions.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for language like:
"Except where such failure would not reasonably be expected to result in a Material Adverse Effect"

RISK SETUP:
If MAE carveouts are narrow AND operating covenants are strict → litigation setup.

Identify:
- MAE language in covenants
- Whether MAE carveouts appear narrow
- Whether operating covenants are strict
- Combined litigation risk"""

        return self._make_structured_call(prompt, MAE_LINKED_SCHEMA, "mae_linked_check")

    def check_financing_cooperation(self, all_clauses_text: str) -> Dict:
        """Check #6: Financing Cooperation Covenant"""

        prompt = f"""Analyze FINANCING COOPERATION requirements.

COVENANT TEXT:
{all_clauses_text[:50000]}

Context: In leveraged deals, if buyer needs financing and target must assist.

Look for:
1. Whether buyer must obtain financing
2. Whether target must assist with financing
3. Cooperation burden (excessive vs reasonable)
4. Whether financing condition is linked to cooperation
5. Buyer's effort standard ("reasonable best efforts" vs weak)

This becomes critical in leveraged deals."""

        return self._make_structured_call(prompt, FINANCING_COOPERATION_SCHEMA, "financing_cooperation_check")

    def check_regulatory(self, all_clauses_text: str) -> Dict:
        """Check #7: Divestiture/Regulatory Restrictions"""

        prompt = f"""Check for REGULATORY and DIVESTITURE restrictions.

COVENANT TEXT:
{all_clauses_text[:50000]}

In regulated deals, look for:
1. Restrictions on integration planning
2. Gun-jumping limitations
3. Mandatory information sharing protocols
4. Weak regulatory coordination covenants

Weak regulatory coordination = clearance risk."""

        return self._make_structured_call(prompt, REGULATORY_SCHEMA, "regulatory_check")

    def check_working_capital(self, all_clauses_text: str) -> Dict:
        """Check #8: Working Capital Management Constraints"""

        prompt = f"""Identify WORKING CAPITAL constraints.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for requirements to maintain:
1. Specific working capital levels
2. Inventory standards
3. CapEx ranges

RISK: This becomes risky in seasonal businesses.

Extract any specific thresholds or requirements."""

        return self._make_structured_call(prompt, WORKING_CAPITAL_SCHEMA, "working_capital_check")

    def check_employee_restrictions(self, all_clauses_text: str) -> Dict:
        """Check #9: Employee/Compensation Restrictions"""

        prompt = f"""Analyze EMPLOYEE and COMPENSATION restrictions.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for:
1. Limits on retention bonuses
2. Restrictions on raises
3. Severance limitations
4. Union agreement restrictions

RISK: In tight labor markets, these can materially disrupt operations pre-close."""

        return self._make_structured_call(prompt, EMPLOYEE_RESTRICTIONS_SCHEMA, "employee_restrictions_check")

    def check_asymmetry(self, all_clauses_text: str) -> Dict:
        """Check #10: Unusual Carveouts or Asymmetry"""

        prompt = f"""Identify ASYMMETRY in covenant obligations.

COVENANT TEXT:
{all_clauses_text[:50000]}

Look for:
1. Whether target is heavily restricted
2. Whether buyer has light obligations
3. Or vice versa

SIGNAL: Asymmetry signals negotiating leverage and sometimes weak buyer commitment.

Identify specific asymmetries and assess their significance."""

        return self._make_structured_call(prompt, ASYMMETRY_SCHEMA, "asymmetry_check")

    def _make_structured_call(self, prompt: str, schema: Dict, check_name: str) -> Dict:
        """Make structured API call with error handling"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an M&A lawyer analyzing covenant provisions for specific risks."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": check_name,
                        "strict": True,
                        "schema": schema
                    }
                },
                temperature=0.1
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"\n⚠️  Error in {check_name}: {e}")
            return {
                "present": False,
                "risk_level": "unknown",
                "confidence": "low",
                "error": str(e)
            }

    def run_all_checks(self, clauses: List[Dict]) -> Dict:
        """Run all 10 provision checks"""

        print("\n🔍 Running 10 specific provision checks...")

        # Combine all clause text for analysis
        all_text = "\n\n".join([
            f"[{c.get('section_title', 'Unknown')}]\n{c.get('text', '')}"
            for c in clauses
        ])

        checks = {}

        # Run each check with progress indication
        checks_to_run = [
            ("1. Ordinary Course (HIGHEST RISK)", self.check_ordinary_course),
            ("2. Parent Consent Rights", self.check_parent_consent),
            ("3. No-Shop / Go-Shop", self.check_no_shop),
            ("4. Unusually Restrictive", self.check_unusually_restrictive),
            ("5. MAE-Linked Covenants", self.check_mae_linked),
            ("6. Financing Cooperation", self.check_financing_cooperation),
            ("7. Regulatory Restrictions", self.check_regulatory),
            ("8. Working Capital Constraints", self.check_working_capital),
            ("9. Employee Restrictions", self.check_employee_restrictions),
            ("10. Asymmetry / Carveouts", self.check_asymmetry)
        ]

        for check_name, check_func in tqdm(checks_to_run, desc="Checking provisions"):
            check_key = check_name.split(". ")[1].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            checks[check_key] = check_func(all_text)

        return checks

    def generate_risk_summary(self, checks: Dict) -> Dict:
        """Generate overall risk summary"""

        high_risk_items = []
        medium_risk_items = []
        low_risk_items = []

        for provision, result in checks.items():
            if result.get('present'):
                risk_level = result.get('risk_level', 'unknown')
                if risk_level == 'high':
                    high_risk_items.append(provision)
                elif risk_level == 'medium':
                    medium_risk_items.append(provision)
                elif risk_level == 'low':
                    low_risk_items.append(provision)

        # Overall risk score
        risk_score = len(high_risk_items) * 3 + len(medium_risk_items) * 1

        if risk_score >= 9:
            overall_risk = "high"
        elif risk_score >= 4:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            'overall_risk_level': overall_risk,
            'risk_score': risk_score,
            'high_risk_provisions': high_risk_items,
            'medium_risk_provisions': medium_risk_items,
            'low_risk_provisions': low_risk_items,
            'provisions_checked': len(checks),
            'provisions_present': sum(1 for r in checks.values() if r.get('present')),
            'top_concerns': self._identify_top_concerns(checks)
        }

    def _identify_top_concerns(self, checks: Dict) -> List[str]:
        """Identify top 3 concerns"""

        concerns = []

        # Check for highest priority risks
        oc_check = checks.get('ordinary_course_(highest_risk)', {})
        if oc_check.get('present') and oc_check.get('risk_level') == 'high':
            concerns.append("🚨 Ordinary Course covenant is highly restrictive (litigation risk)")

        pc_check = checks.get('parent_consent_rights', {})
        if pc_check.get('present') and pc_check.get('consent_standard') == 'sole_discretion':
            concerns.append("🚨 Parent consent is at sole discretion (high control risk)")

        mae_check = checks.get('mae_linked_covenants', {})
        if mae_check.get('litigation_risk_setup'):
            concerns.append("🚨 MAE + strict covenants = litigation setup")

        # Add more if less than 3
        for provision, result in checks.items():
            if len(concerns) >= 3:
                break
            if result.get('present') and result.get('risk_level') == 'high' and provision not in [c.split(':')[0] for c in concerns]:
                concerns.append(f"⚠️  {provision.replace('_', ' ').title()} flagged as high risk")

        return concerns[:3]


def run_stage9(clauses_s3_url, accession, doc_type=None):
    """Run Stage 9 specific provision checks from S3 clauses data."""
    from dotenv import load_dotenv
    load_dotenv()
    from covenant_s3_utils import download_json, upload_json

    clauses_data = download_json(clauses_s3_url)
    deal_id = clauses_data.get("document_id", accession)

    clauses = clauses_data.get("clauses", [])
    combined_text = "\n\n".join([
        clause.get('original_text', '') for clause in clauses
    ])

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set")

    checker = SpecificProvisionChecker(openai_key)

    checks = {}
    checks_to_run = [
        ("1. Ordinary Course (HIGHEST RISK)", checker.check_ordinary_course),
        ("2. Parent Consent Rights", checker.check_parent_consent),
        ("3. No-Shop / Go-Shop", checker.check_no_shop),
        ("4. Unusually Restrictive", checker.check_unusually_restrictive),
        ("5. MAE-Linked Covenants", checker.check_mae_linked),
        ("6. Financing Cooperation", checker.check_financing_cooperation),
        ("7. Regulatory Restrictions", checker.check_regulatory),
        ("8. Working Capital Constraints", checker.check_working_capital),
        ("9. Employee Restrictions", checker.check_employee_restrictions),
        ("10. Asymmetry / Carveouts", checker.check_asymmetry)
    ]

    for check_name, check_func in tqdm(checks_to_run, desc="Checking provisions"):
        check_key = check_name.split(". ")[1].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        checks[check_key] = check_func(combined_text)

    risk_summary = checker.generate_risk_summary(checks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'deal_id': deal_id,
        'check_timestamp': timestamp,
        'track': 'specific_provisions',
        'provisions_checked': checks,
        'risk_summary': risk_summary,
        'token_usage': {
            'input_tokens': checker.total_input_tokens,
            'output_tokens': checker.total_output_tokens,
            'estimated_cost': (checker.total_input_tokens / 1_000_000 * 0.15 +
                             checker.total_output_tokens / 1_000_000 * 0.60)
        }
    }

    _, provision_url = upload_json(output, accession, "specific_provisions_json.json")

    return {"specific_provisions_json": provision_url, "output": output}


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()

    print("="*80)
    print("STAGE 9: SPECIFIC COVENANT PROVISION CHECKS (Track 2)")
    print("="*80)
    print("\nChecking for 10 critical provisions based on litigation patterns")
    print("This is INDEPENDENT of clustering analysis (Track 1)")

    # Check for input file
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide path to deal file")
        print("\nUsage:")
        print("  python3 9_specific_provision_checks.py path/to/openai_response_DEALID_individual_clauses.json")
        return

    deal_file = sys.argv[1]

    if not os.path.exists(deal_file):
        print(f"\n❌ Error: File not found: {deal_file}")
        return

    # Load deal data
    print(f"\n📂 Loading deal: {os.path.basename(deal_file)}")
    with open(deal_file, 'r') as f:
        deal_data = json.load(f)

    if 'clauses' not in deal_data or len(deal_data['clauses']) == 0:
        print("❌ Error: No clauses found in deal file")
        return

    deal_id = deal_data.get('deal_id') or deal_data.get('document_id', 'unknown')
    print(f"  ✓ Deal ID: {deal_id}")
    print(f"  ✓ Clauses: {len(deal_data['clauses'])}")

    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        return

    try:
        # Initialize checker
        checker = SpecificProvisionChecker(openai_key)

        # Run all checks
        checks = checker.run_all_checks(deal_data['clauses'])

        # Generate summary
        risk_summary = checker.generate_risk_summary(checks)

        # Build output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            'deal_id': deal_id,
            'check_timestamp': timestamp,
            'track': 'specific_provisions',
            'provisions_checked': checks,
            'risk_summary': risk_summary,
            'token_usage': {
                'input_tokens': checker.total_input_tokens,
                'output_tokens': checker.total_output_tokens,
                'estimated_cost': (checker.total_input_tokens / 1_000_000 * 0.15 +
                                 checker.total_output_tokens / 1_000_000 * 0.60)
            }
        }

        # Save report
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"specific_provisions_{deal_id}_{timestamp}.json")

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("🚨 SPECIFIC PROVISION CHECK SUMMARY")
        print("="*80)

        print(f"\n🆔 Deal: {deal_id}")
        print(f"📊 Provisions checked: {risk_summary['provisions_checked']}")
        print(f"✅ Provisions found: {risk_summary['provisions_present']}")
        print(f"⚠️  Overall risk level: {risk_summary['overall_risk_level'].upper()}")
        print(f"📈 Risk score: {risk_summary['risk_score']}/30")

        if risk_summary['high_risk_provisions']:
            print(f"\n🔴 HIGH RISK PROVISIONS ({len(risk_summary['high_risk_provisions'])}):")
            for prov in risk_summary['high_risk_provisions']:
                print(f"  • {prov.replace('_', ' ').title()}")

        if risk_summary['medium_risk_provisions']:
            print(f"\n🟡 MEDIUM RISK PROVISIONS ({len(risk_summary['medium_risk_provisions'])}):")
            for prov in risk_summary['medium_risk_provisions']:
                print(f"  • {prov.replace('_', ' ').title()}")

        if risk_summary['top_concerns']:
            print(f"\n💡 TOP CONCERNS:")
            for concern in risk_summary['top_concerns']:
                print(f"  {concern}")

        print(f"\n💰 Token Usage:")
        print(f"  • Input: {checker.total_input_tokens:,} tokens")
        print(f"  • Output: {checker.total_output_tokens:,} tokens")
        print(f"  • Estimated cost: ${output['token_usage']['estimated_cost']:.2f}")

        print("\n" + "="*80)
        print("✅ STAGE 9 COMPLETE!")
        print("="*80)
        print(f"\n📁 Report saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
