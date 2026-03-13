"""
config.py — All constants, prompts, and configuration for the Proxy Comp Pipeline.
"""

import os
import re

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

OUTPUT_FOLDER = "./output"
ENV_FILE = "../.env"

# =============================================================================
# SEC HEADERS
# =============================================================================

SEC_HEADERS = {
    "User-Agent": "Hyperion Technologies (contact@hyperiontechnologies.com)",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# =============================================================================
# LLM MODEL CONFIG
# =============================================================================

MODEL_STANDARD = "claude-sonnet-4-6"
MODEL_THINKING = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
THINKING_BUDGET = 10000

# =============================================================================
# CANONICAL SECTION IDS
# =============================================================================

CANONICAL_SECTION_IDS = [
    "background",
    "consideration_summary",
    "vote_info",
    "regulatory",
    "financing",
    "closing_conditions",
    "termination",
    "risk_factors",
    "interests_conflicts",
    "projections",
    "appraisal_rights",
    "merger_agreement_summary",
    "summary",
    "questions_and_answers",
]

# Tier-1 categories (always important for merger arb)
TIER1_CATEGORIES = {
    "background", "dates", "consideration", "financing",
    "sh_votes", "regulatory", "closing_guidance",
}

# Tier-1 section IDs (sections whose changes are always Tier 1)
TIER1_SECTION_IDS = {
    "background", "consideration_summary", "vote_info", "regulatory",
    "financing", "closing_conditions", "termination",
}

# =============================================================================
# FORM TYPE CONFIGURATION
# =============================================================================

FORM_TYPE_FAMILIES = {
    # proxy_like
    "PREM14A": "proxy_like",
    "PREM14A/A": "proxy_like",
    "DEFM14A": "proxy_like",
    "DEFA14A": "proxy_like",
    # registration_like
    "S-4": "registration_like",
    "S-4/A": "registration_like",
    "F-4": "registration_like",
    "F-4/A": "registration_like",
    # tender_like
    "SC TO-T": "tender_like",
    "SC TO-T/A": "tender_like",
    "SC TO-I": "tender_like",
    "SC TO-I/A": "tender_like",
    "SC 14D-9": "tender_like",
    "SC 14D-9/A": "tender_like",
}

FORM_TYPE_LABELS = {
    "PREM14A": "Preliminary Proxy (PREM14A)",
    "PREM14A/A": "Amended Preliminary Proxy (PREM14A/A)",
    "DEFM14A": "Definitive Proxy (DEFM14A)",
    "DEFA14A": "Additional Definitive Proxy (DEFA14A)",
    "S-4": "Registration Statement (S-4)",
    "S-4/A": "Amended Registration Statement (S-4/A)",
    "F-4": "Registration Statement (F-4)",
    "F-4/A": "Amended Registration Statement (F-4/A)",
    "SC TO-T": "Tender Offer (SC TO-T)",
    "SC TO-T/A": "Amended Tender Offer (SC TO-T/A)",
    "SC TO-I": "Tender Offer (SC TO-I)",
    "SC TO-I/A": "Amended Tender Offer (SC TO-I/A)",
    "SC 14D-9": "Solicitation/Recommendation (SC 14D-9)",
    "SC 14D-9/A": "Amended Solicitation/Recommendation (SC 14D-9/A)",
    "PROXY": "Proxy Statement",
}


def get_form_label(form_type: str) -> str:
    return FORM_TYPE_LABELS.get(form_type, form_type)


def get_form_family(form_type: str) -> str:
    return FORM_TYPE_FAMILIES.get(form_type, "proxy_like")


# =============================================================================
# BLOCK CLASSIFICATION PROMPT
# =============================================================================

BLOCK_CLASSIFY_PROMPT = """\
You are tagging paragraphs from an SEC merger filing by topic. For each paragraph,
assign exactly ONE topic tag from this list:

  dates          — meeting date, record date, mailing date, outside date
  consideration  — merger consideration, exchange ratio, per share price, fractional shares
  financing      — financing commitments, debt/equity, marketing period, limited guarantee
  sh_approval    — shareholder vote thresholds, quorum, abstention effects
  hsr            — HSR Act, Hart-Scott-Rodino, FTC/DOJ antitrust, waiting period, second request
  regulatory     — non-HSR regulatory: CFIUS, EC, foreign competition/investment, state regulators
  closing        — closing conditions, conditions precedent, expected timing
  termination    — termination fees, break-up fees, go-shop, no-shop, reverse termination
  background     — background of the merger, negotiations, board deliberations
  general        — anything else (boilerplate, risk factors, financial statements, etc.)

Return a JSON array with one object per paragraph, in order:
[{{"id": "P1", "topic": "regulatory"}}, {{"id": "P2", "topic": "general"}}, ...]

ONLY return the JSON array. No explanation."""

MODEL_CLASSIFY = "claude-haiku-4-5-20251001"
CLASSIFY_BATCH_SIZE = 15
CLASSIFY_MAX_WORKERS = 15
CLASSIFY_MIN_WORDS = 15

# =============================================================================
# SECTION MAPPING PROMPT
# =============================================================================

SECTION_MAPPING_PROMPT = """\
You are analyzing a SEC merger filing. Below is a list of section headings found in the document.
Map each heading to the most appropriate canonical section ID from this list:

CANONICAL IDS:
- background (Background of the Merger/Acquisition/Transaction, Past Contacts)
- consideration_summary (The Merger Consideration, Terms of the Offer, Exchange Ratio)
- vote_info (Vote Required, Shareholder Approval, Record Date, Meeting Information)
- regulatory (Regulatory Approvals, Antitrust, HSR, CFIUS)
- financing (Financing of the Merger, Source of Funds, Commitment Letters)
- closing_conditions (Conditions to the Merger/Closing, Conditions to the Offer)
- termination (Termination of the Merger Agreement, Break Fees, Go-Shop)
- risk_factors (Risk Factors, Cautionary Statements)
- interests_conflicts (Interests of Directors and Officers, Conflicts of Interest)
- projections (Financial Projections, Forecasts, Unaudited Prospective Financial Information)
- appraisal_rights (Appraisal Rights, Dissenters' Rights)
- merger_agreement_summary (The Merger Agreement, Summary of the Merger Agreement)
- summary (Summary, Summary Term Sheet, Questions and Answers)
- other (anything that doesn't fit the above categories)

SECTION HEADINGS:
{headings_list}

Return a JSON array where each element is {{"heading": "...", "section_id": "..."}}.
Return ONLY the JSON array, no other text.
"""

# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

DATES_EXTRACTION_PROMPT = """\
Extract date-related facts from this merger filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return this exact JSON structure (use null for unknown values):
{{
    "meeting_date": "exact date or null",
    "meeting_time": "exact time with timezone or null",
    "record_date": "exact date or null",
    "mailing_date": "exact date or 'on or about DATE' or null",
    "outside_date": "termination deadline date or null",
    "outside_date_extensions": "extension provisions description or null",
    "expected_closing_timing": "expected timing like 'early 2026' or 'Q1 2026' or null"
}}
"""

CONSIDERATION_EXTRACTION_PROMPT = """\
Extract merger consideration details from this filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return this exact JSON structure (use null for unknown values):
{{
    "type": "all-cash or all-stock or mixed",
    "per_share_cash": "dollar amount per share or null",
    "exchange_ratio": "stock exchange ratio or null",
    "cvr_details": "CVR/earnout description or null",
    "proration": "proration mechanics or null",
    "fractional_shares": "treatment of fractional shares or null",
    "ticking_fee": "ticking fee details or null",
    "total_deal_value": "total deal value if stated or null"
}}
"""

FINANCING_EXTRACTION_PROMPT = """\
Extract financing details from this merger filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return this exact JSON structure (use null for unknown values):
{{
    "is_condition_to_closing": true or false or null,
    "equity_commitments": "equity commitment details or null",
    "debt_commitments": "debt commitment details or null",
    "limited_guarantee": "guarantee amount and guarantor or null",
    "total_amount": "total financing amount or null",
    "marketing_period": "marketing period details or null"
}}
"""

SH_VOTES_EXTRACTION_PROMPT = """\
Extract shareholder vote requirements from this merger filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return this exact JSON structure (use null for unknown values):
{{
    "target_threshold": "vote threshold for target (e.g., 'majority of outstanding shares') or null",
    "acquirer_threshold": "vote threshold for acquirer or 'not required' or null",
    "class_votes": "any class-specific voting requirements or null",
    "tender_condition": "minimum tender condition (for tender offers) or null",
    "insider_ownership_pct": "percentage owned by insiders or null"
}}
"""

REGULATORY_EXTRACTION_PROMPT = """\
Extract regulatory approval requirements from this merger filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return a JSON array of regulatory items. Each item:
{{
    "jurisdiction": "country or region",
    "agency": "regulatory body name",
    "filed_date": "filing date or null",
    "approval_date": "approval date or null",
    "status": "pending or filed or approved or expired or second_request or null",
    "efforts_standard": "reasonable best efforts or hell-or-high-water or null",
    "details": "brief additional details or null"
}}

Include at minimum: HSR (if applicable), CFIUS, and any foreign regulatory approvals mentioned.
Return [] if no regulatory requirements are discussed.
"""

CLOSING_GUIDANCE_EXTRACTION_PROMPT = """\
Extract closing timing guidance from this merger filing section. Return ONLY valid JSON.

FILING TEXT:
{section_text}

Return this exact JSON structure (use null for unknown values):
{{
    "expected_timing": "expected closing timeframe like 'early 2026' or 'first half of 2026' or null",
    "gating_items": ["list of items that must be completed before closing"],
    "updated_expectations": "any updated timing guidance from amendments or null"
}}

IMPORTANT: "expected_timing" is NOT the outside date / termination deadline. Look for language like
"expects to consummate", "expects to close", "anticipated to close" with a timeframe.
"""

# =============================================================================
# COMPARISON PROMPTS
# =============================================================================

_COMPARISON_RULES = """
CRITICAL RULES:
1. ONLY report REAL changes to deal terms, dates, amounts, or status. Filling in a blank placeholder (e.g., "[  ]" -> actual date) counts as a new disclosure, but report it ONCE even if the same value appears multiple times in the text.
2. CONSOLIDATE: If the same fact (e.g., record date) appears in multiple places, report it as ONE item with the key value. Do NOT create separate entries for each mention.
3. BE CONCISE: Values should be the key fact only (e.g., "September 25, 2025"), NOT the full surrounding paragraph or sentence.
4. SKIP UNCHANGED items: If something is the same in both filings, do NOT report it. Do NOT report "no change confirmed" or "language unchanged."
5. SKIP formatting/boilerplate: Ignore page numbers, TOC changes, paragraph reordering, cross-reference updates, or disclosure restructuring that doesn't change the substance.
6. Return a JSON array. If no real differences, return: []"""

_COMPARISON_PROMPTS = {
    "dates": """\
Compare these two SEC merger filing sections for DATE information only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: meeting date, meeting time, record date, mailing date, shares outstanding as of record date, outside date (termination deadline), outside date extensions. Consolidate: "proxy statement date," "letter date," and "mailing date" are the SAME date — report ONCE as "Mailing Date." Do NOT report shareholder vote thresholds (belongs in SH Approval), termination fees (belongs in Termination), or closing conditions.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "concise value"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old value", "now": "new value"}}""",

    "consideration": """\
Compare these two SEC merger filing sections for MERGER CONSIDERATION information only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: per-share cash amount, exchange ratio, total deal value, CVR/earnout details, proration mechanics, ticking fees. Do NOT report dates, financing, or termination fees.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "concise value"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old value", "now": "new value"}}""",

    "financing": """\
Compare these two SEC merger filing sections for FINANCING information only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: debt commitment amounts, lenders, commitment letter dates, equity commitments, limited guarantees, marketing period, financing condition. Do NOT report consideration amounts, termination fees, or regulatory items.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "concise value with $ amounts and lender names"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old value", "now": "new value"}}""",

    "sh_votes": """\
Compare these two SEC merger filing sections for SHAREHOLDER VOTE MECHANICS only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: vote threshold percentages (e.g., "majority of outstanding shares"), class-specific voting requirements, tender conditions, shares outstanding count (report the NUMBER, e.g., "250,106,129 shares"). Do NOT report: dates (belong in Dates), abstention or broker non-vote effects, proxy revocation methods, quorum procedures, virtual meeting URLs, proxy mailing logistics, or how to submit a proxy — those are procedural voting logistics, not deal terms.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "concise value"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old value", "now": "new value"}}""",

    "hsr": """\
Compare these two SEC merger filing sections for U.S. HSR ACT information only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: HSR filing date, waiting period expiration, early termination status, second request status, FTC/DOJ actions. Do NOT report non-U.S. antitrust (Canada, Australia, EU — those belong in Other Regulatory), CFIUS, or state regulatory items.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "HSR - short name", "type": "new", "value": "Status: status | concise detail"}}
- Changed: {{"field": "HSR - short name", "type": "changed", "was": "old status/detail", "now": "new status/detail"}}""",

    "regulatory": """\
Compare these two SEC merger filing sections for NON-HSR REGULATORY information only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: CFIUS, state PSC/PUC approvals, foreign antitrust (Canada, Australia, EU, etc.), FIRB, Investment Canada Act, SEC, money transmitter/financial licensing approvals (state money transmitter licenses, Bank of Spain, UK FCA, other financial regulatory bodies). Do NOT report U.S. HSR Act items (those belong in HSR section). Report ONE entry per agency or approval category — consolidate status + detail into a single item.
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "Agency Name", "type": "new", "value": "Status: status | concise detail"}}
- Status change: {{"field": "Agency Name", "type": "changed", "was": "status: old", "now": "status: new + concise detail"}}""",

    "closing": """\
Compare these two SEC merger filing sections for CLOSING CONDITIONS only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: conditions that must be satisfied for closing (regulatory approvals, shareholder vote, no MAE, accuracy of reps, etc.), and whether any conditions were added, removed, or modified. Do NOT report the outside date (belongs in Dates), termination fees (belongs in Termination), or regulatory filing status (belongs in Regulatory).
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "concise description"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old condition", "now": "new condition"}}
- Removed: {{"field": "short name", "type": "removed", "value": "what was removed"}}""",

    "termination": """\
Compare these two SEC merger filing sections for TERMINATION PROVISIONS only.

EARLIER FILING ({old_label}):
{old_text}

LATER FILING ({new_label}):
{new_text}

SCOPE: company termination fee amount, parent/reverse termination fee amount, regulatory termination fee, fee trigger conditions, go-shop period/dates, fiduciary out provisions. Do NOT report the outside date (belongs in Dates), closing conditions (belongs in Closing), or regulatory status (belongs in Regulatory).
""" + _COMPARISON_RULES + """
Format:
- New: {{"field": "short name", "type": "new", "value": "$ amount and concise trigger"}}
- Changed: {{"field": "short name", "type": "changed", "was": "old value", "now": "new value"}}""",
}

# =============================================================================
# BACKGROUND DIFF PROMPTS
# =============================================================================

BACKGROUND_DIFF_INTERPRET_PROMPT = """\
You are a merger arbitrage analyst. Summarize the material significance of the \
changes below from the Background section. Be terse — one line per change, facts only.

EARLIER: {doc1_label}
LATER: {doc2_label}

INSERTED:
{inserted_text}

DELETED:
{deleted_text}

MODIFIED:
{modified_text}

---
For each material change, write one bullet: "- [what happened/changed]"
Skip immaterial formatting or word-order changes.
If nothing material, write "No material changes."
"""

# =============================================================================
# CHANGE REPORT PROMPT
# =============================================================================

CHANGE_OPENING_PROMPT = """\
Given these changes between {old_label} and {new_label} for {ticker},
write 2-4 sentences summarizing the key updates. Be direct and factual.
Mention: any new dates set, HSR/regulatory status updates, and whether
the Background section changed. Do not use bullet points.

CHANGES:
{changes_summary}
"""

# =============================================================================
# SUMMARY SECTION PROMPTS
# =============================================================================

SECTION_CONFIGS = [
    {"key": "dates",           "topics": ["dates", "sh_approval", "closing", "termination"], "model": "standard", "max_chars": 20000, "thinking": False},
    {"key": "consideration",   "topics": ["consideration"],             "model": "standard", "max_chars": 20000, "thinking": False},
    {"key": "financing",       "topics": ["financing"],                 "model": "standard", "max_chars": 15000, "thinking": False},
    {"key": "sh_approval",     "topics": ["sh_approval", "dates"],      "model": "standard", "max_chars": 15000, "thinking": False},
    {"key": "hsr",             "topics": ["hsr", "regulatory"],         "model": "opus",     "max_chars": 15000, "thinking": True},
    {"key": "other_regulatory","topics": ["regulatory", "hsr", "closing"], "model": "standard", "max_chars": 30000, "thinking": False},
    {"key": "closing",         "topics": ["closing"],                   "model": "standard", "max_chars": 15000, "thinking": False},
    {"key": "conditions",      "topics": ["closing", "regulatory"],     "model": "standard", "max_chars": 20000, "thinking": False},
    {"key": "termination",     "topics": ["termination"],               "model": "opus",     "max_chars": 25000, "thinking": True},
]

_TOPIC_TO_SECTION_IDS = {
    "dates": ["vote_info", "summary"],
    "consideration": ["consideration_summary", "summary"],
    "financing": ["financing", "summary"],
    "sh_approval": ["vote_info", "summary"],
    "hsr": ["regulatory"],
    "other_regulatory": ["regulatory"],
    "closing": ["closing_conditions"],
    "conditions": ["closing_conditions"],
    "termination": ["termination"],
}

SECTION_PROMPTS = {
    "dates": """\
Extract meeting and filing dates from this merger filing text.
Write in plain text, no markdown. Be concise.

FILING TEXT:
{section_text}

Output as bullets, one per date:
- Meeting Date: [date and time]
- Record Date: [date]
- Mailing Date: [date]
- Outside Date: [date] (and any extensions)

ONLY include dates that ARE disclosed in the filing. Skip any items where the date is blank or not yet set.
Use exact dates and language from the filing. No analysis or commentary.""",

    "consideration": """\
Extract the merger consideration from this merger filing text.
Write in plain text, no markdown. Be concise -- 1-3 sentences max.

FILING TEXT:
{section_text}

Output: near-verbatim description of what shareholders receive per share.
Just the core: what does each share convert into, at what price/ratio.
Do NOT include equity award treatment, appraisal rights, or excluded share categories.""",

    "financing": """\
Extract financing information from this merger filing text.
Write in plain text, no markdown. Be concise.

FILING TEXT:
{section_text}

Output:
- If financing is not a condition to closing, write: "Not a condition to closing."
- If N/A, write: "N/A"
- Otherwise: brief structure from filing language (debt commitment letters, equity financing, etc.)
Use near-verbatim filing language.""",

    "sh_approval": """\
Extract shareholder approval requirements from this merger filing text.
Write in plain text, no markdown. Be concise.

FILING TEXT:
{section_text}

Output as bullets:
- {ticker} -- [voting standard: threshold, effect of abstentions/broker non-votes if stated. Use near-verbatim filing language.]
- {acquirer} -- [acquirer voting requirement, or "Not required."]

Only include parties whose vote is required.""",

    "hsr": """\
Extract HSR (Hart-Scott-Rodino) antitrust filing information from this merger filing text.
Write in plain text, no markdown. Be concise -- 2-4 sentences.

FILING TEXT:
{section_text}

Output: when HSR was filed, by whom, waiting period expiration date, whether a second request was issued, and current status.
Use exact dates and filing language. If HSR is not applicable, write "N/A".""",

    "other_regulatory": """\
Extract non-HSR regulatory approval requirements from this merger filing text.
Write in plain text, no markdown. Be concise.

FILING TEXT:
{section_text}

Output as bullets, one per agency or approval category:
- [Agency/Jurisdiction]: [status, timeline, review period if known]

Include ALL: state regulators, CFIUS, foreign antitrust, money transmitter/financial licensing (state licenses, Bank of Spain, UK FCA, etc.).
Do NOT include HSR/Hart-Scott-Rodino -- covered separately.
Use near-verbatim filing language. If no other regulatory approvals are required, write "N/A".""",

    "closing": """\
Extract closing timeline information from this merger filing text.
Write in plain text, no markdown. Be concise -- 2-3 sentences.

EXTRACTED FACTS (structural context only):
{facts_json}

FILING TEXT:
{section_text}

Output: expected closing timing, outside date if known, and brief caveat.
Use exact dates and filing language.""",

    "conditions": """\
Extract conditions to closing from this merger filing text.
Write in plain text, no markdown.

FILING TEXT:
{section_text}

Output as bullets:
- [condition, using near-verbatim filing language]

Include all material conditions. Be concise -- one bullet per condition.""",

    "termination": """\
Extract termination fee information from this merger filing text.
Write in plain text, no markdown.

FILING TEXT:
{section_text}

Output as bullets -- use near-verbatim language from the filing:
- Company termination fee: $[amount] -- [trigger summary]
- Parent/reverse termination fee: $[amount] -- [trigger summary]
- Regulatory termination fee: $[amount] -- [trigger summary] (if applicable)
- Go-shop period: [duration and dates] (or "None")

Only include items that are disclosed. Search the entire text carefully -- fee amounts
are sometimes in different paragraphs than the triggers.""",
}

SECTION_ORDER = ["dates", "consideration", "financing", "sh_approval",
                 "hsr", "other_regulatory", "closing", "conditions", "termination"]
SECTION_HEADERS = {
    "dates": "DATES", "consideration": "CONSIDERATION", "financing": "FINANCING",
    "sh_approval": "SH APPROVAL", "hsr": "HSR", "other_regulatory": "OTHER REGULATORY",
    "closing": "CLOSING", "conditions": "CONDITIONS TO CLOSING",
    "termination": "TERMINATION & FEES",
}

# =============================================================================
# TOPIC PRIORITY
# =============================================================================

# Priority order: minority topics first so they survive truncation
_TOPIC_PRIORITY = [
    "hsr", "termination", "financing", "regulatory", "sh_approval",
    "dates", "consideration", "closing", "background",
]

# Map section IDs to their relevant topics for extraction
_SECTION_ID_TO_TOPICS = {
    "regulatory": ["regulatory", "hsr"],
    "financing": ["financing"],
    "closing_conditions": ["closing"],
    "termination": ["termination"],
    "vote_info": ["sh_approval", "dates"],
    "consideration_summary": ["consideration"],
    "background": ["background"],
}

# Map change categories to the section IDs that contain their source text
_CATEGORY_TO_SECTIONS = {
    "dates": ["vote_info", "merger_agreement_summary"],
    "consideration": ["consideration_summary", "merger_agreement_summary"],
    "financing": ["financing", "merger_agreement_summary"],
    "sh_votes": ["vote_info", "merger_agreement_summary"],
    "hsr": ["regulatory", "merger_agreement_summary"],
    "regulatory": ["regulatory", "merger_agreement_summary"],
    "closing": ["closing_conditions", "merger_agreement_summary"],
    "termination": ["termination", "merger_agreement_summary"],
}

# Direct topic mapping for comparison
_CATEGORY_TO_TOPICS = {
    "dates": ["dates"],
    "consideration": ["consideration"],
    "financing": ["financing"],
    "sh_votes": ["sh_approval", "dates"],
    "hsr": ["hsr"],
    "regulatory": ["regulatory"],
    "closing": ["closing"],
    "termination": ["termination"],
}

FACT_CATEGORY_MAP = {
    "dates": "dates",
    "consideration": "consideration",
    "financing": "financing",
    "sh_votes": "sh_votes",
    "regulatory": "regulatory",
    "closing_guidance": "closing_guidance",
}

# =============================================================================
# DOCX COLOR CONSTANTS
# =============================================================================

_PRESERVE_UPPER = {"HSR", "SH", "NYSE", "SEC", "FTC", "DOJ", "OCC", "CFIUS", "FIRB"}
