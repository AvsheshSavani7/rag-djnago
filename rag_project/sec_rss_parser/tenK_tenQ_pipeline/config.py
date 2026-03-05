"""Constants, colors, headers, batch size, and prompt strings."""

from pathlib import Path

# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_ENV_FILE = Path(".env")

DEFAULT_SEC_URLS = [
    "https://www.sec.gov/Archives/edgar/data/794619/000079461925000107/amwd-20250731.htm",
    "https://www.sec.gov/Archives/edgar/data/794619/000079461925000115/amwd-20251031.htm",
    "https://www.sec.gov/Archives/edgar/data/794619/000079461926000005/amwd-20260131.htm",
]

DEFAULT_TICKER = "AMWD"

# =============================================================================
# HTTP HEADERS
# =============================================================================

SEC_HEADERS = {
    "User-Agent": "Hyperion Technologies josh@yourcompany.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# =============================================================================
# SCORING / PROCESSING
# =============================================================================

BATCH_SIZE = 5

# =============================================================================
# HIGHLIGHT COLORS (hex, no #)
# =============================================================================

COLOR_TIMING = "DCEEFB"
COLOR_REGULATORY = "FCE4EC"
COLOR_BOTH = "FFF3CD"
COLOR_SIGNIFICANT = "FFCDD2"
COLOR_MODERATE = "FFE0B2"
COLOR_MINOR = "FFF9C4"
COLOR_NEW = "C8E6C9"
COLOR_UNCHANGED = "F5F5F5"

# =============================================================================
# PROMPTS
# =============================================================================

DEAL_SETUP_PROMPT = """You are a financial research assistant. Search the web for merger and acquisition information.

Given a stock ticker, find information about any pending, announced, or recently completed merger/acquisition involving that company.

Respond in valid JSON format only:
{
    "target_company": "Full legal name of the company being acquired",
    "target_aliases": ["Common names", "Abbreviations"],
    "acquirer_company": "Full legal name of the acquiring company",
    "acquirer_aliases": ["Common names", "Parent company names"],
    "merger_sub": "Name of merger subsidiary or null",
    "deal_value": "Total deal value, e.g. '$615 million'",
    "announcement_date": "YYYY-MM-DD or null",
    "expected_close": "Expected close date or quarter",
    "deal_type": "cash merger / stock-for-stock / mixed / tender offer",
    "key_terms": ["per share price", "premium", "key conditions"]
}

If no merger is found, still fill in target_company with the company name.
Return ONLY the JSON object."""


BATCH_SCORING_PROMPT = """You are a financial analyst specializing in merger arbitrage.
You will receive multiple paragraphs from an SEC filing for a company with a PENDING MERGER.
Score EACH paragraph for how DIRECTLY relevant it is to the merger/acquisition itself.

IMPORTANT SCORING RULES:
- Score HIGH (8-10) ONLY for paragraphs that directly discuss the merger transaction:
  deal terms, merger agreement provisions, closing conditions, regulatory approvals,
  termination fees, financing commitments, stockholder votes, delisting, timeline,
  merger-related litigation, merger-related costs/expenses, representations/warranties/
  covenants made under the merger agreement, or restrictions on business conduct
  during the pre-closing period.
- Score MEDIUM (6-7) for paragraphs about merger-related risks, restrictions on
  business operations DUE TO the merger, or impacts on the company specifically
  caused by the pending transaction.
- Score LOW (1-3) for GENERIC business content that would exist regardless of the
  merger: revenue descriptions, product/service offerings, competition, market risks,
  general financial performance, tax matters, segment results, franchise performance,
  industry trends, or boilerplate risk factors NOT specific to this merger.
- A paragraph mentioning "acquisitions" or "strategic transactions" in GENERAL terms
  (not about THIS specific merger) should score 1-3.
- Standard business risk factors (competition, key personnel, IP, regulation) that
  do NOT mention the merger or its specific impact should score 1-3.

The test: Would this paragraph be materially different if there were NO pending merger?
If NO (it would read the same), score 1-3. If YES, score 6+.

Scale:
  1-3: Not about the merger (generic business, operations, financials)
  4-5: Tangentially related or boilerplate that mentions the merger in passing
  6-7: Discusses specific impacts or restrictions caused by this merger
  8-9: Critical merger terms, conditions, regulatory status, or material risks TO the deal
  10:  Core deal terms (price, parties, agreement structure, key conditions)

Respond with a JSON array, one object per paragraph, IN THE SAME ORDER as provided:
[
  {"id": "P1", "score": <1-10>, "rationale": "<brief>", "category": "<one of: deal_terms, risk, regulatory, timeline, financial, general>", "key_info": ["fact1", "fact2"]},
  ...
]

key_info should extract important merger facts if score >= 6, otherwise empty list.
Return ONLY the JSON array."""


OPUS_ASSESSMENT_PROMPT = """You are a senior merger arbitrage analyst reviewing excerpts from an SEC filing for a company with a pending merger. For each excerpt, answer TWO questions with precision.

DEAL CONTEXT:
Acquirer: {acquirer}
Target: {target} ({ticker})
Deal Type: {deal_type} | Value: {deal_value}
Expected Close: {expected_close}

For the excerpt below, provide a JSON response with:
{{
  "timing": {{
    "relevant": true/false,
    "assessment": "1-3 sentence analysis of what timing guidance this provides (dates, milestones, expected close, conditions precedent timeline, HSR waiting periods, stockholder meeting dates, outside date/drop-dead date, etc). If not relevant, briefly say why."
  }},
  "regulatory": {{
    "relevant": true/false,
    "assessment": "1-3 sentence analysis of what regulatory information this provides (HSR Act, antitrust review, CFIUS, FCC, state regulatory approvals, DOJ/FTC actions, international competition authorities, consent decrees, etc). If not relevant, briefly say why."
  }}
}}

Be precise. Only mark "relevant": true if the excerpt SPECIFICALLY discusses timing or regulatory matters related to THIS merger — not general business regulation or generic timelines.

Return ONLY the JSON object."""


TIMING_PROMPT = """You are a merger arbitrage analyst. Your ONLY job in this pass is to identify whether specific DATES, DEADLINES, or TIMELINES changed between filings.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

INCLUDE a finding ONLY IF you see one of these specific changes:
• An actual date was added, removed, or changed (e.g., expected close date shifted, outside date extended, HSR waiting period expiry stated or changed)
• A milestone status changed — e.g., "we expect to complete" → "we completed", or a condition "pending" → "satisfied" or "lapsed"
• A deadline was added or removed (e.g., a termination right now has an expiry date that didn't exist before)
• A specific waiting period started, expired, or changed

DO NOT include:
• General merger risk language about closing uncertainty with no specific date change
• Paragraphs where timing language is present in both filings and unchanged
• Routine period/date references (e.g., "as of September 30, 2024" → "as of December 31, 2024")

If NO actual date or timeline changed, DO NOT include the paragraph.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "current_ref": "CURRENT-N",
      "matched_prior": ["PRIOR-N"] or [],
      "match_type": "exact" | "partial" | "new",
      "changed": true/false,
      "severity": "minor" | "moderate" | "significant",
      "analysis": "State what date or timeline specifically changed. Quote the exact old and new language."
    }}
  ]
}}

Return {{"findings": []}} if no actual timing changes exist."""


REGULATORY_PROMPT = """You are a merger arbitrage analyst. Your ONLY job in this pass is to identify whether a specific REGULATORY EVENT changed between filings.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

INCLUDE a finding ONLY IF you see one of these:
• A regulatory approval was received or denied (name the specific jurisdiction/authority)
• A new regulatory authority or jurisdiction was added to or removed from the required approvals list
• A new governmental investigation, inquiry, lawsuit, or challenge was disclosed or resolved
• A consent decree, remedy, divestiture, or behavioral condition was mentioned for the first time or removed
• An agency action changed: DOJ, FTC, CFIUS, EU Commission, UK CMA, or other named body acted

DO NOT include:
• Generic boilerplate about regulatory approvals being required that is unchanged between filings
• Standard HSR filing language present in both filings without change
• General regulatory risk language with no specific new event or change in status

If NO specific regulatory event changed, DO NOT include the paragraph.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "current_ref": "CURRENT-N",
      "matched_prior": ["PRIOR-N"] or [],
      "match_type": "exact" | "partial" | "new",
      "changed": true/false,
      "severity": "minor" | "moderate" | "significant",
      "analysis": "State what regulatory event occurred or changed. Name the specific authority and action."
    }}
  ]
}}

Return {{"findings": []}} if no specific regulatory events changed."""


LEGAL_LANGUAGE_PROMPT = """You are a merger arbitrage analyst with close attention to legal language. Your ONLY job in this pass is to identify words or phrases that were deliberately ADDED or REMOVED between filings — changes that make you ask: why would a lawyer change this specific wording?

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

INCLUDE a finding ONLY IF you see words or phrases meaningfully added or removed, such as:
• A qualifier added or removed: "material", "reasonably", "substantially all", "certain"
• A hedging word changed: "will" → "may", "shall" → "may", "expects" → "intends"
• A condition added to or removed from a list (e.g., a type of approval dropped from a required approvals list)
• A sentence added or deleted within an otherwise similar paragraph
• A tense shift indicating a milestone completed or reversed: "will complete" → "completed", "is expected" → "was expected"
• A specific named party, obligation, or reference added or removed

DO NOT include:
• Routine period/date references (Q3 2024 → Q3 2025)
• Changes where the substance is identical and the edit is purely stylistic/formatting
• Paragraphs where no specific words were actually added or removed

For every finding you MUST identify the exact old phrase AND new phrase. If you cannot quote both precisely, do not include the finding.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "current_ref": "CURRENT-N",
      "matched_prior": ["PRIOR-N"] or [],
      "match_type": "exact" | "partial" | "new",
      "changed": true/false,
      "severity": "minor" | "moderate" | "significant",
      "notable_changes": [
        {{
          "old_phrase": "exact wording from prior filing (empty string if purely added)",
          "new_phrase": "exact wording from current filing (empty string if purely removed)",
          "interpretation": "what specifically changed — e.g., qualifier removed, condition dropped, tense shifted"
        }}
      ],
      "analysis": "Describe what words changed and why a lawyer might have made this specific edit."
    }}
  ]
}}

Return {{"findings": []}} if no meaningful legal language changes exist."""
