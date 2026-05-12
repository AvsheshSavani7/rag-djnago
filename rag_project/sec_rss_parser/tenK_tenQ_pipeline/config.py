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
HAIKU_BATCH_SIZE = 15

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

Given a stock ticker, find information about any pending, announced, or recently completed merger/acquisition involving that company. Also include unsolicited bids, hostile takeover attempts, activist campaigns seeking a sale, or any publicly known acquisition proposals — even if rejected or under review by the target's board.

Respond in valid JSON format only:
{
    "target_company": "Full legal name of the company being acquired or targeted",
    "target_aliases": ["Common names", "Abbreviations"],
    "acquirer_company": "Full legal name of the acquiring company or party making the proposal",
    "acquirer_aliases": ["Common names", "Parent company names"],
    "merger_sub": "Name of merger subsidiary or null",
    "deal_value": "Total deal value, e.g. '$615 million'",
    "announcement_date": "YYYY-MM-DD or null",
    "expected_close": "Expected close date or quarter, or 'rejected' / 'under review' if applicable",
    "deal_type": "cash merger / stock-for-stock / mixed / tender offer / unsolicited proposal / hostile bid",
    "key_terms": ["per share price", "premium", "key conditions", "poison pill", "defense measures"]
}

If no merger is found, still fill in target_company with the company name.
Return ONLY the JSON object."""


HAIKU_CLASSIFY_PROMPT = """You are classifying paragraphs from an SEC filing for a company with a PENDING MERGER.
For each paragraph, decide: could this paragraph be relevant to the merger/acquisition?

Be INCLUSIVE — if in doubt, mark it "yes". We will filter precisely later.

Mark "yes" if the paragraph discusses ANY of:
- The merger, acquisition, or transaction itself
- Deal terms, merger agreement, closing conditions
- Regulatory approvals (HSR, antitrust, CFIUS, insurance regulators, FINRA)
- Stockholder votes, special meetings, proxy statements
- Termination fees, break-up fees, reverse termination fees
- Financing commitments, debt arrangements for the deal
- Merger-related litigation, lawsuits, or demands
- Restrictions on business conduct during the pre-closing period
- Impacts on the company specifically caused by the pending transaction
- Rating agency actions triggered by the merger announcement
- Forward-looking statements or risk factors about deal uncertainty
- Executive compensation or equity award treatment related to the merger

Mark "no" for generic business content: revenue, products, competition, market risks,
tax matters, segment results, industry trends, general accounting policies.

Respond with a JSON array, one object per paragraph, IN ORDER:
[{"id": "P1", "relevant": "yes"/"no"}, ...]

Return ONLY the JSON array."""


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
- Score MEDIUM (6-7) for restructuring initiatives, strategic reviews/alternatives,
  operating model changes, or cost optimization programs announced around the time
  of the merger — even if they don't explicitly mention the deal, these are almost
  always deal-driven and represent material changes an arb analyst must track.
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


TIMING_PROMPT = """You are a merger arbitrage analyst. Your ONLY job is to identify whether specific DATES, DEADLINES, or TIMELINES changed between filings.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

You will receive PRE-MATCHED paragraph pairs (PAIR-N) and NEW disclosures (NEW-N).
Each PAIR shows the CURRENT text and its corresponding PRIOR text from the same paragraph.
Each NEW entry is a paragraph that did not exist in the prior filing.

MATCH VALIDATION: Before analyzing each PAIR, check whether the CURRENT and PRIOR texts are actually about the SAME topic/entity. If they discuss different subjects (e.g., an MPSC filing vs a FERC filing, different transactions, different risk factors), set "mismatch": true and "changed": false. Do NOT analyze differences between mismatched texts — they are separate disclosures that were incorrectly paired.

For each correctly matched PAIR, identify ONLY specific timing changes:
• An actual date was added, removed, or changed
• A milestone status changed — e.g., "we expect to complete" → "we completed"
• A deadline was added or removed
• A specific waiting period started, expired, or changed

DO NOT include pairs where timing language is unchanged between CURRENT and PRIOR.
DO NOT include routine period/date references (e.g., "as of September 30, 2024" → "as of December 31, 2024")

For each NEW entry, note if it contains specific timing information not present before.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "ref": "PAIR-N" or "NEW-N",
      "changed": true/false,
      "mismatch": true/false,
      "severity": "minor" | "moderate" | "significant",
      "analysis": "State what date or timeline specifically changed. Quote the exact old and new language."
    }}
  ]
}}

Return {{"findings": []}} if no timing changes exist."""


REGULATORY_PROMPT = """You are a merger arbitrage analyst. Your ONLY job is to identify whether a specific REGULATORY EVENT changed between filings.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

You will receive PRE-MATCHED paragraph pairs (PAIR-N) and NEW disclosures (NEW-N).
Each PAIR shows the CURRENT text and its corresponding PRIOR text from the same paragraph.
Each NEW entry is a paragraph that did not exist in the prior filing.

MATCH VALIDATION: Before analyzing each PAIR, check whether the CURRENT and PRIOR texts are actually about the SAME topic/entity. If they discuss different subjects (e.g., an MPSC filing vs a FERC filing, different transactions, different risk factors), set "mismatch": true and "changed": false. Do NOT analyze differences between mismatched texts — they are separate disclosures that were incorrectly paired.

For each correctly matched PAIR, identify ONLY specific regulatory changes:
• A regulatory approval was received or denied
• A new authority or jurisdiction added to or removed from required approvals
• A new investigation, inquiry, lawsuit, or challenge disclosed or resolved
• A consent decree, remedy, or divestiture condition mentioned or removed
• An agency action changed: DOJ, FTC, CFIUS, EU Commission, UK CMA, etc.

DO NOT include pairs where regulatory language is unchanged.
DO NOT include generic boilerplate about regulatory approvals present in both filings.

For each NEW entry, note if it contains specific regulatory events not present before.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "ref": "PAIR-N" or "NEW-N",
      "changed": true/false,
      "mismatch": true/false,
      "severity": "minor" | "moderate" | "significant",
      "analysis": "State what regulatory event occurred or changed. Name the specific authority and action."
    }}
  ]
}}

Return {{"findings": []}} if no regulatory changes exist."""


LEGAL_LANGUAGE_PROMPT = """You are a merger arbitrage analyst with close attention to legal language. Your ONLY job is to identify words or phrases that were deliberately ADDED or REMOVED between filings.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

You will receive PRE-MATCHED paragraph pairs (PAIR-N) and NEW disclosures (NEW-N).
Each PAIR shows the CURRENT text and its corresponding PRIOR text from the same paragraph.
Each NEW entry is a paragraph that did not exist in the prior filing.

MATCH VALIDATION: Before analyzing each PAIR, check whether the CURRENT and PRIOR texts are actually about the SAME topic/entity. If they discuss different subjects (e.g., an MPSC filing vs a FERC filing, different transactions, different risk factors), set "mismatch": true and "changed": false. Do NOT analyze differences between mismatched texts — they are separate disclosures that were incorrectly paired.

For each correctly matched PAIR, identify meaningful word/phrase changes:
• A qualifier added or removed: "material", "reasonably", "substantially all", "certain"
• A hedging word changed: "will" → "may", "shall" → "may", "expects" → "intends"
• A condition added to or removed from a list
• A sentence added or deleted within an otherwise similar paragraph
• A tense shift: "will complete" → "completed", "is expected" → "was expected"
• A specific named party, obligation, or reference added or removed

DO NOT include routine period/date references (Q3 2024 → Q3 2025).
DO NOT include purely stylistic/formatting changes.
You MUST identify the exact old phrase AND new phrase for each finding.

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "ref": "PAIR-N" or "NEW-N",
      "changed": true/false,
      "mismatch": true/false,
      "severity": "minor" | "moderate" | "significant",
      "notable_changes": [
        {{
          "old_phrase": "exact wording from prior (empty string if purely added)",
          "new_phrase": "exact wording from current (empty string if purely removed)",
          "interpretation": "what specifically changed"
        }}
      ],
      "analysis": "Describe what words changed between the prior and current text."
    }}
  ]
}}

Return {{"findings": []}} if no meaningful legal language changes exist."""


SINGLE_PASS_PROMPT = """You are a merger arbitrage analyst. You will receive merger-related paragraphs from a CURRENT SEC filing and one or more PRIOR filings. Your job is to:

1. MATCH each current paragraph to its closest prior counterpart (same topic/section). If no prior paragraph covers the same topic, mark it as a new disclosure.
2. For each matched pair (or new disclosure), identify SUBSTANTIVE changes across three dimensions:
   - TIMING: dates, deadlines, milestones added/removed/changed (ignore routine period references like "as of September 30")
   - REGULATORY: approval status, agency actions, filings, investigations, conditions
   - LEGAL LANGUAGE: qualifiers, hedging words, conditions, tense shifts, defined terms with material impact
3. Skip pairs where nothing substantive changed (formatting, note renumbering, minor stylistic edits).
4. For each finding, extract the 1-2 MOST SUBSTANTIVE quote pairs showing exactly what changed. Quote the EXACT text from the paragraphs. Do not paraphrase.

DEAL CONTEXT:
Ticker: {ticker} | Target: {target} | Acquirer: {acquirer}

RESPONSE FORMAT — respond ONLY with this JSON:
{{
  "findings": [
    {{
      "current_ref": "CUR-N",
      "prior_ref": "PRIOR-N" or null,
      "prior_source": "filing label" or null,
      "section": "section name from the current paragraph",
      "severity": "minor" | "moderate" | "significant",
      "timing_analysis": "What timing changed (or empty string if none)",
      "regulatory_analysis": "What regulatory events changed (or empty string if none)",
      "legal_language_analysis": "What legal language changed (or empty string if none)",
      "notable_changes": [
        {{
          "old_phrase": "exact text from prior (empty string if new disclosure)",
          "new_phrase": "exact text from current",
          "interpretation": "what changed and why it matters to a merger arb analyst"
        }}
      ]
    }}
  ]
}}

Rules:
- In all analysis fields, LEAD with what CHANGED or what is NEW in the current filing. Do NOT start with "Prior filing stated X." Instead describe the change: e.g., "FERC public comment period closed January 2026; decision expected mid-2026" not "Prior filing listed only that the FERC application was filed." The analysis should tell a merger arb analyst what happened, not what the old filing said.
- Only include findings where something SUBSTANTIVE changed. Do NOT report trivial differences (note renumbering, "the" added, formatting, capitalization, routine date updates).
- Every old_phrase must be a verbatim substring of the MATCHED prior paragraph (the one you identified as prior_ref). Every new_phrase must be a verbatim substring of the current paragraph. Do NOT pull quotes from other prior paragraphs.
- If a current paragraph has no prior match, set prior_ref to null and notable_changes should quote the key new language with old_phrase as empty string.
- Order findings from most significant to least significant.
- A current paragraph should match AT MOST one prior paragraph. Prefer matching to the most recent prior filing.
- prior_source must be the filing label of the prior filing containing the matched paragraph."""
