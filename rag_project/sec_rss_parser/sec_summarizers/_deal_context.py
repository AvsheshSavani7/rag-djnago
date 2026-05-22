"""
Deal context injection for SEC filing summarizers.

Provides inject_deal_context() which appends pre-confirmed deal metadata
(5 fields) to a SUMMARY_PROMPT, inserted just before the trailing "...TEXT:"
label so the LLM treats these values as ground-truth rather than inferring them.
"""

from __future__ import annotations
import re


def inject_deal_context(prompt: str, deal_context: dict | None) -> str:
    """Return prompt with deal context block inserted before the trailing TEXT label.

    If deal_context is None or empty, returns the prompt unchanged.

    The injected block contains 5 pre-confirmed fields:
      - primary_ticker  : ticker of the company that filed this form (resolved from CIK)
      - target_name     : target company name
      - target_ticker   : target company ticker
      - acquirer_name   : acquirer company name
      - acquirer_ticker : acquirer company ticker
    """
    if not deal_context:
        return prompt

    ctx = deal_context

    def _val(key: str) -> str:
        v = ctx.get(key)
        return str(v).strip() if v else "N/A"

    block = (
        "\n\nDEAL CONTEXT (use provided values when available. If any value is unavailable, extract it only from the filing text. Do not infer, modify, or add deal facts unless directly stated in the filing):\n"
        f"  Primary ticker (filing company): {_val('primary_ticker')}\n"
        f"  Target company:                  {_val('target_name')}\n"
        f"  Target ticker:                   {_val('target_ticker')}\n"
        f"  Acquirer company:                {_val('acquirer_name')} \n"
        f"  Acquirer ticker:                 {_val('acquirer_ticker')}\n"
    )

    # Insert block before the trailing all-caps label (e.g. "6-K TEXT:", "SECTION EXTRACTS:")
    # that appears at the very end of every SUMMARY_PROMPT / SYNTHESIS_PROMPT.
    match = re.search(r'\n[A-Z0-9][A-Z0-9 \.\-/&]*:\s*$', prompt)
    if match:
        return prompt[: match.start()] + block + prompt[match.start():]

    # Fallback: append directly if no trailing label found
    return prompt.rstrip() + block
