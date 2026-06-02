"""
html_builder.py — Email-safe HTML for proxy comparison change reports.
Mirrors styling rules in docx_builder.create_changes_docx.
"""

import html
import re

from .report_format import parse_summary_sections, smart_title

# Hex colors matching docx_builder RGBColor constants
NAVY = "#1F4E79"
DARK_GRAY = "#333333"
GREEN = "#007A33"
AMBER = "#BF8F00"
LIGHT_GRAY = "#999999"
RED = "#CC0000"

_P = 'margin:0 0 8px 0; font-family:Calibri,Arial,sans-serif; font-size:11px; line-height:1.5; color:#333333;'
_H2 = (
    f"margin:18px 0 6px 0; font-family:Calibri,Arial,sans-serif; font-size:13px;"
    f" font-weight:bold; color:{NAVY};"
)
_H3 = (
    f"margin:12px 0 6px 0; font-family:Calibri,Arial,sans-serif; font-size:11px;"
    f" font-weight:bold; color:{NAVY};"
)
_BULLET = f"margin:0 0 6px 0; padding-left:20px; font-family:Calibri,Arial,sans-serif; font-size:11px; line-height:1.5;"


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def _styled_inline_html(text: str) -> str:
    """Inline spans for [NEW] tags and arrow changes."""
    if "[NEW]" in text:
        parts = text.split("[NEW]")
        out = _esc(parts[0].strip())
        out += f' <span style="color:{GREEN}; font-weight:bold; font-size:10px;">[NEW]</span>'
        if len(parts) > 1 and parts[1].strip():
            out += " " + _esc(parts[1].strip())
        return out
    if " -> " in text:
        return f'<span style="color:{AMBER};">{_esc(text)}</span>'
    return _esc(text)


def _is_no_changes_line(stripped: str) -> bool:
    lower = stripped.lower()
    return (
        "no changes" in lower
        or "no material changes" in lower
        or "substantially identical" in lower
        or "not found" in lower
        or "sentences unchanged" in lower
    )


def _render_change_line(stripped: str) -> str:
    """Render one content line from a change report section."""
    sub_match = re.match(
        r"^(NEW SENTENCES|MODIFIED SENTENCES|REMOVED SENTENCES):?$", stripped
    )
    if sub_match:
        return f'<h3 style="{_H3}">{_esc(smart_title(sub_match.group(1)))}</h3>'

    if stripped.startswith("+ "):
        body = _esc(stripped[2:])
        return (
            f'<p style="{_BULLET}">'
            f'<span style="color:{GREEN}; font-size:9px;">&#8226;</span> '
            f'<span style="color:{GREEN}; font-size:9px;">{body}</span></p>'
        )

    if stripped.startswith("- ") and stripped.startswith('- "'):
        body = _esc(stripped[2:])
        return (
            f'<p style="{_BULLET}">'
            f'<span style="color:{RED}; font-size:9px;">&#8226;</span> '
            f'<span style="color:{RED}; font-size:9px;">{body}</span></p>'
        )

    if stripped.startswith("OLD: "):
        return (
            f'<p style="{_P} font-size:9px;">'
            f"<strong>OLD: </strong>"
            f'<span style="color:{LIGHT_GRAY}; font-size:9px;">{_esc(stripped[5:])}</span></p>'
        )

    if stripped.startswith("NEW: "):
        return (
            f'<p style="{_P} font-size:9px;">'
            f"<strong>NEW: </strong>"
            f'<span style="font-size:9px;">{_esc(stripped[5:])}</span></p>'
        )

    if _is_no_changes_line(stripped):
        return (
            f'<p style="{_P}"><em style="color:{LIGHT_GRAY};">{_esc(stripped)}</em></p>'
        )

    if "[NEW]" in stripped or " -> " in stripped:
        return f'<p style="{_P}">{_styled_inline_html(stripped)}</p>'

    if stripped.startswith("- "):
        body = _styled_inline_html(stripped[2:])
        return (
            f'<p style="{_BULLET}">'
            f'<span style="color:{DARK_GRAY};">&#8226;</span> {body}</p>'
        )

    return f'<p style="{_P}">{_esc(stripped)}</p>'


def _render_section_content(content: str) -> str:
    if not content.strip():
        return ""
    parts = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        parts.append(_render_change_line(stripped))
    return "".join(parts)


def _render_title_block(
    ticker: str,
    target: str,
    acquirer: str,
    doc_type_label: str,
    timestamp: str,
) -> str:
    return f"""
<div style="text-align:center; margin:0 0 24px 0; padding-bottom:20px; border-bottom:2px solid #e8e8e8;">
  <p style="margin:0 0 8px 0; font-family:Calibri,Arial,sans-serif; font-size:22px; font-weight:bold; color:{NAVY};">
    {_esc(ticker)} Merger Filing Analysis
  </p>
  <p style="margin:0 0 16px 0; font-family:Calibri,Arial,sans-serif; font-size:16px; color:{DARK_GRAY};">
    {_esc(doc_type_label)}
  </p>
  <p style="margin:4px 0; font-size:12px; color:{DARK_GRAY};">Target: {_esc(target)}</p>
  <p style="margin:4px 0; font-size:12px; color:{DARK_GRAY};">Acquirer: {_esc(acquirer)}</p>
  <p style="margin:12px 0 0 0; font-size:10px; color:{LIGHT_GRAY}; font-style:italic;">
    Generated: {_esc(timestamp)}
  </p>
</div>
"""


def create_changes_html(
    change_text: str,
    ticker: str,
    target: str,
    acquirer: str,
    old_label: str,
    new_label: str,
    timestamp: str,
) -> str:
    """Build email-safe HTML fragment for a proxy comparison change report."""
    doc_label = f"Changes: {old_label} -> {new_label}"
    parts = [
        _render_title_block(ticker, target, acquirer, doc_label, timestamp),
    ]

    opening, sections = parse_summary_sections(change_text)
    if opening:
        parts.append(f'<p style="{_P}">{_esc(opening)}</p>')

    for header, content in sections:
        parts.append(f'<h2 style="{_H2}">{_esc(smart_title(header))}</h2>')
        parts.append(_render_section_content(content))

    return (
        '<div style="margin-top:20px; border-top:2px solid #e0e0e0; padding-top:16px;">'
        + "".join(parts)
        + "</div>"
    )
