"""
Parse DMA summary DOCX files into JSON suitable for the UI.

DOCX structure:
  "Merger Agreement Clause Summaries"   ← title
  "Concise Summary"                     ← marker
    Heading 2  → section name
    Normal (+) → clause text
    Normal (○) → references for previous clause
  "Fulsome Summary"                     ← marker
    (same pattern)
"""

import io
import requests
from docx import Document


def _strip_bullet(text: str) -> str:
    """Remove leading bullet characters (+, ○, •, -, tabs, spaces)."""
    return text.lstrip("+○•-\t ").strip()


def _parse_references(text: str) -> list:
    """Split a references line into a list."""
    # Remove "References:" prefix if present
    text = text.replace("References:", "").strip()
    # Split on semicolons
    return [r.strip() for r in text.split(";") if r.strip()]


def _build_sections(paragraphs: list) -> list:
    """
    Convert a flat list of paragraphs (within one summary block) into
    a list of sections: [{ name, clauses: [{ text, references }] }]
    """
    sections = []
    current_section = None
    current_clause = None

    for para in paragraphs:
        style = para.style.name
        text = para.text.strip()

        if not text:
            continue

        if style == "Heading 2":
            # Save previous clause
            if current_clause and current_section is not None:
                current_section["clauses"].append(current_clause)
                current_clause = None
            # New section
            current_section = {"name": text, "clauses": []}
            sections.append(current_section)

        elif text.startswith("+") or text.startswith("•"):
            if current_section is None:
                continue
            # Save previous clause before starting new one
            if current_clause is not None:
                current_section["clauses"].append(current_clause)
            current_clause = {"text": _strip_bullet(text), "references": []}

        elif text.startswith("○") or text.startswith("◦"):
            if current_clause is not None:
                refs = _parse_references(_strip_bullet(text))
                current_clause["references"].extend(refs)

    # Flush last clause
    if current_clause and current_section is not None:
        current_section["clauses"].append(current_clause)

    return sections


def parse_dma_summary_docx(url: str) -> dict:
    """
    Download and parse a DMA summary DOCX from the given URL.
    Returns:
      {
        "concise_sections": [ { name, clauses: [{ text, references }] } ],
        "fulsome_sections": [ { name, clauses: [{ text, references }] } ]
      }
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    doc = Document(io.BytesIO(resp.content))
    paras = doc.paragraphs

    # Find split points
    concise_start = None
    fulsome_start = None

    for i, p in enumerate(paras):
        t = p.text.strip()
        if t == "Concise Summary":
            concise_start = i + 1
        elif t == "Fulsome Summary":
            fulsome_start = i + 1

    concise_paras = []
    fulsome_paras = []

    if concise_start is not None and fulsome_start is not None:
        concise_paras = paras[concise_start:fulsome_start - 1]
        fulsome_paras = paras[fulsome_start:]
    elif concise_start is not None:
        concise_paras = paras[concise_start:]
    elif fulsome_start is not None:
        fulsome_paras = paras[fulsome_start:]
    else:
        # No markers — treat entire doc as concise
        concise_paras = paras[1:]  # skip title

    return {
        "concise_sections": _build_sections(concise_paras),
        "fulsome_sections": _build_sections(fulsome_paras),
    }
