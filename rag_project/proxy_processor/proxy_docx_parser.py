"""
Parse proxy summary DOCX files — extract the 5 Q&A answers and Chronological Summary.

DOCX structure:
  "Proxy Summary"              ← title (style: Title)
  <question text>              ← Normal, no leading +
  "+   <answer text>"          ← Normal, starts with +
  ... (5 Q&A pairs)
  Heading 1: "Merger Background Analysis - Client Deliverables"
  Heading 1: "Chronological Summary"
    Normal paragraphs...       ← collected until next Heading 1
  Heading 1: "Extraction of other client deliverable sections"
  ...
"""

import io
import json
import os

import requests
from docx import Document

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "quetions.json")


def _load_questions() -> dict:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _strip_answer_bullet(text: str) -> str:
    """Remove leading + and whitespace from answer paragraphs."""
    return text.lstrip("+\t ").strip()


def parse_proxy_summary_docx(url: str) -> dict:
    """
    Download and parse a proxy summary DOCX from the given URL.

    Returns:
      {
        "qa_items": [
          {
            "question_key": "question_1",
            "question": "<full question text from quetions.json>",
            "answer": "<answer extracted from DOCX>"
          },
          ...  (up to 5 items)
        ],
        "chronological_summary": [
          "<paragraph text>",
          ...
        ]  # empty list if section not found
      }
    """
    questions = _load_questions()
    question_keys = [f"question_{n}" for n in range(1, 6)]

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    doc = Document(io.BytesIO(resp.content))
    paras = doc.paragraphs

    # ── Pass 1: collect Q&A pairs before the first Heading ──────────────────
    qa_pairs = []
    i = 0
    while i < len(paras):
        p = paras[i]
        style = p.style.name
        text = p.text.strip()

        if style.startswith("Heading"):
            break

        if not text or text.startswith("+"):
            i += 1
            continue

        # Non-empty, non-answer paragraph — look ahead for the answer line
        j = i + 1
        while j < len(paras) and not paras[j].text.strip():
            j += 1

        if j < len(paras) and paras[j].text.strip().startswith("+"):
            qa_pairs.append({
                "question_text": text,
                "answer": _strip_answer_bullet(paras[j].text.strip()),
            })
            i = j + 1
        else:
            i += 1

    # Map positionally to question keys and canonical question text from JSON
    qa_items = []
    for idx, pair in enumerate(qa_pairs[:5]):
        key = question_keys[idx] if idx < len(question_keys) else f"question_{idx + 1}"
        qa_items.append({
            "question_key": key,
            "question": questions.get(key, pair["question_text"]),
            "answer": pair["answer"],
        })

    # ── Pass 2: find "Chronological Summary" Heading and collect its body ───
    chronological_summary = []
    in_chrono = False

    for p in paras:
        style = p.style.name
        text = p.text.strip()

        if style.startswith("Heading"):
            if text == "Chronological Summary":
                in_chrono = True
                continue
            elif in_chrono:
                # Hit the next heading — stop collecting
                break

        if in_chrono and text:
            # Skip internal validation/format lines
            if text.startswith("✓") or text.startswith("✗"):
                continue
            chronological_summary.append(text)

    return {
        "qa_items": qa_items,
        "chronological_summary": chronological_summary,
    }
