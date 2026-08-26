#!/usr/bin/env python3
"""Extract SC 14D-9 background text from a parsed sections JSON.

Looks up ITEM 4 (The Solicitation or Recommendation) and slices the
chronology between a Background heading and a Reasons heading.

Priority:
1. Text between a Background heading and a Reasons heading.
2. If no Background heading, text from the start of Item 4 up to Reasons.
3. Otherwise None (caller can fall back to Pinecone later).

Usage:
    python extract_sc14d9_background.py path/to/d47680dsc14d9_text.json
    python extract_sc14d9_background.py --input-dir output/sc14d9_new
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE / "output" / "sc14d9_new"

ITEM4_TITLE_RE = re.compile(r"^ITEM\s+4\b", re.I)
ENUM_PREFIX_RE = re.compile(
    r"^(?:"
    r"\(\s*(?:[ivxlcdm]+|[a-z]|\d+)\s*\)"
    r"|[a-z]\."
    r"|\d+\."
    r")\s+",
    re.I,
)
# Prefer "Offer and the Merger" / "Merger Agreement" over a shorter
# "Background of the Offer" line that sometimes appears just above it.
START_SPECIFIC_RE = re.compile(
    r"^background of(?: the)? ("
    r"offers? and(?: the)? .+"
    r"|merger agreements?"
    r"|proposed (?:mergers?|transactions?|acquisitions?)"
    r")$",
    re.I,
)
START_GENERIC_RE = re.compile(
    r"^background of(?: the)? ("
    r"offers?"
    r"|mergers?"
    r"|transactions?"
    r"|acquisitions?"
    r")$",
    re.I,
)
END_RE = re.compile(
    r"^reasons for(?: the)?(?: .+)? recommendation\b",
    re.I,
)


def _normalize(text: str) -> str:
    text = (text or "").replace("\u2014", " ").replace("\u2013", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[“”\"']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_enum_prefix(text: str) -> str:
    return ENUM_PREFIX_RE.sub("", text).strip()


def _strip_heading_punct(text: str) -> str:
    """Headings in 14D-9s are often followed by a period or colon."""
    return text.rstrip(".:")


def _is_combined_header(text: str) -> bool:
    lower = text.lower()
    return "background" in lower and "reasons for" in lower


def _classify_heading(line: str) -> str | None:
    raw = _normalize(line)
    if not raw or len(raw) > 120:
        return None
    if _is_combined_header(raw):
        return "combined"
    if raw[-1] in ",;":
        return None
    body = _strip_heading_punct(_strip_enum_prefix(raw))
    if not body or len(body) > 120:
        return None
    if START_SPECIFIC_RE.fullmatch(body):
        return "start_specific"
    if START_GENERIC_RE.fullmatch(body):
        return "start_generic"
    if END_RE.match(body):
        return "end"
    return None


def find_item4_section(sections: list[dict[str, Any]]) -> dict[str, Any] | None:
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        title = _normalize(section.get("title") or "")
        if ITEM4_TITLE_RE.match(title):
            return section
    return None


def extract_background_from_item4(content: str) -> dict[str, Any]:
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    start_specific: list[int] = []
    start_generic: list[int] = []
    ends: list[int] = []
    labels: dict[int, str] = {}

    for index, line in enumerate(lines):
        kind = _classify_heading(line)
        if kind == "start_specific":
            start_specific.append(index)
            labels[index] = line
        elif kind == "start_generic":
            start_generic.append(index)
            labels[index] = line
        elif kind == "end":
            ends.append(index)
            labels[index] = line

    start_index = start_specific[0] if start_specific else (
        start_generic[0] if start_generic else None
    )
    method = None
    end_index = None

    if start_index is not None:
        end_index = next((index for index in ends if index > start_index), None)
        if end_index is None:
            return {
                "text": None,
                "method": None,
                "start_heading": labels.get(start_index),
                "end_heading": None,
                "reason": "found background heading but no later reasons heading",
            }
        sliced = lines[start_index + 1:end_index]
        method = "between_headings"
    elif ends:
        end_index = ends[0]
        sliced = lines[:end_index]
        method = "item4_start_to_reasons"
    else:
        return {
            "text": None,
            "method": None,
            "start_heading": None,
            "end_heading": None,
            "reason": "no background or reasons heading in item 4",
        }

    text = "\n\n".join(sliced).strip()
    if not text:
        return {
            "text": None,
            "method": method,
            "start_heading": labels.get(start_index) if start_index is not None else None,
            "end_heading": labels.get(end_index) if end_index is not None else None,
            "reason": "slice was empty",
        }
    return {
        "text": text,
        "method": method,
        "start_heading": labels.get(start_index) if start_index is not None else None,
        "end_heading": labels.get(end_index) if end_index is not None else None,
        "reason": None,
    }


def extract_sc14d9_background(sections: list[dict[str, Any]]) -> dict[str, Any]:
    item4 = find_item4_section(sections)
    if not item4:
        return {
            "text": None,
            "method": None,
            "item4_title": None,
            "start_heading": None,
            "end_heading": None,
            "reason": "item 4 section not found",
        }
    result = extract_background_from_item4(item4.get("content") or "")
    result["item4_title"] = item4.get("title")
    result["item4_chars"] = len(item4.get("content") or "")
    return result


def sections_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("sections"), list):
        return payload["sections"]
    raise ValueError("JSON is not a sections array")


def load_sections(path: Path) -> list[dict[str, Any]]:
    return sections_from_payload(json.loads(path.read_text(encoding="utf-8")))


def run_one(path: Path, write_dir: Path | None = None) -> dict[str, Any]:
    sections = load_sections(path)
    result = extract_sc14d9_background(sections)
    result["file"] = path.name
    text = result.get("text")
    result["chars"] = len(text) if text else 0
    if write_dir and text:
        write_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_dir / f"{path.stem}_background.txt"
        out_path.write_text(text + "\n", encoding="utf-8")
        result["output"] = str(out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SC 14D-9 Item 4 background text from parsed JSON.",
    )
    parser.add_argument("paths", nargs="*", help="Parsed *_text.json file(s)")
    parser.add_argument(
        "--input-dir",
        help="Directory of parsed SC 14D-9 JSON files",
    )
    parser.add_argument(
        "--write-dir",
        help="Optional directory to write extracted background .txt files",
    )
    args = parser.parse_args()

    files: list[Path] = [Path(p) for p in args.paths]
    if args.input_dir:
        files.extend(sorted(Path(args.input_dir).glob("*_text.json")))
    elif not files:
        files = sorted(DEFAULT_INPUT_DIR.glob("*_text.json"))

    if not files:
        parser.error("No JSON files found")

    write_dir = Path(args.write_dir) if args.write_dir else None
    ok = 0
    failed = 0
    for path in files:
        result = run_one(path, write_dir=write_dir)
        status = "OK" if result.get("text") else "FAIL"
        if result.get("text"):
            ok += 1
        else:
            failed += 1
        print(
            f"{status:4} {result['file']}: method={result.get('method')} "
            f"chars={result.get('chars')} "
            f"start={result.get('start_heading')!r} "
            f"end={result.get('end_heading')!r}"
        )
        if result.get("text"):
            text = result["text"]
            print(f"     first 100: {text[:100]!r}")
            print(f"     last 100:  {text[-100:]!r}")
        else:
            print(f"     reason: {result.get('reason')}")

    print(f"\nDone. OK={ok} FAIL={failed} files={len(files)}")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
