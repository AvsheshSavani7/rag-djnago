#!/usr/bin/env python3
"""Parse Schedule 14D-9 filings by Item 1-9 headings.

TOC is optional. Body Item headings (no page number) are the source of truth.
Annex/Appendix sections after Item 9 are captured when present.
Subsections are not extracted.

Call from other code the same way as DocumentExtractionPipeline:

    from sec_rss_parser.parse_sc14d9 import parse_sc14d9, SC14D9ExtractionPipeline

    sections = parse_sc14d9(url)
    # or
    result = SC14D9ExtractionPipeline().process(url=url, output_dir=out_dir)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60
MAX_HTML_CHARS = 30_000_000
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "output" / "sc14d9_filings.json"
DEFAULT_OUTPUT_DIR = HERE / "output" / "sc14d9"

CHROMIUM_EXECUTABLE = os.environ.get("MNA_CHROMIUM_EXECUTABLE")
SEC_USER_AGENT = os.environ.get(
    "MNA_SEC_USER_AGENT",
    "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
)

CANONICAL_TITLES = {
    "1": "SUBJECT COMPANY INFORMATION",
    "2": "IDENTITY AND BACKGROUND OF FILING PERSON",
    "3": "PAST CONTACTS, TRANSACTIONS, NEGOTIATIONS AND AGREEMENTS",
    "4": "THE SOLICITATION OR RECOMMENDATION",
    "5": "PERSONS/ASSETS RETAINED, EMPLOYED, COMPENSATED OR USED",
    "6": "INTEREST IN SECURITIES OF THE SUBJECT COMPANY",
    "7": "PURPOSES OF THE TRANSACTION AND PLANS OR PROPOSALS",
    "8": "ADDITIONAL INFORMATION",
    "9": "EXHIBITS",
}

ITEM_HEADING_RE = re.compile(
    r"^ITEM\s+([1-9])\s*[.\-—:]?\s*(.*)$",
    re.I,
)
ANNEX_HEADING_RE = re.compile(
    r"^(?:ANNEX|APPENDIX)\s+"
    r"(?P<label>(?:[IVXLCDM]{1,6}|[A-Z])(?:-\d+)?)"
    r"(?:\s*[.\-—:]?\s*|\s+)(?P<rest>.*)$",
    re.I,
)
EXHIBIT_NO_RE = re.compile(
    r"^\([a-z]\)(?:\(\d+[a-z]?\)|[A-Z0-9.\-]+)",
    re.I,
)
EXHIBIT_PHRASE_RE = re.compile(
    r"included as annex|attached as annex|filed (?:herewith|as)|"
    r"incorporated by reference|see page",
    re.I,
)
SIGNATURE_RE = re.compile(r"^SIGNATURES?$", re.I)
AFTER_INQUIRY_RE = re.compile(
    r"^After due inquiry and to the best of my knowledge",
    re.I,
)
PAGE_LABEL_RE = re.compile(r"^PAGES?$", re.I)
TOC_HEADER_RE = re.compile(r"^TABLE\s+OF\s+CONTENTS$", re.I)
PAGE_TOKEN_RE = re.compile(
    r"^\(?(?:\d{1,4}|[ivxlcdm]{1,10}|[A-Z0-9]+(?:\s*[-–—]\s*[A-Z0-9]+)+)\)?$",
    re.I,
)
TRAILING_PAGE_RE = re.compile(
    r"^(?P<title>.+?)(?:\s+|\.{2,})"
    r"(?P<page>\d{1,4}|[ivxlcdm]{1,10}|[A-Z0-9]+(?:\s*[-–—]\s*[A-Z0-9]+)+)\s*$",
    re.I,
)
PHONE_RE = re.compile(r"^\d{3}-\d{4}$")
ROMAN_RE = re.compile(r"^-?[ivxlcdm]+-?$", re.I)

DOM_EXTRACTOR_JS = r"""
() => {
  const clean = (value) => String(value || "")
    .replace(/[\u00a0\u2000-\u200d\u202f\u205f\ufeff]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n+/g, "\n")
    .trim();

  const visible = (el) => {
    if (!el || !el.getBoundingClientRect) return false;
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" &&
           Number(style.opacity || 1) !== 0 && rect.width > 1 && rect.height > 1;
  };

  const rectData = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return {
      left: r.left + window.scrollX,
      top: r.top + window.scrollY,
      right: r.right + window.scrollX,
      bottom: r.bottom + window.scrollY,
      width: r.width,
      height: r.height,
      paddingLeft: parseFloat(s.paddingLeft) || 0,
      textIndent: parseFloat(s.textIndent) || 0,
      fontWeight: parseInt(s.fontWeight, 10) || (s.fontWeight === "bold" ? 700 : 400),
    };
  };

  const firstTextLeft = (el) => {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (!clean(node.nodeValue)) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      const rects = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
      if (rects.length) return rects[0].left + window.scrollX;
    }
    const data = rectData(el);
    return data.left + data.paddingLeft + data.textIndent;
  };

  const blocks = [];
  let seq = 0;

  document.querySelectorAll("tr").forEach((row) => {
    if (!visible(row)) return;
    const cells = [...row.querySelectorAll(":scope > td, :scope > th")]
      .filter(visible)
      .map((cell) => {
        const a = cell.querySelector("a[href]");
        const href = a ? a.getAttribute("href") : "";
        return { text: clean(cell.innerText), textLeft: firstTextLeft(cell), href, ...rectData(cell) };
      })
      .filter((cell) => cell.text);
    const text = clean(row.innerText);
    if (!text || !cells.length) return;
    const table = row.closest("table");
    const tableRect = table && visible(table) ? rectData(table) : rectData(row.parentElement || row);
    blocks.push({
      seq: seq++, tag: "TR", text, cells,
      containerLeft: tableRect.left,
      ...rectData(row),
    });
  });

  document.querySelectorAll("h1,h2,h3,h4,h5,h6,p,li,div").forEach((el) => {
    if (!visible(el) || el.closest("tr")) return;
    if (el.tagName === "DIV") {
      const nested = el.querySelector(":scope > div, :scope > p, :scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6, :scope > table, :scope > ul, :scope > ol");
      if (nested) return;
    }
    const text = clean(el.innerText);
    if (!text || text.length > 100000) return;
    const parent = el.parentElement;
    const parentRect = parent && visible(parent) ? rectData(parent) : rectData(document.body);
    const a = el.querySelector("a[href]");
    const href = a ? a.getAttribute("href") : "";
    blocks.push({
      seq: seq++, tag: el.tagName, text, cells: [], href,
      containerLeft: parentRect.left,
      textLeft: firstTextLeft(el),
      ...rectData(el),
    });
  });

  blocks.sort((a, b) => (a.top - b.top) || (a.left - b.left) || (a.seq - b.seq));

  const output = [];
  for (const block of blocks) {
    const previous = output[output.length - 1];
    const duplicate = previous && previous.text === block.text &&
      Math.abs(previous.top - block.top) <= 2 && Math.abs(previous.left - block.left) <= 3;
    if (!duplicate) output.push(block);
    else if (block.tag === "TR" && previous.tag !== "TR") output[output.length - 1] = block;
  }
  return output.map((item, index) => ({ ...item, index }));
}
"""


def one_line(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_page_token(value: Any) -> str:
    return re.sub(r"\s*[-–—]\s*", "-", one_line(value))


def is_valid_page_token(value: Any) -> bool:
    token = normalize_page_token(value)
    if not token or PHONE_RE.fullmatch(token):
        return False
    if not PAGE_TOKEN_RE.fullmatch(token):
        return False
    if token.isdigit():
        return 1 <= int(token) <= 1500
    if "-" in token:
        tail = token.rsplit("-", 1)[-1]
        return tail.isdigit() and 1 <= int(tail) <= 1500
    return bool(re.fullmatch(
        r"\(?M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\)?",
        token,
        re.I,
    ))


def split_title_page(value: Any) -> tuple[str, str] | None:
    text = one_line(value)
    match = TRAILING_PAGE_RE.match(text)
    if not match:
        return None
    title = one_line(match.group("title"))
    page_no = normalize_page_token(match.group("page"))
    if not title or not is_valid_page_token(page_no):
        return None
    return title, page_no


def accession_from_url(url: str) -> str:
    raw = url.rstrip("/").split("/")[-1].split(".")[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def fetch_html(url: str) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=REQUEST_TIMEOUT,
    )
    html = response.text or ""
    if (
        response.status_code == 403
        or "Undeclared Automated Tool" in html
        or "Request Originates" in html
    ):
        raise ValueError(f"SEC blocked the request: HTTP {response.status_code}")
    if response.status_code >= 400:
        raise ValueError(f"HTTP {response.status_code}")
    if len(html) < 1000:
        raise ValueError(f"HTML is too short: {len(html)} characters")
    if len(html) > MAX_HTML_CHARS:
        raise ValueError(f"HTML exceeds safety limit: {len(html)} characters")
    return html


def browser_launch_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "headless": True,
        "args": ["--no-sandbox", "--disable-dev-shm-usage"],
    }
    if CHROMIUM_EXECUTABLE:
        kwargs["executable_path"] = CHROMIUM_EXECUTABLE
    return kwargs


def parse_block_row(block: dict[str, Any]) -> dict[str, Any] | None:
    cells = block.get("cells") or []
    title = ""
    page_no = ""
    if block.get("tag") == "TR" and cells:
        page_index: int | None = None
        for index in range(len(cells) - 1, -1, -1):
            cell_text = one_line(cells[index].get("text"))
            if is_valid_page_token(cell_text):
                page_index = index
                page_no = normalize_page_token(cell_text)
                break
        title_cells = cells[:page_index] if page_index is not None else cells
        title_cells = [cell for cell in title_cells if one_line(cell.get("text"))]
        if title_cells:
            title = one_line(" ".join(one_line(cell.get("text")) for cell in title_cells))
        if not page_no:
            split = split_title_page(title or block.get("text"))
            if split:
                title, page_no = split
    else:
        text = one_line(block.get("text"))
        split = split_title_page(text)
        if split:
            title, page_no = split
        else:
            title = text
    title = one_line(title)
    if not title or TOC_HEADER_RE.fullmatch(title) or PAGE_LABEL_RE.fullmatch(title):
        return None
    if title == page_no or is_valid_page_token(title) or ROMAN_RE.fullmatch(title):
        return None
    return {
        "title": title,
        "page-no": page_no,
        "block_index": int(block.get("index", -1)),
        "tag": block.get("tag", ""),
    }


def is_noise_line(line: str) -> bool:
    text = one_line(line)
    if not text:
        return True
    if TOC_HEADER_RE.fullmatch(text) or PAGE_LABEL_RE.fullmatch(text):
        return True
    if is_valid_page_token(text) or ROMAN_RE.fullmatch(text):
        return True
    return False


def block_lines(block: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if block.get("tag") == "TR" and block.get("cells"):
        for cell in block["cells"]:
            for line in normalize_text(cell.get("text")).splitlines():
                line = one_line(line)
                if line:
                    lines.append(line)
    else:
        for line in normalize_text(block.get("text")).splitlines():
            line = one_line(line)
            if line:
                lines.append(line)
    return lines


def is_signature_block(block: dict[str, Any]) -> bool:
    for line in block_lines(block)[:8]:
        text = one_line(line)
        if SIGNATURE_RE.fullmatch(text) or AFTER_INQUIRY_RE.match(text):
            return True
    return False


def find_signature_index(blocks: list[dict[str, Any]], after_index: int) -> int | None:
    for block in blocks:
        index = int(block.get("index", -1))
        if index <= after_index:
            continue
        if is_signature_block(block):
            return index
    return None


def block_content(block: dict[str, Any]) -> str:
    lines: list[str] = []
    if block.get("tag") == "TR" and block.get("cells"):
        for cell in block["cells"]:
            for line in normalize_text(cell.get("text")).splitlines():
                line = one_line(line)
                if line and not is_noise_line(line):
                    lines.append(line)
        return " ".join(lines).strip()
    parts: list[str] = []
    for line in normalize_text(block.get("text")).splitlines():
        line = one_line(line)
        if line and not is_noise_line(line):
            parts.append(line)
    return "\n".join(parts).strip()


def format_item_title(item_no: str, remainder: str) -> str:
    remainder = one_line(remainder).strip(" .-—:")
    canonical = CANONICAL_TITLES.get(item_no, "")
    if not remainder:
        remainder = canonical
    elif canonical and remainder.upper() == canonical:
        remainder = canonical
    return f"ITEM {item_no}. {remainder}".strip()


def parse_item_heading(block: dict[str, Any]) -> dict[str, Any] | None:
    row = parse_block_row(block)
    if not row:
        return None
    match = ITEM_HEADING_RE.match(row["title"])
    if not match:
        return None
    item_no = match.group(1)
    remainder = one_line(match.group(2) or "")
    return {
        "kind": "item",
        "item": item_no,
        "title": format_item_title(item_no, remainder),
        "page-no": row.get("page-no") or "",
        "block_index": int(block["index"]),
    }


def parse_annex_heading(
    block: dict[str, Any],
    *,
    after_signature: bool = False,
) -> dict[str, Any] | None:
    """Match a standalone annex heading on the first significant line of a block.

    Exhibit-table rows such as "Annex A – Opinion of …" are skipped unless we
    are already past the signature page. Running headers reuse the same label
    and are de-duplicated by find_annexes.
    """
    row = parse_block_row(block)
    page_no = (row or {}).get("page-no") or ""
    # "ANNEX I" is a label, not a roman page token. Only skip numeric TOC pages.
    if (
        not after_signature
        and page_no.isdigit()
    ):
        return None

    lines = block_lines(block)
    if not lines:
        return None
    if any(EXHIBIT_NO_RE.match(line) for line in lines[:6]):
        return None

    start = 0
    while start < len(lines) and is_noise_line(lines[start]):
        start += 1
    if start >= len(lines):
        return None

    first = one_line(lines[start])
    match = ANNEX_HEADING_RE.match(first)
    if (
        not match
        and start + 1 < len(lines)
        and re.fullmatch(r"(?:ANNEX|APPENDIX)", first, re.I)
    ):
        match = ANNEX_HEADING_RE.match(f"{first} {one_line(lines[start + 1])}")
    if not match:
        return None

    remainder = one_line(match.group("rest") or "").strip(" .:-—–")
    if len(remainder) > 160:
        remainder = ""
    if EXHIBIT_PHRASE_RE.search(first) or EXHIBIT_PHRASE_RE.search(remainder):
        return None

    if not after_signature:
        # Exhibit index rows are almost always table rows with a description cell.
        cells = [
            one_line(cell.get("text"))
            for cell in (block.get("cells") or [])
            if one_line(cell.get("text"))
        ]
        if block.get("tag") == "TR" and len(cells) >= 2:
            return None
        if remainder and re.search(r"\b(?:opinion of|dated)\b", remainder, re.I):
            return None

    label = match.group("label").upper()
    display = f"ANNEX {label}"
    if remainder:
        display = f"{display}: {remainder}"
    return {
        "kind": "annex",
        "label": label,
        "title": display,
        "block_index": int(block["index"]),
    }


def find_body_items(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block in blocks:
        heading = parse_item_heading(block)
        if heading and not heading["page-no"]:
            candidates.append(heading)

    starts = [index for index, heading in enumerate(candidates) if heading["item"] == "1"]
    if not starts:
        return []

    # If ITEM 1 appears in a TOC-like cluster and again in the body, keep the last.
    found: dict[str, dict[str, Any]] = {}
    expected = 1
    for heading in candidates[starts[-1]:]:
        item_no = int(heading["item"])
        if str(item_no) in found:
            continue
        if item_no == expected or (item_no > expected and expected > 1):
            found[str(item_no)] = heading
            expected = item_no + 1
        if expected > 9:
            break
    return [found[str(n)] for n in range(1, 10) if str(n) in found]


def annex_base_label(label: str) -> str:
    return one_line(label).split("-", 1)[0]


def is_annex_page_header(label: str, seen: set[str]) -> bool:
    """Annex I-1 / B-2 after Annex I / B are running page headers, not new annexes.

    Annex A-1 then A-2 (no plain A) are real sibling annexes.
    """
    label = one_line(label).upper()
    if "-" not in label:
        return False
    base = annex_base_label(label)
    return base in seen


def find_annexes(
    blocks: list[dict[str, Any]],
    after_index: int,
) -> list[dict[str, Any]]:
    signature_index = find_signature_index(blocks, after_index)
    search_after = signature_index if signature_index is not None else after_index
    after_signature = signature_index is not None

    annexes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        if int(block["index"]) <= search_after:
            continue
        heading = parse_annex_heading(block, after_signature=after_signature)
        if not heading:
            continue
        if heading["label"] in seen:
            continue
        if is_annex_page_header(heading["label"], seen):
            continue
        seen.add(heading["label"])
        annexes.append(heading)
    return annexes


def slice_content(
    blocks: list[dict[str, Any]],
    start: int,
    end: int,
    include_heading: bool = False,
) -> str:
    parts: list[str] = []
    begin = start if include_heading else start + 1
    for block in blocks:
        index = int(block["index"])
        if index < begin or index >= end:
            continue
        text = block_content(block)
        if text:
            parts.append(text)
    return normalize_text("\n\n".join(parts))


def extract_sections(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = find_body_items(blocks)
    if not items:
        raise ValueError("Could not find body ITEM 1 heading")

    first_item_index = items[0]["block_index"]
    after_items = items[-1]["block_index"]
    annexes = find_annexes(blocks, after_items)

    stops = [heading["block_index"] for heading in items]
    stops.extend(heading["block_index"] for heading in annexes)
    stops.append(len(blocks))

    sections: list[dict[str, Any]] = []
    preamble = slice_content(blocks, 0, first_item_index, include_heading=True)
    sections.append({
        "title": "Preamble",
        "content": preamble,
    })

    for offset, heading in enumerate(items):
        end = stops[offset + 1]
        sections.append({
            "title": heading["title"],
            "content": slice_content(blocks, heading["block_index"], end),
        })

    annex_offset = len(items)
    for local_index, heading in enumerate(annexes):
        end = stops[annex_offset + local_index + 1]
        sections.append({
            "title": heading["title"],
            "content": slice_content(
                blocks,
                heading["block_index"],
                end,
                include_heading=True,
            ),
        })

    return sections


def render_html(page: Any, html: str) -> None:
    page.set_content(html, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(800)
    try:
        page.wait_for_load_state("networkidle", timeout=5_000)
    except Exception:
        pass


def section_counts(sections: list[dict[str, Any]]) -> tuple[int, int]:
    item_count = sum(
        1 for section in sections
        if one_line(section.get("title")).upper().startswith("ITEM ")
    )
    annex_count = sum(
        1 for section in sections
        if one_line(section.get("title")).upper().startswith("ANNEX ")
    )
    return item_count, annex_count


def extract_sections_from_page(url: str, page: Any) -> list[dict[str, Any]]:
    html = fetch_html(url)
    render_html(page, html)
    blocks = page.evaluate(DOM_EXTRACTOR_JS)
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("No visible DOM blocks were extracted")
    return extract_sections(blocks)


def write_sections_json(
    sections: list[dict[str, Any]],
    output_dir: Path,
    accession: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{accession}_text.json"
    output_path.write_text(
        json.dumps(sections, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_sc14d9(
    url: str,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    page: Any | None = None,
) -> list[dict[str, Any]]:
    """Parse one SC 14D-9 URL and return [{title, content}, ...].

    Same calling style as DocumentExtractionPipeline.process(url=...):
    pass the EDGAR HTML URL. When output_dir is set, also writes
    `{accession}_text.json`. Pass an existing Playwright `page` to reuse a
    browser; otherwise one is launched for this call.
    """
    result = SC14D9ExtractionPipeline().process(
        url=url,
        output_dir=output_dir,
        page=page,
    )
    if result.get("status") != "success":
        raise ValueError(result.get("reason") or "SC 14D-9 parse failed")
    return result["sections"]


class SC14D9ExtractionPipeline:
    """URL-in, sections-JSON-out parser, matching DocumentExtractionPipeline."""

    def process(
        self,
        url: str,
        output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
        page: Any | None = None,
    ) -> dict[str, Any]:
        accession = accession_from_url(url)
        logger.info("parser=sc14d9 start url=%s accession=%s", url, accession)
        try:
            if page is not None:
                sections = extract_sections_from_page(url, page)
            else:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(**browser_launch_kwargs())
                    owned_page = browser.new_page()
                    try:
                        sections = extract_sections_from_page(url, owned_page)
                    finally:
                        browser.close()

            item_count, annex_count = section_counts(sections)
            text_output = None
            if output_dir is not None:
                text_output = str(
                    write_sections_json(sections, Path(output_dir), accession)
                )

            logger.info(
                "parser=sc14d9 done accession=%s items=%s annexes=%s output=%s",
                accession,
                item_count,
                annex_count,
                text_output,
            )
            return {
                "status": "success",
                "url": url,
                "accession": accession,
                "item_count": item_count,
                "annex_count": annex_count,
                "text_output": text_output,
                "sections": sections,
            }
        except Exception as exc:
            logger.warning(
                "parser=sc14d9 failed accession=%s reason=%s",
                accession,
                exc,
            )
            return {
                "status": "error",
                "url": url,
                "accession": accession,
                "reason": str(exc),
                "sections": [],
            }


def process_url(url: str, filing_id: str, page: Any) -> dict[str, Any]:
    result = SC14D9ExtractionPipeline().process(url=url, output_dir=None, page=page)
    if result.get("status") != "success":
        raise ValueError(result.get("reason") or "SC 14D-9 parse failed")
    result["id"] = filing_id
    return result


def load_filings(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    filings: list[dict[str, str]] = []
    for row in payload:
        url = one_line(row.get("url"))
        if not url:
            continue
        filings.append({
            "id": one_line(row.get("id")) or accession_from_url(url),
            "url": url,
        })
    return filings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SC 14D-9 documents by Item 1-9 headings.",
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="SEC document URL(s)",
    )
    parser.add_argument(
        "--input-file",
        help="JSON file with [{id, url}, ...]",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-filing JSON output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filings: list[dict[str, str]] = []
    for url in args.urls or []:
        url = one_line(url)
        if url:
            filings.append({"id": accession_from_url(url), "url": url})
    if args.input_file:
        filings.extend(load_filings(Path(args.input_file)))
    elif not filings:
        filings = load_filings(DEFAULT_INPUT)

    if not filings:
        parser.error("Provide at least one URL or --input-file")

    pipeline = SC14D9ExtractionPipeline()
    successes = 0
    failures = 0
    summary: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**browser_launch_kwargs())
        page = browser.new_page()
        try:
            for index, filing in enumerate(filings, start=1):
                filing_id = filing["id"]
                url = filing["url"]
                logger.info("[%s/%s] start id=%s url=%s", index, len(filings), filing_id, url)
                result = pipeline.process(url=url, output_dir=output_dir, page=page)
                if result.get("status") == "success":
                    logger.info(
                        "[%s/%s] SUCCESS items=%s annexes=%s output=%s",
                        index,
                        len(filings),
                        result["item_count"],
                        result["annex_count"],
                        result["text_output"],
                    )
                    successes += 1
                    annex_titles = [
                        section.get("title")
                        for section in result["sections"]
                        if one_line(section.get("title")).upper().startswith("ANNEX ")
                    ]
                    summary.append({
                        "id": filing_id,
                        "url": url,
                        "status": "success",
                        "item_count": result["item_count"],
                        "annex_count": result["annex_count"],
                        "annex_titles": annex_titles,
                        "output": result["text_output"],
                    })
                else:
                    logger.error(
                        "[%s/%s] FAILED id=%s reason=%s",
                        index,
                        len(filings),
                        filing_id,
                        result.get("reason"),
                    )
                    failures += 1
                    summary.append({
                        "id": filing_id,
                        "url": url,
                        "status": "failed",
                        "reason": result.get("reason"),
                    })
        finally:
            browser.close()

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps({"success": successes, "failed": failures, "results": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Done. Success=%s Failed=%s summary=%s", successes, failures, summary_path)


if __name__ == "__main__":
    main()
