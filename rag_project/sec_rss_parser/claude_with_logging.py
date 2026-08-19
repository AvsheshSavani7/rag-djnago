from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

import requests
from playwright.sync_api import Page, sync_playwright

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRY = 3
REQUEST_TIMEOUT = 60
MAX_HTML_CHARS = 30_000_000

OUTPUT_DIR = Path(
    os.environ.get(
        "MNA_DOCUMENT_OUTPUT_DIR",
        "output",
    )
)

CHROMIUM_EXECUTABLE = os.environ.get(
    "MNA_CHROMIUM_EXECUTABLE"
)

SEC_USER_AGENT = os.environ.get(
    "MNA_SEC_USER_AGENT",
    (
        "MNA-Finder/1.0 "
        "(https://teqnodux.com; "
        "contact: ashish.kachadiya@teqnodux.com)"
    ),
)


# Add SEC document URLs here, pass URLs on the command line,
# or provide a URL text file through --input-file.
URLS: list[str] = [
    "https://www.sec.gov/Archives/edgar/data/1819994/000175392626001452/g085840_s4.htm",
    # "https://www.sec.gov/Archives/edgar/data/1699838/000110465926002481/tm2532777-20_defm14a.htm",
    # "https://www.sec.gov/Archives/edgar/data/1567925/000114036126019138/ny20070999x1_prem14a.htm",
    # "https://www.sec.gov/Archives/edgar/data/2016561/000149315226020967/forms-4a.htm",
    # "https://www.sec.gov/Archives/edgar/data/1206264/000120626426000085/sgis-4.htm"

]


# ============================================================
# SHARED DATA CLASSES
# ============================================================

@dataclass
class Boundary:
    start_block: int
    end_block: int
    start_reason: str
    end_reason: str


@dataclass
class TocEntry:
    title: str
    page_no: str
    href: str
    fragment: str
    path: tuple[int, ...]
    level: int
    entry_type: str = "section"
    annex_label: str = ""


# ============================================================
# SHARED REGULAR EXPRESSIONS
# ============================================================

ANNEX_TITLE_RE = re.compile(
    r"^\s*(?:annex|appendix)\s+"
    r"([A-Z]{1,3})(?:\b|[:.\-])",
    re.IGNORECASE,
)

ANNEX_PAGE_RE = re.compile(
    r"^\s*([A-Z]{1,3})-\d+\s*$",
    re.IGNORECASE,
)


# ============================================================
# SHARED TEXT HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    text = str(value or "")

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")

    text = re.sub(
        r"[\u2000-\u200a\u202f\u205f]",
        " ",
        text,
    )

    text = re.sub(
        r"[\u200c\u200d\ufeff]",
        "",
        text,
    )

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def one_line(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        normalize_text(value),
    ).strip()


def normalize_page_token(value: Any) -> str:
    return re.sub(
        r"\s*[-–—]\s*",
        "-",
        one_line(value),
    )


def accession_from_url(url: str) -> str:
    raw = url.rstrip("/").split("/")[-1].split(".")[0]

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        raw,
    ).strip("_")


def fragment_from_href(href: Any) -> str:
    value = one_line(href)

    if not value:
        return ""

    fragment = urlsplit(value).fragment

    if not fragment and value.startswith("#"):
        fragment = value[1:]

    try:
        fragment = unquote(fragment)
    except Exception:
        pass

    return fragment.strip()


# ============================================================
# TOC ENTRY METADATA
# ============================================================

def detect_entry_metadata(
    title: str,
    page_no: str,
    level: int,
) -> tuple[str, str]:
    if level != 1:
        return "subsection", ""

    title_match = ANNEX_TITLE_RE.search(
        one_line(title)
    )

    if title_match:
        return (
            "annex",
            title_match.group(1).upper(),
        )

    return "section", ""


# ============================================================
# TOC FLATTENING
# ============================================================

def flatten_toc(
    toc: list[dict[str, Any]],
) -> list[TocEntry]:
    entries: list[TocEntry] = []

    for section_index, section in enumerate(toc):
        if not isinstance(section, dict):
            continue

        title = one_line(
            section.get("title")
        )

        if title:
            page_no = one_line(
                section.get("page-no")
            )

            href = one_line(
                section.get("href")
            )

            entry_type, annex_label = (
                detect_entry_metadata(
                    title=title,
                    page_no=page_no,
                    level=1,
                )
            )

            entries.append(
                TocEntry(
                    title=title,
                    page_no=page_no,
                    href=href,
                    fragment=fragment_from_href(href),
                    path=(section_index,),
                    level=1,
                    entry_type=entry_type,
                    annex_label=annex_label,
                )
            )

        subsections = (
            section.get("subsection")
            or []
        )

        if not isinstance(subsections, list):
            continue

        for subsection_index, subsection in enumerate(
            subsections
        ):
            if not isinstance(subsection, dict):
                continue

            subtitle = one_line(
                subsection.get("title")
            )

            if not subtitle:
                continue

            page_no = one_line(
                subsection.get("page-no")
            )

            href = one_line(
                subsection.get("href")
            )

            entries.append(
                TocEntry(
                    title=subtitle,
                    page_no=page_no,
                    href=href,
                    fragment=fragment_from_href(href),
                    path=(
                        section_index,
                        subsection_index,
                    ),
                    level=2,
                    entry_type="subsection",
                    annex_label="",
                )
            )

    if not entries:
        raise ValueError(
            "No titles found in extracted TOC"
        )

    return entries


def get_output_node(
    output_toc: list[dict[str, Any]],
    path: tuple[int, ...],
) -> dict[str, Any]:
    if len(path) == 1:
        return output_toc[path[0]]

    if len(path) == 2:
        return output_toc[path[0]][
            "subsection"
        ][path[1]]

    raise ValueError(
        f"Unsupported TOC path: {path}"
    )


# ============================================================
# SEC HTML FETCHING
# ============================================================

def fetch_html(url: str) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),
        },
        timeout=REQUEST_TIMEOUT,
    )

    html = response.text or ""

    if (
        response.status_code == 403
        or "Undeclared Automated Tool" in html
        or "Request Originates" in html
    ):
        raise ValueError(
            "SEC blocked the request: "
            f"HTTP {response.status_code}"
        )

    if response.status_code >= 400:
        raise ValueError(
            f"HTTP {response.status_code}"
        )

    if len(html) < 1000:
        raise ValueError(
            "HTML is too short: "
            f"{len(html)} characters"
        )

    if len(html) > MAX_HTML_CHARS:
        raise ValueError(
            "HTML exceeds safety limit: "
            f"{len(html)} characters"
        )

    return html


# ============================================================
# PLAYWRIGHT HELPERS
# ============================================================

def browser_launch_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    }

    if CHROMIUM_EXECUTABLE:
        kwargs["executable_path"] = (
            CHROMIUM_EXECUTABLE
        )

    return kwargs


def render_html(
    page: Page,
    html: str,
) -> None:
    page.set_content(
        html,
        wait_until="domcontentloaded",
        timeout=90_000,
    )

    page.wait_for_timeout(800)

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=5_000,
        )
    except Exception:
        pass


class TocExtractor:
    TOC_HEADER_RE = re.compile(r"^TABLE\s+OF\s+CONTENTS$", re.I)

    PAGE_LABEL_RE = re.compile(r"^PAGE$", re.I)

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

    ANNEX_RE = re.compile(
        r"^(?:ANNEX|APPENDIX|EXHIBIT|SCHEDULE)\s+[A-Z0-9]+\b", re.I)

    STRUCTURAL_LABEL_RE = re.compile(
        r"^(?:ANNEX(?:ES)?|APPENDICES|APPENDIX|EXHIBITS?|SCHEDULES?)(?:\s+INDEX)?\s*[:\-–—]?$", re.I)

    BODY_START_RE = re.compile(
        r"^(?:"
        r"SUMMARY(?:\s+TERM\s+SHEET)?"
        r"|TRANSACTION\s+SUMMARY"
        r"|PROXY\s+SUMMARY"
        r"|(?:JOINT\s+)?PROXY\s+STATEMENT"
        r"|YOUR\s+VOTE\s+IS\s+IMPORTANT"
        r"|(?:CERTAIN\s+)?DEFINED\s+TERMS"
        r"|(?:CERTAIN\s+)?DEFINITIONS?"
        r"|(?:COMMONLY|FREQUENTLY)\s+USED\s+TERMS"
        r"|GLOSSARY"
        r"|MARKET\s+AND\s+INDUSTRY\s+DATA"
        r"|ABOUT\s+THIS\s+(?:PROXY\s+STATEMENT/PROSPECTUS|PROSPECTUS/OFFERS\s+TO\s+EXCHANGE)"
        r"|QUESTIONS?\s+AND\s+ANSWERS?.*"
        r"|REFERENCES\s+TO\s+ADDITIONAL\s+INFORMATION"
        r")$",
        re.I,
    )

    BODY_INTRO_RE = re.compile(
        r"^(?:"
        r"THIS\s+SUMMARY\s+(?:TERM\s+SHEET\s+)?HIGHLIGHTS"
        r"|THE\s+FOLLOWING\s+SUMMARY\s+HIGHLIGHTS"
        r"|THIS\s+PROXY\s+STATEMENT"
        r"|UNLESS\s+OTHERWISE\s+INDICATED"
        r"|YOU\s+ARE\s+CORDIALLY\s+INVITED"
        r")\b",
        re.I,
    )

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
          marginLeft: parseFloat(s.marginLeft) || 0,
          textIndent: parseFloat(s.textIndent) || 0,
          fontWeight: parseInt(s.fontWeight, 10) || (s.fontWeight === "bold" ? 700 : 400),
          fontSize: parseFloat(s.fontSize) || 0,
          textAlign: s.textAlign || "",
        };
      };

      // Return the x-position of the first visible character, not merely the
      // containing TD/TR. SEC filings frequently indent text inside a cell using
      // nested FONT, DIV, A, SPAN, padding, or text-indent styles.
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

      // Table rows preserve the title/page columns and their indentation.
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

      // Non-table headings/paragraphs. Leaf-like DIV filtering avoids duplicates.
      document.querySelectorAll("h1,h2,h3,h4,h5,h6,p,li,div").forEach((el) => {
        if (!visible(el) || el.closest("tr")) return;
        if (el.tagName === "DIV") {
          const nested = el.querySelector(":scope > div, :scope > p, :scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6, :scope > table, :scope > ul, :scope > ol");
          if (nested) return;
        }
        const text = clean(el.innerText);
        if (!text || text.length > 1800) return;
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

      // Remove exact visual duplicates, preferring TR records.
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

    def extract_dom_blocks(self, page: Page) -> list[dict[str, Any]]:
        blocks = page.evaluate(self.DOM_EXTRACTOR_JS)
        if not isinstance(blocks, list) or not blocks:
            raise ValueError('No visible DOM blocks were extracted')
        return blocks

    def block_lines(self, block: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if block.get('tag') == 'TR' and block.get('cells'):
            for cell in block['cells']:
                for line in normalize_text(cell.get('text')).splitlines():
                    line = one_line(line)
                    if line:
                        lines.append(line)
        else:
            for line in normalize_text(block.get('text')).splitlines():
                line = one_line(line)
                if line:
                    lines.append(line)
        return lines

    def is_toc_header(self, value: Any) -> bool:
        return bool(self.TOC_HEADER_RE.fullmatch(one_line(value)))

    def is_body_start(self, value: Any) -> bool:
        return bool(self.BODY_START_RE.fullmatch(one_line(value)))

    def is_body_intro(self, value: Any) -> bool:
        return bool(self.BODY_INTRO_RE.match(one_line(value)))

    def is_valid_page_token(self, value: Any) -> bool:
        token = normalize_page_token(value)
        if not token or self.PHONE_RE.fullmatch(token):
            return False
        if not self.PAGE_TOKEN_RE.fullmatch(token):
            return False
        if token.isdigit():
            return 1 <= int(token) <= 1500
        if '-' in token:
            tail = token.rsplit('-', 1)[-1]
            return tail.isdigit() and 1 <= int(tail) <= 1500
        return bool(re.fullmatch('\\(?M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\\)?', token, re.I))

    def split_title_page(self, value: Any) -> tuple[str, str] | None:
        text = one_line(value)
        match = self.TRAILING_PAGE_RE.match(text)
        if not match:
            return None
        title = one_line(match.group('title'))
        page_no = normalize_page_token(match.group('page'))
        if not title or not self.is_valid_page_token(page_no):
            return None
        return (title, page_no)

    def is_body_sentence(self, value: Any) -> bool:
        text = one_line(value)
        words = re.findall('[A-Za-z]+', text)
        if len(text) >= 180 or len(words) >= 28:
            return True
        lower = f' {text.lower()} '
        return len(words) >= 18 and any((marker in lower for marker in (', which ', ' pursuant to ', ' because ', ' although ', ' together with ')))

    def parse_block_row(self, block: dict[str, Any]) -> dict[str, Any] | None:
        cells = block.get('cells') or []
        title = ''
        page_no = ''
        href = ''
        title_left = float(block.get('textLeft', block.get('left', 0.0)))
        container_left = float(block.get('containerLeft', 0.0))
        font_weight = int(block.get('fontWeight', 400) or 400)
        if block.get('tag') == 'TR' and cells:
            page_index: int | None = None
            for index in range(len(cells) - 1, -1, -1):
                cell_text = one_line(cells[index].get('text'))
                if self.is_valid_page_token(cell_text):
                    page_index = index
                    page_no = normalize_page_token(cell_text)
                    break
            title_cells = cells[:page_index] if page_index is not None else cells
            title_cells = [
                cell for cell in title_cells if one_line(cell.get('text'))]
            if title_cells:
                title = one_line(' '.join((one_line(cell.get('text'))
                                 for cell in title_cells)))
                for cell in title_cells:
                    if cell.get('href'):
                        href = cell['href']
                        break
                first = title_cells[0]
                title_left = float(first.get('textLeft', float(first.get('left', title_left)) + float(
                    first.get('paddingLeft', 0.0)) + float(first.get('textIndent', 0.0))))
                font_weight = max(
                    (int(cell.get('fontWeight', 400) or 400) for cell in title_cells))
            if not page_no:
                split = self.split_title_page(title or block.get('text'))
                if split:
                    title, page_no = split
        else:
            text = one_line(block.get('text'))
            split = self.split_title_page(text)
            if split:
                title, page_no = split
            else:
                title = text
            href = block.get('href', '')
        title = one_line(title)
        if not title or self.is_toc_header(title) or self.PAGE_LABEL_RE.fullmatch(title):
            return None
        if title == page_no or self.is_valid_page_token(title):
            return None
        if bool(re.fullmatch(r"-?[ivxlcdm]+-?", title, re.I)) or title.lower() == "(continued)":
            return None
        if bool(re.match('^(?:[\\u2022\\u00B7\\-\\*]\\s*)?[“\\"”\\\'].+?[“\\"”\\\']\\s+(?:means|has the meaning)\\b', title, re.I)):
            return None
        if '☐' in title or '☒' in title or '☑' in title:
            return None
        relative_left = max(0.0, title_left - container_left)
        return {'title': title, 'page-no': page_no, 'left': round(title_left, 2), 'relative_left': round(relative_left, 2), 'top': round(float(block.get('top', 0.0)), 2), 'bottom': round(float(block.get('bottom', 0.0)), 2), 'font_weight': font_weight, 'block_index': int(block['index']), 'tag': block.get('tag', ''), 'href': href}

    def toc_like_score(self, block: dict[str, Any]) -> int:
        row = self.parse_block_row(block)
        if not row:
            return 0
        if row['page-no']:
            return 3
        if self.ANNEX_RE.match(row['title']):
            return 2
        return 0

    def next_significant_line(self, blocks: list[dict[str, Any]], start: int, max_blocks: int = 12) -> tuple[int, str] | None:
        stop = min(len(blocks), start + max_blocks)
        for block_index in range(start, stop):
            for line in self.block_lines(blocks[block_index]):
                line = one_line(line)
                if not line or line in {'* * *', '***'}:
                    continue
                return (block_index, line)
        return None

    def previous_significant_line(self, blocks: list[dict[str, Any]], start: int, max_blocks: int = 40) -> tuple[int, str] | None:
        stop = max(-1, start - max_blocks)
        for block_index in range(start, stop, -1):
            lines = self.block_lines(blocks[block_index])
            for line in reversed(lines):
                line = one_line(line)
                if line and line not in {'* * *', '***'}:
                    return (block_index, line)
        return None

    def iter_significant_lines(self, blocks: list[dict[str, Any]], start: int, max_blocks: int = 20, max_lines: int = 20) -> list[tuple[int, str]]:
        """Return nearby non-empty rendered lines with their block indexes."""
        refs: list[tuple[int, str]] = []
        stop = min(len(blocks), start + max_blocks)
        for block_index in range(start, stop):
            for line in self.block_lines(blocks[block_index]):
                line = one_line(line)
                if not line or line in {'* * *', '***'}:
                    continue
                refs.append((block_index, line))
                if len(refs) >= max_lines:
                    return refs
        return refs

    def is_allowed_start_bridge(self, lines: list[str]) -> bool:
        """Validate the rare text allowed between two real TOC headings."""
        if not lines:
            return True
        combined = one_line(' '.join(lines))
        if not combined or len(combined) > 650:
            return False
        return bool(re.search('(?:THIS\\s+PROXY\\s+STATEMENT\\s+IS\\s+DATED\\b|SPECIAL\\s+MEETING\\s+OF\\s+STOCKHOLDERS\\b.*\\bPROXY\\s+STATEMENT\\b|PRELIMINARY\\s+PROXY\\s+(?:MATERIALS|STATEMENT)\\b.*\\bSUBJECT\\s+TO\\s+COMPLETION\\b)', combined, re.I))

    def get_local_start_evidence(self, blocks: list[dict[str, Any]], header_index: int, max_blocks: int = 80) -> dict[str, int]:
        """
        Inspect only the content immediately following a candidate header.

        The scan stops at body text or another TOC header, preventing an earlier
        cover-page header from borrowing TOC rows found much later in the filing.
        """
        paged_rows = 0
        annex_rows = 0
        body_rows = 0
        scanned_rows = 0
        stop = min(len(blocks), header_index + 1 + max_blocks)
        for block_index in range(header_index + 1, stop):
            block = blocks[block_index]
            lines = self.block_lines(block)
            if not lines:
                continue
            if any((self.is_toc_header(line) for line in lines)):
                break
            if all((self.PAGE_LABEL_RE.fullmatch(line) or self.is_valid_page_token(line) or line in {'* * *', '***'} for line in lines)):
                continue
            row = self.parse_block_row(block)
            if row and row.get('page-no'):
                paged_rows += 1
                scanned_rows += 1
                continue
            if row and self.ANNEX_RE.match(row.get('title', '')):
                annex_rows += 1
                scanned_rows += 1
                continue
            meaningful = [line for line in lines if not self.PAGE_LABEL_RE.fullmatch(
                line) and (not self.is_valid_page_token(line)) and (line not in {'* * *', '***'})]
            if not meaningful:
                continue
            if any((self.is_body_start(line) for line in meaningful)):
                if paged_rows > 0 or annex_rows > 0:
                    break
            if any((self.is_body_sentence(line) for line in meaningful)):
                body_rows += 1
                if paged_rows + annex_rows < 2 or body_rows >= 2:
                    break
            scanned_rows += 1
        return {'paged_rows': paged_rows, 'annex_rows': annex_rows, 'body_rows': body_rows, 'scanned_rows': scanned_rows}

    def has_enough_start_evidence(self, evidence: dict[str, int], minimum_rows: int) -> bool:
        if evidence['body_rows'] > 0:
            return False
        if minimum_rows >= 3 and evidence['paged_rows'] < 2:
            return False
        return int(evidence.get('paged_rows', 0)) + int(evidence.get('annex_rows', 0)) >= minimum_rows and int(evidence.get('body_rows', 0)) <= 1

    def find_start_block(self, blocks: list[dict[str, Any]]) -> tuple[int, str]:
        """
        Detect the TOC start using the manually analysed anchor families.

        Priority:
          1. TABLE OF CONTENTS -> TABLE OF CONTENTS
          2. TABLE OF CONTENTS -> Page
          3. PROXY STATEMENT -> TABLE OF CONTENTS
          4. A single TABLE OF CONTENTS with immediate paged rows
          5. SUMMARY -> 1 fallback
        """
        header_indexes = [int(block['index']) for block in blocks if any(
            (self.is_toc_header(line) for line in self.block_lines(block)))]
        candidates: list[tuple[int, str]] = []
        for first_header in header_indexes:
            refs = self.iter_significant_lines(
                blocks, first_header + 1, max_blocks=18, max_lines=12)
            next_header_position = next((position for position, (_idx, line) in enumerate(
                refs) if self.is_toc_header(line)), None)
            if next_header_position is None:
                continue
            second_header, _line = refs[next_header_position]
            bridge_lines = [line for _idx, line in refs[:next_header_position]]
            if not self.is_allowed_start_bridge(bridge_lines):
                continue
            evidence = self.get_local_start_evidence(blocks, second_header)
            if self.has_enough_start_evidence(evidence, minimum_rows=2):
                candidates.append(
                    (second_header, 'strict_toc_header_then_toc_header'))
        for header_index in header_indexes:
            refs = self.iter_significant_lines(
                blocks, header_index + 1, max_blocks=8, max_lines=4)
            if not refs or not self.PAGE_LABEL_RE.fullmatch(refs[0][1]):
                continue
            evidence = self.get_local_start_evidence(blocks, header_index)
            if self.has_enough_start_evidence(evidence, minimum_rows=2):
                candidates.append(
                    (header_index, 'strict_toc_header_then_page'))
        for header_index in header_indexes:
            previous = self.previous_significant_line(
                blocks, header_index - 1, max_blocks=8)
            if not previous or not re.fullmatch('PROXY\\s+STATEMENT', previous[1], re.I):
                continue
            evidence = self.get_local_start_evidence(blocks, header_index)
            if self.has_enough_start_evidence(evidence, minimum_rows=2):
                candidates.append(
                    (header_index, 'strict_proxy_statement_then_toc_header'))
        for header_index in header_indexes:
            evidence = self.get_local_start_evidence(blocks, header_index)
            if self.has_enough_start_evidence(evidence, minimum_rows=3):
                candidates.append(
                    (header_index, 'single_toc_header_with_local_rows'))
        for block in blocks:
            index = int(block['index'])
            lines = self.block_lines(block)
            for position, line in enumerate(lines):
                if not re.fullmatch('SUMMARY', line, re.I):
                    continue
                page_one_found = any((normalize_page_token(
                    item) == '1' for item in lines[position + 1:]))
                if not page_one_found:
                    next_line = self.next_significant_line(
                        blocks, index + 1, max_blocks=3)
                    page_one_found = bool(
                        next_line and normalize_page_token(next_line[1]) == '1')
                if not page_one_found:
                    continue
                future_rows = sum((self.toc_like_score(
                    item) >= 3 for item in blocks[index:index + 50]))
                if future_rows >= 3:
                    candidates.append((index, 'summary_then_page_1'))
        if candidates:
            return min(candidates, key=lambda c: c[0])
        raise ValueError('Could not detect TOC start')

    def is_toc_continuation_after_header(self, block: dict[str, Any]) -> bool:
        """Return True when a repeated TOC header is followed by more TOC rows.

        This specifically protects continuation pages such as:

            TABLE OF CONTENTS
            ANNEXES:
            ANNEX A ... A-1

        It also covers ordinary continued TOC rows. The check is deliberately local
        and is used only while validating a possible end-header pair.
        """
        lines = self.block_lines(block)
        if any((self.STRUCTURAL_LABEL_RE.fullmatch(line) for line in lines)):
            return True
        row = self.parse_block_row(block)
        if not row:
            return False
        title = one_line(row.get('title', ''))
        return bool(row.get('page-no') or self.ANNEX_RE.match(title))

    def header_end_pair(self, blocks: list[dict[str, Any]], header_index: int) -> tuple[bool, str]:
        for lookahead in range(header_index + 1, min(len(blocks), header_index + 10)):
            block = blocks[lookahead]
            lines = self.block_lines(block)
            row = self.parse_block_row(block)
            has_page = bool(row and row.get('page-no'))
            for line in lines:
                if self.PAGE_LABEL_RE.fullmatch(line) or self.is_toc_header(line):
                    continue
                if self.is_body_start(line) and (not has_page):
                    return (True, f'toc_header_then_{one_line(line)[:60]}')
            if self.is_toc_continuation_after_header(block):
                return (False, '')
            if any((self.is_body_sentence(line) for line in lines)):
                continue
        return (False, '')

    def direct_end_pair(self, blocks: list[dict[str, Any]], index: int) -> tuple[bool, str]:
        lines = self.block_lines(blocks[index])
        heading = next(
            (line for line in lines if self.is_body_start(line)), '')
        if not heading:
            return (False, '')
        row = self.parse_block_row(blocks[index])
        if row and row.get('page-no'):
            return (False, '')
        next_line = self.next_significant_line(blocks, index + 1, max_blocks=4)
        if next_line and (self.is_body_intro(next_line[1]) or self.is_body_sentence(next_line[1])):
            return (True, f'{one_line(heading)[:50]}_then_body_intro')
        return (False, '')

    def find_end_block(self, blocks: list[dict[str, Any]], start_block: int) -> tuple[int, str]:
        toc_row_count = 0
        for index in range(start_block + 1, len(blocks)):
            block = blocks[index]
            toc_row_count += int(self.toc_like_score(block) >= 2)
            if any((self.is_toc_header(line) for line in self.block_lines(block))) and toc_row_count >= 2:
                matched, reason = self.header_end_pair(blocks, index)
                if matched:
                    return (index, reason)
            if toc_row_count >= 2:
                matched, reason = self.direct_end_pair(blocks, index)
                if matched:
                    return (index, reason)
        toc_row_count = 0
        for index in range(start_block + 1, len(blocks)):
            toc_row_count += int(self.toc_like_score(blocks[index]) >= 2)
            if toc_row_count < 4:
                continue
            row = self.parse_block_row(blocks[index])
            for line in self.block_lines(blocks[index]):
                if self.is_body_start(line) and (not (row and row.get('page-no'))):
                    return (index, f'fallback_body_heading_{one_line(line)[:50]}')
        raise ValueError('Could not detect TOC end')

    def detect_boundaries(self, blocks: list[dict[str, Any]]) -> Boundary:
        start, start_reason = self.find_start_block(blocks)
        end, end_reason = self.find_end_block(blocks, start)
        if end <= start:
            raise ValueError(
                f'Invalid TOC boundaries: start={start}, end={end}')
        return Boundary(start, end, start_reason, end_reason)

    def deduplicate_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove only duplicate DOM representations of the same visual TOC row.

        Repeated title/page combinations are valid in many filings. For example,
        multiple proposals may each contain ``General`` or ``Vote Requirement`` on
        the same printed page. Therefore, title and page number must never be used
        as a document-wide uniqueness key.
        """
        output: list[dict[str, Any]] = []
        seen_block_indexes: set[int] = set()
        for row in rows:
            block_index = int(row.get('block_index', -1))
            if block_index >= 0 and block_index in seen_block_indexes:
                continue
            normalized_title = re.sub(
                '\\W+', ' ', one_line(row.get('title', '')).lower()).strip()
            page_no = normalize_page_token(row.get('page-no', ''))
            top = float(row.get('top', 0.0))
            left = float(row.get('left', 0.0))
            visual_duplicate = any((normalized_title == re.sub('\\W+', ' ', one_line(previous.get('title', '')).lower()).strip() and page_no == normalize_page_token(previous.get(
                'page-no', '')) and (abs(top - float(previous.get('top', 0.0))) <= 2.5) and (abs(left - float(previous.get('left', 0.0))) <= 3.0) for previous in output[-6:]))
            if visual_duplicate:
                continue
            output.append(row)
            if block_index >= 0:
                seen_block_indexes.add(block_index)
        return output

    def merge_unpaged_continuations(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in rows:
            title = row['title']
            if not row['page-no'] and self.STRUCTURAL_LABEL_RE.fullmatch(title):
                continue
            if output and (not row['page-no']) and (not self.ANNEX_RE.match(title)):
                previous = output[-1]
                gap = row['top'] - previous['bottom']
                same_indent = abs(row['relative_left'] -
                                  previous['relative_left']) <= 12
                if gap <= 14 and same_indent and (not previous['title'].endswith(('.', ':', ';'))):
                    if not self.is_body_start(title) and (not self.is_body_start(previous['title'])):
                        previous['title'] = one_line(
                            f"{previous['title']} {title}")
                        previous['bottom'] = row['bottom']
                        if not previous.get('href') and row.get('href'):
                            previous['href'] = row['href']
                        continue
            output.append(row)
        return output

    def cluster_centers(self, values: Iterable[float], tolerance: float = 7.0) -> list[tuple[float, int]]:
        clusters: list[list[float]] = []
        for value in sorted((float(v) for v in values)):
            if not clusters or abs(value - median(clusters[-1])) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [(float(median(cluster)), len(cluster)) for cluster in clusters]

    def assign_levels(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        measurable = [row for row in rows if row.get(
            'page-no') or self.ANNEX_RE.match(row.get('title', ''))]
        values = [float(row['left']) for row in measurable]
        centers = self.cluster_centers(values, tolerance=4.0)
        centers.sort()
        base_index = 0
        while base_index + 1 < len(centers) and centers[base_index][1] == 1 and (centers[base_index + 1][0] - centers[base_index][0] <= 5):
            base_index += 1
        base_center = centers[base_index][0] if centers else min(
            values, default=0.0)
        threshold: float | None = None
        for right_center, right_count in centers[base_index + 1:]:
            gap = right_center - base_center
            if right_count >= 2 and gap >= 6 or gap >= 12:
                threshold = (base_center + right_center) / 2
                break
        for row in rows:
            left = float(row.get('left', base_center))
            if self.ANNEX_RE.match(row.get('title', '')):
                level = 1
            elif threshold is None:
                level = 1
            else:
                level = 1 if left <= threshold else 2
            row['level'] = level
            row['indent_from_base'] = round(left - base_center, 2)
        return rows

    def extract_toc_rows(self, blocks: list[dict[str, Any]], boundary: Boundary) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        pending_title: dict[str, Any] | None = None
        for block in blocks[boundary.start_block:boundary.end_block]:
            lines = self.block_lines(block)
            if any((self.is_toc_header(line) for line in lines)):
                continue
            if all((self.PAGE_LABEL_RE.fullmatch(line) for line in lines if line)):
                continue
            row = self.parse_block_row(block)
            if not row:
                page_only = next((normalize_page_token(line)
                                 for line in lines if self.is_valid_page_token(line)), '')
                if page_only and pending_title:
                    pending_title['page-no'] = page_only
                    rows.append(pending_title)
                    pending_title = None
                continue
            title = row['title']
            if self.is_body_sentence(title) and (not row['page-no']):
                continue
            if row['page-no'] or self.ANNEX_RE.match(title):
                if pending_title:
                    if abs(pending_title['relative_left'] - row['relative_left']) <= 18 and (not self.is_body_start(pending_title['title'])):
                        row['title'] = one_line(
                            f"{pending_title['title']} {row['title']}")
                        if pending_title.get('href') and (not row.get('href')):
                            row['href'] = pending_title['href']
                    else:
                        rows.append(pending_title)
                    pending_title = None
                rows.append(row)
            elif len(title) >= 3 and (not self.STRUCTURAL_LABEL_RE.fullmatch(title)):
                if pending_title:
                    rows.append(pending_title)
                pending_title = row
        if pending_title and self.ANNEX_RE.match(pending_title['title']):
            rows.append(pending_title)
        rows = self.deduplicate_rows(rows)
        rows = self.merge_unpaged_continuations(rows)
        rows = self.assign_levels(rows)
        return rows

    def rows_to_nested_json(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        parent: dict[str, Any] | None = None
        for row in rows:
            title = one_line(row['title'])
            page_no = row['page-no']
            level = int(row.get('level', 1))
            href = row.get('href', '')
            if level == 1 or parent is None:
                parent = {'title': title, 'page-no': page_no,
                          'href': href, 'subsection': []}
                output.append(parent)
            else:
                parent['subsection'].append(
                    {'title': title, 'page-no': page_no, 'href': href})
        return output

    def extract(
        self,
        page: Page,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        blocks = self.extract_dom_blocks(page)
        boundary = self.detect_boundaries(blocks)
        rows = self.extract_toc_rows(blocks, boundary)

        if len(rows) < 2:
            raise ValueError(f"Only {len(rows)} TOC row(s) extracted")

        toc = self.rows_to_nested_json(rows)

        diagnostics = {
            "blocks": blocks,
            "boundary": boundary,
            "rows": rows,
        }

        return toc, diagnostics


# ============================================================
# PREAMBLE EXTRACTOR
# ============================================================


class PreambleExtractor:
    """Extract preamble text from document blocks prior to TOC start block."""

    def extract(
        self,
        blocks: list[dict[str, Any]],
        start_block: int,
    ) -> str:
        if start_block <= 0:
            return ""

        preamble_parts: list[str] = []
        for block in blocks[:start_block]:
            text = str(block.get("text") or "").strip()
            if not text:
                continue

            lines: list[str] = []
            if block.get("tag") == "TR" and block.get("cells"):
                for cell in block["cells"]:
                    cell_text = one_line(cell.get("text"))
                    if cell_text:
                        lines.append(cell_text)
                if lines:
                    preamble_parts.append(" ".join(lines))
            else:
                for line in normalize_text(text).splitlines():
                    clean_line = one_line(line)
                    if clean_line:
                        lines.append(clean_line)
                if lines:
                    preamble_parts.append("\n".join(lines))

        raw_preamble = "\n\n".join(preamble_parts)
        return normalize_text(raw_preamble)


# ============================================================
# 3. HREF TEXT EXTRACTOR
# ============================================================


class HrefTextExtractor:
    ANCHOR_EXTRACTION_JS = r"""
    (entries) => {
      const cleanText = (value) => String(value || "")
        .replace(/[\u00a0\u2000-\u200d\u202f\u205f\ufeff]/g, " ")
        .replace(/\r\n?/g, "\n")
        .replace(/[ \t]+/g, " ")
        .replace(/[ \t]*\n[ \t]*/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

      const normalizeMatch = (value) => cleanText(value)
        .toLowerCase()
        .replace(/&/g, " and ")
        .replace(/[’‘]/g, "'")
        .replace(/[–—]/g, "-")
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      const titleWithoutSeePage = (value) => String(value || "")
        .replace(/\s*\(\s*see\s+page(?:s)?\b[^)]*\)\s*$/i, "")
        .trim();

      const BLOCK_TAGS = new Set([
        "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "CAPTION", "DD", "DIV",
        "DL", "DT", "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM",
        "H1", "H2", "H3", "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN",
        "NAV", "OL", "P", "PRE", "SECTION", "TABLE", "TBODY", "TD", "TFOOT",
        "TH", "THEAD", "TR", "UL"
      ]);

      const SKIP_TAGS = new Set([
        "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS"
      ]);

      const append = (state, value) => {
        const text = String(value || "");
        if (!text) return;
        state.parts.push(text);
        state.length += text.length;
      };

      const wantedKeys = new Set(
        entries
          .map((entry) => String(entry.fragment || "").trim().toLowerCase())
          .filter(Boolean)
      );

      const anchorPositions = new Map();
      const state = { parts: [], length: 0 };

      const recordAnchor = (rawKey, element) => {
        const key = String(rawKey || "").trim().toLowerCase();
        if (!key || !wantedKeys.has(key)) return;

        if (!anchorPositions.has(key)) {
          anchorPositions.set(key, []);
        }

        anchorPositions.get(key).push({
          position: state.length,
          tag: element.tagName || "",
        });
      };

      const walk = (node) => {
        if (!node) return;

        if (node.nodeType === Node.TEXT_NODE) {
          append(state, node.nodeValue || "");
          return;
        }

        if (
          node.nodeType === Node.DOCUMENT_NODE ||
          node.nodeType === Node.DOCUMENT_FRAGMENT_NODE
        ) {
          for (const child of node.childNodes) {
            walk(child);
          }

          return;
        }

        if (node.nodeType !== Node.ELEMENT_NODE) {
          return;
        }

        const tag = node.tagName;

        if (SKIP_TAGS.has(tag)) {
          return;
        }

        recordAnchor(node.getAttribute("id"), node);
        recordAnchor(node.getAttribute("name"), node);

        if (tag === "BR") {
          append(state, "\n");
          return;
        }

        const isBlock = BLOCK_TAGS.has(tag);

        if (isBlock) {
          append(state, "\n");
        }

        for (const child of node.childNodes) {
          walk(child);
        }

        if (isBlock) {
          append(state, "\n");
        }
      };

      walk(document.body);

      const documentText = state.parts.join("");

      const regexEscape = (value) => String(value || "")
        .replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

      const buildTitlePattern = (title) => {
        const source = titleWithoutSeePage(title)
          .replace(/[’‘]/g, "'")
          .replace(/[–—]/g, "-")
          .trim();

        if (!source) {
          return null;
        }

        let pattern = regexEscape(source);

        pattern = pattern
          .replace(/'/g, "['’‘]")
          .replace(/-/g, "[-–—]")
          .replace(/&/g, "(?:&|and)")
          .replace(
            / +/g,
            "[\\s\\u00a0\\u2000-\\u200d\\u202f\\u205f\\ufeff]+"
          );

        return pattern;
      };

      const findHeading = (
        title,
        anchorPosition,
        minimumPosition
      ) => {
        const pattern = buildTitlePattern(title);

        if (!pattern) {
          return null;
        }

        const start = Math.max(
          0,
          anchorPosition,
          minimumPosition || 0
        );

        const maxForward = 60000;

        const end = Math.min(
          documentText.length,
          start + maxForward
        );

        const sample = documentText.slice(start, end);

        const standalone = new RegExp(
          "(?:^|\\n)[ \\t]*?(" + pattern + ")" +
          "(?:[ \\t]*\\([^\\n)]{0,100}\\))?" +
          "[ \\t]*(?=\\n|$)",
          "i"
        );

        let match = standalone.exec(sample);

        if (match) {
          const fullStart = start + match.index;

          const headingStart = (
            fullStart +
            match[0].search(/[^\n \t]/)
          );

          return {
            start: Math.max(start, headingStart),
            end: (
              start +
              match.index +
              match[0].length
            ),
            method: "anchor-title-standalone",
          };
        }

        const inline = new RegExp(pattern, "i");

        match = inline.exec(
          sample.slice(0, 5000)
        );

        if (match) {
          return {
            start: start + match.index,
            end: (
              start +
              match.index +
              match[0].length
            ),
            method: "anchor-title-inline",
          };
        }

        return null;
      };

      const occurrenceCursor = new Map();
      const resolved = [];

      let previousHeadingStart = 0;

      for (const entry of entries) {
        const key = String(
          entry.fragment || ""
        ).trim().toLowerCase();

        const occurrences = key
          ? (anchorPositions.get(key) || [])
          : [];

        const cursor = occurrenceCursor.get(key) || 0;

        let occurrence = null;
        let occurrenceIndex = cursor;

        for (
          let index = cursor;
          index < occurrences.length;
          index += 1
        ) {
          if (
            occurrences[index].position >=
            previousHeadingStart
          ) {
            occurrence = occurrences[index];
            occurrenceIndex = index;
            break;
          }
        }

        if (!occurrence && occurrences.length) {
          occurrence = occurrences[
            Math.min(
              cursor,
              occurrences.length - 1
            )
          ];

          occurrenceIndex = Math.min(
            cursor,
            occurrences.length - 1
          );
        }

        if (key && occurrence) {
          occurrenceCursor.set(
            key,
            occurrenceIndex + 1
          );
        }

        const anchorPosition = occurrence
          ? occurrence.position
          : null;

        const heading = anchorPosition === null
          ? null
          : findHeading(
              entry.title,
              anchorPosition,
              previousHeadingStart
            );

        const headingStart = heading
          ? heading.start
          : anchorPosition;

        const contentStart = heading
          ? heading.end
          : anchorPosition;

        if (typeof headingStart === "number") {
          previousHeadingStart = Math.max(
            previousHeadingStart,
            headingStart
          );
        }

        resolved.push({
          ...entry,
          anchorPosition,
          anchorTag: occurrence
            ? occurrence.tag
            : "",
          headingStart,
          contentStart,
          resolutionMethod: heading
            ? heading.method
            : (
                occurrence
                  ? "anchor-only"
                  : (
                      key
                        ? "anchor-not-found"
                        : "missing-href"
                    )
              ),
        });
      }

      const removeLeadingHeading = (
        text,
        title
      ) => {
        const lines = cleanText(text).split("\n");

        if (!lines.length) {
          return "";
        }

        const wanted = normalizeMatch(
          titleWithoutSeePage(title)
        );

        const first = normalizeMatch(lines[0]);

        if (
          wanted &&
          first &&
          (
            first === wanted ||
            first.startsWith(wanted + " ")
          )
        ) {
          lines.shift();
        }

        return cleanText(lines.join("\n"));
      };

      const output = [];

      for (
        let index = 0;
        index < resolved.length;
        index += 1
      ) {
        const current = resolved[index];

        if (!current.fragment) {
          output.push({
            status: "missing-href",
            text: "",
            method: current.resolutionMethod,
            targetTag: current.anchorTag,
            nextIndex: null,
            anchorPosition: null,
            headingStart: null,
            contentStart: null,
          });

          continue;
        }

        if (
          typeof current.contentStart !== "number"
        ) {
          output.push({
            status: "anchor-not-found",
            text: "",
            method: current.resolutionMethod,
            targetTag: current.anchorTag,
            nextIndex: null,
            anchorPosition:
              current.anchorPosition,
            headingStart:
              current.headingStart,
            contentStart:
              current.contentStart,
          });

          continue;
        }

        let nextIndex = null;
        let endPosition = documentText.length;

        for (
          let candidateIndex = index + 1;
          candidateIndex < resolved.length;
          candidateIndex += 1
        ) {
          const candidate = resolved[
            candidateIndex
          ];

          const candidatePosition = (
            typeof candidate.headingStart === "number"
          )
            ? candidate.headingStart
            : candidate.anchorPosition;

          if (
            typeof candidatePosition === "number" &&
            candidatePosition > current.contentStart
          ) {
            endPosition = candidatePosition;
            nextIndex = candidateIndex;
            break;
          }
        }

        if (endPosition < current.contentStart) {
          output.push({
            status: "range-error",
            text: "",
            method: current.resolutionMethod,
            targetTag: current.anchorTag,
            nextIndex,
            anchorPosition:
              current.anchorPosition,
            headingStart:
              current.headingStart,
            contentStart:
              current.contentStart,
            error:
              `Invalid text offsets: ` +
              `start=${current.contentStart}, ` +
              `end=${endPosition}`,
          });

          continue;
        }

        let text = cleanText(
          documentText.slice(
            current.contentStart,
            endPosition
          )
        );

        if (
          current.resolutionMethod ===
          "anchor-only"
        ) {
          text = removeLeadingHeading(
            text,
            current.title
          );
        }

        output.push({
          status: "matched",
          text,
          method: current.resolutionMethod,
          targetTag: current.anchorTag,
          nextIndex,
          anchorPosition:
            current.anchorPosition,
          headingStart:
            current.headingStart,
          contentStart:
            current.contentStart,
          endPosition,
        });
      }

      return output;
    }
    """

    def extract(
        self,
        page: Page,
        entries: list[TocEntry],
    ) -> list[dict[str, Any]]:
        payload = [
            {
                "title": entry.title,
                "href": entry.href,
                "fragment": entry.fragment,
            }
            for entry in entries
        ]

        results = page.evaluate(
            self.ANCHOR_EXTRACTION_JS,
            payload,
        )

        if (
            not isinstance(results, list)
            or len(results) != len(entries)
        ):
            actual = (
                len(results)
                if isinstance(results, list)
                else "invalid"
            )

            raise ValueError(
                "Anchor extraction returned an "
                "invalid result count: "
                f"expected={len(entries)}, "
                f"actual={actual}"
            )

        return results


# ============================================================
# 4. TITLE TEXT EXTRACTOR
# ============================================================


class TitleTextExtractor:
    TITLE_FALLBACK_EXTRACTION_JS = r"""
(entries) => {
  const cleanText = (value) => String(value || "")
    .replace(/[\u00a0\u2000-\u200d\u202f\u205f\ufeff]/g, " ")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const normalizeMatch = (value) => cleanText(value)
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[’‘]/g, "'")
    .replace(/[–—]/g, "-")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const titleWithoutSeePage = (value) => String(value || "")
    .replace(/\s*\(\s*see\s+page(?:s)?\b[^)]*\)\s*$/i, "")
    .replace(/\s*\(\s*page(?:s)?\b[^)]*\)\s*$/i, "")
    .trim();

  const regexEscape = (value) => String(value || "")
    .replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const BLOCK_TAGS = new Set([
    "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "CAPTION", "DD", "DIV",
    "DL", "DT", "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM",
    "H1", "H2", "H3", "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN",
    "NAV", "OL", "P", "PRE", "SECTION", "TABLE", "TBODY", "TD", "TFOOT",
    "TH", "THEAD", "TR", "UL"
  ]);

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS"
  ]);

  const STOP_WORDS = new Set([
    "a", "an", "and", "annex", "appendix", "as", "at", "about", "by",
    "dated", "date", "for", "from", "in", "inc", "incorporated", "into",
    "llc", "lp", "ltd", "of", "on", "or", "the", "to", "with"
  ]);

  const append = (state, value) => {
    const text = String(value || "");
    if (!text) return;
    state.parts.push(text);
    state.length += text.length;
  };

  const state = { parts: [], length: 0 };

  const walk = (node) => {
    if (!node) return;

    if (node.nodeType === Node.TEXT_NODE) {
      append(state, node.nodeValue || "");
      return;
    }

    if (
      node.nodeType === Node.DOCUMENT_NODE ||
      node.nodeType === Node.DOCUMENT_FRAGMENT_NODE
    ) {
      for (const child of node.childNodes) walk(child);
      return;
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const tag = node.tagName;
    if (SKIP_TAGS.has(tag)) return;

    if (tag === "BR") {
      append(state, "\n");
      return;
    }

    const isBlock = BLOCK_TAGS.has(tag);
    if (isBlock) append(state, "\n");
    for (const child of node.childNodes) walk(child);
    if (isBlock) append(state, "\n");
  };

  walk(document.body);
  const documentText = state.parts.join("");

  const buildTitlePattern = (title) => {
    const source = titleWithoutSeePage(title)
      .replace(/[’‘]/g, "'")
      .replace(/[–—]/g, "-")
      .trim();

    if (!source) return null;

    let pattern = regexEscape(source);
    pattern = pattern
      .replace(/'/g, "['’‘]")
      .replace(/-/g, "[-–—]")
      .replace(/&/g, "(?:&|and)")
      .replace(/ +/g, "[\\s\\u00a0\\u2000-\\u200d\\u202f\\u205f\\ufeff]+");

    return pattern;
  };

  const importantTitleKeywords = (title) => {
    const normalized = normalizeMatch(title);
    const tokens = normalized.split(" ").filter(Boolean);
    const seen = new Set();
    const keywords = [];

    for (const token of tokens) {
      if (
        token.length < 3 ||
        STOP_WORDS.has(token) ||
        /^[a-z]{1,3}$/.test(token) ||
        /^\d+$/.test(token)
      ) {
        continue;
      }
      if (!seen.has(token)) {
        seen.add(token);
        keywords.push(token);
      }
    }

    return keywords.slice(0, 18);
  };

  const buildTitleVariants = (entry) => {
    const values = [];
    const add = (value, rank) => {
      const cleaned = cleanText(value);
      if (!cleaned || cleaned.length < 3) return;
      const key = normalizeMatch(cleaned);
      if (!key || values.some((item) => item.key === key)) return;
      values.push({ text: cleaned, key, rank });
    };

    const original = titleWithoutSeePage(entry.title);
    add(original, 0);
    add(original.replace(/^the\s+/i, ""), 1);
    add(original.replace(/\s*\((?:annex|appendix)\s+[A-Z]{1,3}\)\s*$/i, ""), 1);
    add(original.replace(/\s*\([^)]*\)\s*$/i, ""), 2);

    if (entry.entryType === "annex") {
      const core = original.replace(
        /^\s*(?:annex|appendix)\s+[A-Z]{1,3}\s*[:.\-–—]?\s*/i,
        ""
      );

      // Keep controlled variants, but all of them are still accepted only
      // after the correct standalone Annex/Appendix label.
      const withoutDate = core.replace(
        /,?\s+dated\s+as\s+of\b.*$/i,
        ""
      );
      const withoutParties = withoutDate.replace(
        /,?\s+by\s+and\s+(?:among|between)\b.*$/i,
        ""
      );

      add(core, 2);
      add(withoutDate, 2);
      add(withoutParties, 3);

      // Opinion annexes often start with the adviser name rather than the
      // complete TOC wording, e.g. "Wells Fargo Securities, LLC" instead of
      // "Opinion of Wells Fargo Securities, LLC, dated as of ...".
      add(
        withoutDate.replace(/^opinion\s+of\s+/i, ""),
        3
      );
    }

    return values;
  };

  const lines = [];
  let lineCursor = 0;
  for (const rawLine of documentText.split("\n")) {
    const start = lineCursor;
    const end = start + rawLine.length;
    lineCursor = end + 1;

    const text = cleanText(rawLine);
    if (!text) continue;

    const normalized = normalizeMatch(text);
    const tokens = normalized.split(" ").filter(Boolean);
    lines.push({ start, end, text, normalized, tokens });
  }

  const tokenIndex = new Map();
  for (let index = 0; index < lines.length; index += 1) {
    for (const token of new Set(lines[index].tokens)) {
      if (token.length < 4 || STOP_WORDS.has(token)) continue;
      if (!tokenIndex.has(token)) tokenIndex.set(token, []);
      const bucket = tokenIndex.get(token);
      if (bucket.length < 6000) bucket.push(index);
    }
  }

  const normalizePageToken = (value) => cleanText(value)
    .replace(/[–—]/g, "-")
    .trim()
    .toLowerCase();

  const pagePatternFor = (entry) => {
    const pageToken = normalizePageToken(entry.pageNo);
    if (!pageToken) return null;
    const escaped = regexEscape(pageToken).replace(/\\-/g, "[-–—]");
    return new RegExp("(?:^|\\s)" + escaped + "(?=\\s|$)", "i");
  };

  // A compact TOC row has the expected page token immediately after the title,
  // before any meaningful prose. This is deliberately different from merely
  // seeing a page number somewhere near a body heading.
  const compactTocRow = (entry, candidate) => {
    const pattern = pagePatternFor(entry);
    if (!pattern) return null;

    const rawAfter = documentText.slice(
      candidate.end,
      Math.min(documentText.length, candidate.end + 320)
    );
    const match = pattern.exec(cleanText(rawAfter));
    if (!match) return null;

    const beforePage = cleanText(rawAfter.slice(0, match.index));
    const alphabetic = (beforePage.match(/[A-Za-z]/g) || []).length;
    const words = normalizeMatch(beforePage).split(" ").filter(Boolean).length;

    if (alphabetic > 24 || words > 5) return null;

    return {
      pageOffset: match.index,
      rowEnd: Math.min(documentText.length, candidate.end + match.index + match[0].length),
    };
  };

  const collectExactCandidates = (entry) => {
    const candidates = [];
    const byPosition = new Map();
    const variants = buildTitleVariants(entry);

    const add = (candidate) => {
      const key = `${candidate.start}:${candidate.end}`;
      const existing = byPosition.get(key);
      if (!existing || candidate.quality > existing.quality) {
        byPosition.set(key, candidate);
      }
    };

    for (const variant of variants) {
      const pattern = buildTitlePattern(variant.text);
      if (!pattern) continue;

      const standalone = new RegExp(
        "(?:^|\\n)[ \\t\\u00a0\\u2000-\\u200d\\u202f\\u205f\\ufeff]*" +
        "(" + pattern + ")" +
        "(?:[ \\t]*\\([^\\n)]{0,140}\\))?" +
        "[ \\t]*(?=\\n|$)",
        "gi"
      );

      let match;
      while ((match = standalone.exec(documentText)) !== null) {
        const matchedPart = match[1] || "";
        const relative = match[0].indexOf(matchedPart);
        const start = match.index + Math.max(0, relative);
        const end = match.index + match[0].length;
        const quality = variant.rank === 0 ? 4 : (variant.rank === 1 ? 3 : 2);

        add({
          start,
          end,
          method: variant.rank === 0
            ? "title-exact-standalone"
            : "title-variant-standalone",
          quality,
          baseScore: variant.rank === 0 ? 140 : (variant.rank === 1 ? 126 : 114),
          matchedVariant: variant.text,
          matchedKeywords: [],
          similarity: 1,
        });

        if (standalone.lastIndex <= match.index) standalone.lastIndex = match.index + 1;
      }
    }

    candidates.push(...byPosition.values());
    return candidates;
  };

  const isIgnorableAnnexPretitleLine = (value) => {
    const normalized = normalizeMatch(value);
    if (!normalized) return true;

    if (
      normalized === "execution version" ||
      normalized === "execution copy" ||
      normalized === "final form" ||
      normalized === "draft" ||
      normalized === "confidential" ||
      normalized === "confidential draft" ||
      normalized === "subject to completion" ||
      normalized === "table of contents"
    ) {
      return true;
    }

    // Page markers such as A-1, B-1 or AA-12 become "a 1", "b 1"
    // and "aa 12" after normalizeMatch().
    return /^[a-z]{1,3}\s+\d+$/i.test(normalized);
  };

  const collectAnnexLabelCandidates = (entry) => {
    /*
     * Strict annex title matching for TITLE-BASED extraction only.
     *
     * Required order:
     *   ANNEX H
     *   FORM OF LOCK-UP AGREEMENT
     *
     * Also accepted:
     *   ANNEX H - FORM OF LOCK-UP AGREEMENT
     *
     * The separator after the annex label is optional. Generic title-only or
     * fuzzy matches are not used for annexes, so exhibit-index references such
     * as "10.14 Form of ... as Annex I" cannot become annex boundaries.
     */
    if (entry.entryType !== "annex" || !entry.annexLabel) return [];

    const labelText = String(entry.annexLabel || "").trim();
    if (!labelText) return [];

    const escapedLabel = regexEscape(labelText);
    const prefixPattern = new RegExp(
      "^\\s*(?:annex|appendix)\\s+" + escapedLabel +
      "(?=\\s|$|[-–—:.])\\s*(?:[-–—:.]\\s*)?",
      "i"
    );

    const labelLinePattern = new RegExp(
      "^\\s*(?:annex|appendix)\\s+" + escapedLabel +
      "(?=\\s|$|[-–—:.])\\s*(?:[-–—:.]\\s*)?(.*)$",
      "i"
    );

    // Preserve the existing controlled title variants (for example, removal
    // of a trailing "dated as of" phrase), but require every variant to occur
    // AFTER the correct Annex/Appendix label.
    const variants = [];
    const seenVariants = new Set();

    for (const variant of buildTitleVariants(entry)) {
      const coreText = cleanText(
        String(variant.text || "").replace(prefixPattern, "")
      );
      const coreKey = normalizeMatch(coreText);

      if (!coreText || !coreKey || seenVariants.has(coreKey)) continue;
      seenVariants.add(coreKey);
      variants.push({
        text: coreText,
        key: coreKey,
        rank: Number(variant.rank || 0),
      });
    }

    if (!variants.length) return [];

    const candidates = [];
    const seenCandidates = new Set();
    const maxFollowingLines = 8;

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const labelLine = lines[lineIndex];
      const labelMatch = labelLinePattern.exec(labelLine.text);
      if (!labelMatch) continue;

      const sameLineRemainder = cleanText(labelMatch[1] || "");

      for (const variant of variants) {
        const titleParts = [];
        if (
          sameLineRemainder &&
          !isIgnorableAnnexPretitleLine(sameLineRemainder)
        ) {
          titleParts.push(sameLineRemainder);
        }

        let matchedEnd = null;
        let matchedText = "";

        for (let offset = 0; offset <= maxFollowingLines; offset += 1) {
          if (offset > 0) {
            const nextLine = lines[lineIndex + offset];
            if (!nextLine) break;

            // Permit only controlled status/page lines between the standalone
            // Annex label and the actual title. This restores filings such as:
            //   Annex A / EXECUTION VERSION / AGREEMENT AND PLAN OF MERGER
            // without re-enabling generic annex fuzzy matching.
            if (
              !titleParts.length &&
              isIgnorableAnnexPretitleLine(nextLine.text)
            ) {
              continue;
            }

            titleParts.push(nextLine.text);
          }

          if (!titleParts.length) continue;

          const combinedText = cleanText(titleParts.join(" "));
          const combinedKey = normalizeMatch(combinedText);
          if (!combinedKey) continue;

          if (combinedKey === variant.key) {
            const lastLine = lines[lineIndex + offset] || labelLine;
            matchedEnd = lastLine.end;
            matchedText = combinedText;
            break;
          }

          // The text collected so far must remain an exact prefix of the
          // expected title. Any unrelated line immediately rejects this label.
          if (variant.key.startsWith(combinedKey + " ")) {
            continue;
          }

          break;
        }

        if (typeof matchedEnd !== "number") continue;

        const candidateKey = `${labelLine.start}:${matchedEnd}`;
        if (seenCandidates.has(candidateKey)) continue;
        seenCandidates.add(candidateKey);

        const keywords = importantTitleKeywords(entry.title);
        const normalizedMatchedText = normalizeMatch(matchedText);
        const matchedKeywords = keywords.filter(
          (keyword) => normalizedMatchedText.includes(keyword)
        );

        candidates.push({
          start: labelLine.start,
          end: matchedEnd,
          method: "title-annex-ordered-exact",
          quality: variant.rank === 0 ? 5 : (variant.rank === 1 ? 4 : 3),
          baseScore: 160 - Math.min(32, variant.rank * 8),
          matchedVariant:
            `Annex ${entry.annexLabel} ${matchedText}`,
          matchedKeywords,
          similarity: keywords.length
            ? matchedKeywords.length / keywords.length
            : 1,
        });
      }
    }

    return candidates;
  };

  const collectFuzzyCandidates = (entry) => {
    const keywords = importantTitleKeywords(entry.title);
    if (!keywords.length) return [];

    const lineIndexes = new Set();
    const indexedKeywords = [...keywords]
      .sort((left, right) => {
        const leftCount = (tokenIndex.get(left) || []).length;
        const rightCount = (tokenIndex.get(right) || []).length;
        return leftCount - rightCount;
      })
      .slice(0, 8);

    for (const keyword of indexedKeywords) {
      for (const lineIndex of tokenIndex.get(keyword) || []) {
        lineIndexes.add(lineIndex);
        if (lineIndexes.size > 1400) break;
      }
      if (lineIndexes.size > 1400) break;
    }

    const candidates = [];
    const seen = new Set();
    const titleWordCount = normalizeMatch(entry.title).split(" ").filter(Boolean).length;

    const addWindow = (startIndex, endIndex) => {
      if (startIndex < 0 || endIndex >= lines.length || endIndex < startIndex) return;

      const selected = lines.slice(startIndex, endIndex + 1);
      const rawText = selected.map((line) => line.text).join(" ");
      if (!rawText || rawText.length > 560) return;

      const normalized = normalizeMatch(rawText);
      const words = normalized.split(" ").filter(Boolean);
      if (words.length > Math.max(14, titleWordCount + 8)) return;

      const matchedKeywords = keywords.filter((keyword) => normalized.includes(keyword));
      const similarity = matchedKeywords.length / keywords.length;
      const shortTitle = keywords.length <= 3;

      if (shortTitle) {
        if (matchedKeywords.length !== keywords.length) return;
      } else if (matchedKeywords.length < 3 || similarity < 0.72) {
        return;
      }

      const sentenceMarks = (rawText.match(/[.!?](?:\s|$)/g) || []).length;
      if (sentenceMarks > 0 && rawText.length > 160) return;

      const start = selected[0].start;
      const end = selected[selected.length - 1].end;
      const key = `${start}:${end}`;
      if (seen.has(key)) return;
      seen.add(key);

      candidates.push({
        start,
        end,
        method: "title-keyword-fuzzy",
        quality: 1,
        baseScore: 86 + Math.round(similarity * 28) + Math.min(10, matchedKeywords.length),
        matchedVariant: rawText,
        matchedKeywords,
        similarity,
      });
    };

    for (const index of lineIndexes) {
      for (let backward = 0; backward <= 2; backward += 1) {
        const startIndex = index - backward;
        for (let length = 1; length <= 4; length += 1) {
          addWindow(startIndex, startIndex + length - 1);
        }
      }
    }

    return candidates;
  };

  const candidateCache = new Map();
  const getCandidates = (entry, index) => {
    if (candidateCache.has(index)) return candidateCache.get(index);

    // Annexes use only the strict ordered label + title matcher.
    // Normal sections/subsections keep the existing exact and fuzzy logic.
    const combined = entry.entryType === "annex"
      ? collectAnnexLabelCandidates(entry)
      : [
          ...collectExactCandidates(entry),
          ...collectFuzzyCandidates(entry),
        ];

    const byStart = new Map();
    for (const candidate of combined) {
      const existing = byStart.get(candidate.start);
      if (
        !existing ||
        candidate.quality > existing.quality ||
        (candidate.quality === existing.quality && candidate.baseScore > existing.baseScore)
      ) {
        byStart.set(candidate.start, candidate);
      }
    }

    const result = [...byStart.values()].sort(
      (left, right) => left.start - right.start || right.quality - left.quality
    );
    candidateCache.set(index, result);
    return result;
  };

  const findFrontTocEnd = () => {
    const searchLimit = Math.min(
      documentText.length,
      Math.max(180000, Math.floor(documentText.length * 0.42))
    );
    const sample = documentText.slice(0, searchLimit);
    const headerPattern = /(?:^|\n)\s*TABLE\s+OF\s+CONTENTS?\s*(?=\n|$)/gi;
    const headers = [];
    let headerMatch;
    while ((headerMatch = headerPattern.exec(sample)) !== null) {
      headers.push(headerMatch.index + headerMatch[0].length);
      if (headerPattern.lastIndex <= headerMatch.index) headerPattern.lastIndex = headerMatch.index + 1;
    }

    if (!headers.length) return 0;

    let cursor = headers[0];
    let lastEnd = cursor;
    let matched = 0;
    let consecutiveMisses = 0;

    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const compact = getCandidates(entry, index)
        .map((candidate) => ({ candidate, toc: compactTocRow(entry, candidate) }))
        .filter((item) =>
          item.toc &&
          item.candidate.start > cursor &&
          item.candidate.start < searchLimit
        )
        .sort((left, right) => left.candidate.start - right.candidate.start);

      const selected = compact[0];
      if (!selected) {
        consecutiveMisses += 1;
        if (matched >= 5 && consecutiveMisses >= 12) break;
        continue;
      }

      if (matched >= 3 && selected.candidate.start - cursor > 100000) break;

      matched += 1;
      consecutiveMisses = 0;
      cursor = selected.candidate.start;
      lastEnd = Math.max(lastEnd, selected.toc.rowEnd);
    }

    // A genuine front TOC must contain several ordered TOC rows. Returning zero
    // is safer than excluding an uncertain part of the document.
    return matched >= 4 ? Math.min(documentText.length, lastEnd + 1) : 0;
  };

  const frontTocEnd = findFrontTocEnd();

  const referenceLanguageBefore = (entry, candidate) => {
    const before = normalizeMatch(documentText.slice(
      Math.max(0, candidate.start - 260),
      candidate.start
    ));

    if (
      /\b(?:attached as|included as|set forth in|described in|refer to|referred to)\s+(?:the\s+)?(?:section|annex|appendix|exhibit|opinion|agreement)\b/.test(before)
    ) {
      return true;
    }

    if (/\bsee\b/.test(before)) {
      const tail = before.slice(-190);
      const seeIndex = tail.lastIndexOf("see ");
      if (seeIndex >= 0) {
        const afterSee = tail.slice(seeIndex + 4);
        return importantTitleKeywords(entry.title)
          .slice(0, 6)
          .some((keyword) => afterSee.includes(keyword));
      }
    }

    return false;
  };

  const substantiveTextAfter = (candidate) => {
    const after = cleanText(documentText.slice(
      candidate.end,
      Math.min(documentText.length, candidate.end + 900)
    ));
    return (after.match(/[A-Za-z]/g) || []).length;
  };

  const chooseCandidate = (entry, index, minimumPosition, usedStarts) => {
    const candidates = getCandidates(entry, index).filter((candidate) =>
      candidate.start > minimumPosition &&
      candidate.start >= frontTocEnd &&
      !usedStarts.has(candidate.start)
    );

    const exact = candidates.filter((candidate) => candidate.quality >= 2);
    const fuzzy = candidates.filter((candidate) => candidate.quality === 1);

    if (entry.entryType === "annex") {
      const orderedAnnexTitles = exact.filter(
        (candidate) => candidate.method === "title-annex-ordered-exact"
      );

      const acceptableOrderedAnnexTitles = orderedAnnexTitles.filter(
        (candidate) => {
          // If front TOC detection failed, do not select the compact TOC row
          // itself as the body annex heading.
          if (!frontTocEnd && compactTocRow(entry, candidate)) return false;
          return true;
        }
      );

      return acceptableOrderedAnnexTitles.length
        ? acceptableOrderedAnnexTitles[0]
        : null;
    }

    const acceptableExact = exact.filter((candidate) => {
      // When no TOC boundary was found, exclude only candidates that are clearly
      // compact TOC rows. Body headings may legitimately have page numbers near
      // them, so this check is intentionally narrow.
      if (!frontTocEnd && compactTocRow(entry, candidate)) return false;
      return true;
    });

    if (acceptableExact.length) {
      // The TOC itself defines document order. After the front TOC is excluded,
      // the earliest standalone exact/variant heading is the correct duplicate
      // in the overwhelming majority of SEC proxy documents. This avoids the v5
      // bug where a later duplicate won only because it had a higher score.
      const earliestStart = acceptableExact[0].start;
      const sameStart = acceptableExact.filter((candidate) => candidate.start === earliestStart);
      return sameStart.sort(
        (left, right) => right.quality - left.quality || right.baseScore - left.baseScore
      )[0];
    }

    const acceptableFuzzy = fuzzy.filter((candidate) => {
      if (!frontTocEnd && compactTocRow(entry, candidate)) return false;
      if (referenceLanguageBefore(entry, candidate)) return false;
      if (candidate.similarity < 0.72) return false;
      if (substantiveTextAfter(candidate) < 35) return false;
      return true;
    });

    return acceptableFuzzy.length ? acceptableFuzzy[0] : null;
  };

  const resolved = [];
  const usedStarts = new Set();
  let previousHeadingStart = frontTocEnd || 0;

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const selected = chooseCandidate(entry, index, previousHeadingStart, usedStarts);

    if (selected) {
      const preserveHeading = entry.entryType === "annex";
      const evidence = [
        selected.method,
        "earliest-valid-in-toc-order",
        frontTocEnd ? "after-front-toc" : "no-front-toc-boundary",
      ];
      if (selected.quality >= 2) evidence.push("standalone-title");
      if (selected.method === "title-keyword-fuzzy") evidence.push(`similarity:${selected.similarity.toFixed(3)}`);

      resolved.push({
        ...entry,
        headingStart: selected.start,
        contentStart: preserveHeading ? selected.start : selected.end,
        resolutionMethod: selected.method,
        candidateScore: selected.baseScore,
        candidateCount: getCandidates(entry, index).length,
        candidateEvidence: evidence,
        matchedVariant: selected.matchedVariant,
        matchedKeywords: selected.matchedKeywords || [],
        similarity: selected.similarity,
      });
      previousHeadingStart = selected.start;
      usedStarts.add(selected.start);
    } else {
      const available = getCandidates(entry, index).filter(
        (candidate) => candidate.start > previousHeadingStart
      );
      resolved.push({
        ...entry,
        headingStart: null,
        contentStart: null,
        resolutionMethod: available.length ? "title-low-confidence" : "title-not-found",
        candidateScore: available.length ? available[0].baseScore : null,
        candidateCount: available.length,
        candidateEvidence: available.length ? ["no-acceptable-ordered-candidate"] : [],
        matchedVariant: available.length ? available[0].matchedVariant : "",
        matchedKeywords: available.length ? (available[0].matchedKeywords || []) : [],
        similarity: available.length ? available[0].similarity : null,
      });
    }
  }

  const findTrailingNonAnnexBoundary = (text, absoluteStart, annexLabel) => {
    const markers = [
      /\bPRELIMINARY\s+PROXY\s+CARD\b/i,
      /\bYOUR\s+VOTE\s+IS\s+IMPORTANT!?\b/i,
      /\bVOTE\s+BY\s+INTERNET\b/i,
      /\bVOTE\s+BY\s+PHONE\b/i,
      /\bDETACH\s+AND\s+RETURN\b/i,
      /\bTHIS\s+PROXY\s+CARD\s+IS\s+VALID\b/i,
      /\bSCAN\s+TO\s+VIEW\s+MATERIALS\b/i,
      /\bImportant\s+Notice\s+Regarding\s+the\s+Availability\s+of\s+Proxy\s+Materials\b/i,
      /\bCast\s+your\s+vote\s+online\b/i,
      /\bTalkspace\s+Your\s+vote\s+Matters\b/i,
    ];

    let best = null;
    for (const marker of markers) {
      const match = marker.exec(text);
      if (!match || match.index < 1000) continue;

      const prefix = text.slice(0, match.index);
      const pagePattern = annexLabel
        ? new RegExp(`\\b${regexEscape(String(annexLabel))}[-–—]\\d+\\b`, "i")
        : null;
      const prefixStart = normalizeMatch(prefix.slice(0, 500));
      const expectedStart = annexLabel
        ? new RegExp(`^(?:annex|appendix)\\s+${regexEscape(String(annexLabel).toLowerCase())}\\b`, "i")
        : null;
      const hasSubstantialAnnexText =
        (prefix.match(/[A-Za-z]/g) || []).length >= 500 &&
        (!expectedStart || expectedStart.test(prefixStart));
      const hasAnnexEndEvidence =
        (pagePattern && pagePattern.test(prefix.slice(-10000))) ||
        /very\s+truly\s+yours|signature\s+page|\/s\//i.test(prefix.slice(-10000)) ||
        hasSubstantialAnnexText;

      if (!hasAnnexEndEvidence) continue;

      const previousBlankLine = text.lastIndexOf("\n\n", match.index);
      const previousLineBreak = text.lastIndexOf("\n", match.index);
      const cut = previousBlankLine >= 0
        ? previousBlankLine + 2
        : (previousLineBreak >= 0 ? previousLineBreak + 1 : match.index);

      if (best === null || cut < best) best = cut;
    }

    return best === null ? null : absoluteStart + best;
  };

  const output = [];

  for (let index = 0; index < resolved.length; index += 1) {
    const current = resolved[index];

    if (typeof current.contentStart !== "number") {
      output.push({
        status: current.resolutionMethod === "title-low-confidence"
          ? "title-low-confidence"
          : "title-not-found",
        text: "",
        method: current.resolutionMethod,
          nextIndex: null,
        headingStart: null,
        contentStart: null,
        candidateScore: current.candidateScore,
        candidateCount: current.candidateCount,
        candidateEvidence: current.candidateEvidence,
        matchedVariant: current.matchedVariant,
        matchedKeywords: current.matchedKeywords,
        similarity: current.similarity,
        frontTocEnd,
      });
      continue;
    }

    let nextIndex = null;
    let endPosition = documentText.length;

    for (let candidateIndex = index + 1; candidateIndex < resolved.length; candidateIndex += 1) {
      const next = resolved[candidateIndex];
      if (
        typeof next.headingStart === "number" &&
        next.headingStart > current.contentStart
      ) {
        endPosition = next.headingStart;
        nextIndex = candidateIndex;
        break;
      }
    }

    if (endPosition < current.contentStart) {
      output.push({
        status: "range-error",
        text: "",
        method: current.resolutionMethod,
          nextIndex,
        headingStart: current.headingStart,
        contentStart: current.contentStart,
        endPosition,
        candidateScore: current.candidateScore,
        candidateCount: current.candidateCount,
        candidateEvidence: current.candidateEvidence,
        matchedVariant: current.matchedVariant,
        matchedKeywords: current.matchedKeywords,
        similarity: current.similarity,
        frontTocEnd,
        error: `Invalid text offsets: start=${current.contentStart}, end=${endPosition}`,
      });
      continue;
    }

    if (current.entryType === "annex" && nextIndex === null) {
      const finalText = documentText.slice(current.contentStart, endPosition);
      const trailingBoundary = findTrailingNonAnnexBoundary(
        finalText,
        current.contentStart,
        current.annexLabel
      );
      if (typeof trailingBoundary === "number") endPosition = trailingBoundary;
    }

    const text = cleanText(documentText.slice(current.contentStart, endPosition));

    output.push({
      status: "matched",
      text,
      method: current.resolutionMethod,
      nextIndex,
      headingStart: current.headingStart,
      contentStart: current.contentStart,
      endPosition,
      candidateScore: current.candidateScore,
      candidateCount: current.candidateCount,
      candidateEvidence: current.candidateEvidence,
      matchedVariant: current.matchedVariant,
      matchedKeywords: current.matchedKeywords,
      similarity: current.similarity,
      frontTocEnd,
    });
  }

  return output;
}
"""

    def extract(
        self,
        page: Page,
        entries: list[TocEntry],
    ) -> list[dict[str, Any]]:
        payload = [
            {
                "title": entry.title,
                "pageNo": entry.page_no,
                "level": entry.level,
                "entryType": entry.entry_type,
                "annexLabel": entry.annex_label,
            }
            for entry in entries
        ]

        results = page.evaluate(
            self.TITLE_FALLBACK_EXTRACTION_JS,
            payload,
        )

        if (
            not isinstance(results, list)
            or len(results) != len(entries)
        ):
            actual = (
                len(results)
                if isinstance(results, list)
                else "invalid"
            )

            raise ValueError(
                "Title fallback returned an invalid "
                "result count: "
                f"expected={len(entries)}, "
                f"actual={actual}"
            )

        return results

    def validate(
        self,
        entries: list[TocEntry],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach conservative validation errors without changing working text."""
        document_issues: list[dict[str, Any]] = []
        previous_position = -1
        previous_annex_position = -1

        for index, (entry, result) in enumerate(
            zip(entries, results)
        ):
            errors: list[str] = []

            status = one_line(
                result.get("status")
            )

            text = normalize_text(
                result.get("text")
            )

            heading_start = result.get(
                "headingStart"
            )

            content_start = result.get(
                "contentStart"
            )

            end_position = result.get(
                "endPosition"
            )

            has_immediate_subsection = (
                entry.level == 1
                and index + 1 < len(entries)
                and entries[index + 1].level == 2
                and entries[index + 1].path[0]
                == entry.path[0]
            )

            if (
                status == "matched"
                and not text
                and not has_immediate_subsection
            ):
                errors.append(
                    "matched-with-empty-text"
                )

            if isinstance(
                heading_start,
                (int, float),
            ):
                if heading_start < previous_position:
                    errors.append(
                        "non-monotonic-heading-position"
                    )

                previous_position = max(
                    previous_position,
                    int(heading_start),
                )

            if (
                isinstance(
                    content_start,
                    (int, float),
                )
                and isinstance(
                    end_position,
                    (int, float),
                )
                and end_position < content_start
            ):
                errors.append(
                    "invalid-text-range"
                )

            if entry.entry_type == "annex":
                if isinstance(
                    heading_start,
                    (int, float),
                ):
                    if (
                        heading_start
                        <= previous_annex_position
                    ):
                        errors.append(
                            "annex-order-invalid"
                        )

                    previous_annex_position = int(
                        heading_start
                    )

                if status == "matched":
                    prefix = text[:600]
                    expected_label = re.escape(
                        entry.annex_label
                    )

                    if not re.search(
                        rf"\b(?:ANNEX|APPENDIX)\s+"
                        rf"{expected_label}\b",
                        prefix,
                        re.IGNORECASE,
                    ):
                        errors.append(
                            "annex-label-missing-at-start"
                        )

            result["validationErrors"] = errors

            if errors:
                document_issues.append(
                    {
                        "index": index,
                        "title": entry.title,
                        "entry-type":
                            entry.entry_type,
                        "errors": errors,
                    }
                )

                fatal_errors = {
                    "matched-with-empty-text",
                    "invalid-text-range",
                    "annex-order-invalid",
                    "annex-label-missing-at-start",
                }

                if (
                    status == "matched"
                    and fatal_errors.intersection(
                        errors
                    )
                ):
                    result["status"] = (
                        "validation-failed"
                    )

        first_annex_index = next(
            (
                index
                for index, entry
                in enumerate(entries)
                if entry.entry_type == "annex"
            ),
            None,
        )

        if (
            first_annex_index is not None
            and first_annex_index > 0
        ):
            previous_result = results[
                first_annex_index - 1
            ]

            previous_text = normalize_text(
                previous_result.get("text")
            )

            first_label = re.escape(
                entries[
                    first_annex_index
                ].annex_label
            )

            if re.search(
                rf"(?:^|\n)\s*"
                rf"(?:ANNEX|APPENDIX)\s+"
                rf"{first_label}\s*(?:\n|$)",
                previous_text,
                re.IGNORECASE,
            ):
                previous_result.setdefault(
                    "validationErrors",
                    [],
                ).append(
                    "annex-leaked-into-previous-section"
                )

                document_issues.append(
                    {
                        "index":
                            first_annex_index - 1,
                        "title":
                            entries[
                                first_annex_index - 1
                            ].title,
                        "entry-type":
                            entries[
                                first_annex_index - 1
                            ].entry_type,
                        "errors": [
                            "annex-leaked-into-previous-section"
                                ],
                    }
                )

        return document_issues


# ============================================================
# 5. DOCUMENT EXTRACTION PIPELINE
# ============================================================


class DocumentExtractionPipeline:
    RESLICE_MERGED_RESULTS_JS = r"""
    (items) => {
      const cleanText = (value) => String(value || "")
        .replace(/[\u00a0\u2000-\u200d\u202f\u205f\ufeff]/g, " ")
        .replace(/\r\n?/g, "\n")
        .replace(/[ \t]+/g, " ")
        .replace(/[ \t]*\n[ \t]*/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

      const normalizeMatch = (value) => cleanText(value)
        .toLowerCase()
        .replace(/&/g, " and ")
        .replace(/[’‘]/g, "'")
        .replace(/[–—]/g, "-")
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      const titleWithoutSeePage = (value) => String(value || "")
        .replace(/\s*\(\s*see\s+page(?:s)?\b[^)]*\)\s*$/i, "")
        .trim();

      const BLOCK_TAGS = new Set([
        "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "CAPTION", "DD", "DIV",
        "DL", "DT", "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM",
        "H1", "H2", "H3", "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN",
        "NAV", "OL", "P", "PRE", "SECTION", "TABLE", "TBODY", "TD", "TFOOT",
        "TH", "THEAD", "TR", "UL"
      ]);

      const SKIP_TAGS = new Set([
        "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS"
      ]);

      const append = (state, value) => {
        const text = String(value || "");
        if (!text) return;
        state.parts.push(text);
        state.length += text.length;
      };

      const state = { parts: [], length: 0 };

      const walk = (node) => {
        if (!node) return;

        if (node.nodeType === Node.TEXT_NODE) {
          append(state, node.nodeValue || "");
          return;
        }

        if (
          node.nodeType === Node.DOCUMENT_NODE ||
          node.nodeType === Node.DOCUMENT_FRAGMENT_NODE
        ) {
          for (const child of node.childNodes) walk(child);
          return;
        }

        if (node.nodeType !== Node.ELEMENT_NODE) return;

        const tag = node.tagName;
        if (SKIP_TAGS.has(tag)) return;

        if (tag === "BR") {
          append(state, "\n");
          return;
        }

        const isBlock = BLOCK_TAGS.has(tag);
        if (isBlock) append(state, "\n");
        for (const child of node.childNodes) walk(child);
        if (isBlock) append(state, "\n");
      };

      walk(document.body);
      const documentText = state.parts.join("");

      const removeLeadingHeading = (text, title) => {
        const lines = cleanText(text).split("\n");
        if (!lines.length) return "";

        const wanted = normalizeMatch(titleWithoutSeePage(title));
        const first = normalizeMatch(lines[0]);

        if (
          wanted &&
          first &&
          (first === wanted || first.startsWith(wanted + " "))
        ) {
          lines.shift();
        }

        return cleanText(lines.join("\n"));
      };

      return items.map((current, index) => {
        if (
          current.status !== "matched" ||
          typeof current.contentStart !== "number"
        ) {
          return null;
        }

        const originalEnd = typeof current.endPosition === "number"
          ? current.endPosition
          : documentText.length;

        let correctedEnd = originalEnd;
        let nextIndex = current.nextIndex ?? null;

        for (
          let candidateIndex = index + 1;
          candidateIndex < items.length;
          candidateIndex += 1
        ) {
          const candidate = items[candidateIndex];
          const candidatePosition = typeof candidate.headingStart === "number"
            ? candidate.headingStart
            : null;

          if (
            typeof candidatePosition === "number" &&
            candidatePosition > current.contentStart
          ) {
            if (candidatePosition < correctedEnd) {
              correctedEnd = candidatePosition;
              nextIndex = candidateIndex;
            }
            break;
          }
        }

        if (
          correctedEnd >= originalEnd ||
          correctedEnd < current.contentStart
        ) {
          return null;
        }

        let text = cleanText(
          documentText.slice(current.contentStart, correctedEnd)
        );

        if (current.method === "anchor-only") {
          text = removeLeadingHeading(text, current.title);
        }

        return {
          text,
          endPosition: correctedEnd,
          nextIndex,
        };
      });
    }
    """

    def __init__(
        self,
        toc_extractor: TocExtractor | None = None,
        href_extractor: HrefTextExtractor | None = None,
        title_extractor: TitleTextExtractor | None = None,
        preamble_extractor: PreambleExtractor | None = None,
    ) -> None:
        self.toc_extractor = toc_extractor or TocExtractor()
        self.href_extractor = href_extractor or HrefTextExtractor()
        self.title_extractor = title_extractor or TitleTextExtractor()
        self.preamble_extractor = preamble_extractor or PreambleExtractor()

    @staticmethod
    def is_matched(result: dict[str, Any]) -> bool:
        return one_line(result.get("status")) == "matched"

    def merge_results(
        self,
        entries: list[TocEntry],
        href_results: list[dict[str, Any]],
        title_results: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if len(href_results) != len(entries):
            raise ValueError(
                "Href result count does not match TOC entry count"
            )

        if (
            title_results is not None
            and len(title_results) != len(entries)
        ):
            raise ValueError(
                "Title result count does not match TOC entry count"
            )

        merged: list[dict[str, Any]] = []

        for index, href_result in enumerate(href_results):
            if self.is_matched(href_result):
                selected = copy.deepcopy(href_result)
                selected["matchSource"] = "href"
                merged.append(selected)
                continue

            title_result = (
                title_results[index]
                if title_results is not None
                else None
            )

            if title_result is not None:
                selected = copy.deepcopy(title_result)
                selected["matchSource"] = "title"
                selected["hrefStatus"] = one_line(
                    href_result.get("status")
                )
                merged.append(selected)
                continue

            selected = copy.deepcopy(href_result)
            selected["matchSource"] = "href"
            merged.append(selected)

        return merged

    def repair_merged_boundaries(
        self,
        page: Page,
        entries: list[TocEntry],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = [
            {
                "title": entry.title,
                "status": one_line(result.get("status")),
                "method": one_line(result.get("method")),
                "nextIndex": result.get("nextIndex"),
                "headingStart": result.get("headingStart"),
                "contentStart": result.get("contentStart"),
                "endPosition": result.get("endPosition"),
            }
            for entry, result in zip(entries, results)
        ]

        repairs = page.evaluate(
            self.RESLICE_MERGED_RESULTS_JS,
            payload,
        )

        if (
            not isinstance(repairs, list)
            or len(repairs) != len(results)
        ):
            raise ValueError(
                "Merged boundary repair returned an invalid result count"
            )

        repaired_results = copy.deepcopy(results)

        for index, repair in enumerate(repairs):
            if not isinstance(repair, dict):
                continue

            repaired_results[index]["text"] = repair.get(
                "text",
                repaired_results[index].get("text", ""),
            )
            repaired_results[index]["endPosition"] = repair.get(
                "endPosition",
                repaired_results[index].get("endPosition"),
            )
            repaired_results[index]["nextIndex"] = repair.get(
                "nextIndex",
                repaired_results[index].get("nextIndex"),
            )
            repaired_results[index]["boundaryRepaired"] = True

        return repaired_results

    def build_output_json(
        self,
        toc: list[dict[str, Any]],
        entries: list[TocEntry],
        results: list[dict[str, Any]],
        preamble_text: str = "",
    ) -> list[dict[str, Any]]:
        output_toc = copy.deepcopy(toc)

        for entry, result in zip(entries, results):
            node = get_output_node(
                output_toc,
                entry.path,
            )

            text = str(result.get("text") or "")
            text = text.replace("\r\n", "\n")
            text = text.replace("\r", "\n")
            text = re.sub(
                r"\n\nTABLE OF CONTENTS?\b",
                "",
                text,
                flags=re.IGNORECASE,
            )

            node["content"] = normalize_text(text)

            status = (
                one_line(result.get("status"))
                or "unknown"
            )

            source = one_line(
                result.get("matchSource")
            )

            if status == "matched" and source == "href":
                node["match-status"] = "matched by href"
            elif status == "matched" and source == "title":
                node["match-status"] = "matched by title"
            else:
                node["match-status"] = status

            if entry.level == 1 and "subsection" in node:
                node["subsection"] = node.pop("subsection")

        preamble_node = {
            "title": "Preamble",
            "page-no": "1",
            "content": preamble_text,
            "match-status": "",
        }

        return [preamble_node] + output_toc

    def write_toc_debug(
        self,
        url: str,
        accession: str,
        output_dir: Path,
        diagnostics: dict[str, Any],
    ) -> Path:
        boundary: Boundary = diagnostics["boundary"]
        blocks = diagnostics["blocks"]

        debug_path = output_dir / f"{accession}_debug.json"
        debug_payload = {
            "url": url,
            "boundary": {
                "start_block": boundary.start_block,
                "end_block": boundary.end_block,
                "start_reason": boundary.start_reason,
                "end_reason": boundary.end_reason,
            },
            "rows": diagnostics["rows"],
            "start_context": blocks[
                max(0, boundary.start_block - 3):
                boundary.start_block + 5
            ],
            "end_context": blocks[
                max(0, boundary.end_block - 3):
                boundary.end_block + 5
            ],
            "blocks": blocks,
        }

        debug_path.write_text(
            json.dumps(
                debug_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return debug_path

    def write_text_debug(
        self,
        url: str,
        accession: str,
        output_dir: Path,
        entries: list[TocEntry],
        href_results: list[dict[str, Any]],
        title_results: list[dict[str, Any]] | None,
        final_results: list[dict[str, Any]],
        validation_issues: list[dict[str, Any]],
        extraction_mode: str,
    ) -> Path:
        debug_path = output_dir / f"{accession}_text_debug.json"

        entry_payloads: list[dict[str, Any]] = []

        for index, (entry, result) in enumerate(
            zip(entries, final_results)
        ):
            href_result = href_results[index]
            title_result = (
                title_results[index]
                if title_results is not None
                else None
            )

            entry_payloads.append(
                {
                    "title": entry.title,
                    "page-no": entry.page_no,
                    "href": entry.href,
                    "fragment": entry.fragment,
                    "level": entry.level,
                    "entry-type": entry.entry_type,
                    "annex-label": entry.annex_label,
                    "path": list(entry.path),
                    "match-source": one_line(
                        result.get("matchSource")
                    ),
                    "status": one_line(
                        result.get("status")
                    ),
                    "href-status": one_line(
                        href_result.get("status")
                    ),
                    "title-status": (
                        one_line(title_result.get("status"))
                        if title_result is not None
                        else "not-run"
                    ),
                    "resolution-method": one_line(
                        result.get("method")
                    ),
                    "target-tag": one_line(
                        result.get("targetTag")
                    ),
                    "next-entry-index": result.get(
                        "nextIndex"
                    ),
                    "anchor-position": result.get(
                        "anchorPosition"
                    ),
                    "heading-start": result.get(
                        "headingStart"
                    ),
                    "content-start": result.get(
                        "contentStart"
                    ),
                    "end-position": result.get(
                        "endPosition"
                    ),
                    "boundary-repaired": bool(
                        result.get("boundaryRepaired")
                    ),
                    "candidate-score": result.get(
                        "candidateScore"
                    ),
                    "candidate-count": result.get(
                        "candidateCount"
                    ),
                    "candidate-evidence": (
                        result.get("candidateEvidence")
                        or []
                    ),
                    "matched-variant": one_line(
                        result.get("matchedVariant")
                    ),
                    "matched-keywords": (
                        result.get("matchedKeywords")
                        or []
                    ),
                    "similarity": result.get(
                        "similarity"
                    ),
                    "front-toc-end": result.get(
                        "frontTocEnd"
                    ),
                    "validation-errors": (
                        result.get("validationErrors")
                        or []
                    ),
                    "text-length": len(
                        normalize_text(result.get("text"))
                    ),
                    "error": one_line(
                        result.get("error")
                    ),
                }
            )

        matched_count = sum(
            self.is_matched(result)
            for result in final_results
        )

        debug_payload = {
            "url": url,
            "summary": {
                "extraction-mode": extraction_mode,
                "title-count": len(entries),
                "matched-count": matched_count,
                "not-matched-count": (
                    len(entries) - matched_count
                ),
                "href-matched-count": sum(
                    self.is_matched(result)
                    for result in href_results
                ),
                "title-fallback-run": (
                    title_results is not None
                ),
                "title-fallback-matched-count": sum(
                    self.is_matched(result)
                    for result in (title_results or [])
                ),
                "validation-issue-count": len(
                    validation_issues
                ),
            },
            "validation-issues": validation_issues,
            "entries": entry_payloads,
        }

        debug_path.write_text(
            json.dumps(
                debug_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return debug_path

    def process(
        self,
        url: str,
        output_dir: Path = OUTPUT_DIR,
        debug: bool = False,
    ) -> dict[str, Any]:
        """
        Process one SEC document and return compact API-friendly logs.

        Logging is intentionally summarized into four stages:
          - toc
          - href
          - title_fallback
          - final

        `final.failed_titles` contains only TOC titles whose text could not be
        extracted in the final merged result (i.e. href did not recover them
        and title fallback did not recover them either).
        """
        accession = accession_from_url(url)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "parser=claude start url=%s accession=%s output_dir=%s",
            url,
            accession,
            output_dir,
        )
        last_error = ""
        last_stage = "initialization"
        last_logs: dict[str, Any] = {}

        def fresh_logs() -> dict[str, Any]:
            return {
                "toc": {
                    "status": "not_run",
                    "total": 0,
                },
                "href": {
                    "status": "not_run",
                    "matched": 0,
                    "failed": 0,
                },
                "title_fallback": {
                    "status": "not_run",
                    "attempted": 0,
                    "recovered": 0,
                    "failed": 0,
                },
                "final": {
                    "status": "not_run",
                    "matched": 0,
                    "failed": 0,
                    "total": 0,
                    "failed_titles": [],
                },
            }

        def match_status(matched: int, total: int) -> str:
            if total <= 0:
                return "failed"
            if matched == total:
                return "success"
            if matched > 0:
                return "partial"
            return "failed"

        for attempt in range(1, MAX_RETRY + 1):
            logs = fresh_logs()
            stage = "fetch-html"

            try:
                html = fetch_html(url)

                stage = "playwright-start"
                with sync_playwright() as playwright:
                    stage = "browser-launch"
                    browser = playwright.chromium.launch(
                        **browser_launch_kwargs()
                    )

                    try:
                        stage = "browser-context"
                        context = browser.new_context(
                            user_agent=SEC_USER_AGENT,
                            viewport={
                                "width": 1440,
                                "height": 1800,
                            },
                        )
                        page = context.new_page()

                        stage = "render-html"
                        render_html(page, html)

                        # ----------------------------------------------------
                        # TOC
                        # ----------------------------------------------------
                        stage = "toc-extraction"
                        toc, toc_diagnostics = (
                            self.toc_extractor.extract(page)
                        )

                        stage = "toc-flatten"
                        entries = flatten_toc(toc)

                        stage = "preamble-extraction"
                        preamble_start = (
                            toc_diagnostics["boundary"].start_block
                            if "boundary" in toc_diagnostics
                            else 0
                        )
                        preamble_text = self.preamble_extractor.extract(
                            toc_diagnostics.get("blocks", []),
                            preamble_start,
                        )

                        logs["toc"] = {
                            "status": "success",
                            "total": len(entries),
                        }

                        # ----------------------------------------------------
                        # HREF extraction
                        # ----------------------------------------------------
                        stage = "href-extraction"
                        href_results = (
                            self.href_extractor.extract(
                                page,
                                entries,
                            )
                        )

                        href_matched_count = sum(
                            self.is_matched(result)
                            for result in href_results
                        )
                        href_failed_indexes = [
                            index
                            for index, result
                            in enumerate(href_results)
                            if not self.is_matched(result)
                        ]
                        href_failed_count = len(href_failed_indexes)
                        href_failed = href_failed_count > 0

                        logs["href"] = {
                            "status": match_status(
                                href_matched_count,
                                len(entries),
                            ),
                            "matched": href_matched_count,
                            "failed": href_failed_count,
                        }

                        # ----------------------------------------------------
                        # Title fallback
                        # ----------------------------------------------------
                        title_results: (
                            list[dict[str, Any]] | None
                        ) = None
                        validation_issues: list[
                            dict[str, Any]
                        ] = []

                        if href_failed:
                            stage = "title-fallback-extraction"
                            title_results = (
                                self.title_extractor.extract(
                                    page,
                                    entries,
                                )
                            )

                            stage = "title-validation"
                            validation_issues = (
                                self.title_extractor.validate(
                                    entries,
                                    title_results,
                                )
                            )

                            # Only count recovery for entries that actually
                            # failed HREF extraction. Title results for entries
                            # already matched by HREF do not affect fallback
                            # success/failure.
                            recovered_indexes = [
                                index
                                for index in href_failed_indexes
                                if (
                                    index < len(title_results)
                                    and self.is_matched(
                                        title_results[index]
                                    )
                                )
                            ]
                            recovered_count = len(recovered_indexes)
                            fallback_failed_count = (
                                href_failed_count - recovered_count
                            )

                            if recovered_count == href_failed_count:
                                fallback_status = "success"
                            elif recovered_count > 0:
                                fallback_status = "partial"
                            else:
                                fallback_status = "failed"

                            logs["title_fallback"] = {
                                "status": fallback_status,
                                "attempted": href_failed_count,
                                "recovered": recovered_count,
                                "failed": fallback_failed_count,
                            }

                            extraction_mode = (
                                "href-with-title-fallback"
                            )
                        else:
                            logs["title_fallback"] = {
                                "status": "not_required",
                                "attempted": 0,
                                "recovered": 0,
                                "failed": 0,
                            }
                            extraction_mode = "href-only"

                        # ----------------------------------------------------
                        # Merge + final boundaries
                        # ----------------------------------------------------
                        stage = "merge-results"
                        final_results = self.merge_results(
                            entries,
                            href_results,
                            title_results,
                        )

                        if title_results is not None:
                            stage = "boundary-repair"
                            final_results = (
                                self.repair_merged_boundaries(
                                    page,
                                    entries,
                                    final_results,
                                )
                            )

                        matched_count = sum(
                            self.is_matched(result)
                            for result in final_results
                        )
                        failed_indexes = [
                            index
                            for index, result
                            in enumerate(final_results)
                            if not self.is_matched(result)
                        ]
                        failed_titles = [
                            entries[index].title
                            for index in failed_indexes
                        ]

                        logs["final"] = {
                            "status": match_status(
                                matched_count,
                                len(entries),
                            ),
                            "matched": matched_count,
                            "failed": len(failed_indexes),
                            "total": len(entries),
                            "failed_titles": failed_titles,
                        }

                    finally:
                        browser.close()

                # ------------------------------------------------------------
                # Output files
                # ------------------------------------------------------------
                stage = "write-toc-output"
                toc_path = output_dir / f"{accession}_toc.json"
                toc_path.write_text(
                    json.dumps(
                        toc,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                stage = "build-output-json"
                output_json = self.build_output_json(
                    toc,
                    entries,
                    final_results,
                    preamble_text=preamble_text,
                )

                stage = "write-text-output"
                output_path = (
                    output_dir /
                    f"{accession}_text.json"
                )
                output_path.write_text(
                    json.dumps(
                        output_json,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                toc_debug_path: Path | None = None
                text_debug_path: Path | None = None

                if debug:
                    stage = "write-toc-debug"
                    toc_debug_path = self.write_toc_debug(
                        url,
                        accession,
                        output_dir,
                        toc_diagnostics,
                    )

                    stage = "write-text-debug"
                    text_debug_path = self.write_text_debug(
                        url,
                        accession,
                        output_dir,
                        entries,
                        href_results,
                        title_results,
                        final_results,
                        validation_issues,
                        extraction_mode,
                    )

                if matched_count == len(entries):
                    status = "success"
                elif matched_count > 0:
                    status = "partial"
                else:
                    status = "error"

                title_matched_count = sum(
                    self.is_matched(result)
                    for result in (title_results or [])
                )

                logger.info(
                    "parser=claude done status=%s accession=%s "
                    "matched=%s/%s mode=%s output=%s",
                    status,
                    accession,
                    matched_count,
                    len(entries),
                    extraction_mode,
                    output_path,
                )
                logger.info(
                    "parser=claude logs=%s",
                    json.dumps(
                        logs,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )

                return {
                    "status": status,
                    "url": url,
                    "accession": accession,
                    "toc_output": str(toc_path),
                    "text_output": str(output_path),
                    "toc_debug": (
                        str(toc_debug_path)
                        if toc_debug_path
                        else None
                    ),
                    "text_debug": (
                        str(text_debug_path)
                        if text_debug_path
                        else None
                    ),
                    "title_count": len(entries),
                    "matched_count": matched_count,
                    "not_matched_count": (
                        len(entries) - matched_count
                    ),
                    "href_matched_count": href_matched_count,
                    "title_matched_count": title_matched_count,
                    "extraction_mode": extraction_mode,
                    "logs": logs,
                }

            except Exception as error:
                last_error = str(error)
                last_stage = stage

                # Mark only the stage that actually failed. Keep the payload
                # small so it can be returned directly by an API.
                if stage in {"toc-extraction", "toc-flatten"}:
                    logs["toc"] = {
                        "status": "failed",
                        "total": 0,
                    }
                elif stage == "href-extraction":
                    logs["href"] = {
                        "status": "failed",
                        "matched": 0,
                        "failed": logs["toc"].get("total", 0),
                    }
                elif stage in {
                    "title-fallback-extraction",
                    "title-validation",
                }:
                    attempted = logs["href"].get("failed", 0)
                    logs["title_fallback"] = {
                        "status": "failed",
                        "attempted": attempted,
                        "recovered": 0,
                        "failed": attempted,
                    }

                logs["error"] = {
                    "stage": stage,
                    "message": last_error,
                }
                last_logs = logs

                if attempt < MAX_RETRY:
                    time.sleep(2)

        if not last_logs:
            last_logs = fresh_logs()
            last_logs["error"] = {
                "stage": last_stage,
                "message": last_error,
            }

        logger.warning(
            "parser=claude failed accession=%s stage=%s reason=%s",
            accession,
            last_stage,
            last_error,
        )
        return {
            "status": "error",
            "url": url,
            "accession": accession,
            "reason": last_error,
            "logs": last_logs,
        }


# ============================================================
# 6. CLI / MAIN
# ============================================================


def read_urls(
    args: argparse.Namespace,
) -> list[str]:
    urls = list(args.urls or URLS)

    if args.input_file:
        urls.extend(
            line.strip()
            for line in Path(
                args.input_file
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            if (
                line.strip()
                and not line.lstrip().startswith("#")
            )
        )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SEC TOC and section text "
            "using href extraction with "
            "title-based fallback"
        )
    )

    parser.add_argument(
        "urls",
        nargs="*",
        help="SEC document URL(s)",
    )

    parser.add_argument(
        "--input-file",
        help=(
            "Text file containing one "
            "SEC URL per line"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for generated JSON files",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save TOC, extraction, validation, "
            "and boundary diagnostics"
        ),
    )

    args = parser.parse_args()

    urls = read_urls(args)

    if not urls:
        parser.error(
            "Provide at least one URL, "
            "--input-file, or populate URLS"
        )

    pipeline = DocumentExtractionPipeline()

    successes = 0
    partials = 0
    failures = 0

    for index, url in enumerate(
        urls,
        start=1,
    ):
        result = pipeline.process(
            url=url,
            output_dir=Path(args.output_dir),
            debug=args.debug,
        )

        status = result.get("status")

        logger.info(
            "LOGS %s",
            json.dumps(
                result.get("logs", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        if status == "success":
            successes += 1

            logger.info(
                "[%s/%s] SUCCESS %s | matched=%s/%s | mode=%s | output=%s",
                index,
                len(urls),
                result["accession"],
                result["matched_count"],
                result["title_count"],
                result["extraction_mode"],
                result["text_output"],
            )

        elif status == "partial":
            partials += 1

            logger.info(
                "[%s/%s] PARTIAL %s | matched=%s/%s | mode=%s | output=%s",
                index,
                len(urls),
                result["accession"],
                result["matched_count"],
                result["title_count"],
                result["extraction_mode"],
                result["text_output"],
            )

        else:
            failures += 1

            logger.error(
                "[%s/%s] FAILED %s | %s",
                index,
                len(urls),
                result.get("accession", "unknown"),
                result.get("reason", "Unknown error"),
            )

    logger.info(
        "Done. Success=%s, Partial=%s, Failed=%s",
        successes,
        partials,
        failures,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
