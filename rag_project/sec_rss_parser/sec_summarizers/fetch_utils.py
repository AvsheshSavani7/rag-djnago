"""
Shared text extraction utility for SEC filing summarizers.
Handles HTML pages, local files, and PDFs transparently.
"""

import logging
import re
import sys
from pathlib import Path
from io import BytesIO

import requests
from bs4 import BeautifulSoup

# Suppress noisy pypdf warnings about malformed PDF cross-references
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Auto-install pypdf if missing (matches existing pattern in summarizers)
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pypdf", "-q"])
    import pypdf


def fetch_text(source: str, word_limit: int = 10000) -> str:
    """
    Fetch and extract text from a URL or local file.
    Automatically handles HTML pages and PDFs.
    """
    is_url = source.startswith("http")
    is_pdf = source.lower().endswith(".pdf")

    if is_url:
        headers = {
            "User-Agent": "MergerArbDashboard/1.0 (merger-arb-research@outlook.com)"}

        resp = requests.get(source, headers=headers, timeout=60)
        resp.raise_for_status()

        # Detect PDF from content-type or URL
        content_type = resp.headers.get("Content-Type", "")
        if is_pdf or "application/pdf" in content_type:
            return _extract_pdf_bytes(resp.content, word_limit)

        raw = resp.text
    else:
        path = Path(source)
        if path.suffix.lower() == ".pdf":
            return _extract_pdf_file(path, word_limit)
        raw = path.read_text()

    return _extract_html(raw, word_limit)


def _extract_html(html: str, word_limit: int) -> str:
    """Parse HTML and extract clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    words = text.split()
    if word_limit and len(words) > word_limit:

        text = " ".join(words[:word_limit])

    return text


def _extract_pdf_bytes(data: bytes, word_limit: int) -> str:
    """Extract text from PDF bytes (downloaded from URL)."""
    reader = pypdf.PdfReader(BytesIO(data))
    return _read_pdf_pages(reader, word_limit)


def _extract_pdf_file(path: Path, word_limit: int) -> str:
    """Extract text from a local PDF file."""
    reader = pypdf.PdfReader(str(path))
    return _read_pdf_pages(reader, word_limit)


def _read_pdf_pages(reader: pypdf.PdfReader, word_limit: int) -> str:
    """Read pages from a PdfReader and return cleaned text."""
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    text = "\n\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    words = text.split()
    if word_limit and len(words) > word_limit:

        text = " ".join(words[:word_limit])

    return text
