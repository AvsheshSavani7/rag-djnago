"""
8-K Filing Summarizer — Multi-level summaries via Claude API.

Can be used as a function: summarize_8k_filing(url, output_path)
Or run as script: python -m sec_rss_parser.8k_summary <url> <output_path>
"""

from pathlib import Path
import os
import sys
import json
import re
from typing import Union

import anthropic

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install",
         "requests", "beautifulsoup4", "python-dotenv", "python-docx", "-q"]
    )
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

# Load .env from project root (rag_project/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


SUMMARY_PROMPT = """You are an expert analyst summarizing SEC 8-K filings for a merger arbitrage desk.

Given the 8-K text below, produce summaries at 3 levels. Respond ONLY in valid JSON (no markdown fences).

{
  "ticker": "<ticker symbol>",
  "filing_date": "<MM/DD/YY>",
  "items_reported": ["<Item numbers, e.g. Item 5.07, Item 8.01>"],
  
  "L1_headline": "<ticker> – <key event in ≤8 words>. | <date>",
  
  "L2_brief": "<2-3 sentence summary covering: what happened, key numbers, what it means for the deal>",
  
  "L3_detailed": {
    "event": "<what happened>",
    "key_figures": ["<vote %, dollar amounts, dates, conditions>"],
    "deal_implications": "<impact on deal timeline/probability>",
    "remaining_conditions": ["<what still needs to happen>"],
    "risks_flagged": ["<any risks, litigation, regulatory issues>"]
  }
}

Rules:
- L1 format MUST be: + <TICKER> – <event>. | <date>
- For merger-related 8-Ks, focus on deal probability impact
- Extract exact vote percentages, dollar figures, dates
- Flag any conditions precedent still outstanding
- Note any litigation or regulatory mentions

8-K TEXT:
"""


def _filename_from_sec_url(url: str) -> str:
    """Extract base filename from SEC filing URL (last path segment, no extension).
    e.g. .../form4-02092026_110254.xml -> form4-02092026_110254
    """
    path = url.rstrip("/").split("/")[-1]
    return path.rsplit(".", 1)[0] if "." in path else path


def _get_s3_client():
    """S3 client using same bucket/env as document_processor S3Service."""
    if not boto3:
        raise ValueError(
            "boto3 is required for S3 upload. Install with: pip install boto3")
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is not set in environment")
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    ), bucket


def _upload_to_s3(
    local_file_path: Union[str, Path],
    s3_key: str,
    content_type: str,
) -> str:
    """Upload a file to S3 using same bucket/env as document_processor S3Service. Returns S3 URL."""
    client, bucket = _get_s3_client()
    client.upload_file(
        str(local_file_path),
        bucket,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"https://{bucket}.s3.amazonaws.com/{s3_key}"


def _upload_json_to_s3(data: dict, s3_key: str) -> str:
    """Upload JSON to S3 and return the full URL (same pattern as S3Service.upload_json)."""
    client, bucket = _get_s3_client()
    body = json.dumps(data, indent=2).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=body,
        ContentType="application/json",
    )
    return f"https://{bucket}.s3.amazonaws.com/{s3_key}"


def fetch_8k_text(source: str) -> str:
    """Fetch and extract text from an 8-K filing (URL or local file)."""
    if source.startswith("http"):
        headers = {"User-Agent": "ResearchBot/1.0 (research@example.com)"}
        resp = requests.get(source, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text
    else:
        html = Path(source).read_text()

    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Truncate to ~6k words to stay within token budget
    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000])

    return text


def summarize(text: str, model: str = "claude-opus-4-5-20251101") -> dict:
    """Call Claude API to produce multi-level summary."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in your .env or environment."
        )
    client = anthropic.Anthropic()

    msg = client.messages.create(
        model=model,
        max_tokens=2500,
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT + "\n\n" + text
        }]
    )

    raw = msg.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def print_summary(s: dict):
    """Pretty-print the multi-level summary."""
    print("\n" + "=" * 70)
    print("  8-K SUMMARY")
    print("=" * 70)

    # L1 — Headline
    print(f"\n📌 L1 | HEADLINE")
    print(f"   {s['L1_headline']}")

    # L2 — Brief
    print(f"\n📋 L2 | BRIEF")
    print(f"   {s['L2_brief']}")

    # L3 — Detailed
    d = s["L3_detailed"]
    print(f"\n📊 L3 | DETAILED")
    print(f"   Event:       {d['event']}")
    print(f"   Key Figures:")
    for f in d["key_figures"]:
        print(f"     • {f}")
    print(f"   Deal Impact: {d['deal_implications']}")
    if d.get("remaining_conditions"):
        print(f"   Remaining Conditions:")
        for c in d["remaining_conditions"]:
            print(f"     • {c}")
    if d.get("risks_flagged"):
        print(f"   Risks:")
        for r in d["risks_flagged"]:
            print(f"     • {r}")

    print(f"\n   Items: {', '.join(s.get('items_reported', []))}")
    print("=" * 70)


def export_docx(s: dict, filepath: str = None):
    """Export summary to a formatted Word document."""
    ticker = s.get("ticker", "UNKNOWN")
    date = s.get("filing_date", "")

    if filepath is None:
        filepath = f"8K_Summary_{ticker}_{date.replace('/', '-')}.docx"

    doc = DocxDocument()

    # -- Styles --
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    # -- Title --
    title = doc.add_heading(f"8-K Summary: {ticker}", level=0)
    title.runs[0].font.size = Pt(20)

    # Filing metadata
    meta = doc.add_paragraph()
    meta.add_run(f"Filing Date: ").bold = True
    meta.add_run(date)
    meta.add_run(f"    Items: ").bold = True
    meta.add_run(", ".join(s.get("items_reported", [])))

    # -- L1: Headline --
    doc.add_heading("L1 — Headline", level=1)
    p = doc.add_paragraph()
    run = p.add_run(s["L1_headline"])
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0, 51, 102)

    # -- L2: Brief --
    doc.add_heading("L2 — Brief", level=1)
    doc.add_paragraph(s["L2_brief"])

    # -- L3: Detailed --
    doc.add_heading("L3 — Detailed", level=1)
    d = s["L3_detailed"]

    doc.add_heading("Event", level=2)
    doc.add_paragraph(d["event"])

    doc.add_heading("Key Figures", level=2)
    for fig in d.get("key_figures", []):
        doc.add_paragraph(fig, style="List Bullet")

    doc.add_heading("Deal Implications", level=2)
    doc.add_paragraph(d.get("deal_implications", "N/A"))

    if d.get("remaining_conditions"):
        doc.add_heading("Remaining Conditions", level=2)
        for c in d["remaining_conditions"]:
            doc.add_paragraph(c, style="List Bullet")

    if d.get("risks_flagged"):
        doc.add_heading("Risks Flagged", level=2)
        for r in d["risks_flagged"]:
            doc.add_paragraph(r, style="List Bullet")

    doc.save(filepath)
    return filepath


def summarize_8k_filing(
    url: str,
    output_path: Union[str, Path],
    *,
    verbose: bool = True,
    model: str = "claude-opus-4-5-20251101",
    upload_to_s3: bool = True,
    s3_folder: str = "8k",
) -> dict:
    """
    Fetch an 8-K from the given URL, summarize it via Claude, save JSON + DOCX, and optionally upload to S3.

    Args:
        url: SEC 8-K filing URL (or path to local HTML/file).
        output_path: Directory to save outputs, or path to a specific file.
                     If directory: saves 8k_summary.json and 8K_Summary_<ticker>_<date>.docx.
                     If file path (e.g. .docx or .json): uses that path for one file and
                     places the other in the same directory with default names.
        verbose: If True, print progress and the summary to stdout.
        model: Claude model name.
        upload_to_s3: If True, upload DOCX (and JSON) to S3. S3 key is {s3_folder}/{filename}.docx
                      where filename is derived from the URL (e.g. form4-02092026_110254).
        s3_folder: S3 prefix/folder for the file, e.g. "8k" or "99_1". Default "8k".

    Returns:
        Summary dict with keys: ticker, filing_date, items_reported,
        L1_headline, L2_brief, L3_detailed. If upload_to_s3 is True, also includes
        "s3_url" (URL of the uploaded DOCX) and "s3_json_url" (URL of the uploaded JSON).
    """
    output_path = Path(output_path)
    if output_path.suffix in (".json", ".docx"):
        out_dir = output_path.parent
        single_file = output_path
    else:
        out_dir = output_path
        single_file = None

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Fetching 8-K from: {url}")
    text = fetch_8k_text(url)
    if verbose:
        print(f"Extracted {len(text.split())} words of text")
        print("Generating summary via Claude Opus 4.5...")

    result = summarize(text, model=model)

    if verbose:
        print_summary(result)

    # Save JSON
    if single_file and single_file.suffix == ".json":
        json_path = single_file
    else:
        json_path = out_dir / "8k_summary.json"
    json_path.write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"\nRaw JSON saved to: {json_path}")

    # Save DOCX
    ticker = result.get("ticker", "UNKNOWN")
    date = result.get("filing_date", "")
    safe_ticker = re.sub(r"[^\w\-.]", "_", ticker)
    safe_date = date.replace("/", "-")
    default_docx_name = f"8K_Summary_{safe_ticker}_{safe_date}.docx"

    if single_file and single_file.suffix == ".docx":
        docx_path = single_file
    else:
        docx_path = out_dir / default_docx_name
    export_docx(result, str(docx_path))
    if verbose:
        print(f"DOCX saved to: {docx_path}")

    if upload_to_s3:
        base_name = _filename_from_sec_url(url)
        docx_key = f"{s3_folder}/{base_name}.docx"
        json_key = f"{s3_folder}/{base_name}.json"
        try:
            result["s3_url"] = _upload_to_s3(
                docx_path, docx_key, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            if verbose:
                print(f"Uploaded DOCX to S3: {result['s3_url']}")
            # Upload JSON to same folder
            result["s3_json_url"] = _upload_json_to_s3(
                result, json_key
            )
            if verbose:
                print(f"Uploaded JSON to S3: {result['s3_json_url']}")
        except Exception as e:
            if verbose:
                print(f"S3 upload failed: {e}", file=sys.stderr)
            result["s3_url"] = None
            result["s3_json_url"] = None

    return result


def main():
    if len(sys.argv) >= 3:
        url = sys.argv[1]
        output_path = sys.argv[2]
    else:
        url = "https://www.sec.gov/Archives/edgar/data/2022236/000202223626000002/xslF345X05/form4-02092026_110254.xml"
        output_path = ""
        print(f"Usage: python -m sec_rss_parser.8k_summary <url> <output_path>")
        print(f"Using defaults: url={url!r}, output_path={output_path}\n")

    try:
        summarize_8k_filing(url, output_path)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
