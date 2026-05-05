"""
Quick test: parse the DMA DOCX, generate the 8-K summary email HTML, save locally.
"""
import sys
import os

# Allow direct imports from sec_rss_parser without Django setup
sys.path.insert(0, os.path.dirname(__file__))

from sec_rss_parser.docx_parser import parse_dma_summary_docx
from sec_rss_parser.email_templates import generate_8k_summary_email_html

DOCX_URL = "https://rag-mna-doc.s3.amazonaws.com/summaries-engine/0001104659-26-054379_tm2613446d1_ex2-1.docx"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "test_8k_email_preview.html")

print(f"Parsing DOCX from: {DOCX_URL}")
parsed = parse_dma_summary_docx(DOCX_URL)
concise_sections = parsed.get("concise_sections", [])
fulsome_sections = parsed.get("fulsome_sections", [])

print(f"  concise_sections: {len(concise_sections)} section(s)")
print(f"  fulsome_sections: {len(fulsome_sections)} section(s)")

if concise_sections:
    first = concise_sections[0]
    print(f"  First section: '{first['name']}' — {len(first['clauses'])} clause(s)")

print("\nGenerating email HTML...")
subject, html_email = generate_8k_summary_email_html(
    company_name="Modiv Industrial, Inc.",
    form_type="8-K",
    summary_doc_url=DOCX_URL,
    cik_number="0001234567",
    sec_url="https://www.sec.gov/Archives/edgar/data/1234567/000110465926054379/0001104659-26-054379-index.htm",
    accession_number="0001104659-26-054379",
    summary_kind="DMA",
    concise_sections=concise_sections,
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_email)

print(f"Subject : {subject}")
print(f"HTML saved to: {OUTPUT_FILE}")
print("\nOpen the file in your browser to preview the email.")
