import sys
from pathlib import Path

# Add project root so sec_rss_parser is importable (run from rag_project or set PYTHONPATH)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

past_doc = {
    "_id": "7047e871-2860-4b87-8282-f92d3b11b013",
    "accession_number": "3c144fdd-dd6a-4285-8f49-12514c846213",
    "cik_number": "0001685040",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/1685040/000114036125046438/ny20060599x1_prem14a.htm",
    "filing_date": {
        "$date": "2025-12-23T00:00:00.000Z"
    },
    "deal_id": "69303b08c0aa46c32884888a",
    "created_at": {
        "$date": "2026-01-09T07:51:47.618Z"
    },
    "updated_at": {
        "$date": "2026-03-05T05:32:34.237Z"
    },
    "form_type": "PREM14A",
    "proxy": {
        "proxy_parsing_status": "completed",
        "empty_percentage": 5.263157894736842,
        "processing_state": {
            "pdf_created": True,
            "toc_found": True,
            "toc_extracted": True,
            "sections_extracted": True,
            "empty_percentage": 5.263157894736842,
            "iteration_count": 2
        },
        "s3_urls": {
            "pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf/ny20060599x1_prem14a.pdf",
            "toc_pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf-toc/ny20060599x1_prem14a_toc_pages.pdf",
            "toc_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/table_of_contents_new_ny20060599x1_prem14a.json",
            "sections_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/sections_with_content_html_ny20060599x1_prem14a.json"
        },
        "pinecone_processing_status": "completed",
        "pinecone_processed_at": {
            "$date": "2026-01-09T08:02:46.119Z"
        },
        "pinecone_error_message": None
    },
    "ten_k_ten_q": None,
    "8_k": None,
    "other_filings": None
}
latest_doc = {
    "_id": "30944bc3-ccae-4d40-861d-5adc84f23731",
    "accession_number": "82cab331-8bb8-47d4-afc2-78dfe9294779",
    "cik_number": "0001685040",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/1685040/000114036126000522/ny20060599x2_defm14a.htm",
    "filing_date": {
        "$date": "2026-01-07T00:00:00.000Z"
    },
    "deal_id": "69303b08c0aa46c32884888a",
    "created_at": {
        "$date": "2026-01-07T22:02:08.100Z"
    },
    "updated_at": {
        "$date": "2026-03-05T05:32:34.173Z"
    },
    "form_type": "DEFM14A",
    "proxy": {
        "proxy_parsing_status": "completed",
        "empty_percentage": 10.526315789473683,
        "processing_state": {
            "pdf_created": True,
            "toc_found": True,
            "toc_extracted": True,
            "sections_extracted": True,
            "empty_percentage": 10.526315789473683,
            "iteration_count": 3
        },
        "s3_urls": {
            "pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf/ny20060599x2_defm14a.pdf",
            "toc_pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf-toc/ny20060599x2_defm14a_toc_pages.pdf",
            "toc_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/table_of_contents_new_ny20060599x2_defm14a.json",
            "sections_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/sections_with_content_html_ny20060599x2_defm14a.json"
        },
        "pinecone_processing_status": "completed",
        "pinecone_processed_at": {
            "$date": "2026-01-07T22:09:14.046Z"
        },
        "pinecone_error_message": None
    },
    "ten_k_ten_q": None,
    "8_k": None,
    "other_filings": None
}

if __name__ == "__main__":
    from sec_rss_parser.proxy_comparision.orchestrator import run_comparison

    # S3 + MongoDB only. Ensure .env has ANTHROPIC_API_KEY, AWS_*, MONGODB_CONNECTION_STRING.
    result = run_comparison(
        latest_doc_record=latest_doc,
        past_doc_record=past_doc,
        env_path=Path(__file__).resolve().parent.parent.parent / ".env",
    )
    print(result)
