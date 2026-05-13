import sys
from pathlib import Path

# Add project root so sec_rss_parser is importable (run from rag_project or set PYTHONPATH)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

past_doc = {
    "_id": "1d6908ba-ad57-4c1a-8401-95ce049ea468",
    "accession_number": "0001140361-26-016785",
    "cik_number": "0000723254",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/723254/000114036126016785/ny20069194x1_s4.htm",
    "filing_date": {
        "$date": "2026-04-27T00:00:00.000Z"
    },
    "deal_id": "69b15c2254958e923c2cb92a",
    "created_at": {
        "$date": "2026-04-27T10:15:06.165Z"
    },
    "updated_at": {
        "$date": "2026-05-11T12:25:16.863Z"
    },
    "form_type": "S-4",
    "items_reported": [],
    "L1_headline": "+ UNF – S-4 filed for Cintas cash-and-stock merger. | 04/24/26",
    "L2_brief": "Cintas Corporation is acquiring UniFirst Corporation in a two-step merger whereby each share of UniFirst stock (common and class B) will be converted into $155.00 in cash plus 0.7720 shares of CTAS common stock. The Croatti family, controlling approximately two-thirds of UniFirst's combined voting power, has signed a voting and support agreement to vote in favor. The companies expect to close in the second half of calendar year 2026, subject to HSR clearance, one foreign antitrust approval, and UniFirst shareholder approval (two-thirds supermajority).",
    "L3_detailed": {
        "deal_structure": "Two-step merger: (1) Bruin Merger Sub I, Inc. (Delaware corp, wholly owned Cintas subsidiary) merges with and into UniFirst (UniFirst survives as wholly owned Cintas subsidiary); (2) immediately thereafter, UniFirst merges into Bruin Merger Sub II, LLC (Delaware LLC, wholly owned Cintas subsidiary). UniFirst ceases to exist as a separate entity.",
        "consideration": {
            "type": "Cash & Stock",
            "per_share_value": "$155.00 cash + 0.7720 shares of CTAS common stock per share of UniFirst stock (common and class B treated identically)",
            "exchange_ratio": "0.7720 shares of CTAS per share of UNF (fixed; not adjusted for stock price changes, but adjusted for stock splits, dividends, etc.)",
            "cash_component": "$155.00 per share",
            "total_deal_value": "Not explicitly stated as an aggregate dollar figure; approximately 18,473,xxx shares of UniFirst stock outstanding (implied from ~14,261,683 CTAS shares to be issued ÷ 0.7720 exchange ratio ≈ 18.47M UNF shares). Former UniFirst shareholders expected to hold ~3.4% of pro forma Cintas.",
            "premium": "Not explicitly stated in the filing with a reference price/date"
        },
        "conditions_precedent": [
            "UniFirst shareholder approval: affirmative vote of two-thirds of combined voting power of UniFirst common stock and class B common stock, voting together as a single class",
            "Expiration or early termination of HSR Act waiting period",
            "Authorization or consent from a certain foreign regulator under its antitrust laws",
            "Authorization or consent of applicable governmental authorities in respect of certain UniFirst permits",
            "Effectiveness of the Form S-4 registration statement (no stop order or pending/threatened written action seeking one)",
            "Authorization for listing on NASDAQ of CTAS common stock to be issued in the mergers",
            "Absence of any order or law enacted after the merger agreement date enjoining or prohibiting the mergers",
            "Accuracy of representations and warranties of the other party (subject to materiality standards in the merger agreement)",
            "Performance by the other party of its covenants and agreements in all material respects",
            "Absence of any event between signing and closing that would reasonably be expected to result in a material adverse effect for either party",
            "Delivery of officer's certificate by the other party certifying satisfaction of the foregoing conditions"
        ],
        "regulatory_approvals": {
            "required": [
                "HSR Act (DOJ/FTC) — U.S. antitrust",
                "One foreign antitrust/competition regulator (unspecified jurisdiction)",
                "Governmental authority approvals for certain UniFirst permits"
            ],
            "status": "HSR notification and report forms filed by both Cintas and UniFirst on April 8, 2026. Foreign and permit approvals pending."
        },
        "deal_protections": {
            "breakup_fee_target": "$213,300,000 (UniFirst termination fee), payable if: (1) Cintas terminates for UniFirst breach and a competing proposal was disclosed and not withdrawn, and within 12 months a competing transaction is consummated or a definitive agreement entered into; (2) termination due to Termination Date passing or failure to obtain UniFirst shareholder approval under similar competing proposal circumstances; (3) UniFirst terminates to accept a superior proposal; or (4) Cintas terminates due to UniFirst adverse recommendation change or related solicitation breaches.",
            "breakup_fee_acquirer": "$350,000,000 (Cintas termination fee), payable if: (1) termination due to Termination Date passing or permanent antitrust-related restraint, where antitrust conditions are the sole unsatisfied conditions; or (2) UniFirst terminates for Cintas' uncured material breach of regulatory/antitrust covenants. UniFirst must elect to accept or decline within 7 business days; accepting constitutes waiver of other claims including for knowing breach or fraud.",
            "go_shop": "None",
            "matching_rights": "Cintas has matching rights — UniFirst Board must allow Cintas to respond before changing recommendation or terminating for a superior proposal (specific negotiation period details from merger agreement not fully excerpted in filing text)",
            "no_shop": "UniFirst agreed to cease solicitation and not solicit, discuss, or enter into agreements regarding acquisition proposals. Exceptions exist for intervening events (allowing recommendation change) and superior proposals (allowing termination to accept), subject to compliance with notice and negotiation procedures and payment of UniFirst termination fee."
        },
        "expected_timeline": "Expected closing in the second half of calendar year 2026. HSR filings made April 8, 2026. S-4 filed April 24, 2026. Special meeting date and record date not yet set. Initial Termination Date is January 10, 2027, with two automatic four-month extensions available (up to September 10, 2027) if antitrust conditions are the sole impediment.",
        "shareholder_vote": "UniFirst shareholders only. Approval requires affirmative vote of two-thirds of combined voting power of UniFirst common stock (1 vote/share) and class B common stock (10 votes/share), voting together as a single class. Record date not yet set. Special meeting date not yet set. Croatti family (supporting shareholders) control approximately two-thirds of combined voting power and have agreed to vote in favor per the voting and support agreement, so approval is expected.",
        "pro_forma_highlights": [
            "Unaudited pro forma condensed combined financial statements are included in the filing (page 105) but specific combined revenue, EPS accretion/dilution, and synergy figures are not excerpted in the provided text.",
            "Former UniFirst shareholders expected to hold approximately 3.4% of outstanding Cintas common stock post-merger; current Cintas shareholders approximately 96.6%.",
            "Cintas expects to issue approximately 14,261,683 shares of CTAS common stock to UniFirst shareholders.",
            "Acquisition to be accounted for under ASC 805 (acquisition method); purchase price allocation is preliminary."
        ],
        "risk_factors": [
            "The fixed exchange ratio means the market value of the stock consideration will fluctuate with Cintas' stock price and could be worth less at closing than at announcement.",
            "Regulatory risk: mergers require HSR clearance and foreign antitrust approval; the Termination Date can extend up to September 2027 if antitrust conditions are not met.",
            "UniFirst shareholders have no appraisal rights under Massachusetts law.",
            "Completion of the mergers is not conditioned on the mergers qualifying as a tax-free reorganization under Section 368(a) of the Code; the IRS could challenge tax treatment.",
            "The voting and support agreement with the Croatti family has termination triggers including any adverse recommendation change by the UniFirst Board or certain amendments to the merger agreement, which could jeopardize the expected shareholder vote outcome."
        ],
        "fairness_opinion": "J.P. Morgan Securities LLC and Goldman Sachs & Co. LLC each delivered fairness opinions to the UniFirst Board on March 10, 2026, concluding that the merger consideration to be paid to holders of UniFirst stock was fair from a financial point of view. UniFirst agreed to pay each advisor an estimated transaction fee of approximately $42,000,000 ($5,000,000 payable at opinion delivery/signing; remainder contingent on consummation).",
        "background_summary": "A confidentiality agreement between Cintas and UniFirst was executed on January 26, 2025, indicating discussions began at least by that date. On March 10, 2026, UniFirst, Cintas, Merger Sub Inc., and Merger Sub LLC executed the merger agreement. Concurrently, the Croatti family (controlling approximately two-thirds of UniFirst's combined voting power) entered into a voting and support agreement with Cintas to vote in favor of the transaction. The UniFirst Board unanimously approved the merger agreement after receiving fairness opinions from both J.P. Morgan and Goldman Sachs. Detailed negotiation history is described in the 'Background of the Mergers' section beginning on page 37 of the filing."
    },
    "s3_docx_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_docx/S4_Summary_UNF_04-24-26_26016785.docx",
    "s3_json_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_json/s4_summary_26016785.json",
    "99_1": None,
    "proxy": {
        "proxy_parsing_status": "completed",
        "empty_percentage": 0,
        "processing_state": {
            "pdf_created": True,
            "toc_found": True,
            "toc_extracted": True,
            "sections_extracted": True,
            "empty_percentage": 0,
            "iteration_count": 0
        },
        "s3_urls": {
            "pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf/ny20069194x1_s4.pdf",
            "toc_pdf_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-pdf-toc/ny20069194x1_s4_toc_pages.pdf",
            "toc_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/table_of_contents_new_ny20069194x1_s4.json",
            "sections_json_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-parse-json/sections_with_content_html_ny20069194x1_s4.json"
        },
        "pinecone_processing_status": "completed",
        "pinecone_processed_at": {
            "$date": "2026-04-27T10:17:47.961Z"
        },
        "pinecone_error_message": None,
        "summary_generation_status": "completed",
        "summary_docx_url": "https://rag-mna-doc.s3.amazonaws.com/proxy-summaries/1d6908ba-ad57-4c1a-8401-95ce049ea468_summary.docx",
        "summary_generated_at": {
            "$date": "2026-04-27T10:24:39.690Z"
        },
        "agent_response": "The SEC document has been fully processed following the requested workflow. The Table of Contents was extracted and cleaned, all sections' content was extracted with 0% empty sections after applying content splitting corrections, and all tables in the final content were cleaned and formatted. The document is now ready for further analysis or use.",
        "error_message": None,
        "completed_at": {
            "$date": "2026-04-27T10:16:32.025Z"
        },
        "total_sections": 0,
        "empty_sections": 0,
        "iteration_count": 0,
        "company_name": "CINTAS CORP",
        "comparison": {
            "cache": {
                "status": "pending",
                "filing_date": "unknown",
                "form_family": "registration_like",
                "form_type": "S-4",
                "priority_facts_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_json/proxy_comp_69b15c2254958e923c2cb92a_1d6908ba-ad57-4c1a-8401-95ce049ea468_priority_facts.json",
                "sections_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_json/proxy_comp_69b15c2254958e923c2cb92a_1d6908ba-ad57-4c1a-8401-95ce049ea468_sections.json",
                "topic_blocks_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_json/proxy_comp_69b15c2254958e923c2cb92a_1d6908ba-ad57-4c1a-8401-95ce049ea468_topic_blocks.json"
            }
        }
    },
    "ten_k_ten_q": None,
    "8_k": None,
    "other_filings": None
}
latest_doc = {
    "_id": "f2456adb-44db-4241-86a0-4d50e5b62591",
    "accession_number": "0001140361-26-020380",
    "cik_number": "0000717954",
    "sec_document_url": "https://www.sec.gov/Archives/edgar/data/717954/000114036126020380/ny20072400x1_defm14a.htm",
    "filing_date": {
        "$date": "2026-05-11T00:00:00.000Z"
    },
    "deal_id": "69b15c2254958e923c2cb92a",
    "created_at": {
        "$date": "2026-05-11T12:23:14.739Z"
    },
    "updated_at": {
        "$date": "2026-05-11T12:29:07.234Z"
    },
    "form_type": "DEFM14A",
    "items_reported": [],
    "L1_headline": "+ UNF – Cintas merger vote set June 11; $155+0.772 CTAS/share. | 05/11/26",
    "L2_brief": "UniFirst filed a definitive merger proxy statement/prospectus for its acquisition by Cintas Corporation. Each share of UniFirst stock will be converted into $155.00 cash plus 0.7720 shares of Cintas common stock, with a shareholder vote scheduled for June 11, 2026 requiring two-thirds of combined voting power. The Croatti family, controlling approximately two-thirds of combined voting power, has entered a voting and support agreement to vote in favor, making approval expected.",
    "L3_detailed": {
        "filing_purpose": "Definitive proxy statement/prospectus filed by UniFirst in connection with its proposed acquisition by Cintas Corporation. The filing solicits UniFirst shareholder approval of the merger agreement dated March 10, 2026, and serves as a prospectus for the Cintas common stock to be issued as part of the merger consideration. Filed as part of Cintas' S-4 registration statement.",
        "key_information": [
            "Merger consideration is $155.00 cash plus 0.7720 shares of Cintas common stock per share of UniFirst stock (both common and class B), with the exchange ratio fixed and not subject to adjustment for stock price changes",
            "Croatti family-affiliated entities controlling approximately two-thirds of combined voting power signed a voting and support agreement to vote in favor of the merger, making shareholder approval expected",
            "Cintas will issue approximately 14,261,683 shares of Cintas common stock to UniFirst shareholders, resulting in former UniFirst holders owning approximately 3.4% of pro forma Cintas shares outstanding",
            "UniFirst termination fee payable to Cintas is $213,300,000; Cintas reverse termination fee payable to UniFirst is $350,000,000",
            "UniFirst shareholders are NOT entitled to appraisal rights in connection with the mergers; shareholder approval requires affirmative vote of two-thirds of combined voting power of common and class B stock voting as single class"
        ],
        "financial_highlights": [
            "Cash consideration: $155.00 per share of UniFirst stock",
            "Stock consideration: 0.7720 shares of Cintas common stock per share of UniFirst stock (fixed exchange ratio)",
            "UniFirst termination fee: $213,300,000",
            "Cintas reverse termination fee: $350,000,000",
            "UniFirst outstanding shares as of April 16, 2026: approximately 14,532,640 common shares and 3,551,265 class B common shares",
            "UniFirst permitted quarterly dividend: up to $0.3650/share common and $0.292/share class B through October 2026",
            "Cintas last dividend: $0.45/share paid March 13, 2026"
        ],
        "deal_relevance": "This is the definitive proxy for the Cintas/UniFirst merger. The deal structure is a two-step merger (forward triangular merger followed by a downstream merger into an LLC). The fixed exchange ratio of 0.7720 CTAS shares plus $155.00 cash means spread dynamics are driven by both CTAS stock price movements and deal certainty. The Croatti family voting agreement covering approximately two-thirds of voting power effectively guarantees the shareholder vote, removing the principal vote risk. Cintas' obligation is not subject to a financing condition. The transaction is expected to close in the second half of calendar year 2026.",
        "regulatory_mentions": "HSR Act filing and other applicable U.S. or non-U.S. competition, antitrust, merger control or investment laws are referenced as conditions to closing. The DOJ and FTC are specifically mentioned. SEC declared the S-4 registration statement effective. No specific regulatory challenges or second requests are disclosed in this excerpt.",
        "timeline_or_dates": [
            "March 10, 2026 – Merger agreement signed; voting and support agreement executed",
            "January 26, 2025 – Confidentiality agreement between Cintas and UniFirst",
            "May 11, 2026 – Record date for special meeting; proxy statement/prospectus dated",
            "May 12, 2026 – Proxy materials first mailed to UniFirst shareholders",
            "June 4, 2026 – Deadline to request incorporated documents for timely delivery before special meeting",
            "June 11, 2026 – Special meeting of UniFirst shareholders at 10:00 a.m. ET",
            "Second half of calendar year 2026 – Expected merger completion"
        ],
        "conditions_or_requirements": [
            "Approval of merger agreement by affirmative vote of two-thirds of combined voting power of UniFirst common stock and class B common stock, voting as a single class",
            "Receipt of required regulatory approvals under HSR Act and other applicable antitrust laws",
            "Effectiveness of the S-4 registration statement",
            "Other customary closing conditions as set forth in the merger agreement"
        ],
        "risks_flagged": [
            "Market value of Cintas stock consideration will fluctuate due to fixed exchange ratio; value at closing could differ materially from value at announcement or proxy date",
            "UniFirst shareholders have no appraisal rights",
            "Litigation related to the mergers is referenced (details in full filing at page 76)",
            "Certain UniFirst directors and executive officers have interests in the mergers that may differ from those of shareholders generally",
            "If mergers are not completed, UniFirst may owe a $213,300,000 termination fee to Cintas under specified circumstances",
            "Broker non-votes on the merger proposal will have the same effect as a vote AGAINST the merger"
        ]
    },
    "s3_docx_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_docx/DEFM14A_Summary_UNF_05-11-26_26020380.docx",
    "s3_json_url": "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/summary_json/sec_filing_summary_26020380.json",
    "99_1": None,
    "proxy": {},
    "ten_k_ten_q":    None,
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
