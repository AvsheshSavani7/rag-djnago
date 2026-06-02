"""
Preview proxy comparison summary email with inline change report HTML.

Run from rag_project/:
    python test_proxy_comparison_email_preview.py

Optional: pass path to change_report.txt or a saved change_text file.
"""
from sec_rss_parser.proxy_comparision.html_builder import create_changes_html
from sec_rss_parser.email_templates import generate_proxy_comparison_summary_email_html
import django
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")


django.setup()


OUTPUT_FILE = os.path.join(os.path.dirname(
    __file__), "test_proxy_comparison_email_preview.html")

SAMPLE_CHANGE_TEXT = """========================================================================
  BLD -- 69e50eee8d446ade643d9b7d
  Changes: Amended Registration Statement (S-4/A) -> Definitive Proxy (DEFM14A)
  Generated: May 29, 2026 - 09:35 PM
========================================================================

The Definitive Proxy (DEFM14A) establishes a mailing date of May 29, 2026, which was not present in the Amended Registration Statement. No material changes were made to the Background section between the two filings.

DATES
  [From Amended Registration Statement (S-4/A):]
  No meeting dates, record dates, mailing dates, or outside dates are explicitly disclosed in the provided filing text.

  [Changes in Definitive Proxy (DEFM14A):]
  - Mailing Date: May 29, 2026  [NEW]

CONSIDERATION
  [From Amended Registration Statement (S-4/A):]
  Each TopBuild share will be converted into the right to receive, at the holder's election, either $505.00 in cash per share or 20.200 QXO shares per share. Both elections are subject to proration, with cash consideration capped at 45% and stock consideration capped at 55% of total TopBuild shares outstanding. Shares for which no election is made will be treated as having elected stock consideration.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

FINANCING
  [From Amended Registration Statement (S-4/A):]
  Not a condition to closing.
  QXO Building Products, Inc. entered into a commitment letter dated April 15, 2026 with Morgan Stanley Senior Funding, Inc., Wells Fargo Bank, National Association and Wells Fargo Securities, LLC, Barclays Bank PLC, Apollo Capital Management, L.P., Citigroup Global Markets Inc. and Credit Agricole Corporate and Investment Bank to provide (i) a $3.0 billion senior secured term loan facility and (ii) $3.0 billion of senior unsecured bridge financing. Financing is for the purposes of funding the cash consideration, paying fees, costs and expenses, repaying certain existing indebtedness of TopBuild and/or its subsidiaries, and paying other transaction costs. The bridge facilities will be available to be drawn upon to the extent that QXO has not prior to or concurrently with the consummation of the mergers received proceeds from one or more debt offerings or loan facility transactions sufficient to pay the required amounts.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

SH APPROVAL
  [From Amended Registration Statement (S-4/A):]
  - 69e50eee8d446ade643d9b7d -- TopBuild stockholder approval means the affirmative vote of holders of a majority of the outstanding TopBuild shares entitled to vote thereon in favor of the adoption of the merger agreement. Failure to vote TopBuild shares will have the same effect as a vote "AGAINST" the TopBuild merger proposal.
  - TBD -- QXO stockholder approval means the approval of the QXO share issuance by a majority of the votes cast by holders of shares of QXO voting stock present in person or represented by proxy at the QXO stockholder meeting. Approval of the QXO share issuance proposal is a condition to closing.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

HSR
  [From Amended Registration Statement (S-4/A):]
  QXO and TopBuild each filed HSR Act notifications with the FTC and the DOJ on April 24, 2026, pursuant to the merger agreement requirement to file within 10 business days of execution. The HSR Act waiting period expired on May 26, 2026, with no second request mentioned in the filing. Additionally, QXO and TopBuild each filed for Canadian Competition Bureau clearance on May 1, 2026, and received a no-action letter from the Canadian Competition Bureau on May 28, 2026, stating it does not intend to oppose completion of the merger. HSR clearance has been obtained and the antitrust condition to closing has been satisfied.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

OTHER REGULATORY
  [From Amended Registration Statement (S-4/A):]
  - Canadian Competition Bureau: Antitrust clearance required; QXO and TopBuild each filed a request for an advanced ruling certificate and notifications under Part IX of the Competition Act (Canada) on May 1, 2026; on May 28, 2026, QXO received a "no-action letter" from the Canadian Competition Bureau stating that it does not intend to oppose completion of the merger. Alternatively, expiration of the statutory waiting period under Part IX of the Competition Act (Canada) would satisfy this condition.
  - SEC (Form S-4 / Securities Act): QXO filed a registration statement on Form S-4 with the SEC under the Securities Act, of which the joint proxy statement/prospectus forms a part; this registration statement must be declared effective by the SEC in order for the mergers to be completed.
  - SEC (Exchange Act): Filings with the SEC under the Exchange Act and the rules and regulations promulgated thereunder are required.
  - NYSE: Filings and/or notices under the rules of the New York Stock Exchange are required.
  - Foreign or state securities or blue sky laws: Filings pursuant to any applicable foreign or state securities or blue sky laws are required.
  - Other applicable foreign antitrust, foreign direct investment, or competition law jurisdictions: The parties are required to make any other applicable foreign antitrust, foreign direct investment, or competition law filings as promptly as practicable; specific jurisdictions are set forth in the confidential TopBuild disclosure letter and are not identified by name in the filing text.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

CONDITIONS
  [From Amended Registration Statement (S-4/A):]
  - The registration statement on Form S-4 has been declared effective by the SEC, and no stop order suspending the effectiveness of the Form S-4 has been issued and no proceedings by the SEC for that purpose have been initiated or threatened.
  - Antitrust clearance from the Canadian Competition Bureau, or expiration of the statutory waiting period under Part IX of the Competition Act (Canada), is required (note: a no-action letter was received from the Canadian Competition Bureau on May 28, 2026).
  - QXO stockholder approval of the QXO share issuance proposal must be obtained.
  - TopBuild stockholder approval of the adoption of the merger agreement must be obtained.
  - All governmental, regulatory or other consents and approvals necessary for the consummation of the transactions contemplated by the merger agreement must be obtained.
  - Each party must use reasonable best efforts to obtain required regulatory approvals; QXO is not required to agree to any divestitures, hold-separate arrangements, terminations of existing relationships or contractual rights, or other restructurings that would, individually or in the aggregate, reasonably be expected to be material to QXO or TopBuild, measured on the basis of a hypothetical company of the same size and scale as TopBuild and its subsidiaries as of the date of the merger agreement.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

CLOSING
  [From Amended Registration Statement (S-4/A):]
  The mergers are expected to be completed during the third quarter of 2026, subject to satisfaction or waiver of closing conditions, though neither TopBuild nor QXO can predict the actual date on which the mergers will be completed. The outside date is January 17, 2027, after which either party may terminate the merger agreement if the Titanium Merger has not been consummated (except with respect to a party whose failure to fulfill any obligation under the merger agreement caused such failure). Key remaining gating items include SEC effectiveness of the Form S-4 registration statement and receipt of stockholder approvals from both QXO and TopBuild at their respective June 29, 2026 special meetings.

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

TERMINATION & FEES
  [From Amended Registration Statement (S-4/A):]
  - Company termination fee (TopBuild pays QXO): $600,000,000 -- payable if the merger agreement is terminated under certain circumstances, including due to a TopBuild adverse recommendation change, the consummation of (or entry into a definitive agreement with respect to) an alternative transaction within twelve months following certain terminations where an acquisition proposal had been publicly announced and not withdrawn, or a material and uncured breach of the non-solicitation obligations
  - Parent/reverse termination fee (QXO pays TopBuild): $600,000,000 -- payable if the merger agreement is terminated under certain circumstances, including due to a QXO adverse recommendation change, the consummation of (or entry into a definitive agreement with respect to) an alternative transaction within twelve months following certain terminations where an acquisition proposal had been publicly announced and not withdrawn, or a material and uncured breach of the non-solicitation obligations; representing approximately 4.2% of the nominal value of the transaction as of signing
  - Regulatory termination fee: Not disclosed in the provided text (note: the filing references that Jones Day proposed "the addition of a reverse termination fee payable by QXO in the event of a failure to obtain required regulatory approvals," but no separate regulatory termination fee amount is disclosed in the excerpted text)
  - Go-shop period: None

  [Changes in Definitive Proxy (DEFM14A):]
  No changes.

BACKGROUND
No material changes.

========================================================================
"""

TICKER = "AFBI"
TARGET = "Affinity Bancshares, Inc."
ACQUIRER = "Fidelity BancShares (N.C.), Inc."
OLD_LABEL = "Preliminary Proxy (PREM14A)"
NEW_LABEL = "Definitive Proxy (DEFM14A)"
TIMESTAMP = "June 2, 2026 - 10:30 AM"


def _load_change_text() -> str:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        if "====" in raw:
            parts = raw.split("=" * 72)
            return parts[-1].strip() if len(parts) > 1 else raw
        return raw
    return SAMPLE_CHANGE_TEXT


def main():
    change_text = _load_change_text()
    change_report_html = create_changes_html(
        change_text,
        TICKER,
        TARGET,
        ACQUIRER,
        OLD_LABEL,
        NEW_LABEL,
        TIMESTAMP,
    )

    subject, html = generate_proxy_comparison_summary_email_html(
        company_name=TARGET,
        form_type="DEFM14A",
        ticker=TICKER,
        label=TARGET,
        deal_id="69e50eee8d446ade643d9b7d",
        cik_number="0001234567",
        change_report_html=change_report_html,
        tier1_changes=2,
        tier2_changes=1,
        target_ticker=TICKER,
        target_name=TARGET,
        matched_cik_label="(target)",
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Subject: {subject}")
    print(f"HTML:    {OUTPUT_FILE}")
    webbrowser.open(f"file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
