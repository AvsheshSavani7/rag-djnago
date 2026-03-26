#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proxy Background Analyzer - FINAL COMPLETE VERSION
- Addresses all critical assessment findings
- Professional Word tables
- Complete client deliverables (Sections 1-13 including missing 4 & 9)
- Fixed chronological summary (Points 1-8 with correct content)
- Supplemental analysis document

Requires:
  pip install anthropic python-dotenv python-docx
"""

import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

import anthropic
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

load_dotenv()

# File paths
INPUT_FILE = "/Users/kaushal/Desktop/My Code/Merger Background/test_ghld.background.txt"
OUTPUT_FILE = "/Users/kaushal/Desktop/My Code/Merger Background/qa_output.json"
OUTPUT_DOCX_A = "/Users/kaushal/Desktop/My Code/Merger Background/client_deliverables.docx"
OUTPUT_DOCX_B = "/Users/kaushal/Desktop/My Code/Merger Backgroundsupplemental_analysis.docx"

# Model
MODEL_NAME = "claude-sonnet-4-5-20250929"

# -----------------------------------------------------------------------------
# UPDATED Prompts with Enhanced Instructions
# -----------------------------------------------------------------------------

STAGE_1_EXTRACTION_PROMPT = """You are a financial analyst specializing in M&A transactions. Your task is to extract comprehensive information from the Background section of a merger proxy statement.

# CRITICAL: Determining the "Starting Point"

The "starting point" is the SPECIFIC EVENT that directly spurred the sales process - typically:
- A written proposal or formal indication of interest (not just preliminary discussions)
- A board resolution to explore alternatives
- An unsolicited approach that triggered formal process
- The first concrete action after any preliminary/historical context

Look for the FIRST SUBSTANTIVE WRITTEN PROPOSAL or board decision, NOT preliminary industry conference discussions.

Everything before this starting point should be noted as "Pre-Starting Point Context."

# EXTRACTION REQUIREMENTS

## 1. Pre-Starting Point Summary
Summarize in 2 sentences or less all events that occurred before the starting point. Include any prior contact between parties, historical context, or earlier strategic reviews.

## 2. Starting Point
- Date of starting point event (use first written proposal or formal board decision)
- Description of the precipitating event
- Nature (e.g., "Company A delivered written proposal" or "Board authorized exploration")

## 3. Process Structure
- Process type: Broad auction / Limited auction / Bilateral negotiation / Other
- Exclusivity status: Formal (with agreement) / Informal (de facto) / Non-exclusive / Mixed
- If no market check conducted: Board's stated rationale for not contacting other buyers
- Timeline of process (start to signing)

## 4. Bidder Universe (Complete Census)
For EVERY potential buyer mentioned, create an entry with:
- Identifier: Use the proxy's naming (Party A, Party B, Company X, etc.)
- Classification: Financial sponsor / Strategic buyer / Other
- Detailed description: Include ALL qualitative details provided:
  * Size descriptors (large, mid-size, etc.)
  * Geographic indicators (international, North American, etc.)
  * Business description (industry, products, services)
  * Executive information (if any names or titles mentioned)
  * Other activities or characteristics
- First contact date (if specified)
- CA/NDA signed: Yes (date) / No / Not mentioned
- Participation level: Full participant / Declined initially / Withdrew mid-process / Never engaged
- Reason for decline/withdrawal: Quote exact rationale if provided
- Any other identifying information

## 5. Complete Bid Timeline
Create a chronological table of EVERY bid, offer, IOI, or proposal:
- Date (exact date or timeframe)
- Party identifier
- Offer amount ($ per share and/or total enterprise value)
- Form of consideration: All cash / All stock / Mixed (specify structure)
- Nature: Non-binding IOI / Indication of interest / Preliminary proposal / Revised offer / Best and final / Binding offer / Other
- Key conditions: Financing contingency / Due diligence / Regulatory approval / Board approval / Stockholder vote / Other
- Material terms: Collar structure, breakup fees, go-shop provisions, timing, etc.
- Any premium/discount to prior offers

## 6. Sales Process Metrics
For EACH distinct outbound process described (be specific):
- Date range of initial outreach
- Total parties contacted (exact number)
- Breakdown: X financial sponsors, Y strategic buyers
- CA/NDA signings: Number signed
- IOIs received: Number and from which parties
- Data room access: Number receiving access
- Parties advancing to each subsequent round
- Reasons parties dropped out at each stage

CRITICAL: If there are multiple outreach efforts (e.g. "Spring 2025" and "August 2025"), document EACH separately with complete metrics.

## 7. Final Round Analysis
Identify the "final round" (last bids received before board decision):
- Date of final round deadline
- Number of parties submitting final bids
- For each final bidder:
  * Party identifier
  * Final bid amount and terms
  * Comparison to their prior bids (percentage change)
- Overall trend: "up-bid" (higher than prior round) or "down-bid" (lower)?
- Spread between highest and second-highest final bids ($ and %)

## 8. Board Selection Rationale
- Winning party identifier
- Winning bid amount and terms
- Was it the highest bid? Yes / No (if no, by how much?)
- Board's stated reasons for selection (list ALL factors):
  * Valuation/price
  * Certainty of closing
  * Regulatory risk
  * Financing certainty
  * Strategic fit
  * Management continuity
  * Speed to close
  * Other factors
- Relative weighting or priority of factors (if discernible)
- Any dissenting director views

## 9. Risk Analysis - Regulatory/Antitrust
- Did regulatory/antitrust considerations factor into board deliberations? Yes / No
- If yes, provide detailed analysis:
  * Which specific buyers had regulatory concerns identified?
  * Nature of regulatory concerns for each (HSR timing, substantive issues, etc.)
  * How regulatory risk was characterized (low/medium/high)
  * COMPARATIVE analysis between bidders on regulatory risk
  * Specific terms: reverse termination fees, obligations to obtain approvals, efforts standards
  * Whether regulatory risk affected final decision and how
  * Any regulatory commitments or remedies discussed

## 10. Risk Analysis - Financing
- Did financing certainty factor into board deliberations? Yes / No
- If yes, provide detailed analysis:
  * Which buyers had financing structures identified?
  * Nature of financing for each
  * How financing risk was characterized
  * Comparative analysis between bidders
  * Whether financing risk affected final decision

## 11. Merger Agreement Negotiations
- Key terms most heavily negotiated
- Evolution of terms through negotiation
- Final agreed terms vs. initial proposals

## 12. Process Events and Timeline
- Press leaks: Date(s), source if identified, impact on process
- Material process changes or disruptions
- Timing pressures or deadlines
- Any unusual procedural elements
- Management presentations or site visits
- Due diligence scope and findings

## 13. Key Dates Summary
Chronological list of all major milestone dates from starting point through signing.

# OUTPUT FORMAT

Provide analysis in clearly labeled sections. Use tables for Bidder Universe, Bid Timeline, and Key Dates. Be exhaustive - include every detail."""

STAGE_2A_STRICT_SUMMARY_PROMPT = """You are a financial analyst creating a chronological summary from a proxy statement background section.

Using the detailed extraction provided, create a summary following this EXACT format:

# CRITICAL FORMATTING RULES:

1. Start with 1-2 sentences summarizing events BEFORE the starting point, including any informal approaches by individual consortium members that predate the formal starting point (no header)
2. Then provide numbered points in chronological order
3. Each numbered item is EXACTLY ONE SENTENCE (maximum 35 words), EXCEPT Point 8 which allows two sentences
4. DO NOT add bold headers - just “1. [sentence]”
5. If an item doesn’t apply, SKIP that number entirely
5. If an item doesn’t apply, OMIT it
6. Format the items as bullet points instead of numbered points. The final output should use consistent bullet styling throughout, with no numbering.

# NUMBERED POINTS REQUIREMENTS:

**Point 1: Sales Process Metrics**
IF there was a sales process, include ALL of:
- When it occurred (date range)
- How many parties contacted
- Breakdown (financial vs. strategic)
- How many signed CAs/NDAs
- How many submitted IOIs or received data room access
- If a go-shop was conducted post-signing, include as a second sentence: parties contacted, NDAs signed, and proposals received

IF no formal process, state:
- Whether negotiations were exclusive (formal or informal)
- Why board didn’t conduct market check

**Point 2: Other Parties**
IF not captured in points 1 and 3 to 9, list other buyers with:
- Detailed descriptions
- What happened with them

**Point 3: Exclusivity (if no sales process)**
Only include if there was NO sales process. Otherwise SKIP.

**Point 4: Final Bidders**
IF multiple bidders, include:
- Who submitted final bids
- DESCRIPTION of each party (e.g., “large global pharmaceutical company”)
- Amount of each final bid as submitted - CRITICAL: If the winning bidder’s final submitted bid differs from the ultimate signed merger consideration, you MUST note both figures in this point using the format: “$X.XX/share as submitted, subsequently negotiated to $Y.YY prior to signing” - Failure to include both figures when they differ will create a contradiction with Point 4

**Point 5: Board Selection Rationale**
Summarize why board selected the acquirer’s bid:
- Focus on PRIMARY reasons (price, timing, certainty)
- Note if they did NOT select highest offer
- Keep regulatory details for Point 5

**Point 6: Regulatory Considerations**
IF antitrust/regulatory was a factor, include:
- How regulatory considerations differentiated bidders
- Specific terms (reverse termination fees, approval obligations)
- How this factored into the decision
SKIP if not applicable.

**Point 7: Financing Considerations**
IF financing certainty was a factor, state whether the merger agreement includes a financing condition and how the buyer expects to fund the transaction.
Do not compare the buyer’s financing to the target’s standalone financing needs.

**Point 8: Press Leaks**
IF process leaked to press, indicate when.
SKIP if no leak.

**Point 9: Bid Trajectory — TWO SENTENCES ALLOWED FOR THIS POINT ONLY**
Sentence 1: List each proposal by the winning bidder in chronological order with date and price.
Sentence 2: State the final accepted offer relative to the bidder’s initial proposal and indicate whether the final price was higher or lower than the initial proposal.



# EXAMPLE (CORRECT FORMAT):

Following preliminary 2023 discussions at conferences, Company executed NDAs with three parties but received no formal proposals before 2024.

1. The company conducted a targeted auction from March to May 2024, contacting 15 parties (10 strategic, 5 financial), with 8 signing NDAs and 5 submitting IOIs. A 30-day go-shop contacted 20 parties with two signing NDAs but none submitting proposals.

4. Two parties submitted final bids: Party A (large multinational pharmaceutical) at $52/share all-cash and Party B (financial sponsor) at $48/share.

5. The board selected Party A’s $52/share bid as the highest offer with superior execution certainty and favorable timeline.

6. Regulatory risk was minimal for both parties, with Party A offering a $200M reverse termination fee versus Party B’s $150M fee.

9. Party A withdrew its offer on October 15 citing market deterioration before re-tabling at $50/share on November 1. The final $52/share offer exceeded Party A’s initial $44/share proposal, characterizing the final round as an up-bid overall.

# CRITICAL REMINDERS:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- Do not describe actions that did not occur (e.g., “did not withdraw”, “did not walk away”, “remained committed”). Only summarize actions explicitly described in the document.
- Avoid narrative verbs such as: remained committed, demonstrated confidence, stayed engaged, did not withdraw. Use only transactional verbs such as: proposed, increased, reduced, withdrew, re-tabled, accepted.
- NO bold headers (just “1. [sentence]“)
- Maximum 35 words per sentence EXCEPT Point 8 which allows two sentences
- Skip inapplicable numbers
- Include ALL sales process metrics in Point 1, including go-shop if applicable
- Include party DESCRIPTIONS in Point 4
- Separate board rationale (Point 4) from regulatory details (Point 6)
- Point 9 MUST distinguish between a withdrawal+re-tabling and a simple down-bid if a withdrawal occurred

Now create your summary."""

STAGE_2B_NARRATIVE_SUMMARY_PROMPT = """You are a financial analyst creating a chronological summary from a proxy statement background section.

Using the detailed extraction provided, create a summary following this EXACT format:

# CRITICAL FORMATTING RULES:

1. Start with 1-2 sentences summarizing events BEFORE the starting point, including any informal approaches by individual consortium members that predate the formal starting point (no header)
2. Then provide numbered points in chronological order
3. Each numbered item is EXACTLY ONE SENTENCE (maximum 35 words), EXCEPT Point 8 which allows two sentences
4. DO NOT add bold headers - just “1. [sentence]”
5. If an item doesn’t apply, SKIP that number entirely
5. If an item doesn’t apply, OMIT it
6. Format the items as bullet points instead of numbered points. The final output should use consistent bullet styling throughout, with no numbering.

# NUMBERED POINTS REQUIREMENTS:

**Point 1: Sales Process Metrics**
IF there was a sales process, include ALL of:
- When it occurred (date range)
- How many parties contacted
- Breakdown (financial vs. strategic)
- How many signed CAs/NDAs
- How many submitted IOIs or received data room access
- If a go-shop was conducted post-signing, include as a second sentence: parties contacted, NDAs signed, and proposals received

IF no formal process, state:
- Whether negotiations were exclusive (formal or informal)
- Why board didn’t conduct market check

**Point 2: Other Parties**
IF not captured in points 1 and 3 to 9, list other buyers with:
- Detailed descriptions
- What happened with them

**Point 3: Exclusivity (if no sales process)**
Only include if there was NO sales process. Otherwise SKIP.

**Point 4: Final Bidders**
IF multiple bidders, include:
- Who submitted final bids
- DESCRIPTION of each party (e.g., “large global pharmaceutical company”)
- Amount of each final bid as submitted - CRITICAL: If the winning bidder’s final submitted bid differs from the ultimate signed merger consideration, you MUST note both figures in this point using the format: “$X.XX/share as submitted, subsequently negotiated to $Y.YY prior to signing” - Failure to include both figures when they differ will create a contradiction with Point 4

**Point 5: Board Selection Rationale**
Summarize why board selected the acquirer’s bid:
- Focus on PRIMARY reasons (price, timing, certainty)
- Note if they did NOT select highest offer
- Keep regulatory details for Point 5

**Point 6: Regulatory Considerations**
IF antitrust/regulatory was a factor, include:
- How regulatory considerations differentiated bidders
- Specific terms (reverse termination fees, approval obligations)
- How this factored into the decision
SKIP if not applicable.

**Point 7: Financing Considerations**
IF financing certainty was a factor, state whether the merger agreement includes a financing condition and how the buyer expects to fund the transaction.
Do not compare the buyer’s financing to the target’s standalone financing needs.

**Point 8: Press Leaks**
IF process leaked to press, indicate when.
SKIP if no leak.

**Point 9: Bid Trajectory — TWO SENTENCES ALLOWED FOR THIS POINT ONLY**
Sentence 1: List each proposal by the winning bidder in chronological order with date and price.
Sentence 2: State the final accepted offer relative to the bidder’s initial proposal and indicate whether the final price was higher or lower than the initial proposal.



# EXAMPLE (CORRECT FORMAT):

Following preliminary 2023 discussions at conferences, Company executed NDAs with three parties but received no formal proposals before 2024.

1. The company conducted a targeted auction from March to May 2024, contacting 15 parties (10 strategic, 5 financial), with 8 signing NDAs and 5 submitting IOIs. A 30-day go-shop contacted 20 parties with two signing NDAs but none submitting proposals.

4. Two parties submitted final bids: Party A (large multinational pharmaceutical) at $52/share all-cash and Party B (financial sponsor) at $48/share.

5. The board selected Party A’s $52/share bid as the highest offer with superior execution certainty and favorable timeline.

6. Regulatory risk was minimal for both parties, with Party A offering a $200M reverse termination fee versus Party B’s $150M fee.

9. Party A withdrew its offer on October 15 citing market deterioration before re-tabling at $50/share on November 1. The final $52/share offer exceeded Party A’s initial $44/share proposal, characterizing the final round as an up-bid overall.

# CRITICAL REMINDERS:
- TONE: State only facts from the filing. Do NOT speculate on motives, interpret what actions "signal" or "suggest", assess confidence levels, or draw conclusions beyond what is explicitly stated. GOOD: "Company suspended earnings calls due to pending transaction." BAD: "Company suspended earnings calls, signaling high confidence in deal completion."
- Do not describe actions that did not occur (e.g., “did not withdraw”, “did not walk away”, “remained committed”). Only summarize actions explicitly described in the document.
- Avoid narrative verbs such as: remained committed, demonstrated confidence, stayed engaged, did not withdraw. Use only transactional verbs such as: proposed, increased, reduced, withdrew, re-tabled, accepted.
- NO bold headers (just “1. [sentence]“)
- Maximum 35 words per sentence EXCEPT Point 8 which allows two sentences
- Skip inapplicable numbers
- Include ALL sales process metrics in Point 1, including go-shop if applicable
- Include party DESCRIPTIONS in Point 4
- Separate board rationale (Point 4) from regulatory details (Point 6)
- Point 9 MUST distinguish between a withdrawal+re-tabling and a simple down-bid if a withdrawal occurred

Now create your summary."""

STAGE_3_RED_FLAGS_PROMPT = """You are a merger arbitrage analyst reviewing this transaction for risk factors. Based on the detailed extraction and executive summary provided, identify potential red flags and points of interest that could affect deal certainty or arbitrage returns.

# ANALYSIS FRAMEWORK

Evaluate and flag the following categories:

## 1. Valuation Red Flags
- Board selected non-highest bid (quantify discount)
- Wide spread between final bids (suggests valuation uncertainty)
- Significant bid decreases in final rounds ("down-bid")
- Unusual valuation methodology or fairness opinion qualifications

## 2. Process Red Flags
- Limited or no market check conducted
- Exclusive negotiations without competitive process
- Rushed timeline or artificial deadlines
- Process leaks that disrupted competitive dynamics
- Unusual procedural elements

## 3. Regulatory/Antitrust Risks
- Significant regulatory concerns identified for winning bidder
- Regulatory issues contributed to selecting lower bid
- Lack of reverse termination fee for regulatory failure
- Foreign investment (CFIUS) concerns
- Complex antitrust issues in concentrated industries

## 4. Financing Risks
- Uncommitted or contingent financing
- High leverage or uncertain debt markets
- Buyer financing issues identified during process
- No financing guarantees or limited parent guarantees

## 5. Deal Protection Concerns
- Very high termination fees (>4%)
- Weak or no go-shop provisions
- Strong matching rights favoring buyer
- Unusual deal protection mechanisms

## 6. Board Deliberation Issues
- Dissenting directors
- Board selected bid over objections from financial advisor
- Inadequate deliberation process
- Conflicts of interest noted

## 7. Execution Risks
- Long timeline to close (>6 months)
- Multiple complex conditions
- Integration concerns flagged
- Material adverse effect definition concerns

## 8. Positive Signals (Deal Certainty)
- All-cash transaction with no financing condition
- Reverse termination fee provided
- Comprehensive market check conducted
- Strong competitive process with multiple bidders
- Short timeline to close
- Buyer board approval obtained pre-signing

# OUTPUT FORMAT

Provide a prioritized list with:

**HIGH RISK FLAGS** (could threaten deal completion or returns)
- [Flag 1]: Brief description and impact

**MEDIUM RISK FLAGS** (worth monitoring)
- [Flag 1]: Brief description

**POSITIVE SIGNALS** (support deal certainty)
- [Signal 1]: Brief description

**OVERALL RISK ASSESSMENT**: [2-3 sentences on whether this appears to be a high-certainty or risky transaction based on the background]

Keep each item concise (1-2 sentences). Focus on factors material to deal completion risk and arbitrage analysis."""

# -----------------------------------------------------------------------------
# Enhanced DOCX Formatter Class with Professional Tables
# -----------------------------------------------------------------------------


class DOCXFormatter:
    """Handles all DOCX formatting with professional styling including proper tables."""

    def __init__(self, doc: Document):
        self.doc = doc
        self._setup_styles()

    def _setup_styles(self):
        """Set up document-wide styles."""
        style = self.doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

        h1_style = self.doc.styles['Heading 1']
        h1_style.font.name = 'Calibri'
        h1_style.font.size = Pt(16)
        h1_style.font.bold = True
        h1_style.font.color.rgb = RGBColor(0, 70, 127)
        h1_style.paragraph_format.space_before = Pt(18)
        h1_style.paragraph_format.space_after = Pt(12)

        h2_style = self.doc.styles['Heading 2']
        h2_style.font.name = 'Calibri'
        h2_style.font.size = Pt(13)
        h2_style.font.bold = True
        h2_style.font.color.rgb = RGBColor(0, 70, 127)
        h2_style.paragraph_format.space_before = Pt(12)
        h2_style.paragraph_format.space_after = Pt(6)

    def add_title(self, title: str):
        """Add document title."""
        title_para = self.doc.add_heading(title, 0)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.runs[0].font.color.rgb = RGBColor(0, 70, 127)
        title_para.runs[0].font.size = Pt(24)

    def add_subtitle(self, subtitle: str):
        """Add document subtitle."""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(subtitle)
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0, 70, 127)
        run.italic = True
        para.paragraph_format.space_after = Pt(12)

    def add_metadata(self, model: str):
        """Add metadata section."""
        metadata = self.doc.add_paragraph()
        metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER

        date_run = metadata.add_run(
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        date_run.italic = True
        date_run.font.size = Pt(9)
        date_run.font.color.rgb = RGBColor(89, 89, 89)

        metadata.add_run("\n")

        model_run = metadata.add_run(f"Model: {model}")
        model_run.italic = True
        model_run.font.size = Pt(9)
        model_run.font.color.rgb = RGBColor(89, 89, 89)

        metadata.paragraph_format.space_after = Pt(24)

    def add_document_purpose(self, purpose_text: str):
        """Add document purpose box."""
        para = self.doc.add_paragraph()
        para.paragraph_format.space_after = Pt(16)
        para.paragraph_format.left_indent = Inches(0.5)
        para.paragraph_format.right_indent = Inches(0.5)

        run = para.add_run(purpose_text)
        run.italic = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(89, 89, 89)

        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), 'F0F0F0')
        para._element.get_or_add_pPr().append(shading_elm)

    def add_horizontal_line(self):
        """Add a horizontal line separator."""
        para = self.doc.add_paragraph()
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(6)

        pPr = para._element.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        pPr.insert_element_before(pBdr,
                                  'w:shd', 'w:tabs', 'w:suppressAutoHyphens', 'w:kinsoku', 'w:wordWrap',
                                  'w:overflowPunct', 'w:topLinePunct', 'w:autoSpaceDE', 'w:autoSpaceDN',
                                  'w:bidi', 'w:adjustRightInd', 'w:snapToGrid', 'w:spacing', 'w:ind',
                                  'w:contextualSpacing', 'w:mirrorIndents', 'w:suppressOverlap', 'w:jc',
                                  'w:textDirection', 'w:textAlignment', 'w:textboxTightWrap',
                                  'w:outlineLvl', 'w:divId', 'w:cnfStyle', 'w:rPr', 'w:sectPr',
                                  'w:pPrChange'
                                  )
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), 'CCCCCC')
        pBdr.append(bottom)

    def add_professional_table(self, text: str):
        """Convert markdown table to professional Word table."""
        lines = [line.strip() for line in text.split('\n')
                 if line.strip() and '|' in line]

        if len(lines) < 2:
            return

        data_lines = [line for line in lines if not re.match(
            r'^\|[\s\-:]+\|$', line)]

        if len(data_lines) < 2:
            return

        def parse_row(line):
            return [cell.strip() for cell in line.split('|') if cell.strip()]

        header = parse_row(data_lines[0])
        rows = [parse_row(line) for line in data_lines[1:]]

        num_cols = len(header)
        num_rows = len(rows)

        if num_cols == 0 or num_rows == 0:
            return

        table = self.doc.add_table(rows=num_rows + 1, cols=num_cols)
        table.style = 'Light Grid Accent 1'

        for row in table.rows:
            for cell in row.cells:
                cell.width = Inches(6.5 / num_cols)

        header_cells = table.rows[0].cells
        for i, header_text in enumerate(header):
            if i < len(header_cells):
                cell = header_cells[i]
                cell.text = header_text
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.size = Pt(10)
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E2F3')
                cell._element.get_or_add_tcPr().append(shading_elm)

        for row_idx, row_data in enumerate(rows):
            cells = table.rows[row_idx + 1].cells
            for col_idx, cell_text in enumerate(row_data):
                if col_idx < len(cells):
                    cells[col_idx].text = cell_text
                    for paragraph in cells[col_idx].paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(9)

        para = self.doc.add_paragraph()
        para.paragraph_format.space_before = Pt(12)

    def add_shaded_box(self, title: str, content: str, color: Tuple[int, int, int] = (240, 240, 240)):
        """Add a shaded text box with title and content."""
        title_para = self.doc.add_paragraph()
        title_para.paragraph_format.space_before = Pt(12)
        title_para.paragraph_format.space_after = Pt(6)
        title_run = title_para.add_run(title)
        title_run.bold = True
        title_run.font.size = Pt(11)
        title_run.font.color.rgb = RGBColor(0, 70, 127)

        content_para = self.doc.add_paragraph()
        content_para.paragraph_format.left_indent = Inches(0.25)
        content_para.paragraph_format.space_after = Pt(8)
        content_run = content_para.add_run(content)
        content_run.font.size = Pt(10)

        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), '%02x%02x%02x' % color)
        content_para._element.get_or_add_pPr().append(shading_elm)

        pBdr = OxmlElement('w:pBdr')
        for border_name in ['top', 'left', 'bottom', 'right']:
            border = OxmlElement(f'w:{border_name}')
            border.set(qn('w:val'), 'single')
            border.set(qn('w:sz'), '4')
            border.set(qn('w:space'), '0')
            border.set(qn('w:color'), '%02x%02x%02x' %
                       tuple(max(0, c - 30) for c in color))
            pBdr.append(border)
        content_para._element.get_or_add_pPr().append(pBdr)

    def add_simple_numbered_summary(self, text: str):
        """Add summary in simple numbered format (no bold headers)."""
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^\d+\.', line):
                para = self.doc.add_paragraph(line)
                para.paragraph_format.left_indent = Inches(0.25)
                para.paragraph_format.space_after = Pt(6)
            else:
                para = self.doc.add_paragraph(line)
                para.paragraph_format.space_after = Pt(8)

    def add_formatted_text(self, text: str, parse_structure: bool = True):
        """Add text with proper formatting."""
        text = self._clean_markdown(text)

        if not parse_structure:
            paragraphs = text.split('\n\n')
            for para_text in paragraphs:
                if para_text.strip():
                    self.doc.add_paragraph(para_text.strip())
            return

        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            para = self.doc.add_paragraph(line)
            para.paragraph_format.space_after = Pt(6)

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting."""
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('>', '')
        return text

# -----------------------------------------------------------------------------
# Main Analyzer Class
# -----------------------------------------------------------------------------


class ProxyBackgroundAnalyzer:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = MODEL_NAME

    def stage_1_extraction(self, document_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Stage 1: Deep extraction of all structured data and details."""

        print("\n" + "="*80)
        print("STAGE 1: DEEP EXTRACTION")
        print("="*80)

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"\n🔍 Running extraction (attempt {attempt}/{max_retries})...")

                t0 = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    messages=[{
                        "role": "user",
                        "content": f"{STAGE_1_EXTRACTION_PROMPT}\n\n# DOCUMENT TO ANALYZE:\n\n{document_text}"
                    }],
                    temperature=0,
                    timeout=500
                )
                elapsed = time.time() - t0

                extraction = response.content[0].text.strip()

                print(f"✅ Extraction completed in {elapsed:.1f}s")
                print(f"📊 Extraction length: {len(extraction):,} characters")

                return {
                    "extraction_text": extraction,
                    "response_time_seconds": round(elapsed, 2),
                    "model": self.model,
                    "attempt": attempt,
                    "success": True
                }

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error on attempt {attempt}: {error_msg}")

                if attempt < max_retries:
                    wait_time = 10 * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"❌ All {max_retries} attempts failed for extraction")
                    return {
                        "extraction_text": None,
                        "error": error_msg,
                        "attempts": max_retries,
                        "success": False
                    }

    def stage_2a_strict_summary(self, extraction_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Stage 2a: Create strict simple numbered summary."""

        print("\n" + "="*80)
        print("STAGE 2A: STRICT CHRONOLOGICAL SUMMARY")
        print("="*80)

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"\n📝 Generating strict summary (attempt {attempt}/{max_retries})...")

                t0 = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"{STAGE_2A_STRICT_SUMMARY_PROMPT}\n\n# DETAILED EXTRACTION:\n\n{extraction_text}"
                    }],
                    temperature=0,
                    timeout=180
                )
                elapsed = time.time() - t0

                summary = response.content[0].text.strip()

                print(f"✅ Strict summary completed in {elapsed:.1f}s")
                print(f"📊 Summary length: {len(summary):,} characters")

                validation = self._validate_strict_summary(summary)
                print(
                    f"📋 Format validation: {validation['score']}/{validation['total']} checks passed")

                return {
                    "summary_text": summary,
                    "response_time_seconds": round(elapsed, 2),
                    "model": self.model,
                    "attempt": attempt,
                    "validation": validation,
                    "success": True
                }

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error on attempt {attempt}: {error_msg}")

                if attempt < max_retries:
                    wait_time = 10 * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"❌ All {max_retries} attempts failed for strict summary")
                    return {
                        "summary_text": None,
                        "error": error_msg,
                        "attempts": max_retries,
                        "success": False
                    }

    def stage_2b_narrative_summary(self, extraction_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Stage 2b: Create narrative flowing summary."""

        print("\n" + "="*80)
        print("STAGE 2B: NARRATIVE SUMMARY")
        print("="*80)

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"\n📝 Generating narrative summary (attempt {attempt}/{max_retries})...")

                t0 = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"{STAGE_2B_NARRATIVE_SUMMARY_PROMPT}\n\n# DETAILED EXTRACTION:\n\n{extraction_text}"
                    }],
                    temperature=0,
                    timeout=180
                )
                elapsed = time.time() - t0

                summary = response.content[0].text.strip()

                print(f"✅ Narrative summary completed in {elapsed:.1f}s")
                print(f"📊 Summary length: {len(summary):,} characters")

                return {
                    "summary_text": summary,
                    "response_time_seconds": round(elapsed, 2),
                    "model": self.model,
                    "attempt": attempt,
                    "success": True
                }

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error on attempt {attempt}: {error_msg}")

                if attempt < max_retries:
                    wait_time = 10 * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"❌ All {max_retries} attempts failed for narrative summary")
                    return {
                        "summary_text": None,
                        "error": error_msg,
                        "attempts": max_retries,
                        "success": False
                    }

    def stage_3_red_flags(self, extraction_text: str, summary_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Stage 3: Identify red flags and risk factors."""

        print("\n" + "="*80)
        print("STAGE 3: RED FLAGS & RISK ANALYSIS")
        print("="*80)

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"\n🚩 Analyzing risks (attempt {attempt}/{max_retries})...")

                t0 = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"{STAGE_3_RED_FLAGS_PROMPT}\n\n# STRICT SUMMARY:\n\n{summary_text}\n\n# DETAILED EXTRACTION:\n\n{extraction_text}"
                    }],
                    temperature=0,
                    timeout=180
                )
                elapsed = time.time() - t0

                red_flags = response.content[0].text.strip()

                print(f"✅ Risk analysis completed in {elapsed:.1f}s")
                print(f"📊 Analysis length: {len(red_flags):,} characters")

                return {
                    "red_flags_text": red_flags,
                    "response_time_seconds": round(elapsed, 2),
                    "model": self.model,
                    "attempt": attempt,
                    "success": True
                }

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error on attempt {attempt}: {error_msg}")

                if attempt < max_retries:
                    wait_time = 10 * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ All {max_retries} attempts failed for red flags")
                    return {
                        "red_flags_text": None,
                        "error": error_msg,
                        "attempts": max_retries,
                        "success": False
                    }

    def _validate_strict_summary(self, summary_text: str) -> Dict[str, Any]:
        """Validate that strict summary follows required format."""
        checks = {
            "has_numbered_items": bool(re.search(r'^\d+\.', summary_text, re.MULTILINE)),
            "no_bold_headers": '**' not in summary_text,
            "no_n_a_responses": "N/A" not in summary_text and "Not applicable" not in summary_text.lower(),
            "starts_with_context": not summary_text.startswith('1.') and not summary_text.startswith('**'),
        }

        # Check sentence length
        numbered_items = re.findall(
            r'^\d+\.\s+(.+)$', summary_text, re.MULTILINE)
        sentence_length_ok = True
        max_words = 0
        for item in numbered_items:
            word_count = len(item.split())
            max_words = max(max_words, word_count)
            if word_count > 40:
                sentence_length_ok = False

        checks["sentence_length_compliant"] = sentence_length_ok
        checks["max_sentence_length"] = max_words

        score = sum(1 for k, v in checks.items() if k not in [
                    'max_sentence_length'] and v)
        checks["score"] = score
        checks["total"] = len([k for k in checks.keys() if k not in [
                              'score', 'total', 'max_sentence_length']])
        checks["passed"] = score >= checks["total"] - 1

        return checks

    def create_document_a(self, extraction_result: Dict, strict_summary_result: Dict, output_path: str):
        """Create Document A: Client Deliverables Only."""

        print(f"\n📝 Creating Document A (Client Deliverables)...")

        doc = Document()
        formatter = DOCXFormatter(doc)

        # Title
        formatter.add_title('Merger Proxy Background Analysis')
        formatter.add_subtitle('Client Deliverables Package')
        formatter.add_metadata(self.model)

        # Document purpose
        purpose = ("This document contains the specific deliverables requested in your specifications: "
                   "chronological summary (numbered format), complete bidder census, bid timeline, "
                   "sales process metrics, and supporting documentation.\n\n"
                   "For additional strategic analysis and insights, see companion document: "
                   "\"Supplemental Analysis & Insights\"")
        formatter.add_document_purpose(purpose)

        # =====================================================================
        # SECTION 1: CHRONOLOGICAL SUMMARY
        # =====================================================================
        doc.add_page_break()
        doc.add_heading('Chronological Summary', 1)

        if strict_summary_result.get('success'):
            formatter.add_simple_numbered_summary(
                strict_summary_result['summary_text'])

            # Validation
            if 'validation' in strict_summary_result:
                formatter.add_horizontal_line()
                val_para = doc.add_paragraph()
                val_para.paragraph_format.space_before = Pt(6)

                val_color = RGBColor(
                    0, 128, 0) if strict_summary_result['validation']['passed'] else RGBColor(255, 0, 0)

                val_run = val_para.add_run(
                    f"✓ Format Validation: {strict_summary_result['validation']['score']}/{strict_summary_result['validation']['total']} checks passed"
                )
                val_run.font.size = Pt(9)
                val_run.font.color.rgb = val_color

                if 'max_sentence_length' in strict_summary_result['validation']:
                    val_para.add_run(
                        f" | Max sentence: {strict_summary_result['validation']['max_sentence_length']} words")
        else:
            doc.add_paragraph(
                f"❌ Error: {strict_summary_result.get('error', 'Unknown error')}")

        # =====================================================================
        # EXTRACT REQUIRED SECTIONS FROM EXTRACTION
        # =====================================================================
        if extraction_result.get('success'):
            extraction_text = extraction_result['extraction_text']
            sections = self._parse_extraction_for_document_a(extraction_text)

            for section_title, section_content in sections:
                doc.add_page_break()
                doc.add_heading(section_title, 1)

                # Check if it's a table
                if self._is_table(section_content):
                    formatter.add_professional_table(section_content)
                else:
                    # Regular text
                    paragraphs = section_content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            clean_para = formatter._clean_markdown(
                                para.strip())
                            doc.add_paragraph(clean_para)

        doc.save(output_path)
        print(f"✅ Document A saved: {output_path}")

    def create_document_b(self, extraction_result: Dict, narrative_summary_result: Dict,
                          red_flags_result: Dict, output_path: str):
        """Create Document B: Supplemental Analysis."""

        print(f"\n📝 Creating Document B (Supplemental Analysis)...")

        doc = Document()
        formatter = DOCXFormatter(doc)

        # Title
        formatter.add_title('Merger Proxy Background Analysis')
        formatter.add_subtitle('Supplemental Analysis & Insights')
        formatter.add_metadata(self.model)

        # Document purpose
        purpose = ("This document provides additional analysis and strategic perspective beyond the client-requested deliverables. "
                   "It includes: narrative summary for executive consumption, risk analysis framework for merger arbitrage assessment, "
                   "and process insights.\n\n"
                   "This analysis is based on the same source material documented in the \"Client Deliverables Package.\"")
        formatter.add_document_purpose(purpose)

        # =====================================================================
        # SECTION 1: NARRATIVE SUMMARY
        # =====================================================================
        doc.add_page_break()
        doc.add_heading('Executive Narrative', 1)

        note_para = doc.add_paragraph()
        note_para.paragraph_format.space_after = Pt(12)
        note_run = note_para.add_run(
            "This narrative presents the same transaction information in flowing prose format for executive review."
        )
        note_run.italic = True
        note_run.font.size = Pt(9)
        note_run.font.color.rgb = RGBColor(89, 89, 89)

        formatter.add_horizontal_line()

        if narrative_summary_result.get('success'):
            formatter.add_formatted_text(
                narrative_summary_result['summary_text'], parse_structure=False)
        else:
            doc.add_paragraph(
                f"❌ Error: {narrative_summary_result.get('error', 'Unknown error')}")

        # =====================================================================
        # SECTION 2: RISK ANALYSIS
        # =====================================================================
        doc.add_page_break()
        doc.add_heading('Risk Analysis Framework', 1)

        note_para = doc.add_paragraph()
        note_para.paragraph_format.space_after = Pt(12)
        note_run = note_para.add_run(
            "Merger arbitrage perspective on deal certainty factors and potential risks."
        )
        note_run.italic = True
        note_run.font.size = Pt(9)
        note_run.font.color.rgb = RGBColor(89, 89, 89)

        formatter.add_horizontal_line()

        if red_flags_result.get('success'):
            sections = self._parse_red_flags_sections(
                red_flags_result['red_flags_text'])

            for section_type, section_content in sections:
                if section_type == 'high_risk':
                    formatter.add_shaded_box(
                        '🔴 HIGH RISK FLAGS', section_content, (255, 230, 230))
                elif section_type == 'medium_risk':
                    formatter.add_shaded_box(
                        '🟡 MEDIUM RISK FLAGS', section_content, (255, 255, 230))
                elif section_type == 'positive':
                    formatter.add_shaded_box(
                        '🟢 POSITIVE SIGNALS', section_content, (230, 255, 230))
                elif section_type == 'overall':
                    formatter.add_horizontal_line()
                    overall_para = doc.add_paragraph()
                    overall_para.paragraph_format.space_before = Pt(12)
                    overall_run = overall_para.add_run(
                        'OVERALL RISK ASSESSMENT')
                    overall_run.bold = True
                    overall_run.font.size = Pt(12)
                    overall_run.font.color.rgb = RGBColor(0, 70, 127)

                    content_para = doc.add_paragraph(section_content)
                    content_para.paragraph_format.left_indent = Inches(0.25)
        else:
            doc.add_paragraph(
                f"❌ Error: {red_flags_result.get('error', 'Unknown error')}")

        # =====================================================================
        # SECTION 3: PROCESS INSIGHTS
        # =====================================================================
        if extraction_result.get('success'):
            extraction_text = extraction_result['extraction_text']
            supplemental_sections = self._parse_extraction_for_document_b(
                extraction_text)

            if supplemental_sections:
                doc.add_page_break()
                doc.add_heading('Process Insights', 1)

                for section_title, section_content in supplemental_sections:
                    doc.add_heading(section_title, 2)

                    paragraphs = section_content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            clean_para = formatter._clean_markdown(
                                para.strip())
                            doc.add_paragraph(clean_para)

        doc.save(output_path)
        print(f"✅ Document B saved: {output_path}")

    def _is_table(self, text: str) -> bool:
        """Check if text appears to be a table."""
        lines = text.split('\n')
        pipe_lines = [line for line in lines if '|' in line]

        # If more than 50% of lines have pipes, and we have at least 3 lines, it's probably a table
        return len(pipe_lines) >= 3 and len(pipe_lines) / max(len(lines), 1) > 0.5

    def _parse_extraction_for_document_a(self, extraction_text: str) -> List[Tuple[str, str]]:
        """Extract only sections needed for Document A (client deliverables)."""
        required_sections = {
            '2. Starting Point': [],
            '3. Process Structure': [],
            '4. Bidder Universe': [],  # CRITICAL - NOW INCLUDED
            '5. Complete Bid Timeline': [],
            '6. Sales Process Metrics': [],
            '7. Final Round Analysis': [],
            '8. Board Selection Rationale': [],
            '9. Risk Analysis - Regulatory': [],  # CRITICAL - NOW INCLUDED
            '10. Risk Analysis - Financing': [],
            '13. Key Dates Summary': []
        }

        lines = extraction_text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            if line.startswith('## '):
                if current_section and current_section in required_sections:
                    required_sections[current_section] = '\n'.join(
                        current_content)

                section_name = line.replace('##', '').strip()
                if section_name in required_sections:
                    current_section = section_name
                    current_content = []
                else:
                    current_section = None
            elif current_section:
                current_content.append(line)

        if current_section and current_section in required_sections:
            required_sections[current_section] = '\n'.join(current_content)

        return [(k, v) for k, v in required_sections.items() if v]

    def _parse_extraction_for_document_b(self, extraction_text: str) -> List[Tuple[str, str]]:
        """Extract supplemental sections for Document B."""
        supplemental_sections = {
            '11. Merger Agreement Negotiations': [],
            '12. Process Events and Timeline': []
        }

        lines = extraction_text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            if line.startswith('## '):
                if current_section and current_section in supplemental_sections:
                    supplemental_sections[current_section] = '\n'.join(
                        current_content)

                section_name = line.replace('##', '').strip()
                if section_name in supplemental_sections:
                    current_section = section_name
                    current_content = []
                else:
                    current_section = None
            elif current_section:
                current_content.append(line)

        if current_section and current_section in supplemental_sections:
            supplemental_sections[current_section] = '\n'.join(current_content)

        return [(k, v) for k, v in supplemental_sections.items() if v]

    def _parse_red_flags_sections(self, text: str) -> List[Tuple[str, str]]:
        """Parse red flags text into categorized sections."""
        sections = []
        current_type = None
        current_content = []

        for line in text.split('\n'):
            line = line.strip()

            if 'HIGH RISK' in line.upper():
                if current_type and current_content:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = 'high_risk'
                current_content = []
            elif 'MEDIUM RISK' in line.upper():
                if current_type and current_content:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = 'medium_risk'
                current_content = []
            elif 'POSITIVE SIGNAL' in line.upper():
                if current_type and current_content:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = 'positive'
                current_content = []
            elif 'OVERALL RISK' in line.upper():
                if current_type and current_content:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = 'overall'
                current_content = []
            elif line and current_type:
                current_content.append(line)

        if current_type and current_content:
            sections.append((current_type, '\n'.join(current_content)))

        return sections

    def run(self, input_file: str, output_file: str, output_docx_a: str, output_docx_b: str):
        """Run the complete analysis workflow creating both documents."""

        print("="*80)
        print("PROXY BACKGROUND ANALYZER - FINAL COMPLETE VERSION")
        print("="*80)

        # Load input
        print(f"\n📂 Loading input file: {input_file}")
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            document_text = f.read()
        print(f"✅ Loaded document: {len(document_text):,} characters")

        # Stage 1: Deep extraction
        extraction_result = self.stage_1_extraction(document_text)

        # Stage 2a: Strict summary
        if extraction_result.get('success'):
            strict_summary_result = self.stage_2a_strict_summary(
                extraction_result['extraction_text'])
        else:
            strict_summary_result = {"summary_text": None,
                                     "error": "Extraction failed", "success": False}

        # Stage 2b: Narrative summary
        if extraction_result.get('success'):
            narrative_summary_result = self.stage_2b_narrative_summary(
                extraction_result['extraction_text'])
        else:
            narrative_summary_result = {
                "summary_text": None, "error": "Extraction failed", "success": False}

        # Stage 3: Red flags
        if extraction_result.get('success') and strict_summary_result.get('success'):
            red_flags_result = self.stage_3_red_flags(
                extraction_result['extraction_text'],
                strict_summary_result['summary_text']
            )
        else:
            red_flags_result = {"red_flags_text": None,
                                "error": "Prior stage failed", "success": False}

        # Create both documents
        if extraction_result.get('success') and strict_summary_result.get('success'):
            self.create_document_a(
                extraction_result, strict_summary_result, output_docx_a)

        if extraction_result.get('success') and narrative_summary_result.get('success') and red_flags_result.get('success'):
            self.create_document_b(
                extraction_result, narrative_summary_result, red_flags_result, output_docx_b)

        # Save JSON
        output = {
            "input_file": input_file,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "stage_1_extraction": extraction_result,
            "stage_2a_strict_summary": strict_summary_result,
            "stage_2b_narrative_summary": narrative_summary_result,
            "stage_3_red_flags": red_flags_result
        }

        print(f"\n💾 Saving JSON output to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved!")

        # Save individual text files
        if extraction_result.get('success'):
            extraction_file = output_file.replace(".json", "_extraction.txt")
            with open(extraction_file, "w", encoding="utf-8") as f:
                f.write(extraction_result['extraction_text'])
            print(f"✅ Extraction text saved: {extraction_file}")

        if strict_summary_result.get('success'):
            strict_file = output_file.replace(".json", "_strict_summary.txt")
            with open(strict_file, "w", encoding="utf-8") as f:
                f.write(strict_summary_result['summary_text'])
            print(f"✅ Strict summary saved: {strict_file}")

        if narrative_summary_result.get('success'):
            narrative_file = output_file.replace(
                ".json", "_narrative_summary.txt")
            with open(narrative_file, "w", encoding="utf-8") as f:
                f.write(narrative_summary_result['summary_text'])
            print(f"✅ Narrative summary saved: {narrative_file}")

        if red_flags_result.get('success'):
            red_flags_file = output_file.replace(".json", "_red_flags.txt")
            with open(red_flags_file, "w", encoding="utf-8") as f:
                f.write(red_flags_result['red_flags_text'])
            print(f"✅ Red flags saved: {red_flags_file}")

        # Print summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)

        total_time = 0

        if extraction_result.get('success'):
            print(
                f"\n✅ Stage 1 (Extraction): SUCCESS - {extraction_result['response_time_seconds']}s")
            total_time += extraction_result['response_time_seconds']

        if strict_summary_result.get('success'):
            print(
                f"✅ Stage 2a (Strict Summary): SUCCESS - {strict_summary_result['response_time_seconds']}s")
            print(
                f"   Validation: {strict_summary_result['validation']['score']}/{strict_summary_result['validation']['total']}, Max: {strict_summary_result['validation']['max_sentence_length']} words")
            total_time += strict_summary_result['response_time_seconds']

        if narrative_summary_result.get('success'):
            print(
                f"✅ Stage 2b (Narrative): SUCCESS - {narrative_summary_result['response_time_seconds']}s")
            total_time += narrative_summary_result['response_time_seconds']

        if red_flags_result.get('success'):
            print(
                f"✅ Stage 3 (Red Flags): SUCCESS - {red_flags_result['response_time_seconds']}s")
            total_time += red_flags_result['response_time_seconds']

        print(f"\n⏱️  Total: {total_time:.1f}s")

        print("\n" + "="*80)
        print("OUTPUT FILES:")
        print("="*80)
        print(f"📄 Document A (Client Deliverables): {output_docx_a}")
        print(f"📄 Document B (Supplemental Analysis): {output_docx_b}")
        print(f"📄 Full JSON: {output_file}")
        print("="*80)


def main():
    analyzer = ProxyBackgroundAnalyzer()
    analyzer.run(INPUT_FILE, OUTPUT_FILE, OUTPUT_DOCX_A, OUTPUT_DOCX_B)


if __name__ == "__main__":
    main()
