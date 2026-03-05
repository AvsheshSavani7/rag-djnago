"""DOCX generation: single-filing reports and multi-filing comparison reports."""

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import List

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from .config import (
    COLOR_TIMING, COLOR_REGULATORY, COLOR_BOTH,
    COLOR_SIGNIFICANT, COLOR_MODERATE, COLOR_MINOR,
    COLOR_NEW, COLOR_UNCHANGED,
)
from .models import DealContext


# =============================================================================
# SHARED DOCX HELPERS
# =============================================================================

def _apply_shading(run, hex_color: str) -> None:
    rPr = run._r.get_or_add_rPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    rPr.append(shd)


def _shade_cell(cell, hex_color: str) -> None:
    """Apply background fill color to a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def _set_cell_width(cell, width_inches: float) -> None:
    """Set fixed column width on a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = OxmlElement('w:tcW')
    tcW.set(qn('w:w'), str(int(width_inches * 1440)))
    tcW.set(qn('w:type'), 'dxa')
    tcPr.append(tcW)


def _add_diff_text(para, old_text: str, new_text: str, pt_size: int = 8) -> None:
    """
    Add word-level redline diff inline to a paragraph.
    Deletions: red strikethrough. Insertions/replacements: green bold + highlight.
    """
    old_words = (old_text or "").split()
    new_words = (new_text or "").split()
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            chunk = ' '.join(old_words[i1:i2])
            if chunk:
                r = para.add_run(chunk + ' ')
                r.font.size = Pt(pt_size)
        elif tag in ('replace', 'delete'):
            old_chunk = ' '.join(old_words[i1:i2])
            if old_chunk:
                r = para.add_run(old_chunk + ' ')
                r.font.size = Pt(pt_size)
                r.font.strike = True
                r.font.color.rgb = RGBColor(183, 28, 28)
            if tag == 'replace':
                new_chunk = ' '.join(new_words[j1:j2])
                if new_chunk:
                    r = para.add_run(new_chunk + ' ')
                    r.font.size = Pt(pt_size)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(27, 94, 32)
                    _apply_shading(r, 'C8E6C9')
        elif tag == 'insert':
            new_chunk = ' '.join(new_words[j1:j2])
            if new_chunk:
                r = para.add_run(new_chunk + ' ')
                r.font.size = Pt(pt_size)
                r.font.bold = True
                r.font.color.rgb = RGBColor(27, 94, 32)
                _apply_shading(r, 'C8E6C9')


def _make_doc_base(deal: DealContext, filing_labels_str: str) -> Document:
    """Create a Document with standard header."""
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    hdr = doc.add_paragraph()
    r = hdr.add_run(f"{deal.acquirer_company} / {deal.target_company} ({deal.ticker})")
    r.font.size = Pt(13); r.bold = True

    fl = doc.add_paragraph()
    r = fl.add_run(filing_labels_str)
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()
    return doc


# =============================================================================
# SINGLE-FILING REPORTS
# =============================================================================

def generate_single_filing_report_fulsome(
    excerpts: list,
    deal: DealContext,
    filing_label: str,
    output_path: Path,
) -> None:
    """Generate a fulsome DOCX for a single filing with all commentary."""
    priority_categories = {"risk", "deal_terms", "timeline", "regulatory"}

    filtered = [
        e for e in excerpts
        if (e.get("timing_flag") or e.get("regulatory_flag"))
        and e.get("category", "general") in priority_categories
    ]

    if not filtered:
        filtered = [e for e in excerpts if e.get("timing_flag") or e.get("regulatory_flag")]

    def sort_key(e):
        cat = e.get("category", "general")
        if cat == "risk": return (0, -e.get("relevance_score", 0))
        elif cat == "deal_terms": return (1, -e.get("relevance_score", 0))
        else: return (2, -e.get("relevance_score", 0))

    filtered.sort(key=sort_key)

    doc = _make_doc_base(deal, f"Filing: {filing_label}")
    doc.add_heading('Key Excerpts', 1)

    for idx, excerpt in enumerate(filtered, 1):
        timing = excerpt.get("timing_flag", False)
        regulatory = excerpt.get("regulatory_flag", False)
        category = excerpt.get("category", "general")
        score = excerpt.get("relevance_score", 0)

        if timing and regulatory:
            bg_color = COLOR_BOTH
            flag_label = "TIMING + REGULATORY"
        elif timing:
            bg_color = COLOR_TIMING
            flag_label = "TIMING"
        else:
            bg_color = COLOR_REGULATORY
            flag_label = "REGULATORY"

        header_p = doc.add_paragraph()
        tag = header_p.add_run(f" {idx}. [{flag_label}] ")
        _apply_shading(tag, bg_color)
        tag.bold = True; tag.font.size = Pt(9)
        tag.font.color.rgb = (
            RGBColor(0, 0, 0) if timing and regulatory
            else RGBColor(21, 101, 192) if timing
            else RGBColor(173, 20, 87)
        )

        cat_score = header_p.add_run(f"  {category.upper()} (Score: {score}/10)")
        cat_score.font.size = Pt(8); cat_score.font.color.rgb = RGBColor(120, 120, 120)

        sec_p = doc.add_paragraph()
        sec_p.paragraph_format.left_indent = Inches(0.15)
        sec_r = sec_p.add_run(f"Section: {excerpt.get('section', 'Unknown')}")
        sec_r.font.size = Pt(8); sec_r.italic = True; sec_r.font.color.rgb = RGBColor(100, 100, 100)

        if timing and excerpt.get("timing_assessment"):
            assess_p = doc.add_paragraph()
            assess_p.paragraph_format.left_indent = Inches(0.15)
            assess_r = assess_p.add_run(f"Timing: {excerpt['timing_assessment']}")
            assess_r.font.size = Pt(9); assess_r.font.color.rgb = RGBColor(21, 101, 192)

        if regulatory and excerpt.get("regulatory_assessment"):
            assess_p = doc.add_paragraph()
            assess_p.paragraph_format.left_indent = Inches(0.15)
            assess_r = assess_p.add_run(f"Regulatory: {excerpt['regulatory_assessment']}")
            assess_r.font.size = Pt(9); assess_r.font.color.rgb = RGBColor(173, 20, 87)

        text_p = doc.add_paragraph()
        text_p.paragraph_format.left_indent = Inches(0.25)
        text_p.paragraph_format.space_before = Pt(4)
        clean = excerpt.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()
        text_r = text_p.add_run(clean)
        text_r.font.size = Pt(9)

        doc.add_paragraph()

    doc.save(output_path)
    print(f"  Fulsome report saved: {output_path}")


def generate_single_filing_report_concise(
    excerpts: list,
    deal: DealContext,
    filing_label: str,
    output_path: Path,
) -> None:
    """Generate a concise DOCX for a single filing — no commentary or scores."""
    priority_categories = {"risk", "deal_terms", "timeline", "regulatory"}

    filtered = [
        e for e in excerpts
        if (e.get("timing_flag") or e.get("regulatory_flag"))
        and e.get("category", "general") in priority_categories
    ]

    if not filtered:
        filtered = [e for e in excerpts if e.get("timing_flag") or e.get("regulatory_flag")]

    def sort_key(e):
        cat = e.get("category", "general")
        if cat == "risk": return (0, -e.get("relevance_score", 0))
        elif cat == "deal_terms": return (1, -e.get("relevance_score", 0))
        else: return (2, -e.get("relevance_score", 0))

    filtered.sort(key=sort_key)

    doc = _make_doc_base(deal, f"Filing: {filing_label}")
    doc.add_heading('Key Excerpts', 1)

    for idx, excerpt in enumerate(filtered, 1):
        timing = excerpt.get("timing_flag", False)
        regulatory = excerpt.get("regulatory_flag", False)
        category = excerpt.get("category", "general")

        if timing and regulatory:
            bg_color = COLOR_BOTH
            flag_label = "TIMING + REGULATORY"
        elif timing:
            bg_color = COLOR_TIMING
            flag_label = "TIMING"
        else:
            bg_color = COLOR_REGULATORY
            flag_label = "REGULATORY"

        header_p = doc.add_paragraph()
        tag = header_p.add_run(f" {idx}. [{flag_label}] ")
        _apply_shading(tag, bg_color)
        tag.bold = True; tag.font.size = Pt(9)
        tag.font.color.rgb = (
            RGBColor(0, 0, 0) if timing and regulatory
            else RGBColor(21, 101, 192) if timing
            else RGBColor(173, 20, 87)
        )

        cat_label = header_p.add_run(f"  {category.upper()}")
        cat_label.font.size = Pt(8); cat_label.font.color.rgb = RGBColor(120, 120, 120)

        sec_p = doc.add_paragraph()
        sec_p.paragraph_format.left_indent = Inches(0.15)
        sec_r = sec_p.add_run(f"Section: {excerpt.get('section', 'Unknown')}")
        sec_r.font.size = Pt(8); sec_r.italic = True; sec_r.font.color.rgb = RGBColor(100, 100, 100)

        text_p = doc.add_paragraph()
        text_p.paragraph_format.left_indent = Inches(0.25)
        text_p.paragraph_format.space_before = Pt(4)
        clean = excerpt.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()
        text_r = text_p.add_run(clean)
        text_r.font.size = Pt(9)

        doc.add_paragraph()

    doc.save(output_path)
    print(f"  Concise report saved: {output_path}")


# =============================================================================
# MULTI-FILING COMPARISON REPORTS
# =============================================================================

def generate_client_report(
    all_comparison_steps: list,
    deal: DealContext,
    filing_labels: List[str],
    output_path: Path,
) -> None:
    """Generate a clean client-ready DOCX showing only new/changed findings."""
    doc = _make_doc_base(deal, ' → '.join(filing_labels))

    PASS_COLORS = {
        "timing":         (RGBColor(21, 101, 192), "TIMING"),
        "regulatory":     (RGBColor(173, 20, 87), "REGULATORY"),
        "legal_language": (RGBColor(156, 110, 0), "LEGAL LANGUAGE"),
    }

    for step_idx, step in enumerate(all_comparison_steps):
        current_label = step["current_label"]
        prior_labels = step["prior_labels"]
        merged = step["merged_results"]

        doc.add_heading(f"{current_label}  vs  {', '.join(prior_labels)}", 1)

        if not merged:
            np = doc.add_paragraph()
            np.add_run("No material changes detected.").font.size = Pt(10)
            np.runs[0].font.color.rgb = RGBColor(100, 100, 100)
            doc.add_paragraph()
            continue

        sig = sum(1 for r in merged if r["overall_severity"] == "significant")
        mod = sum(1 for r in merged if r["overall_severity"] == "moderate")
        minor_ct = sum(1 for r in merged if r["overall_severity"] == "minor")
        new_ct = sum(1 for r in merged
                     if any((r.get(k) or {}).get("match_type") == "new"
                            for k in ["timing", "regulatory", "legal_language"]))

        stat = doc.add_paragraph()
        stat.add_run(
            f"{len(merged)} item(s): {sig} significant, {mod} moderate, "
            f"{minor_ct} minor, {new_ct} new disclosure(s)"
        ).font.size = Pt(9)
        doc.add_paragraph()

        for idx, result in enumerate(merged, 1):
            severity = result["overall_severity"]
            is_new = any(
                (result.get(k) or {}).get("match_type") == "new"
                for k in ["timing", "regulatory", "legal_language"]
            )

            header_p = doc.add_paragraph()
            sev_label = "NEW" if is_new and severity == "none" else severity.upper()
            bg_color = COLOR_NEW if (is_new and severity == "none") else {
                "significant": COLOR_SIGNIFICANT, "moderate": COLOR_MODERATE,
                "minor": COLOR_MINOR,
            }.get(severity, COLOR_UNCHANGED)
            txt_color = (
                RGBColor(27, 94, 32) if (is_new and severity == "none")
                else {
                    "significant": RGBColor(183, 28, 28),
                    "moderate": RGBColor(230, 81, 0),
                    "minor": RGBColor(156, 110, 0),
                }.get(severity, RGBColor(120, 120, 120))
            )

            tag = header_p.add_run(f" {idx}. [{sev_label}] ")
            _apply_shading(tag, bg_color)
            tag.bold = True; tag.font.size = Pt(9); tag.font.color.rgb = txt_color

            sec = header_p.add_run(f"  {result.get('section', '')[:60]}")
            sec.font.size = Pt(8); sec.font.color.rgb = RGBColor(120, 120, 120)

            clause_text = result.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()
            if clause_text:
                clause_p = doc.add_paragraph()
                clause_p.paragraph_format.left_indent = Inches(0.25)
                clause_p.paragraph_format.space_before = Pt(4)
                clause_label = clause_p.add_run("Filing text: ")
                clause_label.bold = True
                clause_label.font.size = Pt(8)
                clause_label.font.color.rgb = RGBColor(80, 80, 80)
                clause_r = clause_p.add_run(clause_text)
                clause_r.font.size = Pt(8)
                clause_r.font.color.rgb = RGBColor(50, 50, 50)

            for pass_key, (color, label) in PASS_COLORS.items():
                finding = result.get(pass_key)
                if not finding:
                    continue
                if not finding.get("changed") and finding.get("match_type") != "new":
                    continue

                fp = doc.add_paragraph()
                fp.paragraph_format.left_indent = Inches(0.15)
                lr = fp.add_run(f"{label}: ")
                lr.bold = True; lr.font.size = Pt(9); lr.font.color.rgb = color
                ar = fp.add_run(finding.get("analysis", ""))
                ar.font.size = Pt(9)

                if pass_key == "legal_language":
                    for change in (finding.get("notable_changes") or [])[:5]:
                        cp = doc.add_paragraph()
                        cp.paragraph_format.left_indent = Inches(0.35)

                        if change.get("old_phrase"):
                            old_r = cp.add_run(f'"{change["old_phrase"]}"')
                            old_r.font.size = Pt(8)
                            old_r.font.color.rgb = RGBColor(183, 28, 28)
                            old_r.font.strike = True

                        if change.get("old_phrase") and change.get("new_phrase"):
                            cp.add_run("  →  ").font.size = Pt(8)

                        if change.get("new_phrase"):
                            new_r = cp.add_run(f'"{change["new_phrase"]}"')
                            new_r.font.size = Pt(8)
                            new_r.font.color.rgb = RGBColor(27, 94, 32)
                            new_r.bold = True

                        if change.get("interpretation"):
                            ip2 = doc.add_paragraph()
                            ip2.paragraph_format.left_indent = Inches(0.35)
                            ir = ip2.add_run(change["interpretation"])
                            ir.font.size = Pt(8); ir.font.italic = True
                            ir.font.color.rgb = RGBColor(100, 100, 100)

            sep = doc.add_paragraph()
            sep.paragraph_format.space_before = Pt(2)
            sep.paragraph_format.space_after = Pt(2)

        if step_idx < len(all_comparison_steps) - 1:
            doc.add_page_break()

    doc.save(output_path)
    print(f"  Client report saved: {output_path}")


def generate_raw_text_report(
    all_comparison_steps: list,
    deal: DealContext,
    filing_labels: List[str],
    output_path: Path,
) -> None:
    """Generate a DOCX containing only the raw filing text excerpts."""
    doc = _make_doc_base(deal, ' → '.join(filing_labels))

    for step_idx, step in enumerate(all_comparison_steps):
        current_label = step["current_label"]
        prior_labels = step["prior_labels"]
        merged = step["merged_results"]

        doc.add_heading(f"{current_label}  vs  {', '.join(prior_labels)}", 1)

        if not merged:
            np = doc.add_paragraph()
            np.add_run("No excerpts.").font.size = Pt(10)
            np.runs[0].font.color.rgb = RGBColor(100, 100, 100)
            doc.add_paragraph()
            continue

        for idx, result in enumerate(merged, 1):
            section = result.get("section", "")
            clause_text = result.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()

            sec_p = doc.add_paragraph()
            sr = sec_p.add_run(f"{idx}. {section}")
            sr.bold = True
            sr.font.size = Pt(9)
            sr.font.color.rgb = RGBColor(60, 60, 60)

            if clause_text:
                text_p = doc.add_paragraph()
                text_p.paragraph_format.left_indent = Inches(0.25)
                tr = text_p.add_run(clause_text)
                tr.font.size = Pt(9)
                tr.font.color.rgb = RGBColor(30, 30, 30)

            sep = doc.add_paragraph()
            sep.paragraph_format.space_before = Pt(2)
            sep.paragraph_format.space_after = Pt(2)

        if step_idx < len(all_comparison_steps) - 1:
            doc.add_page_break()

    doc.save(output_path)
    print(f"  Raw text report saved: {output_path}")


def generate_redline_report(
    all_comparison_steps: list,
    deal: DealContext,
    filing_labels: List[str],
    output_path: Path,
) -> None:
    """Generate a redline-style DOCX with side-by-side word-level diffs."""
    doc = _make_doc_base(deal, ' → '.join(filing_labels))

    PASS_LABEL = {
        "timing":         ("TIMING",         RGBColor(21, 101, 192)),
        "regulatory":     ("REGULATORY",     RGBColor(173, 20, 87)),
        "legal_language": ("LEGAL LANGUAGE", RGBColor(156, 110, 0)),
    }

    COL_W = 3.2  # inches per column

    for step_idx, step in enumerate(all_comparison_steps):
        current_label = step["current_label"]
        prior_labels  = step["prior_labels"]
        merged        = step["merged_results"]

        doc.add_heading(
            f"Current Filing: {current_label}  |  Compared to: {', '.join(prior_labels)}", 1
        )

        if not merged:
            np_p = doc.add_paragraph()
            np_p.add_run("No material changes detected.").font.size = Pt(10)
            np_p.runs[0].font.color.rgb = RGBColor(100, 100, 100)
            doc.add_paragraph()
            continue

        for idx, result in enumerate(merged, 1):
            section      = result.get("section", "")[:80]
            severity     = result.get("overall_severity", "none")
            current_text = result.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()

            active_passes = [
                pk for pk in ["timing", "regulatory", "legal_language"]
                if (result.get(pk) or {}).get("changed") or
                   (result.get(pk) or {}).get("match_type") == "new"
            ]
            if not active_passes:
                continue

            is_new = any(
                (result.get(pk) or {}).get("match_type") == "new"
                for pk in ["timing", "regulatory", "legal_language"]
            )

            sev_bg, sev_fg = {
                "significant": (COLOR_SIGNIFICANT, RGBColor(183, 28, 28)),
                "moderate":    (COLOR_MODERATE,    RGBColor(230, 81, 0)),
                "minor":       (COLOR_MINOR,        RGBColor(156, 110, 0)),
            }.get(severity, (
                COLOR_NEW            if is_new else COLOR_UNCHANGED,
                RGBColor(27, 94, 32) if is_new else RGBColor(80, 80, 80),
            ))
            sev_label = "NEW" if (is_new and severity == "none") else severity.upper()

            hdr_p = doc.add_paragraph()
            tag_run = hdr_p.add_run(f" {idx}. [{sev_label}] ")
            _apply_shading(tag_run, sev_bg)
            tag_run.bold = True; tag_run.font.size = Pt(9)
            tag_run.font.color.rgb = sev_fg
            types_str = "  |  ".join(PASS_LABEL[pk][0] for pk in active_passes)
            meta_run = hdr_p.add_run(f"  {types_str}  —  {section}")
            meta_run.font.size = Pt(8)
            meta_run.font.color.rgb = RGBColor(80, 80, 80)

            prior_excerpts_for_para = None
            prior_label_for_para    = ", ".join(prior_labels)
            prior_ref               = None

            for pk in active_passes:
                finding = result.get(pk) or {}
                if prior_excerpts_for_para is None and "_prior_excerpts" in finding:
                    prior_excerpts_for_para = finding["_prior_excerpts"]
                    prior_label_for_para    = finding.get("_prior_label", prior_label_for_para)
                if prior_ref is None:
                    matched = finding.get("matched_prior", [])
                    if matched:
                        prior_ref = matched[0]

            prior_data = None
            if prior_excerpts_for_para and prior_ref:
                lookup = {f"PRIOR-{i+1}": p for i, p in enumerate(prior_excerpts_for_para)}
                prior_data = lookup.get(prior_ref)

            prior_text = (
                prior_data.get("text", "").replace('\n', ' ').replace('  ', ' ').strip()
                if prior_data else None
            )

            left_hdr_str  = f"Current: {current_label}"
            right_hdr_str = f"Prior: {prior_label_for_para}"

            if prior_text and current_text:
                tbl = doc.add_table(rows=2, cols=2)
                tbl.style = 'Table Grid'
                for row in tbl.rows:
                    for c in row.cells:
                        _set_cell_width(c, COL_W)

                h_current = tbl.cell(0, 0)
                h_prior   = tbl.cell(0, 1)
                _shade_cell(h_current, 'CCFFCC')
                _shade_cell(h_prior,   'FFCCCC')
                cr_h = h_current.paragraphs[0].add_run(left_hdr_str)
                cr_h.bold = True; cr_h.font.size = Pt(8)
                cr_h.font.color.rgb = RGBColor(0, 100, 0)
                pr_h = h_prior.paragraphs[0].add_run(right_hdr_str)
                pr_h.bold = True; pr_h.font.size = Pt(8)
                pr_h.font.color.rgb = RGBColor(120, 0, 0)

                c_current = tbl.cell(1, 0)
                c_prior   = tbl.cell(1, 1)
                _shade_cell(c_current, 'F8FFF8')
                _shade_cell(c_prior,   'FFF8F8')

                _add_diff_text(c_current.paragraphs[0], prior_text, current_text, pt_size=8)

                old_r = c_prior.paragraphs[0].add_run(prior_text)
                old_r.font.size = Pt(8)
                old_r.font.color.rgb = RGBColor(100, 20, 20)

            elif current_text:
                tbl = doc.add_table(rows=2, cols=1)
                tbl.style = 'Table Grid'
                _set_cell_width(tbl.cell(0, 0), COL_W * 2)
                _set_cell_width(tbl.cell(1, 0), COL_W * 2)

                h_new = tbl.cell(0, 0)
                _shade_cell(h_new, 'CCFFCC')
                hr2 = h_new.paragraphs[0].add_run(f"{left_hdr_str} — New Disclosure")
                hr2.bold = True; hr2.font.size = Pt(8)
                hr2.font.color.rgb = RGBColor(0, 100, 0)

                c_new = tbl.cell(1, 0)
                _shade_cell(c_new, 'F8FFF8')
                cr3 = c_new.paragraphs[0].add_run(current_text)
                cr3.font.size = Pt(8); cr3.font.bold = True
                cr3.font.color.rgb = RGBColor(0, 100, 0)

            ll = result.get("legal_language") or {}
            notable = ll.get("notable_changes") or []
            if notable and "legal_language" in active_passes:
                ph_hdr = doc.add_paragraph()
                ph_hdr.paragraph_format.left_indent = Inches(0.15)
                ph_hdr.paragraph_format.space_before = Pt(6)
                phr = ph_hdr.add_run("Specific Phrase Changes:")
                phr.bold = True; phr.font.size = Pt(8)
                phr.font.color.rgb = RGBColor(156, 110, 0)

                for change in notable[:6]:
                    old_phrase = (change.get("old_phrase") or "").strip()
                    new_phrase = (change.get("new_phrase") or "").strip()
                    interp     = (change.get("interpretation") or "").strip()
                    if not old_phrase and not new_phrase:
                        continue

                    ptbl = doc.add_table(rows=2, cols=2)
                    ptbl.style = 'Table Grid'
                    for row in ptbl.rows:
                        for c in row.cells:
                            _set_cell_width(c, COL_W)

                    _shade_cell(ptbl.cell(0, 0), 'E5FFE5')
                    _shade_cell(ptbl.cell(0, 1), 'FFE5E5')
                    rh0 = ptbl.cell(0, 0).paragraphs[0].add_run("Current Phrase")
                    rh0.bold = True; rh0.font.size = Pt(7)
                    rh1 = ptbl.cell(0, 1).paragraphs[0].add_run("Prior Phrase")
                    rh1.bold = True; rh1.font.size = Pt(7)

                    if new_phrase:
                        nr_ = ptbl.cell(1, 0).paragraphs[0].add_run(f'"{new_phrase}"')
                        nr_.font.size = Pt(8); nr_.font.bold = True
                        nr_.font.color.rgb = RGBColor(27, 94, 32)
                        _apply_shading(nr_, 'C8E6C9')

                    if old_phrase:
                        or_ = ptbl.cell(1, 1).paragraphs[0].add_run(f'"{old_phrase}"')
                        or_.font.size = Pt(8); or_.font.strike = True
                        or_.font.color.rgb = RGBColor(183, 28, 28)

                    if interp:
                        ip3 = doc.add_paragraph()
                        ip3.paragraph_format.left_indent = Inches(0.35)
                        ir_ = ip3.add_run(interp)
                        ir_.font.size = Pt(8); ir_.font.italic = True
                        ir_.font.color.rgb = RGBColor(100, 100, 100)

            for pk in active_passes:
                finding  = result.get(pk) or {}
                analysis = finding.get("analysis", "").strip()
                if not analysis:
                    continue
                ap = doc.add_paragraph()
                ap.paragraph_format.left_indent = Inches(0.15)
                ap.paragraph_format.space_before = Pt(3)
                lbl, col = PASS_LABEL[pk]
                bullet_lbl = ap.add_run(f"• {lbl}: ")
                bullet_lbl.bold = True; bullet_lbl.font.size = Pt(8)
                bullet_lbl.font.color.rgb = col
                ar = ap.add_run(analysis)
                ar.font.size = Pt(8)

            doc.add_paragraph()

        if step_idx < len(all_comparison_steps) - 1:
            doc.add_page_break()

    doc.save(output_path)
    print(f"  Redline report saved: {output_path}")


def generate_full_comparison_json(
    all_comparison_steps: list,
    deal: DealContext,
    filing_labels: List[str],
    output_path: Path,
) -> None:
    output = {
        "metadata": {
            "ticker": deal.ticker,
            "target": deal.target_company,
            "acquirer": deal.acquirer_company,
            "deal_value": deal.deal_value,
            "filings_analyzed": filing_labels,
            "generated": datetime.now().isoformat(),
            "model": "claude-opus-4-6",
        },
        "comparison_steps": [],
    }

    for step in all_comparison_steps:
        merged = step["merged_results"]
        step_data = {
            "current_filing": step["current_label"],
            "compared_against": step["prior_labels"],
            "summary": {
                "total_changes": len(merged),
                "significant": sum(1 for r in merged if r["overall_severity"] == "significant"),
                "moderate": sum(1 for r in merged if r["overall_severity"] == "moderate"),
                "minor": sum(1 for r in merged if r["overall_severity"] == "minor"),
            },
            "pass_summaries": {
                name: {"findings": data["total_findings"], "changes": data["changes_detected"]}
                for name, data in step["pass_results"].items()
            },
            "findings": merged,
        }
        output["comparison_steps"].append(step_data)

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Full comparison JSON: {output_path}")


def generate_exec_summary_report(
    all_comparison_steps: list,
    deal: DealContext,
    filing_labels: List[str],
    output_path: Path,
    anthropic_key: str = "",
) -> None:
    """Generate a crisp executive summary DOCX: synthesized bullets via Claude."""
    from .comparator import summarize_findings_to_bullets

    print("  Generating executive summary bullets via Claude...")
    bullets = []
    if anthropic_key:
        bullets = summarize_findings_to_bullets(
            all_comparison_steps, deal, filing_labels, anthropic_key
        )

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.line_spacing = 1.15

    hdr = doc.add_paragraph()
    r = hdr.add_run(f"{deal.acquirer_company} / {deal.target_company} ({deal.ticker})")
    r.font.size = Pt(13); r.bold = True

    fl = doc.add_paragraph()
    r = fl.add_run(' → '.join(filing_labels))
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()

    CATEGORY_COLORS = {
        "Timing":        RGBColor(21, 101, 192),
        "Regulatory":    RGBColor(173, 20, 87),
        "Business/Risk": RGBColor(46, 125, 50),
        "Legal":         RGBColor(60, 60, 60),
        "Other":         RGBColor(60, 60, 60),
    }

    if not bullets:
        p = doc.add_paragraph()
        p.add_run("No material changes identified across all filings.").font.size = Pt(10)
    else:
        headline_items = [b for b in bullets if b.get("category") == "__headline__"]
        body_bullets   = [b for b in bullets if b.get("category") != "__headline__"]

        if headline_items:
            hl_p = doc.add_paragraph()
            hl_p.paragraph_format.space_before = Pt(0)
            hl_p.paragraph_format.space_after = Pt(10)
            hl_r = hl_p.add_run(headline_items[0]["bullet"])
            hl_r.bold = True
            hl_r.font.size = Pt(11)
            hl_r.font.color.rgb = RGBColor(30, 30, 30)

        cat_order = ["Timing", "Regulatory", "Business/Risk", "Legal", "Other"]
        grouped: dict = {}
        for item in body_bullets:
            cat = item.get("category", "Other")
            grouped.setdefault(cat, []).append(item["bullet"])

        for cat in cat_order:
            if cat not in grouped:
                continue
            color = CATEGORY_COLORS.get(cat, RGBColor(60, 60, 60))

            hdr_p = doc.add_paragraph()
            hdr_p.paragraph_format.space_before = Pt(8)
            hdr_p.paragraph_format.space_after = Pt(2)
            hdr_r = hdr_p.add_run(cat.upper())
            hdr_r.bold = True
            hdr_r.font.size = Pt(8)
            hdr_r.font.color.rgb = color

            for bullet in grouped[cat]:
                bp = doc.add_paragraph()
                bp.paragraph_format.left_indent = Inches(0.15)
                bp.paragraph_format.space_before = Pt(2)
                bp.paragraph_format.space_after = Pt(3)

                dot_r = bp.add_run("• ")
                dot_r.font.size = Pt(10)
                dot_r.font.color.rgb = color

                txt_r = bp.add_run(bullet)
                txt_r.font.size = Pt(10)

    doc.save(output_path)
    print(f"  Exec summary saved: {output_path}")
