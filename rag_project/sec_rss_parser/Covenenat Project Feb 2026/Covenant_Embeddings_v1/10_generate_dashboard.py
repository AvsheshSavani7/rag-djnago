#!/usr/bin/env python3
"""
Stage 10: HTML Dashboard Generator for Covenant Analysis
Focus: Show what matters. Outliers first. Risk-based presentation.

Philosophy:
- Outlier clauses with "WHY unusual" explanations at the top
- Show restrictiveness scores and risk levels
- Specific provision checks highlighted
- Full table with filtering
- Clean, professional design for client presentation
"""

import json
import os
import webbrowser
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import glob


class CovenantDashboardGenerator:
    """Generate HTML dashboard for covenant analysis"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_dashboard(self, deal_name: Optional[str] = None) -> str:
        """Generate interactive HTML dashboard"""

        print("\n" + "="*80)
        print("🚀 STAGE 10: HTML DASHBOARD GENERATOR")
        print("="*80)

        classification, assessment, comparison, provisions = self._load_input_files(deal_name)

        if not classification:
            print("❌ Could not find classification file (Stage 6 output)")
            return None

        if not assessment:
            print("❌ Could not find assessment file (Stage 7 output)")
            return None

        deal_id = classification.get('deal_id', 'unknown')

        print(f"\n📊 Generating dashboard for: {deal_id}")
        print(f"   Classification (Stage 6): ✅")
        print(f"   Assessment (Stage 7): ✅")
        print(f"   Comparison (Stage 8): {'✅' if comparison else '⚠️  Optional'}")
        print(f"   Provisions (Stage 9): {'✅' if provisions else '⚠️  Optional'}")

        output_dir = f"{self.base_dir}/dashboard_output"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n🎨 Generating covenant analysis dashboard...")

        html_content = self._generate_html(
            classification, assessment, comparison, provisions, deal_id
        )

        filename = f"{output_dir}/covenant_dashboard_{deal_id}_{self.timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n✅ Dashboard generated: {filename}")
        print(f"\n🌐 Opening in browser...")

        webbrowser.open('file://' + os.path.abspath(filename))

        print(f"\n💡 Dashboard opened!")

        return filename

    def _load_input_files(self, deal_name: Optional[str] = None) -> Tuple[
        Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]
    ]:
        """Load Stage 6, 7, 8, and 9 outputs"""

        print(f"\n📂 Loading input files...")

        reports_dir = f"{self.base_dir}/new_deal_reports"

        # Find classification file (Stage 6)
        classification_files = sorted(
            glob.glob(f"{reports_dir}/deal_classification_*.json"), reverse=True
        )

        if deal_name:
            classification_files = [
                f for f in classification_files
                if deal_name.lower() in f.lower()
            ]

        classification = None
        if classification_files:
            with open(classification_files[0], 'r') as f:
                classification = json.load(f)
                print(f"   ✅ Classification: {os.path.basename(classification_files[0])}")
        else:
            return None, None, None, None

        actual_deal_id = classification.get('deal_id', 'unknown')

        # Find assessment file (Stage 7)
        assessment_files = sorted(
            glob.glob(f"{reports_dir}/deal_assessment_{actual_deal_id}_*.json"),
            reverse=True
        )

        assessment = None
        if assessment_files:
            with open(assessment_files[0], 'r') as f:
                assessment = json.load(f)
                print(f"   ✅ Assessment: {os.path.basename(assessment_files[0])}")

        # Find comparison file (Stage 8 - optional)
        comparison_files = sorted(
            glob.glob(f"{reports_dir}/benchmark_comparison_{actual_deal_id}_*.json"),
            reverse=True
        )

        comparison = None
        if comparison_files:
            with open(comparison_files[0], 'r') as f:
                comparison = json.load(f)
                print(f"   ✅ Comparison: {os.path.basename(comparison_files[0])}")
        else:
            print(f"   ⚠️  Comparison: Not found (optional)")

        # Find provisions file (Stage 9 - optional)
        provisions_files = sorted(
            glob.glob(f"{reports_dir}/specific_provisions_{actual_deal_id}_*.json"),
            reverse=True
        )

        provisions = None
        if provisions_files:
            with open(provisions_files[0], 'r') as f:
                provisions = json.load(f)
                print(f"   ✅ Provisions: {os.path.basename(provisions_files[0])}")
        else:
            print(f"   ⚠️  Provisions: Not found (optional)")

        return classification, assessment, comparison, provisions

    def _generate_html(
        self,
        classification: Dict,
        assessment: Dict,
        comparison: Optional[Dict],
        provisions: Optional[Dict],
        deal_id: str
    ) -> str:
        """Generate complete HTML dashboard"""

        # Calculate metrics
        total = len(assessment.get('assessed_clauses', []))
        outliers = sum(
            1 for c in assessment.get('assessed_clauses', [])
            if c.get('is_outlier', False)
        )

        # Get high restrictiveness count
        high_restrictiveness = sum(
            1 for c in assessment.get('assessed_clauses', [])
            if c.get('restrictiveness_score', 0) >= 8
        )

        # Get provision flags
        prov_flags = self._get_provision_flags(provisions)

        # Prepare data for JS
        clause_data_json = json.dumps(
            self._prepare_clause_data(classification, assessment, provisions),
            indent=2
        )

        # Generate alert bar
        alert_bar = self._generate_alert_bar(
            deal_id, outliers, high_restrictiveness, prov_flags
        )

        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            assessment, comparison, provisions
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{deal_id} - Covenant Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e14;
            --bg-secondary: #0f1419;
            --bg-tertiary: #151c24;
            --bg-hover: #1a232e;
            --text-primary: #e6e6e6;
            --text-secondary: #8a919a;
            --text-muted: #5c6370;
            --accent-blue: #5ccfe6;
            --accent-green: #87d96c;
            --accent-yellow: #ffcc66;
            --accent-red: #f07178;
            --accent-orange: #ff8c42;
            --font-mono: 'JetBrains Mono', monospace;
            --font-sans: 'IBM Plex Sans', sans-serif;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: var(--font-sans); background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 16px; }}

        /* Alert Bar */
        .alert-bar {{
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 4px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 16px;
            font-family: var(--font-mono);
            flex-wrap: wrap;
        }}

        .deal-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .alert-flag {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .alert-flag.outliers {{
            background: rgba(240, 113, 120, 0.2);
            color: var(--accent-red);
        }}

        .alert-flag.restrictive {{
            background: rgba(255, 140, 66, 0.2);
            color: var(--accent-orange);
        }}

        .alert-flag.provision {{
            background: rgba(255, 204, 102, 0.15);
            color: var(--accent-yellow);
        }}

        /* Executive Summary */
        .exec-summary {{
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 4px;
            padding: 16px;
            margin-bottom: 16px;
        }}

        .exec-summary h2 {{
            font-family: var(--font-mono);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
            color: var(--accent-blue);
        }}

        .exec-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 12px;
        }}

        .exec-item {{
            background: var(--bg-tertiary);
            padding: 10px;
            border-radius: 4px;
        }}

        .exec-label {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .exec-value {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .exec-insights {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Table Section */
        .table-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 4px;
            margin-bottom: 16px;
        }}

        .section-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--bg-tertiary);
            font-family: var(--font-mono);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .filters {{
            padding: 12px 16px;
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--bg-tertiary);
            flex-wrap: wrap;
        }}

        .filter-btn {{
            font-family: var(--font-mono);
            padding: 6px 12px;
            border: 1px solid var(--bg-tertiary);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
        }}

        .filter-btn:hover {{
            background: var(--bg-hover);
            border-color: var(--accent-blue);
        }}

        .filter-btn.active {{
            background: var(--accent-blue);
            color: var(--bg-primary);
            border-color: var(--accent-blue);
        }}

        .export-btn {{
            font-family: var(--font-mono);
            padding: 6px 12px;
            background: var(--accent-green);
            color: var(--bg-primary);
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 500;
            transition: all 0.2s;
            margin-left: auto;
        }}

        .export-btn:hover {{
            background: var(--accent-blue);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        thead {{
            background: var(--bg-tertiary);
        }}

        th {{
            font-family: var(--font-mono);
            text-align: left;
            padding: 10px 12px;
            font-weight: 500;
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        td {{
            font-family: var(--font-mono);
            padding: 10px 12px;
            border-bottom: 1px solid var(--bg-tertiary);
            font-size: 0.75rem;
        }}

        tbody tr:hover {{
            background: var(--bg-hover);
        }}

        .row-outlier {{
            background: rgba(240, 113, 120, 0.05);
            border-left: 3px solid var(--accent-red);
        }}

        .row-outlier:hover {{
            background: rgba(240, 113, 120, 0.08);
        }}

        .row-highly-restrictive {{
            background: rgba(255, 140, 66, 0.05);
            border-left: 3px solid var(--accent-orange);
        }}

        .row-highly-restrictive:hover {{
            background: rgba(255, 140, 66, 0.08);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.7rem;
            font-weight: 500;
        }}

        .badge-outlier {{
            background: rgba(240, 113, 120, 0.15);
            color: var(--accent-red);
        }}

        .badge-high {{
            background: rgba(255, 140, 66, 0.15);
            color: var(--accent-orange);
        }}

        .badge-medium {{
            background: rgba(255, 204, 102, 0.15);
            color: var(--accent-yellow);
        }}

        .badge-low {{
            background: rgba(135, 217, 108, 0.15);
            color: var(--accent-green);
        }}

        .expand-btn {{
            font-family: var(--font-mono);
            padding: 4px 10px;
            background: var(--bg-tertiary);
            color: var(--accent-blue);
            border: 1px solid var(--bg-tertiary);
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
        }}

        .expand-btn:hover {{
            background: var(--accent-blue);
            color: var(--bg-primary);
        }}

        .clause-detail {{
            display: none;
            background: var(--bg-tertiary);
            padding: 12px;
            border-radius: 3px;
            border-left: 3px solid var(--accent-blue);
        }}

        .clause-detail.show {{
            display: block;
        }}

        .detail-section {{
            margin-bottom: 12px;
        }}

        .detail-section:last-child {{
            margin-bottom: 0;
        }}

        .detail-section h4 {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
        }}

        .detail-section p {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            line-height: 1.6;
        }}

        .detail-section ul {{
            list-style: none;
            padding: 0;
        }}

        .detail-section li {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            line-height: 1.6;
            padding-left: 12px;
            position: relative;
        }}

        .detail-section li:before {{
            content: "•";
            position: absolute;
            left: 0;
            color: var(--accent-red);
        }}

        .full-text {{
            background: var(--bg-secondary);
            padding: 10px;
            border-radius: 3px;
            font-style: italic;
            color: var(--text-primary);
            border: 1px solid var(--bg-tertiary);
        }}

        .red-flags {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 4px;
        }}

        .red-flag {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            padding: 3px 8px;
            background: rgba(240, 113, 120, 0.15);
            color: var(--accent-red);
            border-radius: 3px;
        }}

        .clickable-row {{ cursor: pointer; }}
        .clickable-row td {{ user-select: none; }}

        .cluster-meta {{
            font-size: 0.68rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            padding: 4px 8px;
            background: var(--bg-primary);
            border-radius: 3px;
            margin-bottom: 10px;
            display: inline-block;
        }}

        .text-preview {{
            color: var(--text-secondary);
            font-size: 0.72rem;
            line-height: 1.4;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Alert Bar -->
        {alert_bar}

        <!-- Executive Summary -->
        {exec_summary}

        <!-- Clause Analysis Table -->
        <div class="table-section">
            <div class="section-header">COVENANT CLAUSE ANALYSIS ({total})</div>
            <div class="filters">
                <button class="filter-btn active" onclick="filterClauses('all')">All ({total})</button>
                <button class="filter-btn" onclick="filterClauses('outlier')">🔴 Outliers (&lt;70% sim)</button>
                <button class="filter-btn" onclick="filterClauses('borderline')">🟡 Borderline (70–80%)</button>
                <button class="filter-btn" onclick="filterClauses('matched')">🟢 Well Matched (&gt;80%)</button>
                <button class="export-btn" onclick="exportData()">📥 Export JSON</button>
            </div>
            <table id="clauseTable">
                <thead>
                    <tr>
                        <th style="width:40px">#</th>
                        <th>Section / Covenant</th>
                        <th>Preview</th>
                        <th style="width:100px">Similarity</th>
                        <th style="width:160px">Status</th>
                    </tr>
                </thead>
                <tbody id="clauseTableBody"></tbody>
            </table>
        </div>
    </div>

    <script>
        const clauseData = {clause_data_json};

        function populateTable(filter = 'all') {{
            const tbody = document.getElementById('clauseTableBody');
            tbody.innerHTML = '';

            let filtered = clauseData;
            if (filter === 'outlier') {{
                filtered = clauseData.filter(c => c.similarity < 0.70);
            }} else if (filter === 'borderline') {{
                filtered = clauseData.filter(c => c.similarity >= 0.70 && c.similarity < 0.80);
            }} else if (filter === 'matched') {{
                filtered = clauseData.filter(c => c.similarity >= 0.80);
            }}

            filtered.forEach((clause, index) => {{
                // ── Main row (click anywhere to expand) ──
                const row = document.createElement('tr');
                const rowClasses = ['clickable-row'];
                if (clause.similarity < 0.70) rowClasses.push('row-outlier');
                else if (clause.similarity < 0.80) rowClasses.push('row-highly-restrictive');
                row.className = rowClasses.join(' ');
                row.onclick = () => toggleDetail(index);

                // Similarity badge — this is the grounded signal
                const simPct = clause.similarity * 100;
                const simBadge = simPct < 70 ? 'badge-outlier' :
                                 simPct < 80 ? 'badge-medium' : 'badge-low';

                const status = [];
                if (clause.similarity < 0.70) status.push('<span class="badge badge-outlier">OUTLIER</span>');
                else if (clause.similarity < 0.80) status.push('<span class="badge badge-medium">BORDERLINE</span>');
                else status.push('<span class="badge badge-low">MATCHED</span>');

                const preview = clause.plain_english || (clause.text ? clause.text.substring(0, 120) : '');

                row.innerHTML = `
                    <td style="color:var(--text-muted)">${{index + 1}}</td>
                    <td>
                        <div style="font-weight:500;color:var(--text-primary);margin-bottom:2px">${{clause.section_title || clause.clause_id}}</div>
                        <div style="font-size:0.68rem;color:var(--text-muted)">§${{clause.section_number}} · id ${{clause.clause_id}}</div>
                    </td>
                    <td><div class="text-preview">${{preview}}</div></td>
                    <td><span class="badge ${{simBadge}}">${{simPct.toFixed(1)}}%</span></td>
                    <td>${{status.join(' ')}}</td>
                `;
                tbody.appendChild(row);

                // ── Detail row (hidden until row clicked) ──
                const detailRow = document.createElement('tr');
                detailRow.className = clause.similarity < 0.70 ? 'row-outlier' : clause.similarity < 0.80 ? 'row-highly-restrictive' : '';
                detailRow.innerHTML = `
                    <td colspan="5">
                        <div class="clause-detail" id="detail-${{index}}">

                            <!-- BENCHMARK DATA (grounded) -->
                            <div class="detail-section">
                                <h4>Benchmark Cluster Match</h4>
                                <p><strong>Cluster:</strong> ${{clause.cluster_theme}}</p>
                                <p><strong>Similarity to nearest cluster:</strong>
                                    <span class="badge ${{simBadge}}">${{simPct.toFixed(1)}}%</span>
                                    ${{clause.similarity < 0.70 ? ' — below 70% threshold (outlier)' :
                                       clause.similarity < 0.80 ? ' — borderline match' : ' — good match'}}
                                </p>
                            </div>

                            <div class="detail-section">
                                <h4>Full Text</h4>
                                <p class="full-text">${{clause.text}}</p>
                            </div>

                            ${{clause.is_outlier && clause.outlier_analysis ? `
                                <div class="detail-section">
                                    <h4>Why This Doesn't Match the Benchmark</h4>
                                    <p>${{clause.outlier_analysis.why_unusual}}</p>
                                </div>
                                <div class="detail-section">
                                    <h4>Specific Differences</h4>
                                    <ul>${{clause.outlier_analysis.specific_differences.map(d => `<li>${{d}}</li>`).join('')}}</ul>
                                </div>
                                <div class="detail-section">
                                    <h4>vs. Typical ${{clause.cluster_theme}} Clause</h4>
                                    <p>${{clause.outlier_analysis.comparison_to_cluster}}</p>
                                </div>
                                ${{clause.outlier_analysis.requires_lawyer_review ? `
                                    <div class="detail-section">
                                        <h4>Suggested Review Points</h4>
                                        <ul>${{clause.outlier_analysis.lawyer_should_review_for.map(item => `<li>${{item}}</li>`).join('')}}</ul>
                                    </div>
                                ` : ''}}
                            ` : ''}}

                            <!-- LLM ASSESSMENT (uncalibrated — for reference only) -->
                            <div class="detail-section" style="opacity:0.6;border-top:1px solid var(--bg-hover);padding-top:10px;margin-top:4px">
                                <h4 style="color:var(--text-muted)">LLM Assessment (uncalibrated)</h4>
                                <p style="font-size:0.72rem;color:var(--text-muted)">
                                    Restrictiveness: ${{clause.restrictiveness}}/10 &nbsp;·&nbsp; Priority: ${{clause.priority}}
                                    ${{clause.themes && clause.themes.length > 0 ? ' &nbsp;·&nbsp; Themes: ' + clause.themes.join(', ') : ''}}
                                </p>
                                ${{clause.red_flags && clause.red_flags.length > 0 ? `
                                    <p style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">
                                        Flags: ${{clause.red_flags.join(' · ')}}
                                    </p>
                                ` : ''}}
                            </div>

                        </div>
                    </td>
                `;
                tbody.appendChild(detailRow);
            }});
        }}

        function toggleDetail(index) {{
            document.getElementById(`detail-${{index}}`).classList.toggle('show');
        }}

        function filterClauses(filter) {{
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            populateTable(filter);
        }}

        function exportData() {{
            const dataStr = JSON.stringify({{clauses: clauseData}}, null, 2);
            const blob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{deal_id}_covenant_analysis.json';
            a.click();
        }}

        populateTable();
    </script>
</body>
</html>"""

        return html

    def _get_provision_flags(self, provisions: Optional[Dict]) -> List[str]:
        """Get list of provision flags"""
        if not provisions:
            return []

        flags = []
        risk_summary = provisions.get('risk_summary', {})

        high_risk = risk_summary.get('high_risk_provisions', [])
        if high_risk:
            flags.append(f"🔴 {len(high_risk)} HIGH RISK")

        medium_risk = risk_summary.get('medium_risk_provisions', [])
        if medium_risk:
            flags.append(f"🟡 {len(medium_risk)} MEDIUM RISK")

        return flags

    def _generate_alert_bar(
        self, deal_id: str, outliers: int, high_restrictiveness: int, prov_flags: List[str]
    ) -> str:
        """Generate alert bar HTML"""

        flags_html = []

        if outliers > 0:
            flags_html.append(
                f'<span class="alert-flag outliers">⚠️ {outliers} OUTLIER{"S" if outliers > 1 else ""}</span>'
            )

        if high_restrictiveness > 0:
            flags_html.append(
                f'<span class="alert-flag restrictive">🟠 {high_restrictiveness} HIGHLY RESTRICTIVE</span>'
            )

        for flag in prov_flags:
            flags_html.append(f'<span class="alert-flag provision">{flag}</span>')

        return f'''
        <div class="alert-bar">
            <span class="deal-name">{deal_id}</span>
            {" ".join(flags_html) if flags_html else '<span style="color: var(--accent-green); font-size: 0.75rem;">✓ NO MAJOR FLAGS</span>'}
        </div>
        '''

    def _generate_executive_summary(
        self, assessment: Dict, comparison: Optional[Dict], provisions: Optional[Dict]
    ) -> str:
        """Generate executive summary section"""

        summary = assessment.get('summary', {})

        # Calculate metrics
        total_clauses = summary.get('total_clauses', 0)
        avg_restrictiveness = summary.get('avg_restrictiveness', 0)
        median_restrictiveness = summary.get('median_restrictiveness', 0)

        # Outlier statistics
        outlier_stats = summary.get('outlier_statistics', {})
        total_outliers = outlier_stats.get('total_outliers', 0)
        high_risk_outliers = outlier_stats.get('high_risk_outliers', 0)

        # Benchmark comparison if available
        benchmark_html = ''
        if comparison:
            exec_sum = comparison.get('executive_summary', {})
            overall = exec_sum.get('overall_assessment', 'N/A')
            percentile = exec_sum.get('percentile_rank', 'N/A')

            benchmark_html = f'''
            <div class="exec-item">
                <div class="exec-label">Benchmark Position</div>
                <div class="exec-value">{percentile}</div>
            </div>
            <div class="exec-item">
                <div class="exec-label">Assessment</div>
                <div class="exec-value" style="font-size: 0.9rem;">{overall}</div>
            </div>
            '''

        # Provisions summary if available
        provisions_html = ''
        if provisions:
            risk_sum = provisions.get('risk_summary', {})
            risk_score = risk_sum.get('risk_score', 0)
            risk_level = risk_sum.get('overall_risk_level', 'unknown')

            provisions_html = f'''
            <div class="exec-item">
                <div class="exec-label">Provision Risk Score</div>
                <div class="exec-value">{risk_score}/30</div>
            </div>
            <div class="exec-item">
                <div class="exec-label">Overall Risk</div>
                <div class="exec-value" style="font-size: 0.9rem;">{risk_level.upper()}</div>
            </div>
            '''

        return f'''
        <div class="exec-summary">
            <h2>📊 Executive Summary</h2>
            <div class="exec-grid">
                <div class="exec-item">
                    <div class="exec-label">Total Clauses</div>
                    <div class="exec-value">{total_clauses}</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Avg Restrictiveness</div>
                    <div class="exec-value">{avg_restrictiveness:.1f}/10</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Median Restrictiveness</div>
                    <div class="exec-value">{median_restrictiveness:.1f}/10</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">Outliers Found</div>
                    <div class="exec-value">{total_outliers}</div>
                </div>
                <div class="exec-item">
                    <div class="exec-label">High-Risk Outliers</div>
                    <div class="exec-value">{high_risk_outliers}</div>
                </div>
                {benchmark_html}
                {provisions_html}
            </div>
            <div class="exec-insights">
                <strong>Key Insights:</strong><br>
                {self._generate_insights(assessment, comparison, provisions)}
            </div>
        </div>
        '''

    def _generate_insights(
        self, assessment: Dict, comparison: Optional[Dict], provisions: Optional[Dict]
    ) -> str:
        """Generate key insights text"""

        insights = []

        summary = assessment.get('summary', {})

        # Restrictiveness insight
        avg_restrict = summary.get('avg_restrictiveness', 0)
        if avg_restrict >= 7.5:
            insights.append("• Deal contains highly restrictive covenants overall")
        elif avg_restrict <= 5:
            insights.append("• Deal has relatively permissive covenant structure")

        # Outlier insight
        outlier_stats = summary.get('outlier_statistics', {})
        if outlier_stats.get('high_risk_outliers', 0) > 0:
            insights.append(
                f"• {outlier_stats['high_risk_outliers']} high-risk outlier clause(s) require immediate lawyer review"
            )

        # Benchmark insight
        if comparison:
            exec_sum = comparison.get('executive_summary', {})
            insights.append(f"• {exec_sum.get('overall_assessment', 'Position uncertain')}")

        # Provisions insight
        if provisions:
            risk_sum = provisions.get('risk_summary', {})
            high_risk = risk_sum.get('high_risk_provisions', [])
            if high_risk:
                insights.append(f"• Critical provisions flagged: {', '.join(high_risk)}")

        return '<br>'.join(insights) if insights else 'No major concerns identified'

    def _prepare_clause_data(
        self, classification: Dict, assessment: Dict, provisions: Optional[Dict]
    ) -> List[Dict]:
        """Prepare clause data for JavaScript"""

        clause_data = []

        for clause in assessment.get('assessed_clauses', []):
            clause_id = clause.get('clause_id', '')

            data = {
                'clause_id': clause_id,
                'section_title': clause.get('section_title', ''),
                'section_number': clause.get('section_number', ''),
                'text': clause.get('original_text', clause.get('text', clause.get('processed_text', ''))),
                'plain_english': clause.get('covenant_explanation', ''),
                'cluster_theme': clause.get('cluster_theme', 'Unknown'),
                'similarity': clause.get('similarity_score', 0),
                'restrictiveness': clause.get('restrictiveness_score', 0),
                'restrictiveness_category': clause.get('restrictiveness_category', 'unknown'),
                'priority': clause.get('investigation_priority', 'unknown'),
                'is_outlier': clause.get('is_outlier', False),
                'themes': clause.get('covenant_themes', clause.get('themes', [])),
                'red_flags': clause.get('red_flags', [])
            }

            # Add outlier analysis if present
            if clause.get('is_outlier') and 'outlier_analysis' in clause:
                data['outlier_analysis'] = clause['outlier_analysis']

            clause_data.append(data)

        return clause_data


def run_stage10(accession, classification_url, assessment_url, comparison_url=None, provisions_url=None, deal_name=None):
    """Run Stage 10 dashboard generation from S3 URLs and upload result back to S3."""
    from covenant_s3_utils import download_json, upload_text

    print("\n" + "="*80)
    print("🚀 STAGE 10: HTML DASHBOARD GENERATOR (S3 Pipeline)")
    print("="*80)

    print(f"\n📂 Loading data from S3...")

    classification = download_json(classification_url)
    print(f"   ✅ Classification: loaded from S3")

    assessment = download_json(assessment_url)
    print(f"   ✅ Assessment: loaded from S3")

    comparison = None
    if comparison_url:
        comparison = download_json(comparison_url)
        print(f"   ✅ Comparison: loaded from S3")
    else:
        print(f"   ⚠️  Comparison: not provided (optional)")

    provisions = None
    if provisions_url:
        provisions = download_json(provisions_url)
        print(f"   ✅ Provisions: loaded from S3")
    else:
        print(f"   ⚠️  Provisions: not provided (optional)")

    deal_id = classification.get("deal_id", accession)

    base_dir = str(Path(__file__).resolve().parent)
    gen = CovenantDashboardGenerator(base_dir)

    print(f"\n🎨 Generating dashboard for: {deal_id}")
    html = gen._generate_html(classification, assessment, comparison, provisions, deal_id)

    _, dashboard_url = upload_text(html, accession, "dashboard_html.html", content_type="text/html; charset=utf-8")
    print(f"\n✅ Dashboard uploaded to S3: {dashboard_url}")

    return {"dashboard_html": dashboard_url}


def main():
    """Main execution"""
    import sys

    BASE_DIR = f"{Path(__file__).resolve().parent.parent}/Covenant_Embeddings_v1"

    generator = CovenantDashboardGenerator(BASE_DIR)

    # Optional: specify deal name if provided
    deal_name = sys.argv[1] if len(sys.argv) > 1 else None

    html_file = generator.generate_dashboard(deal_name)

    if html_file:
        print(f"\n✅ Dashboard ready!")
        print(f"\n📖 View in browser to see:")
        print(f"   • Outlier clauses with 'WHY unusual' explanations")
        print(f"   • Restrictiveness scores and risk levels")
        print(f"   • Specific provision check results")
        print(f"   • Filterable table with full analysis")


if __name__ == "__main__":
    main()
