#!/usr/bin/env python3
"""
Stage 7: LLM Risk Analysis for New Deal (UPDATED for Stage 6 v2)
Performs targeted LLM analysis on classified clauses, focusing on unusual and high-risk provisions.
Cost: ~$0.05 per analyzed clause (only analyzes flagged clauses), Time: ~30 seconds per deal
Django-compatible version
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from anthropic import Anthropic


class NewDealRiskAnalyzer:
    """Perform targeted LLM risk analysis on new deal"""

    def __init__(self, anthropic_key: str, benchmark_file: Optional[str] = None):
        """
        Initialize risk analyzer

        Args:
            anthropic_key: Anthropic API key for Claude
            benchmark_file: Optional benchmark for comparison
        """
        self.client = Anthropic(api_key=anthropic_key)
        self.benchmark = None
        self.benchmark_file = benchmark_file

        if benchmark_file and os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                self.benchmark = json.load(f)
            print(f"✅ Loaded benchmark for comparison analysis")

    def analyze_new_deal(self, classification_file: str, output_dir: str,
                         analyze_all: bool = False) -> Dict:
        """
        Analyze deal with focus on unusual and high-risk clauses

        Args:
            classification_file: Output from Stage 6
            output_dir: Directory to save results
            analyze_all: If True, analyze all clauses; if False, only outliers

        Returns:
            Comprehensive risk analysis
        """

        # Load classification results
        print(f"📂 Loading classification from: {classification_file}")
        with open(classification_file, 'r') as f:
            classification = json.load(f)

        deal_name = classification['deal_name']
        # NEW: 'results' instead of 'clauses'
        clauses = classification['results']

        print(f"\n📄 Analyzing deal: {deal_name}")
        print(f"   Total clauses: {len(clauses)}")

        # Generate initial risk assessment from Stage 6 zones
        initial_risk = self._generate_initial_risk_assessment(clauses)

        print(f"\n📊 Initial Risk Assessment:")
        print(f"   🟢 Typical: {initial_risk['typical_count']}")
        print(f"   🟡 Atypical: {initial_risk['atypical_count']}")
        print(f"   🔴 Outliers: {initial_risk['outlier_count']}")

        # Select clauses for analysis
        clauses_to_analyze = self._select_clauses_for_analysis(
            clauses,
            initial_risk['risk_flags'],
            analyze_all
        )

        print(
            f"\n🔍 Clauses selected for LLM analysis: {len(clauses_to_analyze)}")
        estimated_cost = len(clauses_to_analyze) * 0.05
        print(f"   Estimated cost: ${estimated_cost:.2f}")

        if len(clauses_to_analyze) > 20:
            print(f"\n⚠️  This will analyze {len(clauses_to_analyze)} clauses")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Analysis cancelled.")
                return None

        # Perform risk analysis on selected clauses
        analyzed_clauses = []
        high_risk_clauses = []
        medium_risk_clauses = []
        disclosure_references = []
        unusual_provisions = []

        print("\n🔍 Analyzing clauses with Claude Sonnet 4...")
        for i, clause in enumerate(clauses_to_analyze, 1):
            print(
                f"   [{i:2d}/{len(clauses_to_analyze)}] {clause.get('label', 'N/A')}: {clause['best_match']['cluster_name'][:50]}...")

            # Perform LLM risk analysis
            risk_analysis = self._analyze_clause_risk(
                clause['text'],
                deal_name,
                clause.get('label', ''),
                clause['best_match']['cluster_name']
            )

            # Build analyzed clause object
            analyzed_clause = {
                'clause_id': clause.get('label', f"clause_{i}"),
                'label': clause.get('label', ''),
                'text': clause['text'],
                'text_preview': clause['text'][:100] + "..." if len(clause['text']) > 100 else clause['text'],
                'cluster_match': clause['best_match']['cluster_name'],
                'cluster_category': clause['best_match']['category'],
                'distance_ratio': clause['best_match']['ratio'],
                'zone': clause['zone'],
                'risk_analysis': risk_analysis
            }

            analyzed_clauses.append(analyzed_clause)

            # Categorize by risk
            risk_level = risk_analysis.get('risk_level', 'low')
            if risk_level == 'high':
                high_risk_clauses.append(analyzed_clause)
            elif risk_level == 'medium':
                medium_risk_clauses.append(analyzed_clause)

            # Track specific flags
            if risk_analysis.get('disclosure_references'):
                disclosure_references.append(analyzed_clause)

            if risk_analysis.get('unusual_provisions'):
                unusual_provisions.extend(risk_analysis['unusual_provisions'])

        print(f"\n✅ Analysis complete!")
        print(f"   High risk: {len(high_risk_clauses)}")
        print(f"   Medium risk: {len(medium_risk_clauses)}")
        print(f"   Disclosure references: {len(disclosure_references)}")

        # Generate final risk assessment
        final_risk_assessment = self._generate_final_risk_assessment(
            analyzed_clauses,
            high_risk_clauses,
            medium_risk_clauses
        )

        # Generate investigation report
        investigation_report = self._generate_investigation_report(
            high_risk_clauses,
            disclosure_references
        )

        # Compare to benchmark if available
        benchmark_comparison = None
        if self.benchmark:
            benchmark_comparison = self._compare_to_benchmark(
                clauses,
                analyzed_clauses
            )

        # Generate negotiation recommendations
        negotiation_points = self._generate_negotiation_recommendations(
            high_risk_clauses,
            medium_risk_clauses,
            unusual_provisions
        )

        # Prepare comprehensive results
        results = {
            'deal_name': deal_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clauses': len(clauses),
            'clauses_analyzed': len(analyzed_clauses),
            'analysis_coverage': f"{(len(analyzed_clauses)/len(clauses)*100):.1f}%",
            'initial_risk_assessment': initial_risk,
            'risk_summary': {
                'final_risk_level': final_risk_assessment['overall_risk_level'],
                'risk_score': final_risk_assessment['risk_score'],
                'high_risk_count': len(high_risk_clauses),
                'medium_risk_count': len(medium_risk_clauses),
                'low_risk_count': len(analyzed_clauses) - len(high_risk_clauses) - len(medium_risk_clauses),
                'disclosure_references_count': len(disclosure_references),
                'key_concerns': final_risk_assessment['key_concerns']
            },
            'investigation_report': investigation_report,
            'negotiation_recommendations': negotiation_points,
            'benchmark_comparison': benchmark_comparison,
            'detailed_analysis': {
                'high_risk_clauses': self._format_clause_details(high_risk_clauses),
                'medium_risk_clauses': self._format_clause_details(medium_risk_clauses),
                'disclosure_references': self._format_clause_details(disclosure_references),
                'all_analyzed_clauses': analyzed_clauses
            },
            'unusual_provisions_summary': list(set(unusual_provisions)),
            'red_flags': self._extract_red_flags(analyzed_clauses),
            'classification_file': os.path.basename(classification_file)
        }

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main analysis file
        output_file = f"{output_dir}/risk_analysis_{deal_name}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save high-risk summary CSV
        if high_risk_clauses:
            csv_file = f"{output_dir}/high_risk_clauses_{deal_name}_{timestamp}.csv"
            df_high_risk = pd.DataFrame([
                {
                    'clause_id': c['clause_id'],
                    'label': c.get('label', ''),
                    'text_preview': c['text_preview'],
                    'cluster_match': c['cluster_match'],
                    'distance_ratio': f"{c['distance_ratio']:.2f}",
                    'zone': c['zone'],
                    'risk_level': c['risk_analysis']['risk_level'],
                    'key_issues': ', '.join(c['risk_analysis'].get('risk_factors', [])),
                    'investigation_priority': c['risk_analysis'].get('investigation_priority', 'unknown'),
                    'explanation': c['risk_analysis'].get('explanation', '')
                }
                for c in high_risk_clauses
            ])
            df_high_risk.to_csv(csv_file, index=False)
            print(f"\n📄 High-risk summary: {csv_file}")

        print(f"\n✅ Risk analysis saved: {output_file}")

        # Print summary
        self._print_analysis_summary(results)

        return results

    def analyze_new_deal_from_data(self, classification: dict, analyze_all: bool = False,
                                   auto_confirm: bool = True) -> Optional[Dict]:
        """
        Analyze deal from in-memory classification dict. No file I/O.
        classification: {deal_name, results} from Stage 6.
        auto_confirm: if True, skip interactive prompt when >20 clauses.
        Returns: risk analysis dict (same as analyze_new_deal).
        """
        deal_name = classification['deal_name']
        clauses = classification['results']

        print(f"\n📄 Analyzing deal: {deal_name}")
        print(f"   Total clauses: {len(clauses)}")

        initial_risk = self._generate_initial_risk_assessment(clauses)
        print(f"\n📊 Initial Risk Assessment:")
        print(f"   🟢 Typical: {initial_risk['typical_count']}")
        print(f"   🟡 Atypical: {initial_risk['atypical_count']}")
        print(f"   🔴 Outliers: {initial_risk['outlier_count']}")

        clauses_to_analyze = self._select_clauses_for_analysis(
            clauses, initial_risk['risk_flags'], analyze_all
        )
        print(
            f"\n🔍 Clauses selected for LLM analysis: {len(clauses_to_analyze)}")
        estimated_cost = len(clauses_to_analyze) * 0.05
        print(f"   Estimated cost: ${estimated_cost:.2f}")

        if len(clauses_to_analyze) > 20 and not auto_confirm:
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Analysis cancelled.")
                return None

        analyzed_clauses = []
        high_risk_clauses = []
        medium_risk_clauses = []
        disclosure_references = []
        unusual_provisions = []

        print("\n🔍 Analyzing clauses with Claude Sonnet 4...")
        for i, clause in enumerate(clauses_to_analyze, 1):
            print(
                f"   [{i:2d}/{len(clauses_to_analyze)}] {clause.get('label', 'N/A')}: {clause['best_match']['cluster_name'][:50]}...")
            risk_analysis = self._analyze_clause_risk(
                clause['text'], deal_name, clause.get('label', ''),
                clause['best_match']['cluster_name']
            )
            analyzed_clause = {
                'clause_id': clause.get('label', f"clause_{i}"),
                'label': clause.get('label', ''),
                'text': clause['text'],
                'text_preview': clause['text'][:100] + "..." if len(clause['text']) > 100 else clause['text'],
                'cluster_match': clause['best_match']['cluster_name'],
                'cluster_category': clause['best_match']['category'],
                'distance_ratio': clause['best_match']['ratio'],
                'zone': clause['zone'],
                'risk_analysis': risk_analysis
            }
            analyzed_clauses.append(analyzed_clause)
            risk_level = risk_analysis.get('risk_level', 'low')
            if risk_level == 'high':
                high_risk_clauses.append(analyzed_clause)
            elif risk_level == 'medium':
                medium_risk_clauses.append(analyzed_clause)
            if risk_analysis.get('disclosure_references'):
                disclosure_references.append(analyzed_clause)
            if risk_analysis.get('unusual_provisions'):
                unusual_provisions.extend(risk_analysis['unusual_provisions'])

        print(
            f"\n✅ Analysis complete! High risk: {len(high_risk_clauses)}, Medium: {len(medium_risk_clauses)}")

        final_risk_assessment = self._generate_final_risk_assessment(
            analyzed_clauses, high_risk_clauses, medium_risk_clauses
        )
        investigation_report = self._generate_investigation_report(
            high_risk_clauses, disclosure_references
        )
        benchmark_comparison = None
        if self.benchmark:
            benchmark_comparison = self._compare_to_benchmark(
                clauses, analyzed_clauses)
        negotiation_points = self._generate_negotiation_recommendations(
            high_risk_clauses, medium_risk_clauses, unusual_provisions
        )

        results = {
            'deal_name': deal_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clauses': len(clauses),
            'clauses_analyzed': len(analyzed_clauses),
            'analysis_coverage': f"{(len(analyzed_clauses)/len(clauses)*100):.1f}%" if clauses else "0%",
            'initial_risk_assessment': initial_risk,
            'risk_summary': {
                'final_risk_level': final_risk_assessment['overall_risk_level'],
                'risk_score': final_risk_assessment['risk_score'],
                'high_risk_count': len(high_risk_clauses),
                'medium_risk_count': len(medium_risk_clauses),
                'low_risk_count': len(analyzed_clauses) - len(high_risk_clauses) - len(medium_risk_clauses),
                'disclosure_references_count': len(disclosure_references),
                'key_concerns': final_risk_assessment['key_concerns']
            },
            'investigation_report': investigation_report,
            'negotiation_recommendations': negotiation_points,
            'benchmark_comparison': benchmark_comparison,
            'detailed_analysis': {
                'high_risk_clauses': self._format_clause_details(high_risk_clauses),
                'medium_risk_clauses': self._format_clause_details(medium_risk_clauses),
                'disclosure_references': self._format_clause_details(disclosure_references),
                'all_analyzed_clauses': analyzed_clauses
            },
            'unusual_provisions_summary': list(set(unusual_provisions)),
            'red_flags': self._extract_red_flags(analyzed_clauses),
            'classification_file': None
        }
        self._print_analysis_summary(results)
        return results

    def _generate_initial_risk_assessment(self, clauses: List[Dict]) -> Dict:
        """Generate initial risk assessment from Stage 6 classification zones"""

        typical_count = sum(1 for c in clauses if c['zone'] == 'typical')
        atypical_count = sum(1 for c in clauses if c['zone'] == 'atypical')
        outlier_count = sum(1 for c in clauses if c['zone'] == 'outlier')

        # Identify risk flags (atypical + outliers)
        risk_flags = []
        for clause in clauses:
            if clause['zone'] in ['atypical', 'outlier']:
                risk_flags.append({
                    'clause_id': clause.get('label', ''),
                    'reason': f"{clause['zone'].upper()}: {clause['best_match']['cluster_name']}",
                    'ratio': clause['best_match']['ratio']
                })

        # Calculate overall risk level
        if outlier_count > len(clauses) * 0.3:
            overall_risk = 'high'
        elif outlier_count > 0 or atypical_count > len(clauses) * 0.2:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        return {
            'typical_count': typical_count,
            'atypical_count': atypical_count,
            'outlier_count': outlier_count,
            'risk_flags': risk_flags,
            'overall_risk': overall_risk,
            'requires_review': outlier_count > 0 or atypical_count > 0
        }

    def _select_clauses_for_analysis(self, clauses: List[Dict],
                                     risk_flags: List[Dict],
                                     analyze_all: bool) -> List[Dict]:
        """
        Intelligently select which clauses need LLM analysis
        """

        if analyze_all:
            return clauses

        # Create set of flagged clause IDs
        flagged_ids = {flag['clause_id'] for flag in risk_flags}

        clauses_to_analyze = []

        for clause in clauses:
            clause_id = clause.get('label', '')

            # Always analyze flagged clauses (atypical + outliers)
            if clause_id in flagged_ids:
                clauses_to_analyze.append(clause)

            # Also analyze outliers based on zone
            elif clause['zone'] == 'outlier':
                clauses_to_analyze.append(clause)

        return clauses_to_analyze

    def _analyze_clause_risk(self, text: str, deal_name: str,
                             label: str, cluster_label: str) -> Dict:
        """Perform LLM risk analysis on a single clause"""

        prompt = f"""Analyze this Material Adverse Effect (MAE) exclusion clause for risk.

CLAUSE TEXT:
{text}

DEAL: {deal_name}
LABEL: {label}
CLUSTER: {cluster_label}

Provide analysis in this JSON format:
{{
    "risk_level": "low|medium|high",
    "requires_investigation": true/false,
    "risk_factors": ["factor1", "factor2"],
    "unusual_provisions": ["provision1", "provision2"], 
    "disclosure_references": true/false,
    "specific_company_issues": ["issue1", "issue2"],
    "red_flags": ["flag1", "flag2"],
    "investigation_priority": "routine|review|urgent",
    "explanation": "detailed explanation of concerns",
    "negotiation_impact": "how this affects deal negotiations",
    "similar_to_standard": true/false,
    "confidence": 1-10
}}

HIGH RISK INDICATORS:
- Disclosure schedule/letter references (suggests private material issues)
- Specific company problems (SEC filings, litigation, regulatory issues)
- Unusual specificity (detailed dates, specific transactions, named parties)
- Vague matter references that could hide material issues
- Financial reporting problems (10-K amendments, disclosure controls)
- References to specific ongoing investigations or proceedings
- Carve-outs for known but undisclosed issues

MEDIUM RISK:
- Industry-specific carve-outs that seem tailored to known issues
- Unusual timing restrictions or conditions
- References to specific regulatory actions or changes
- Non-standard language compared to typical MAE exclusions
- Overly broad exclusions that could encompass material issues

LOW RISK (Standard exclusions):
- General economic conditions, market changes
- War, terrorism, natural disasters  
- Changes in GAAP or applicable law
- Industry-wide effects
- Stock price or trading volume changes
- Failure to meet projections (without more)
- Announcement of the transaction itself

Focus on identifying what makes this clause unusual or concerning compared to standard MAE exclusions. Be specific about any red flags."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # Parse JSON response
            response_text = response.content[0].text

            # Clean up response to extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]

            # Clean control characters
            response_text = response_text.strip()
            import string
            printable = set(string.printable)
            response_text = ''.join(
                filter(lambda x: x in printable, response_text))
            response_text = response_text.replace(
                '\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')

            result = json.loads(response_text)

            # Ensure all required fields
            required_fields = {
                'risk_level': 'unknown',
                'requires_investigation': False,
                'risk_factors': [],
                'unusual_provisions': [],
                'disclosure_references': False,
                'specific_company_issues': [],
                'red_flags': [],
                'investigation_priority': 'routine',
                'explanation': 'Analysis pending',
                'negotiation_impact': '',
                'similar_to_standard': True,
                'confidence': 5
            }

            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default

            return result

        except Exception as e:
            print(f"\n❌ Error analyzing clause: {e}")
            return {
                'risk_level': 'unknown',
                'requires_investigation': False,
                'risk_factors': [f"analysis_error: {str(e)[:100]}"],
                'unusual_provisions': [],
                'disclosure_references': False,
                'specific_company_issues': [],
                'red_flags': [],
                'investigation_priority': 'routine',
                'explanation': f"Analysis failed: {str(e)[:100]}",
                'negotiation_impact': 'Unknown due to analysis error',
                'similar_to_standard': True,
                'confidence': 0
            }

    def _generate_final_risk_assessment(self, analyzed_clauses: List[Dict],
                                        high_risk_clauses: List[Dict],
                                        medium_risk_clauses: List[Dict]) -> Dict:
        """Generate final overall risk assessment"""

        # Calculate risk score (0-100)
        if not analyzed_clauses:
            risk_score = 0
        else:
            high_weight = len(high_risk_clauses) * 10
            medium_weight = len(medium_risk_clauses) * 5
            risk_score = min(
                100, (high_weight + medium_weight) / len(analyzed_clauses) * 10)

        # Determine overall risk level
        if risk_score >= 30:
            overall_risk = 'high'
        elif risk_score >= 15:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        # Extract key concerns
        key_concerns = []
        for clause in high_risk_clauses[:3]:  # Top 3
            key_concerns.append({
                'clause': clause['label'],
                'concern': clause['risk_analysis'].get('explanation', 'Unknown')[:100]
            })

        return {
            'overall_risk_level': overall_risk,
            'risk_score': risk_score,
            'key_concerns': key_concerns
        }

    def _generate_investigation_report(self, high_risk_clauses: List[Dict],
                                       disclosure_references: List[Dict]) -> Dict:
        """Generate detailed investigation report"""

        report = {
            'executive_summary': '',
            'immediate_actions_required': [],
            'key_negotiation_points': [],
            'red_flag_summary': [],
            'disclosure_schedule_concerns': [],
            'unusual_provisions_detail': []
        }

        # Collect medium risk clauses for total count
        medium_risk = [c for c in [] if c.get('risk_analysis', {}).get(
            'risk_level') == 'medium']  # Will be populated in analyze_new_deal

        # Executive summary
        total_concerning = len(high_risk_clauses)
        if len(high_risk_clauses) > 5 or len(disclosure_references) > 2:
            report['executive_summary'] = (
                f"⚠️ SIGNIFICANT CONCERNS: This deal contains {len(high_risk_clauses)} high-risk provisions "
                f"and {len(disclosure_references)} references to undisclosed materials. "
                f"Immediate legal review recommended before proceeding."
            )
            report['immediate_actions_required'] = [
                "Request and review all disclosure schedules",
                "Conduct detailed due diligence on flagged items",
                "Legal team review of non-standard provisions"
            ]
        elif total_concerning > 3:
            report['executive_summary'] = (
                f"MODERATE CONCERNS: {total_concerning} provisions require additional review. "
                f"Standard due diligence with focus on flagged items recommended."
            )
            report['immediate_actions_required'] = [
                "Review flagged provisions with legal team",
                "Verify no undisclosed material issues"
            ]
        else:
            report['executive_summary'] = (
                f"STANDARD RISK PROFILE: Most provisions align with market standards. "
                f"Only {total_concerning} items flagged for routine review."
            )
            report['immediate_actions_required'] = [
                "Standard due diligence process",
                "Routine review of flagged items"
            ]

        # Key negotiation points from high-risk items
        for clause in high_risk_clauses[:5]:  # Top 5 high-risk items
            if clause.get('risk_analysis', {}).get('negotiation_impact'):
                report['key_negotiation_points'].append({
                    'clause_id': clause.get('label', clause['clause_id']),
                    'issue': clause['risk_analysis'].get('explanation', ''),
                    'recommendation': clause['risk_analysis'].get('negotiation_impact', '')
                })

        # Red flags summary
        all_red_flags = set()
        for clause in high_risk_clauses:
            flags = clause.get('risk_analysis', {}).get('red_flags', [])
            all_red_flags.update(flags)
        report['red_flag_summary'] = list(all_red_flags)

        # Disclosure schedule concerns
        for clause in disclosure_references:
            report['disclosure_schedule_concerns'].append({
                'clause_id': clause.get('label', clause['clause_id']),
                'text_preview': clause['text_preview'],
                'concern': clause.get('risk_analysis', {}).get('explanation', '')
            })

        return report

    def _compare_to_benchmark(self, clauses: List[Dict],
                              analyzed_clauses: List[Dict]) -> Dict:
        """Compare deal to benchmark statistics"""

        comparison = {
            'vs_typical_deals': {},
            'statistical_outliers': [],
            'category_distribution': {},
            'risk_distribution': {}
        }

        # Calculate our statistics
        total = len(clauses)
        typical = sum(1 for c in clauses if c['zone'] == 'typical')
        atypical = sum(1 for c in clauses if c['zone'] == 'atypical')
        outlier = sum(1 for c in clauses if c['zone'] == 'outlier')

        comparison['vs_typical_deals']['typical_rate'] = {
            'this_deal': f"{(typical/total*100):.1f}%",
            'note': 'Higher is better - most clauses match standard patterns'
        }

        comparison['vs_typical_deals']['outlier_rate'] = {
            'this_deal': f"{(outlier/total*100):.1f}%",
            'note': 'Lower is better - fewer unusual provisions'
        }

        # Category distribution
        categories = {}
        for clause in clauses:
            cat = clause['best_match']['category']
            categories[cat] = categories.get(cat, 0) + 1

        comparison['category_distribution'] = {
            cat: f"{count}/{total} ({count/total*100:.1f}%)"
            for cat, count in categories.items()
        }

        # Risk distribution from analyzed clauses
        if analyzed_clauses:
            risk_levels = {}
            for clause in analyzed_clauses:
                risk = clause['risk_analysis'].get('risk_level', 'unknown')
                risk_levels[risk] = risk_levels.get(risk, 0) + 1

            comparison['risk_distribution'] = risk_levels

        return comparison

    def _generate_negotiation_recommendations(self, high_risk_clauses: List[Dict],
                                              medium_risk_clauses: List[Dict],
                                              unusual_provisions: List[str]) -> List[str]:
        """Generate negotiation points"""

        recommendations = []

        if high_risk_clauses:
            recommendations.append(
                f"Request clarification on {len(high_risk_clauses)} high-risk provisions")
            for clause in high_risk_clauses[:2]:
                if clause['risk_analysis'].get('negotiation_impact'):
                    recommendations.append(
                        f"  • {clause['label']}: {clause['risk_analysis']['negotiation_impact'][:100]}")

        if len(unusual_provisions) > 3:
            recommendations.append(
                f"Review {len(unusual_provisions)} unusual provisions for potential narrowing")

        if not recommendations:
            recommendations.append(
                "No significant negotiation concerns identified")

        return recommendations

    def _format_clause_details(self, clauses: List[Dict]) -> List[Dict]:
        """Format clause details for output"""
        return [
            {
                'clause_id': c['clause_id'],
                'label': c['label'],
                'text_preview': c['text_preview'],
                'cluster': c['cluster_match'],
                'category': c['cluster_category'],
                'distance_ratio': c['distance_ratio'],
                'zone': c['zone'],
                'risk_level': c['risk_analysis']['risk_level'],
                'investigation_priority': c['risk_analysis']['investigation_priority'],
                'key_issues': ', '.join(c['risk_analysis'].get('risk_factors', []))
            }
            for c in clauses
        ]

    def _extract_red_flags(self, analyzed_clauses: List[Dict]) -> List[str]:
        """Extract all red flags from analyzed clauses"""
        red_flags = []
        for clause in analyzed_clauses:
            flags = clause['risk_analysis'].get('red_flags', [])
            if flags:
                for flag in flags:
                    red_flags.append(f"{clause['label']}: {flag}")
        return red_flags

    def _print_analysis_summary(self, results: Dict):
        """Print analysis summary to console"""
        print("\n" + "="*80)
        print(f"📊 RISK ANALYSIS SUMMARY: {results['deal_name']}")
        print("="*80)

        summary = results['risk_summary']
        print(f"\n🎯 Overall Risk: {summary['final_risk_level'].upper()}")
        print(f"   Risk Score: {summary['risk_score']:.1f}/100")
        print(f"\n📋 Analysis Coverage: {results['analysis_coverage']}")
        print(
            f"   Clauses analyzed: {results['clauses_analyzed']}/{results['total_clauses']}")

        print(f"\n⚠️  Risk Distribution:")
        print(f"   🔴 High risk: {summary['high_risk_count']}")
        print(f"   🟡 Medium risk: {summary['medium_risk_count']}")
        print(f"   🟢 Low risk: {summary['low_risk_count']}")

        if summary['disclosure_references_count'] > 0:
            print(
                f"\n📄 Disclosure References: {summary['disclosure_references_count']}")

        if summary['key_concerns']:
            print(f"\n🚨 Key Concerns:")
            for concern in summary['key_concerns']:
                print(f"   • {concern['clause']}: {concern['concern']}")

        print("="*80)


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()

    # Configuration
    ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY')
    if not ANTHROPIC_KEY:
        print("❌ ERROR: ANTHROPIC_API_KEY not set in .env")
        return

    # Paths - relative to script (project root = parent of MAE v2)
    _script_dir = Path(__file__).resolve().parent
    _base_dir = _script_dir.parent
    _classification_dir = _base_dir / "classification_output"
    _classification_files = sorted(_classification_dir.glob(
        "classification_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    CLASSIFICATION_FILE = str(
        _classification_files[0]) if _classification_files else ""
    BENCHMARK_FILE = str(_base_dir / "final_results" /
                         "benchmark_20260211_151459.json")
    OUTPUT_DIR = str(_base_dir / "risk_analysis_output")

    # Check files exist
    if not CLASSIFICATION_FILE or not Path(CLASSIFICATION_FILE).exists():
        print("❌ No classification file found in classification_output/")
        print(f"   Looked in: {_classification_dir}")
        print("\n💡 Run Stage 6 first to generate classification")
        return

    # Initialize analyzer
    analyzer = NewDealRiskAnalyzer(ANTHROPIC_KEY, BENCHMARK_FILE)

    # Run analysis
    print("\n🚀 Starting Stage 7: Risk Analysis")
    print("="*80)

    results = analyzer.analyze_new_deal(
        classification_file=CLASSIFICATION_FILE,
        output_dir=OUTPUT_DIR,
        analyze_all=False  # Only analyze outliers/atypical by default
    )

    if results:
        print(f"\n✅ Stage 7 complete!")
        print(f"\n💡 Next: Run Stage 8 to generate Excel reports")


if __name__ == "__main__":
    main()
