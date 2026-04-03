#!/usr/bin/env python3
"""
Stage 8: Compare New Deal to Benchmark

Takes the assessed new deal from Stage 7 and generates a comprehensive
comparison report against the benchmark statistics.

Answers questions like:
- Is this deal more/less restrictive than benchmark average?
- Which covenant categories are over/under-represented?
- How does this compare to typical market deals?
- What are the key differences?

Usage:
    python3 8_compare_to_benchmark.py path/to/deal_assessment_DEALID_*.json
"""

import json
import os
import sys
from typing import Dict, List
from datetime import datetime
import pandas as pd
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CACHE_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/embeddings_cache"
OUTPUT_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/new_deal_reports"


class BenchmarkComparator:
    """Compares new deal to benchmark statistics"""

    def __init__(self):
        self.benchmark_stats = None
        self.benchmark_data = None

    def load_benchmark_stats(self, cache_dir: str):
        """Load benchmark statistics"""
        print("\n📂 Loading benchmark statistics...")

        # Find most recent assessment summary
        assessment_files = sorted([f for f in os.listdir(cache_dir)
                                  if f.startswith('covenant_assessment_summary_')])

        if not assessment_files:
            raise FileNotFoundError("No benchmark assessment summary found")

        summary_file = assessment_files[-1]
        summary_path = os.path.join(cache_dir, summary_file)

        with open(summary_path, 'r') as f:
            self.benchmark_stats = json.load(f)

        print(f"  ✓ Loaded: {summary_file}")
        print(f"    • Benchmark clauses: {self.benchmark_stats['quality_summary']['valid_covenants']}")
        print(f"    • Avg restrictiveness: {self.benchmark_stats['covenant_summary']['avg_restrictiveness']:.2f}")

        # Also load full risk assessment for clause-level comparison
        risk_files = sorted([f for f in os.listdir(cache_dir)
                            if f.startswith('covenant_risk_assessment_')])

        if risk_files:
            risk_file = risk_files[-1]
            risk_path = os.path.join(cache_dir, risk_file)

            with open(risk_path, 'r') as f:
                self.benchmark_data = json.load(f)

            print(f"  ✓ Loaded risk assessment: {risk_file}")

    def compare_restrictiveness(self, deal_summary: Dict) -> Dict:
        """Compare restrictiveness scores"""

        benchmark_avg = self.benchmark_stats['covenant_summary']['avg_restrictiveness']
        benchmark_median = self.benchmark_stats['covenant_summary']['median_restrictiveness']

        deal_avg = deal_summary['avg_restrictiveness']
        deal_median = deal_summary['median_restrictiveness']

        diff_avg = deal_avg - benchmark_avg
        diff_median = deal_median - benchmark_median

        # Categorize
        if diff_avg >= 1.5:
            category = "significantly_more_restrictive"
        elif diff_avg >= 0.5:
            category = "more_restrictive"
        elif diff_avg <= -1.5:
            category = "significantly_less_restrictive"
        elif diff_avg <= -0.5:
            category = "less_restrictive"
        else:
            category = "similar_to_benchmark"

        return {
            'benchmark_avg': benchmark_avg,
            'benchmark_median': float(benchmark_median),
            'deal_avg': deal_avg,
            'deal_median': float(deal_median),
            'difference_avg': diff_avg,
            'difference_median': diff_median,
            'category': category,
            'percentile_rank': self._calculate_percentile(deal_avg, benchmark_avg)
        }

    def _calculate_percentile(self, deal_avg: float, benchmark_avg: float) -> str:
        """Estimate percentile rank"""
        # Simplified percentile calculation
        if deal_avg >= benchmark_avg + 1.5:
            return "top_10%_most_restrictive"
        elif deal_avg >= benchmark_avg + 0.5:
            return "top_25%_most_restrictive"
        elif deal_avg <= benchmark_avg - 1.5:
            return "top_10%_least_restrictive"
        elif deal_avg <= benchmark_avg - 0.5:
            return "top_25%_least_restrictive"
        else:
            return "middle_50%"

    def compare_categories(self, deal_summary: Dict) -> Dict:
        """Compare covenant category distributions"""

        benchmark_themes = self.benchmark_stats['covenant_summary']['theme_distribution']
        deal_themes = deal_summary['theme_distribution']

        # Calculate percentages
        benchmark_total = sum(benchmark_themes.values())
        deal_total = deal_summary['total_clauses']

        comparison = {}

        all_themes = set(list(benchmark_themes.keys()) + list(deal_themes.keys()))

        for theme in all_themes:
            bench_count = benchmark_themes.get(theme, 0)
            deal_count = deal_themes.get(theme, 0)

            bench_pct = (bench_count / benchmark_total * 100) if benchmark_total > 0 else 0
            deal_pct = (deal_count / deal_total * 100) if deal_total > 0 else 0

            diff_pct = deal_pct - bench_pct

            comparison[theme] = {
                'benchmark_count': bench_count,
                'benchmark_percentage': bench_pct,
                'deal_count': deal_count,
                'deal_percentage': deal_pct,
                'difference_percentage': diff_pct,
                'assessment': self._assess_theme_difference(diff_pct)
            }

        return comparison

    def _assess_theme_difference(self, diff_pct: float) -> str:
        """Assess significance of theme difference"""
        if diff_pct >= 10:
            return "significantly_more_than_benchmark"
        elif diff_pct >= 5:
            return "more_than_benchmark"
        elif diff_pct <= -10:
            return "significantly_less_than_benchmark"
        elif diff_pct <= -5:
            return "less_than_benchmark"
        else:
            return "similar_to_benchmark"

    def compare_priorities(self, deal_summary: Dict) -> Dict:
        """Compare investigation priorities"""

        benchmark_priorities = self.benchmark_stats['covenant_summary']['priority_distribution']
        deal_priorities = deal_summary['priority_distribution']

        benchmark_total = sum(benchmark_priorities.values())
        deal_total = deal_summary['total_clauses']

        return {
            'urgent': {
                'benchmark_pct': benchmark_priorities.get('urgent', 0) / benchmark_total * 100,
                'deal_pct': deal_priorities.get('urgent', 0) / deal_total * 100,
                'deal_count': deal_priorities.get('urgent', 0)
            },
            'routine': {
                'benchmark_pct': benchmark_priorities.get('routine', 0) / benchmark_total * 100,
                'deal_pct': deal_priorities.get('routine', 0) / deal_total * 100,
                'deal_count': deal_priorities.get('routine', 0)
            },
            'review': {
                'benchmark_pct': benchmark_priorities.get('review', 0) / benchmark_total * 100,
                'deal_pct': deal_priorities.get('review', 0) / deal_total * 100,
                'deal_count': deal_priorities.get('review', 0)
            }
        }

    def identify_outliers(self, assessed_clauses: List[Dict]) -> List[Dict]:
        """Identify clauses that are unusual compared to benchmark"""

        outliers = []

        for clause in assessed_clauses:
            is_outlier = False
            reasons = []

            # Very high restrictiveness (9-10) in a typically permissive category
            if clause['restrictiveness_score'] >= 9 and clause.get('similarity_score', 1.0) < 0.75:
                is_outlier = True
                reasons.append("Highly restrictive clause with low benchmark similarity")

            # Red flags present
            if clause.get('red_flags') and len(clause['red_flags']) > 0:
                is_outlier = True
                reasons.append(f"Contains {len(clause['red_flags'])} red flag(s)")

            # Unusual provisions
            if clause.get('unusual_provisions') and len(clause['unusual_provisions']) > 0:
                is_outlier = True
                reasons.append(f"Contains {len(clause['unusual_provisions'])} unusual provision(s)")

            if is_outlier:
                outliers.append({
                    'clause_id': clause.get('clause_id', ''),
                    'section_title': clause.get('section_title', ''),
                    'restrictiveness_score': clause['restrictiveness_score'],
                    'similarity_to_benchmark': clause.get('similarity_score', 0),
                    'reasons': reasons,
                    'red_flags': clause.get('red_flags', []),
                    'unusual_provisions': clause.get('unusual_provisions', [])
                })

        return outliers

    def generate_insights(self, restrictiveness_comp: Dict, category_comp: Dict,
                         priority_comp: Dict, outliers: List[Dict]) -> List[str]:
        """Generate key insights"""

        insights = []

        # Restrictiveness insights
        if restrictiveness_comp['category'] == 'significantly_more_restrictive':
            insights.append(f"🔴 This deal is SIGNIFICANTLY MORE RESTRICTIVE than benchmark "
                          f"(avg {restrictiveness_comp['deal_avg']:.1f} vs {restrictiveness_comp['benchmark_avg']:.1f})")
        elif restrictiveness_comp['category'] == 'more_restrictive':
            insights.append(f"🟡 This deal is MORE RESTRICTIVE than benchmark "
                          f"(avg {restrictiveness_comp['deal_avg']:.1f} vs {restrictiveness_comp['benchmark_avg']:.1f})")
        elif restrictiveness_comp['category'] == 'significantly_less_restrictive':
            insights.append(f"🟢 This deal is SIGNIFICANTLY LESS RESTRICTIVE than benchmark "
                          f"(avg {restrictiveness_comp['deal_avg']:.1f} vs {restrictiveness_comp['benchmark_avg']:.1f})")
        elif restrictiveness_comp['category'] == 'less_restrictive':
            insights.append(f"🟢 This deal is LESS RESTRICTIVE than benchmark "
                          f"(avg {restrictiveness_comp['deal_avg']:.1f} vs {restrictiveness_comp['benchmark_avg']:.1f})")
        else:
            insights.append(f"✅ This deal has SIMILAR restrictiveness to benchmark "
                          f"(avg {restrictiveness_comp['deal_avg']:.1f} vs {restrictiveness_comp['benchmark_avg']:.1f})")

        # Category insights - find biggest differences
        category_diffs = [(theme, data['difference_percentage'])
                         for theme, data in category_comp.items()]
        category_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

        for theme, diff in category_diffs[:3]:
            if abs(diff) >= 5:
                if diff > 0:
                    insights.append(f"📊 More {theme.replace('_', ' ')} clauses than benchmark (+{diff:.1f}%)")
                else:
                    insights.append(f"📊 Fewer {theme.replace('_', ' ')} clauses than benchmark ({diff:.1f}%)")

        # Outlier insights
        if len(outliers) > 0:
            insights.append(f"⚠️  {len(outliers)} outlier clauses detected (unusual vs benchmark)")

        # Priority insights
        urgent_deal = priority_comp['urgent']['deal_pct']
        urgent_bench = priority_comp['urgent']['benchmark_pct']

        if urgent_deal > urgent_bench + 10:
            insights.append(f"🔥 Higher proportion of urgent items ({urgent_deal:.0f}% vs {urgent_bench:.0f}% benchmark)")

        return insights

    def generate_comparison_report(self, assessment_data: Dict, output_dir: str) -> str:
        """Generate comprehensive comparison report"""

        print("\n📊 Generating benchmark comparison report...")

        deal_id = assessment_data['deal_id']
        deal_summary = assessment_data['summary']
        assessed_clauses = assessment_data['assessed_clauses']

        # Perform comparisons
        restrictiveness_comp = self.compare_restrictiveness(deal_summary)
        category_comp = self.compare_categories(deal_summary)
        priority_comp = self.compare_priorities(deal_summary)
        outliers = self.identify_outliers(assessed_clauses)

        # Generate insights
        insights = self.generate_insights(restrictiveness_comp, category_comp,
                                         priority_comp, outliers)

        # Build report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            'deal_id': deal_id,
            'comparison_timestamp': timestamp,
            'assessment_file': assessment_data.get('assessment_timestamp', 'unknown'),
            'benchmark_info': {
                'total_clauses': self.benchmark_stats['quality_summary']['valid_covenants'],
                'total_deals': self.benchmark_stats['quality_summary']['total_clauses'] // 50,  # estimate
                'avg_restrictiveness': self.benchmark_stats['covenant_summary']['avg_restrictiveness']
            },
            'restrictiveness_comparison': restrictiveness_comp,
            'category_comparison': category_comp,
            'priority_comparison': priority_comp,
            'outlier_clauses': outliers,
            'key_insights': insights,
            'executive_summary': self._generate_executive_summary(
                restrictiveness_comp, category_comp, outliers, insights
            )
        }

        # Save report
        output_file = os.path.join(output_dir, f"benchmark_comparison_{deal_id}_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Report saved: {output_file}")

        # Generate summary CSV
        self._generate_summary_csv(report, deal_id, timestamp, output_dir)

        return output_file

    def _generate_executive_summary(self, restrictiveness_comp: Dict,
                                   category_comp: Dict, outliers: List[Dict],
                                   insights: List[str]) -> Dict:
        """Generate executive summary for quick review"""

        return {
            'overall_assessment': restrictiveness_comp['category'].replace('_', ' ').title(),
            'percentile_rank': restrictiveness_comp['percentile_rank'].replace('_', ' ').title(),
            'key_findings': insights[:5],  # Top 5 insights
            'outliers_count': len(outliers),
            'recommendation': self._generate_recommendation(restrictiveness_comp, outliers)
        }

    def _generate_recommendation(self, restrictiveness_comp: Dict, outliers: List[Dict]) -> str:
        """Generate recommendation based on comparison"""

        if restrictiveness_comp['category'] == 'significantly_more_restrictive' and len(outliers) > 5:
            return "CAUTION: Deal is significantly more restrictive than benchmark with multiple unusual clauses. Recommend detailed negotiation."
        elif restrictiveness_comp['category'] in ['significantly_more_restrictive', 'more_restrictive']:
            return "REVIEW: Deal is more restrictive than benchmark. Recommend reviewing key covenants for negotiation opportunities."
        elif len(outliers) > 5:
            return "REVIEW: Multiple unusual clauses detected. Recommend detailed review even though overall restrictiveness is typical."
        else:
            return "PROCEED: Deal covenants are within normal market parameters."

    def _generate_summary_csv(self, report: Dict, deal_id: str, timestamp: str, output_dir: str):
        """Generate summary CSV"""

        rows = []

        # Add restrictiveness comparison
        rows.append({
            'Metric': 'Overall Restrictiveness',
            'Deal Value': f"{report['restrictiveness_comparison']['deal_avg']:.2f}",
            'Benchmark Value': f"{report['restrictiveness_comparison']['benchmark_avg']:.2f}",
            'Difference': f"{report['restrictiveness_comparison']['difference_avg']:+.2f}",
            'Assessment': report['restrictiveness_comparison']['category'].replace('_', ' ').title()
        })

        # Add category comparisons (top differences)
        category_diffs = [(theme, data) for theme, data in report['category_comparison'].items()]
        category_diffs.sort(key=lambda x: abs(x[1]['difference_percentage']), reverse=True)

        for theme, data in category_diffs[:10]:
            rows.append({
                'Metric': theme.replace('_', ' ').title(),
                'Deal Value': f"{data['deal_percentage']:.1f}%",
                'Benchmark Value': f"{data['benchmark_percentage']:.1f}%",
                'Difference': f"{data['difference_percentage']:+.1f}%",
                'Assessment': data['assessment'].replace('_', ' ').title()
            })

        df = pd.DataFrame(rows)
        csv_file = os.path.join(output_dir, f"comparison_summary_{deal_id}_{timestamp}.csv")
        df.to_csv(csv_file, index=False)

        print(f"  ✓ Summary CSV: {csv_file}")

    def print_summary(self, report: Dict):
        """Print summary to console"""

        print("\n" + "="*80)
        print("📊 BENCHMARK COMPARISON SUMMARY")
        print("="*80)

        exec_summary = report['executive_summary']

        print(f"\n🆔 Deal: {report['deal_id']}")
        print(f"📊 Overall Assessment: {exec_summary['overall_assessment']}")
        print(f"📈 Percentile Rank: {exec_summary['percentile_rank']}")

        print(f"\n💡 Key Insights:")
        for insight in exec_summary['key_findings']:
            print(f"  {insight}")

        print(f"\n⚠️  Outlier Clauses: {exec_summary['outliers_count']}")

        if report['outlier_clauses']:
            print(f"\n📋 Top Outliers:")
            for outlier in report['outlier_clauses'][:5]:
                print(f"  • {outlier['section_title']} (score: {outlier['restrictiveness_score']}/10)")
                print(f"    Similarity: {outlier['similarity_to_benchmark']:.1%}")
                for reason in outlier['reasons']:
                    print(f"    - {reason}")

        print(f"\n💼 Recommendation:")
        print(f"  {exec_summary['recommendation']}")


def run_stage8(assessment_s3_url: str, accession: str, doc_type: str = None) -> Dict:
    """Run Stage 8 as part of the S3-based pipeline.

    Downloads assessment data from S3, compares against local benchmark stats,
    and uploads the comparison report + summary CSV back to S3.

    Returns dict with S3 URLs for the uploaded artifacts.
    """
    from covenant_s3_utils import download_json, upload_json, upload_text

    print("=" * 80)
    print("STAGE 8 (S3): COMPARE NEW DEAL TO BENCHMARK")
    print("=" * 80)

    assessment_data = download_json(assessment_s3_url)
    if not assessment_data:
        raise ValueError(f"Failed to download assessment from {assessment_s3_url}")

    deal_id = assessment_data['deal_id']
    deal_summary = assessment_data['summary']
    assessed_clauses = assessment_data['assessed_clauses']

    print(f"  ✓ Deal ID: {deal_id}")
    print(f"  ✓ Clauses: {deal_summary['total_clauses']}")

    comparator = BenchmarkComparator()
    comparator.load_benchmark_stats(CACHE_DIR)

    restrictiveness_comp = comparator.compare_restrictiveness(deal_summary)
    category_comp = comparator.compare_categories(deal_summary)
    priority_comp = comparator.compare_priorities(deal_summary)
    outliers = comparator.identify_outliers(assessed_clauses)
    insights = comparator.generate_insights(
        restrictiveness_comp, category_comp, priority_comp, outliers
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        'deal_id': deal_id,
        'comparison_timestamp': timestamp,
        'assessment_file': assessment_data.get('assessment_timestamp', 'unknown'),
        'benchmark_info': {
            'total_clauses': comparator.benchmark_stats['quality_summary']['valid_covenants'],
            'total_deals': comparator.benchmark_stats['quality_summary']['total_clauses'] // 50,
            'avg_restrictiveness': comparator.benchmark_stats['covenant_summary']['avg_restrictiveness']
        },
        'restrictiveness_comparison': restrictiveness_comp,
        'category_comparison': category_comp,
        'priority_comparison': priority_comp,
        'outlier_clauses': outliers,
        'key_insights': insights,
        'executive_summary': comparator._generate_executive_summary(
            restrictiveness_comp, category_comp, outliers, insights
        )
    }

    _, comparison_url = upload_json(report, accession, "benchmark_comparison_json.json")
    print(f"  ✓ Uploaded benchmark comparison JSON to S3")

    # Build CSV content (mirrors _generate_summary_csv but returns a string)
    rows = []
    rows.append({
        'Metric': 'Overall Restrictiveness',
        'Deal Value': f"{report['restrictiveness_comparison']['deal_avg']:.2f}",
        'Benchmark Value': f"{report['restrictiveness_comparison']['benchmark_avg']:.2f}",
        'Difference': f"{report['restrictiveness_comparison']['difference_avg']:+.2f}",
        'Assessment': report['restrictiveness_comparison']['category'].replace('_', ' ').title()
    })

    category_diffs = [(theme, data) for theme, data in report['category_comparison'].items()]
    category_diffs.sort(key=lambda x: abs(x[1]['difference_percentage']), reverse=True)

    for theme, data in category_diffs[:10]:
        rows.append({
            'Metric': theme.replace('_', ' ').title(),
            'Deal Value': f"{data['deal_percentage']:.1f}%",
            'Benchmark Value': f"{data['benchmark_percentage']:.1f}%",
            'Difference': f"{data['difference_percentage']:+.1f}%",
            'Assessment': data['assessment'].replace('_', ' ').title()
        })

    csv_content = pd.DataFrame(rows).to_csv(index=False)
    _, summary_url = upload_text(
        csv_content, accession, "benchmark_summary_csv.csv",
        content_type="text/csv; charset=utf-8"
    )
    print(f"  ✓ Uploaded benchmark summary CSV to S3")

    print("\n" + "=" * 80)
    print("✅ STAGE 8 (S3) COMPLETE!")
    print("=" * 80)

    return {
        "benchmark_comparison_json": comparison_url,
        "benchmark_summary_csv": summary_url,
    }


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()

    print("="*80)
    print("STAGE 8: COMPARE NEW DEAL TO BENCHMARK")
    print("="*80)

    # Check for input file
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide path to Stage 7 assessment file")
        print("\nUsage:")
        print("  python3 8_compare_to_benchmark.py path/to/deal_assessment_DEALID_*.json")
        print("\nExample:")
        print("  python3 8_compare_to_benchmark.py new_deal_reports/deal_assessment_d12345_*.json")
        return

    assessment_file = sys.argv[1]

    if not os.path.exists(assessment_file):
        print(f"\n❌ Error: File not found: {assessment_file}")
        return

    # Load assessment data
    print(f"\n📂 Loading assessment: {os.path.basename(assessment_file)}")
    with open(assessment_file, 'r') as f:
        assessment_data = json.load(f)

    print(f"  ✓ Deal ID: {assessment_data['deal_id']}")
    print(f"  ✓ Clauses: {assessment_data['summary']['total_clauses']}")

    try:
        # Initialize comparator
        comparator = BenchmarkComparator()

        # Load benchmark
        comparator.load_benchmark_stats(CACHE_DIR)

        # Generate comparison report
        report_file = comparator.generate_comparison_report(assessment_data, OUTPUT_DIR)

        # Load and print summary
        with open(report_file, 'r') as f:
            report = json.load(f)

        comparator.print_summary(report)

        print("\n" + "="*80)
        print("✅ STAGE 8 COMPLETE!")
        print("="*80)
        print(f"\n📁 Comparison report: {report_file}")
        print(f"\n🎉 New deal analysis complete! All reports in: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
