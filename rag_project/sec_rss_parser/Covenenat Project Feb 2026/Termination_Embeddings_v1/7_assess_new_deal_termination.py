#!/usr/local/bin/python3
"""
Stage 7: Termination Risk Assessment for New Deal (WITH OUTLIER ANALYSIS)

Takes the classified new deal from Stage 6 and performs detailed risk assessment:
- Deal completion risk scoring (0-10) for ALL termination trigger clauses
- Trigger characteristics identification
- Party-favorable aspects analysis
- Red flags and investigation priorities
- Comparison against termination benchmark

CRITICAL ENHANCEMENT:
- OUTLIER FEEDBACK LOOP: For clauses with low similarity to benchmark (< 70%),
  performs targeted LLM analysis asking "WHY is this unusual?"
- Explains specific differences vs benchmark termination patterns
- Identifies high-risk vs medium-risk outliers (novel walkaway triggers)
- Most actionable insight for M&A lawyers reviewing deal termination provisions

Usage:
    python3 7_assess_new_deal_termination.py path/to/termination_classification_DEALID_*.json
"""

import json
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import openai
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ================================
# CONFIGURATION
# ================================
_BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(_BASE_DIR / "new_deal_reports")

# OpenAI settings (same as Stage 2)
OPENAI_MODEL = "gpt-4o-mini"

# Deal risk assessment schema
DEAL_RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "deal_risk_score": {
            "type": "integer",
            "description": "Score 0-10: deal completion risk from acquirer's perspective (0=very low walkaway risk, 10=high deal-walkaway risk)"
        },
        "trigger_characteristics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Trigger type characteristics: mutual_right, unilateral_acquirer, unilateral_target, regulatory_trigger, fiduciary_out, breach_trigger, outside_date, fee_bearing, tail_provision, sole_remedy, specific_performance, other"
        },
        "requires_investigation": {
            "type": "boolean",
            "description": "Whether this clause requires detailed legal investigation"
        },
        "unusual_provisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Non-standard or unusual termination provisions"
        },
        "party_favorable": {
            "type": "string",
            "enum": ["buyer_favorable", "seller_favorable", "mutual", "neutral"],
            "description": "Which party this termination provision favors"
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Potential red flags or deal completion concerns"
        },
        "investigation_priority": {
            "type": "string",
            "enum": ["urgent", "routine", "review"],
            "description": "Priority level for investigation"
        },
        "termination_explanation": {
            "type": "string",
            "description": "Brief explanation of what termination right this clause creates and its deal risk"
        }
    },
    "required": [
        "deal_risk_score",
        "trigger_characteristics",
        "requires_investigation",
        "unusual_provisions",
        "party_favorable",
        "red_flags",
        "investigation_priority",
        "termination_explanation"
    ],
    "additionalProperties": False
}


class NewDealTerminationAssessor:
    """Performs deal risk assessment on new deal termination trigger clauses"""

    def __init__(self, openai_key: str):
        self.client = openai.OpenAI(api_key=openai_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.outlier_threshold = 0.70  # Clauses below this similarity are outliers

    def assess_clause(self, clause: Dict) -> Dict:
        """Assess a single termination trigger clause for deal risk"""

        text = clause.get('original_text', '') or clause.get(
            'text', '') or clause.get('processed_text', '')
        trigger_type = clause.get('trigger_type', 'Unknown')
        cluster_category = clause.get('cluster_trigger_category', 'unknown')
        cluster_theme = clause.get('cluster_theme', 'Unknown')

        prompt = f"""Analyze this termination trigger clause from a merger agreement for deal completion risk.

TERMINATION CLAUSE TEXT:
{text}

CONTEXT:
- Trigger type: {trigger_type}
- Benchmark cluster category: {cluster_category}
- Cluster theme: {cluster_theme}

Provide a detailed risk assessment from the ACQUIRER'S perspective:

1. DEAL RISK SCORE (0-10) — likelihood this clause causes deal failure or walk-away:
   - 0-2: Very low risk (routine provision, hard to trigger, easily cured)
   - 3-5: Moderate risk (could be triggered under adverse conditions, manageable)
   - 6-8: High risk (realistic trigger scenario, difficult to cure, deal-threatening)
   - 9-10: Extreme risk (easy to trigger, no cure period, likely deal break)

2. TRIGGER CHARACTERISTICS (select ALL that apply):
   - mutual_right: Either party may terminate
   - unilateral_acquirer: Only acquirer/parent may terminate
   - unilateral_target: Only target/company may terminate
   - regulatory_trigger: Triggered by regulatory/antitrust failure
   - fiduciary_out: Board fiduciary duty-based right
   - breach_trigger: Triggered by breach of reps, warranties, or covenants
   - outside_date: Triggered by failure to close by outside date
   - fee_bearing: Termination fee payment associated
   - tail_provision: Post-termination fee obligations
   - sole_remedy: Limits remedies to termination fee
   - specific_performance: Allows equity-forced performance before termination
   - other: Does not fit above

3. PARTY FAVORABLE: Which party does this provision favor?
   - buyer_favorable: Benefits/protects the acquirer
   - seller_favorable: Benefits/protects the target
   - mutual: Balanced, applies equally
   - neutral: Neither clearly favored

4. UNUSUAL PROVISIONS: Non-standard language, unusual carve-outs, or atypical specificity

5. RED FLAGS: Conditions that could allow easy walkaway, ambiguous trigger language,
   or provisions that create unexpected deal completion risk

6. INVESTIGATION PRIORITY:
   - urgent: High risk, immediate attention required
   - review: Low priority, review for completeness
   - routine: Standard provision, no immediate concerns

7. TERMINATION EXPLANATION: One sentence explaining what termination right this creates
   and why it matters from a deal completion perspective.
"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an M&A lawyer analyzing termination provisions in merger agreements for deal completion risk."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "termination_assessment",
                        "strict": True,
                        "schema": DEAL_RISK_ASSESSMENT_SCHEMA
                    }
                },
                temperature=0.3
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

            assessment = json.loads(response.choices[0].message.content)
            return assessment

        except Exception as e:
            print(f"\n  Warning: Error assessing clause: {e}")
            return self._default_assessment()

    def _default_assessment(self) -> Dict:
        """Return default assessment on error"""
        return {
            "deal_risk_score": 5,
            "trigger_characteristics": ["other"],
            "requires_investigation": True,
            "unusual_provisions": ["Error in assessment"],
            "party_favorable": "neutral",
            "red_flags": ["Could not complete assessment"],
            "investigation_priority": "review",
            "termination_explanation": "Assessment failed - manual review required"
        }

    def analyze_outlier_with_context(self, clause: Dict) -> Dict:
        """
        CRITICAL: Analyze outliers with feedback loop

        For termination trigger clauses that don't match benchmark well,
        ask LLM WHY they're unusual. This identifies novel walkaway triggers
        not seen in the benchmark deal set.
        """

        text = clause.get('original_text', '') or clause.get(
            'text', '') or clause.get('processed_text', '')
        similarity = clause.get('similarity_score', 1.0)
        assigned_cluster = clause.get('assigned_cluster', 'unknown')
        cluster_category = clause.get('cluster_trigger_category', 'unknown')
        cluster_theme = clause.get('cluster_theme', 'Unknown')
        trigger_type = clause.get('trigger_type', 'Unknown')

        prompt = f"""This termination trigger clause is an OUTLIER — it has LOW SIMILARITY to benchmark patterns.

OUTLIER DETAILS:
- Similarity to best match: {similarity:.2%} (LOW - threshold is 70%)
- Best match cluster: #{assigned_cluster} - {cluster_theme}
- Cluster category: {cluster_category}
- Trigger type label: {trigger_type}

CLAUSE TEXT:
{text}

CRITICAL ANALYSIS NEEDED:

This termination provision was classified but has low similarity to the benchmark cluster.
Explain WHY this termination right is unusual compared to standard market patterns.

Answer these questions:

1. WHY IS THIS UNUSUAL?
   What specific elements make this termination provision different from standard
   termination clause patterns? Be specific — what is structurally or substantively different?

2. WHAT MAKES IT DIFFERENT FROM THE MATCHED CLUSTER?
   It was assigned to "{cluster_theme}" but with low confidence.
   How does this clause differ from typical {cluster_category} termination provisions?

3. DEAL RISK ASSESSMENT:
   - Is this HIGH RISK or MEDIUM RISK as an outlier from the acquirer's perspective?
   - Should a deal lawyer specifically review this provision?
   - What should they look for?

4. SPECIFIC CONCERNS:
   List 2-3 specific concerns about this unusual termination provision.

Return JSON format:
{{
    "why_unusual": "Explanation of what makes this termination provision unusual",
    "specific_differences": ["difference 1", "difference 2", "difference 3"],
    "comparison_to_cluster": "How this differs from typical {cluster_category} termination clauses",
    "outlier_risk_level": "high|medium",
    "requires_lawyer_review": true,
    "lawyer_should_review_for": ["concern 1", "concern 2"],
    "unusual_elements": ["element 1", "element 2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an M&A lawyer analyzing unusual termination provisions that don't match benchmark patterns in merger agreements."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

            outlier_analysis = json.loads(response.choices[0].message.content)
            return outlier_analysis

        except Exception as e:
            print(f"\n  Warning: Error analyzing outlier: {e}")
            return {
                "why_unusual": "Error analyzing outlier",
                "specific_differences": ["Analysis failed"],
                "comparison_to_cluster": "Unknown",
                "outlier_risk_level": "medium",
                "requires_lawyer_review": True,
                "lawyer_should_review_for": ["Manual review needed"],
                "unusual_elements": ["Error in analysis"]
            }

    def assess_all_clauses(self, classified_clauses: List[Dict]) -> List[Dict]:
        """Assess all termination trigger clauses in the deal (with outlier analysis)"""

        print(
            f"\nAssessing {len(classified_clauses)} termination clauses for deal risk...")

        # Identify outliers first
        outliers = [c for c in classified_clauses
                    if c.get('similarity_score', 1.0) < self.outlier_threshold]

        if outliers:
            print(
                f"   Found {len(outliers)} outlier clauses (similarity < {self.outlier_threshold:.0%})")
            print(
                f"   These will receive detailed analysis to explain WHY they differ from benchmark")

        assessed_clauses = []

        for clause in tqdm(classified_clauses, desc="Assessing termination risk"):
            # Standard risk assessment for all clauses
            assessment = self.assess_clause(clause)

            # CRITICAL: For outliers, add specific analysis explaining the difference
            is_outlier = clause.get(
                'similarity_score', 1.0) < self.outlier_threshold
            if is_outlier:
                outlier_analysis = self.analyze_outlier_with_context(clause)
                assessment['is_outlier'] = True
                assessment['outlier_analysis'] = outlier_analysis
            else:
                assessment['is_outlier'] = False

            # Merge assessment with clause data
            assessed_clause = {**clause, **assessment}
            assessed_clauses.append(assessed_clause)

        # Print outlier summary
        if outliers:
            print(f"\nOutlier Analysis Complete:")
            print(f"   {len(outliers)} clauses analyzed for unusual patterns")
            high_risk_outliers = sum(1 for c in assessed_clauses
                                     if c.get('is_outlier') and
                                     c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high')
            if high_risk_outliers:
                print(
                    f"   {high_risk_outliers} HIGH-RISK outliers requiring lawyer review")

        return assessed_clauses

    def generate_summary(self, assessed_clauses: List[Dict]) -> Dict:
        """Generate summary statistics with outlier metrics"""

        scores = [c['deal_risk_score'] for c in assessed_clauses]

        deal_risk_dist = {
            'very_low_risk (0-2)': sum(1 for s in scores if s <= 2),
            'moderate_risk (3-5)': sum(1 for s in scores if 3 <= s <= 5),
            'high_risk (6-8)': sum(1 for s in scores if 6 <= s <= 8),
            'extreme_risk (9-10)': sum(1 for s in scores if s >= 9)
        }

        # Trigger characteristic counts (handling semicolon-separated or list values)
        characteristic_counts = {}
        for clause in assessed_clauses:
            chars = clause.get('trigger_characteristics', [])
            if isinstance(chars, str):
                chars = [c.strip() for c in chars.split(';') if c.strip()]
            for char in chars:
                characteristic_counts[char] = characteristic_counts.get(
                    char, 0) + 1

        priority_counts = {
            'urgent': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'urgent'),
            'routine': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'routine'),
            'review': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'review')
        }

        party_counts = {}
        for clause in assessed_clauses:
            party = clause.get('party_favorable', 'neutral')
            party_counts[party] = party_counts.get(party, 0) + 1

        # Outlier statistics
        outliers = [c for c in assessed_clauses if c.get('is_outlier', False)]
        high_risk_outliers = [c for c in outliers
                              if c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high']

        return {
            'total_clauses': len(assessed_clauses),
            'avg_deal_risk_score': sum(scores) / len(scores) if scores else 0,
            'median_deal_risk_score': sorted(scores)[len(scores)//2] if scores else 0,
            'deal_risk_distribution': deal_risk_dist,
            'trigger_characteristic_distribution': characteristic_counts,
            'investigation_required': sum(1 for c in assessed_clauses if c['requires_investigation']),
            'priority_distribution': priority_counts,
            'party_favorable_distribution': party_counts,
            'clauses_with_red_flags': sum(1 for c in assessed_clauses if c['red_flags']),
            'outlier_statistics': {
                'total_outliers': len(outliers),
                'high_risk_outliers': len(high_risk_outliers),
                'outlier_percentage': (len(outliers) / len(assessed_clauses) * 100) if assessed_clauses else 0,
                'outlier_threshold': self.outlier_threshold
            }
        }

    def compare_to_benchmark(self, assessed_clauses: List[Dict],
                             benchmark_file: Optional[str] = None) -> Dict:
        """
        Compare deal's termination provisions to the benchmark.
        Loads termination_benchmark_*.json from final_results if available.
        """
        # Locate benchmark
        final_results_dir = OUTPUT_DIR.replace(
            'new_deal_reports', 'final_results')
        if benchmark_file and os.path.exists(benchmark_file):
            bm_path = benchmark_file
        else:
            bm_files = []
            if os.path.exists(final_results_dir):
                bm_files = sorted([f for f in os.listdir(final_results_dir)
                                   if f.startswith('termination_benchmark_') and f.endswith('.json')])
            if not bm_files:
                return {'note': 'No termination benchmark file found. Run Stage 5 first.'}
            bm_path = os.path.join(final_results_dir, bm_files[-1])

        try:
            with open(bm_path, 'r') as f:
                benchmark = json.load(f)
        except Exception as e:
            return {'note': f'Could not load benchmark: {e}'}

        # Benchmark stats from cluster_metadata
        cluster_meta = benchmark.get('cluster_metadata', {})
        bm_categories = {}
        bm_party_dist = {}
        for cid, meta in cluster_meta.items():
            cat = meta.get('trigger_category', 'unknown')
            party = meta.get('party_favored', 'unknown')
            bm_categories[cat] = bm_categories.get(
                cat, 0) + meta.get('size', 0)
            bm_party_dist[party] = bm_party_dist.get(
                party, 0) + meta.get('size', 0)

        # Deal stats
        deal_categories = {}
        deal_party_dist = {}
        for c in assessed_clauses:
            cat = c.get('cluster_trigger_category', 'unknown')
            party = c.get('party_favorable', c.get('party_favored', 'neutral'))
            deal_categories[cat] = deal_categories.get(cat, 0) + 1
            deal_party_dist[party] = deal_party_dist.get(party, 0) + 1

        # High-risk clause comparison
        deal_high_risk = sum(
            1 for c in assessed_clauses if c.get('deal_risk_score', 0) >= 8)
        deal_high_risk_pct = (
            deal_high_risk / len(assessed_clauses) * 100) if assessed_clauses else 0

        return {
            'benchmark_file': os.path.basename(bm_path),
            'benchmark_total_clusters': len(cluster_meta),
            'deal_category_distribution': deal_categories,
            'benchmark_category_distribution': bm_categories,
            'deal_party_distribution': deal_party_dist,
            'benchmark_party_distribution': bm_party_dist,
            'deal_high_risk_clauses': deal_high_risk,
            'deal_high_risk_percentage': round(deal_high_risk_pct, 1),
            'avg_deal_risk_score': (sum(c.get('deal_risk_score', 0) for c in assessed_clauses) /
                                    len(assessed_clauses)) if assessed_clauses else 0,
        }


def run_stage7(classification_s3_url: str, accession: str, doc_type: str) -> Dict:
    """
    New-flow entry point: download classification from S3, assess, upload results to S3.
    Returns dict with assessment_json S3 URL and the output data.
    """
    from termination_s3_utils import download_json, upload_json

    print("="*80)
    print("STAGE 7: TERMINATION TRIGGER RISK ASSESSMENT (S3 FLOW)")
    print("="*80)

    classification_data = download_json(classification_s3_url)
    if not classification_data:
        raise ValueError(
            f"Invalid classification data from {classification_s3_url}")

    deal_id = classification_data.get('deal_id', accession)
    classified_clauses = classification_data.get('classified_clauses', [])
    print(f"  Accession: {accession}")
    print(f"  Clauses to assess: {len(classified_clauses)}")

    openai_key = os.getenv('OPENAI_API_KEY_SEC_FILING')
    if not openai_key:
        raise ValueError("OPENAI_API_KEY_SEC_FILING not set")

    assessor = NewDealTerminationAssessor(openai_key)
    assessed_clauses = assessor.assess_all_clauses(classified_clauses)
    summary = assessor.generate_summary(assessed_clauses)
    benchmark_comparison = assessor.compare_to_benchmark(assessed_clauses)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'deal_id': deal_id,
        'assessment_timestamp': timestamp,
        'classification_source': classification_s3_url,
        'summary': summary,
        'benchmark_comparison': benchmark_comparison,
        'assessed_clauses': assessed_clauses,
        'token_usage': {
            'input_tokens': assessor.total_input_tokens,
            'output_tokens': assessor.total_output_tokens,
            'estimated_cost': (assessor.total_input_tokens / 1_000_000 * 0.15 +
                               assessor.total_output_tokens / 1_000_000 * 0.60)
        }
    }

    _, assessment_url = upload_json(
        output, accession, doc_type, "assessment_json.json")
    print(f"  Assessment uploaded to S3: {assessment_url}")

    print("\n" + "="*80)
    print("STAGE 7 COMPLETE!")
    print("="*80)

    return {
        "assessment_json": assessment_url,
        "output": output,
    }


def main():
    """Main execution"""
    print("="*80)
    print("STAGE 7: TERMINATION TRIGGER RISK ASSESSMENT FOR NEW DEAL")
    print("="*80)

    # Check for input file
    if len(sys.argv) < 2:
        print("\nError: Please provide path to Stage 6 classification file")
        print("\nUsage:")
        print("  python3 7_assess_new_deal_termination.py path/to/termination_classification_DEALID_*.json")
        print("\nExample:")
        print("  python3 7_assess_new_deal_termination.py new_deal_reports/termination_classification_d12345_*.json")
        return

    classification_file = sys.argv[1]

    if not os.path.exists(classification_file):
        print(f"\nError: File not found: {classification_file}")
        return

    # Load classification data
    print(f"\nLoading classification: {os.path.basename(classification_file)}")
    with open(classification_file, 'r') as f:
        classification_data = json.load(f)

    deal_id = classification_data['deal_id']
    classified_clauses = classification_data['classified_clauses']

    print(f"  Deal ID: {deal_id}")
    print(f"  Termination clauses to assess: {len(classified_clauses)}")

    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY_SEC_FILING')
    if not openai_key:
        print("\nERROR: OPENAI_API_KEY_SEC_FILING not set")
        print("   Please add to .env file")
        return

    try:
        # Initialize assessor
        assessor = NewDealTerminationAssessor(openai_key)

        # Assess all clauses
        assessed_clauses = assessor.assess_all_clauses(classified_clauses)

        # Generate summary
        summary = assessor.generate_summary(assessed_clauses)

        # Compare to benchmark
        benchmark_comparison = assessor.compare_to_benchmark(assessed_clauses)

        # Build output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            'deal_id': deal_id,
            'assessment_timestamp': timestamp,
            'classification_file': os.path.basename(classification_file),
            'summary': summary,
            'benchmark_comparison': benchmark_comparison,
            'assessed_clauses': assessed_clauses,
            'token_usage': {
                'input_tokens': assessor.total_input_tokens,
                'output_tokens': assessor.total_output_tokens,
                'estimated_cost': (assessor.total_input_tokens / 1_000_000 * 0.15 +
                                   assessor.total_output_tokens / 1_000_000 * 0.60)
            }
        }

        # Save report
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(
            OUTPUT_DIR, f"termination_assessment_{deal_id}_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("TERMINATION RISK ASSESSMENT SUMMARY")
        print("="*80)

        print(f"\nDeal: {deal_id}")
        print(f"Clauses assessed: {summary['total_clauses']}")
        print(f"Avg deal risk score: {summary['avg_deal_risk_score']:.2f}/10")
        print(
            f"Median deal risk score: {summary['median_deal_risk_score']}/10")

        print(f"\nDeal Risk Distribution:")
        for category, count in summary['deal_risk_distribution'].items():
            pct = count / summary['total_clauses'] * 100
            print(f"  - {category}: {count} ({pct:.1f}%)")

        print(f"\nInvestigation Priorities:")
        for priority, count in summary['priority_distribution'].items():
            print(f"  - {priority}: {count} clauses")

        print(f"\nParty Favorable:")
        for party, count in sorted(summary['party_favorable_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {party}: {count} clauses")

        print(f"\nTop Trigger Characteristics:")
        for char, count in sorted(summary['trigger_characteristic_distribution'].items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  - {char}: {count}")

        print(f"\nKey Findings:")
        print(
            f"  - Requires investigation: {summary['investigation_required']} clauses")
        print(
            f"  - Red flags identified: {summary['clauses_with_red_flags']} clauses")

        # Benchmark comparison
        if benchmark_comparison and 'note' not in benchmark_comparison:
            print(f"\nBenchmark Comparison:")
            print(
                f"  - Benchmark clusters: {benchmark_comparison['benchmark_total_clusters']}")
            print(
                f"  - High risk clauses (8-10): {benchmark_comparison['deal_high_risk_clauses']} ({benchmark_comparison['deal_high_risk_percentage']}%)")
            print(f"\n  Deal trigger categories vs benchmark:")
            deal_cats = benchmark_comparison.get(
                'deal_category_distribution', {})
            for cat, count in sorted(deal_cats.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {cat}: {count}")

        # Outlier analysis
        outlier_stats = summary.get('outlier_statistics', {})
        if outlier_stats.get('total_outliers', 0) > 0:
            print(f"\nOUTLIER ANALYSIS (LOW SIMILARITY TO BENCHMARK):")
            print(
                f"  - Total outliers: {outlier_stats['total_outliers']} ({outlier_stats['outlier_percentage']:.1f}%)")
            print(
                f"  - High-risk outliers: {outlier_stats['high_risk_outliers']}")
            print(
                f"  - Similarity threshold: {outlier_stats['outlier_threshold']:.0%}")

            # Show top outlier concerns
            high_risk_outliers = [c for c in assessed_clauses
                                  if c.get('is_outlier') and
                                  c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high']

            if high_risk_outliers:
                print(f"\n  HIGH-RISK OUTLIERS (Require Lawyer Review):")
                for i, outlier in enumerate(high_risk_outliers[:3], 1):
                    analysis = outlier.get('outlier_analysis', {})
                    print(
                        f"\n  {i}. Trigger type: {outlier.get('trigger_type', 'Unknown')}")
                    print(
                        f"     Similarity: {outlier.get('similarity_score', 0):.1%}")
                    why = analysis.get('why_unusual', 'N/A')
                    print(f"     Why unusual: {str(why)[:120]}...")

        print(f"\nToken Usage:")
        print(f"  - Input: {assessor.total_input_tokens:,} tokens")
        print(f"  - Output: {assessor.total_output_tokens:,} tokens")
        print(
            f"  - Estimated cost: ${output['token_usage']['estimated_cost']:.2f}")

        print("\n" + "="*80)
        print("STAGE 7 COMPLETE!")
        print("="*80)
        print(f"\nAssessment saved to: {output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
