#!/usr/bin/env python3
"""
Stage 7: Risk Assessment for New Deal (WITH OUTLIER ANALYSIS)

Takes the classified new deal from Stage 6 and performs detailed risk assessment:
- Restrictiveness scoring (0-10) for ALL clauses
- Covenant themes identification
- Seller-favorable aspects
- Red flags and investigation priorities

CRITICAL ENHANCEMENT (Matches MAE Stage 7):
- OUTLIER FEEDBACK LOOP: For clauses with low similarity to benchmark (< 70%),
  performs targeted LLM analysis asking "WHY is this unusual?"
- Explains specific differences vs benchmark patterns
- Identifies high-risk vs medium-risk outliers
- Most actionable insight for lawyers

This matches the MAE tool architecture where Stage 7 focuses on explaining
outliers rather than just flagging them.

Usage:
    python3 7_assess_new_deal.py path/to/deal_classification_DEALID_*.json
"""

import json
import os
import sys
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
import openai
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/new_deal_reports"

# OpenAI settings (same as Stage 2)
OPENAI_MODEL = "gpt-4o-mini"

# Restrictiveness scoring schema
RESTRICTIVENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "restrictiveness_score": {
            "type": "integer",
            "description": "Score from 0-10 indicating how restrictive this covenant is"
        },
        "covenant_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of covenant themes (e.g., 'debt_and_financing', 'mergers_acquisitions')"
        },
        "requires_investigation": {
            "type": "boolean",
            "description": "Whether this clause requires detailed investigation"
        },
        "unusual_provisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of unusual or non-standard provisions"
        },
        "seller_favorable_aspects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Aspects that favor the seller"
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Potential red flags or concerns"
        },
        "investigation_priority": {
            "type": "string",
            "enum": ["urgent", "routine", "review"],
            "description": "Priority level for investigation"
        },
        "covenant_explanation": {
            "type": "string",
            "description": "Brief explanation of what this covenant restricts"
        }
    },
    "required": [
        "restrictiveness_score",
        "covenant_themes",
        "requires_investigation",
        "unusual_provisions",
        "seller_favorable_aspects",
        "red_flags",
        "investigation_priority",
        "covenant_explanation"
    ],
    "additionalProperties": False
}


class NewDealRiskAssessor:
    """Performs risk assessment on new deal clauses"""

    def __init__(self, openai_key: str):
        self.client = openai.OpenAI(api_key=openai_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.outlier_threshold = 0.70  # Clauses below this similarity are outliers

    def assess_clause(self, clause: Dict) -> Dict:
        """Assess a single clause for restrictiveness and risk"""

        text = clause.get('text', '') or clause.get('processed_text', '')
        section_title = clause.get('section_title', 'Unknown')
        cluster_category = clause.get('cluster_category', 'unknown')
        cluster_theme = clause.get('cluster_theme', 'Unknown')

        prompt = f"""Analyze this merger agreement covenant clause for restrictiveness and risk.

CLAUSE TEXT:
{text}

CONTEXT:
- Section: {section_title}
- Cluster Category: {cluster_category}
- Cluster Theme: {cluster_theme}

Provide a detailed risk assessment:

1. RESTRICTIVENESS SCORE (0-10):
   - 0-2: Very permissive (broad carveouts, seller discretion)
   - 3-5: Moderate (balanced restrictions)
   - 6-8: Restrictive (limited carveouts, buyer consent required)
   - 9-10: Extremely restrictive (absolute prohibitions, no exceptions)

2. COVENANT THEMES: Identify themes (select from):
   - operations_and_conduct
   - mergers_and_acquisitions
   - debt_and_financing
   - compensation_and_benefits
   - dividends_and_distributions
   - asset_dispositions
   - capital_expenditures
   - contracts_and_commitments
   - tax_matters
   - intellectual_property
   - litigation_and_settlements
   - accounting_and_reporting
   - related_party_transactions
   - insurance
   - other

3. REQUIRES INVESTIGATION: Does this need detailed legal review?

4. UNUSUAL PROVISIONS: Non-standard or unique terms

5. SELLER-FAVORABLE ASPECTS: Carveouts, exceptions, materiality thresholds that favor seller

6. RED FLAGS: Concerns, overly broad restrictions, potential deal-breakers

7. INVESTIGATION PRIORITY:
   - urgent: High risk, requires immediate attention
   - routine: Standard review during diligence
   - review: Low priority, review for completeness

8. COVENANT EXPLANATION: One sentence explaining what this restricts.
"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an M&A lawyer analyzing covenant restrictiveness."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "covenant_assessment",
                        "strict": True,
                        "schema": RESTRICTIVENESS_SCHEMA
                    }
                },
                temperature=0.3
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

            assessment = json.loads(response.choices[0].message.content)
            return assessment

        except Exception as e:
            print(f"\n⚠️  Error assessing clause: {e}")
            return self._default_assessment()

    def _default_assessment(self) -> Dict:
        """Return default assessment on error"""
        return {
            "restrictiveness_score": 5,
            "covenant_themes": ["other"],
            "requires_investigation": True,
            "unusual_provisions": ["Error in assessment"],
            "seller_favorable_aspects": [],
            "red_flags": ["Could not complete assessment"],
            "investigation_priority": "review",
            "covenant_explanation": "Assessment failed"
        }

    def analyze_outlier_with_context(self, clause: Dict) -> Dict:
        """
        CRITICAL: Analyze outliers with feedback loop (matches MAE Stage 7)

        For clauses that don't match benchmark well, ask LLM WHY they're unusual.
        This is the key insight that makes the analysis actionable.
        """

        text = clause.get('text', '') or clause.get('processed_text', '')
        similarity = clause.get('similarity_score', 1.0)
        assigned_cluster = clause.get('assigned_cluster', 'unknown')
        cluster_category = clause.get('cluster_category', 'unknown')
        cluster_theme = clause.get('cluster_theme', 'Unknown')
        section_title = clause.get('section_title', 'Unknown')

        prompt = f"""This covenant clause is an OUTLIER - it has LOW SIMILARITY to benchmark patterns.

OUTLIER DETAILS:
- Similarity to best match: {similarity:.2%} (LOW - threshold is 70%)
- Best match cluster: #{assigned_cluster} - {cluster_theme}
- Cluster category: {cluster_category}

CLAUSE TEXT:
{text}

CONTEXT:
- Section: {section_title}

CRITICAL ANALYSIS NEEDED:

This clause was classified but has low similarity to the benchmark cluster.
Your job is to explain WHY this is unusual compared to typical covenant patterns.

Answer these questions:

1. WHY IS THIS UNUSUAL?
   What specific elements make this clause different from benchmark patterns?
   Be specific - don't just say "it's complex" - explain WHAT is different.

2. WHAT MAKES IT DIFFERENT FROM THE MATCHED CLUSTER?
   It was assigned to "{cluster_theme}" but with low confidence.
   How does this clause differ from typical {cluster_category} covenants?

3. RISK ASSESSMENT:
   - Is this HIGH RISK or MEDIUM RISK as an outlier?
   - Should a lawyer review this specific clause?
   - What should they look for?

4. SPECIFIC CONCERNS:
   List 2-3 specific concerns about this unusual clause.

Return JSON format:
{{
    "why_unusual": "Explanation of what makes this unusual",
    "specific_differences": ["difference 1", "difference 2", "difference 3"],
    "comparison_to_cluster": "How this differs from typical {cluster_category} covenants",
    "outlier_risk_level": "high|medium",
    "requires_lawyer_review": true|false,
    "lawyer_should_review_for": ["concern 1", "concern 2"],
    "unusual_elements": ["element 1", "element 2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an M&A lawyer analyzing unusual covenant clauses that don't match benchmark patterns."},
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
            print(f"\n⚠️  Error analyzing outlier: {e}")
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
        """Assess all clauses in the deal (with outlier analysis)"""

        print(f"\n🔍 Assessing {len(classified_clauses)} clauses for risk...")

        # Identify outliers first
        outliers = [c for c in classified_clauses
                   if c.get('similarity_score', 1.0) < self.outlier_threshold]

        if outliers:
            print(f"   ⚠️  Found {len(outliers)} outlier clauses (similarity < {self.outlier_threshold:.0%})")
            print(f"   These will receive detailed analysis to explain WHY they're unusual")

        assessed_clauses = []

        for clause in tqdm(classified_clauses, desc="Assessing risk"):
            # Standard risk assessment for all clauses
            assessment = self.assess_clause(clause)

            # CRITICAL: For outliers, add specific analysis (MAE Stage 7 approach)
            is_outlier = clause.get('similarity_score', 1.0) < self.outlier_threshold
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
            print(f"\n📊 Outlier Analysis Complete:")
            print(f"   {len(outliers)} clauses analyzed for unusual patterns")
            high_risk_outliers = sum(1 for c in assessed_clauses
                                    if c.get('is_outlier') and
                                    c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high')
            if high_risk_outliers:
                print(f"   🔴 {high_risk_outliers} HIGH-RISK outliers requiring lawyer review")

        return assessed_clauses

    def generate_summary(self, assessed_clauses: List[Dict]) -> Dict:
        """Generate summary statistics (with outlier metrics)"""

        scores = [c['restrictiveness_score'] for c in assessed_clauses]

        restrictiveness_dist = {
            'very_permissive (0-2)': sum(1 for s in scores if s <= 2),
            'moderate (3-5)': sum(1 for s in scores if 3 <= s <= 5),
            'restrictive (6-8)': sum(1 for s in scores if 6 <= s <= 8),
            'extremely_restrictive (9-10)': sum(1 for s in scores if s >= 9)
        }

        theme_counts = {}
        for clause in assessed_clauses:
            for theme in clause['covenant_themes']:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        priority_counts = {
            'urgent': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'urgent'),
            'routine': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'routine'),
            'review': sum(1 for c in assessed_clauses if c['investigation_priority'] == 'review')
        }

        # Outlier statistics (NEW - matches MAE Stage 7)
        outliers = [c for c in assessed_clauses if c.get('is_outlier', False)]
        high_risk_outliers = [c for c in outliers
                             if c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high']

        return {
            'total_clauses': len(assessed_clauses),
            'avg_restrictiveness': sum(scores) / len(scores) if scores else 0,
            'median_restrictiveness': sorted(scores)[len(scores)//2] if scores else 0,
            'restrictiveness_distribution': restrictiveness_dist,
            'theme_distribution': theme_counts,
            'investigation_required': sum(1 for c in assessed_clauses if c['requires_investigation']),
            'priority_distribution': priority_counts,
            'seller_favorable_count': sum(1 for c in assessed_clauses if c['seller_favorable_aspects']),
            'clauses_with_red_flags': sum(1 for c in assessed_clauses if c['red_flags']),
            'outlier_statistics': {
                'total_outliers': len(outliers),
                'high_risk_outliers': len(high_risk_outliers),
                'outlier_percentage': (len(outliers) / len(assessed_clauses) * 100) if assessed_clauses else 0,
                'outlier_threshold': self.outlier_threshold
            }
        }


def run_stage7(classification_s3_url, accession, doc_type=None):
    """Run Stage 7 assessment from S3 classification data and upload results back to S3."""
    from dotenv import load_dotenv
    from covenant_s3_utils import download_json, upload_json
    load_dotenv()

    print(f"\n{'='*80}")
    print("STAGE 7: RISK ASSESSMENT (S3 PIPELINE)")
    print(f"{'='*80}")
    print(f"  Accession: {accession}")

    classification_data = download_json(classification_s3_url)
    deal_id = classification_data['deal_id']
    classified_clauses = classification_data['classified_clauses']

    print(f"  Deal ID: {deal_id}")
    print(f"  Clauses to assess: {len(classified_clauses)}")

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    assessor = NewDealRiskAssessor(openai_key)
    assessed_clauses = assessor.assess_all_clauses(classified_clauses)
    summary = assessor.generate_summary(assessed_clauses)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'deal_id': deal_id,
        'assessment_timestamp': timestamp,
        'summary': summary,
        'assessed_clauses': assessed_clauses,
        'token_usage': {
            'input_tokens': assessor.total_input_tokens,
            'output_tokens': assessor.total_output_tokens,
            'estimated_cost': (assessor.total_input_tokens / 1_000_000 * 0.15 +
                             assessor.total_output_tokens / 1_000_000 * 0.60)
        }
    }

    _, assessment_url = upload_json(output, accession, "assessment_json.json")
    print(f"  Assessment uploaded to S3: {assessment_url}")

    return {"assessment_json": assessment_url, "output": output}


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()

    print("="*80)
    print("STAGE 7: RISK ASSESSMENT FOR NEW DEAL")
    print("="*80)

    # Check for input file
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide path to Stage 6 classification file")
        print("\nUsage:")
        print("  python3 7_assess_new_deal.py path/to/deal_classification_DEALID_*.json")
        print("\nExample:")
        print("  python3 7_assess_new_deal.py new_deal_reports/deal_classification_d12345_*.json")
        return

    classification_file = sys.argv[1]

    if not os.path.exists(classification_file):
        print(f"\n❌ Error: File not found: {classification_file}")
        return

    # Load classification data
    print(f"\n📂 Loading classification: {os.path.basename(classification_file)}")
    with open(classification_file, 'r') as f:
        classification_data = json.load(f)

    deal_id = classification_data['deal_id']
    classified_clauses = classification_data['classified_clauses']

    print(f"  ✓ Deal ID: {deal_id}")
    print(f"  ✓ Clauses to assess: {len(classified_clauses)}")

    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        print("   Please add to .env file")
        return

    try:
        # Initialize assessor
        assessor = NewDealRiskAssessor(openai_key)

        # Assess all clauses
        assessed_clauses = assessor.assess_all_clauses(classified_clauses)

        # Generate summary
        summary = assessor.generate_summary(assessed_clauses)

        # Build output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            'deal_id': deal_id,
            'assessment_timestamp': timestamp,
            'classification_file': os.path.basename(classification_file),
            'summary': summary,
            'assessed_clauses': assessed_clauses,
            'token_usage': {
                'input_tokens': assessor.total_input_tokens,
                'output_tokens': assessor.total_output_tokens,
                'estimated_cost': (assessor.total_input_tokens / 1_000_000 * 0.15 +
                                 assessor.total_output_tokens / 1_000_000 * 0.60)
            }
        }

        # Save report
        output_file = os.path.join(OUTPUT_DIR, f"deal_assessment_{deal_id}_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("📊 RISK ASSESSMENT SUMMARY")
        print("="*80)

        print(f"\n🆔 Deal: {deal_id}")
        print(f"📊 Clauses assessed: {summary['total_clauses']}")
        print(f"📈 Avg restrictiveness: {summary['avg_restrictiveness']:.2f}/10")
        print(f"📊 Median restrictiveness: {summary['median_restrictiveness']}/10")

        print(f"\n📊 Restrictiveness Distribution:")
        for category, count in summary['restrictiveness_distribution'].items():
            pct = count / summary['total_clauses'] * 100
            print(f"  • {category}: {count} ({pct:.1f}%)")

        print(f"\n🎯 Investigation Priorities:")
        for priority, count in summary['priority_distribution'].items():
            print(f"  • {priority}: {count} clauses")

        print(f"\n💡 Key Findings:")
        print(f"  • Requires investigation: {summary['investigation_required']} clauses")
        print(f"  • Seller-favorable aspects: {summary['seller_favorable_count']} clauses")
        print(f"  • Red flags: {summary['clauses_with_red_flags']} clauses")

        # Outlier analysis (NEW - matches MAE Stage 7)
        outlier_stats = summary.get('outlier_statistics', {})
        if outlier_stats.get('total_outliers', 0) > 0:
            print(f"\n⚠️  OUTLIER ANALYSIS (LOW SIMILARITY TO BENCHMARK):")
            print(f"  • Total outliers: {outlier_stats['total_outliers']} ({outlier_stats['outlier_percentage']:.1f}%)")
            print(f"  • High-risk outliers: {outlier_stats['high_risk_outliers']}")
            print(f"  • Similarity threshold: {outlier_stats['outlier_threshold']:.0%}")

            # Show top outlier concerns
            high_risk_outliers = [c for c in assessed_clauses
                                 if c.get('is_outlier') and
                                 c.get('outlier_analysis', {}).get('outlier_risk_level') == 'high']

            if high_risk_outliers:
                print(f"\n  🔴 HIGH-RISK OUTLIERS (Require Lawyer Review):")
                for i, outlier in enumerate(high_risk_outliers[:3], 1):
                    analysis = outlier.get('outlier_analysis', {})
                    print(f"\n  {i}. {outlier.get('section_title', 'Unknown')}")
                    print(f"     Similarity: {outlier.get('similarity_score', 0):.1%}")
                    print(f"     Why unusual: {analysis.get('why_unusual', 'N/A')[:100]}...")

        print(f"\n💰 Token Usage:")
        print(f"  • Input: {assessor.total_input_tokens:,} tokens")
        print(f"  • Output: {assessor.total_output_tokens:,} tokens")
        print(f"  • Estimated cost: ${output['token_usage']['estimated_cost']:.2f}")

        print("\n" + "="*80)
        print("✅ STAGE 7 COMPLETE!")
        print("="*80)
        print(f"\n📁 Assessment saved to: {output_file}")
        print(f"\n💡 Next: Run Stage 8 to compare against benchmark")
        print(f"   python3 8_compare_to_benchmark.py {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
