#!/usr/local/bin/python3
"""
Stage 6: Classify New Deal Termination Triggers Against Benchmark

Takes a new deal's termination trigger clauses and classifies them against the
existing benchmark clusters without re-running the entire pipeline.

Input format: termination_response_{accession}_triggers.json
  {
    "document_id": "...",
    "total_clauses": N,
    "clauses": [
      {
        "section_number": ...,
        "clause_id": ...,
        "trigger_type": ...,
        "original_text": ...
      }
    ]
  }

Preamble clauses (trigger_type == "preamble") are skipped automatically.

Usage:
    python3 6_classify_new_deal_termination.py path/to/termination_response_ACCESSION_triggers.json
"""

import json
import numpy as np
import os
import re
import sys
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import cohere
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cosine

from dotenv import load_dotenv
load_dotenv()

# ================================
# CONFIGURATION
# ================================
_BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(_BASE_DIR / "embeddings_cache")
OUTPUT_DIR = str(_BASE_DIR / "new_deal_reports")

# Cohere settings (must match Stage 1)
COHERE_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "search_document"
COHERE_DIMS = 1536


class TerminationPreprocessor:
    """Text preprocessing for termination trigger clauses (same as Stage 1)"""

    def preprocess(self, text: str) -> str:
        """Clean text with termination-specific substitutions"""

        substitutions = {
            # Party normalization
            r'\b(?:the\s+)?(?:company|target)\b': 'COMPANY_ENTITY',
            r'\b(?:the\s+)?(?:parent|acquirer|buyer|purchaser)\b': 'PARENT_ENTITY',
            r'\bmerger\s+sub(?:sidiary)?\b': 'MERGER_SUB',

            # Date/deadline references
            r'\b(?:the\s+)?(?:outside\s+date|termination\s+date|long[\-\s]stop\s+date|end\s+date)\b': 'OUTSIDE_DATE',
            r'\b(?:the\s+)?(?:effective\s+time|closing\s+date|closing)\b': 'CLOSING_DATE',

            # Regulatory references
            r'\b(?:antitrust|competition|hsr\s+act|hart[\-\s]scott[\-\s]rodino)\b': 'ANTITRUST_LAW',
            r'\b(?:governmental\s+authority|regulatory\s+authority|governmental\s+entity)\b': 'GOV_AUTHORITY',
            r'\b(?:injunction|order|decree|judgment|law|statute)\b': 'LEGAL_ORDER',

            # Stockholder vote
            r'\b(?:requisite\s+)?(?:company\s+)?stockholder\s+(?:approval|vote)\b': 'STOCKHOLDER_APPROVAL',
            r'\b(?:requisite\s+)?(?:parent\s+)?stockholder\s+(?:approval|vote)\b': 'PARENT_STOCKHOLDER_APPROVAL',

            # Board recommendation
            r'\b(?:adverse\s+)?(?:recommendation\s+change|change\s+(?:of|in)\s+recommendation)\b': 'ADVERSE_REC_CHANGE',
            r'\b(?:superior\s+proposal|competing\s+transaction)\b': 'SUPERIOR_PROPOSAL',
            r'\b(?:intervening\s+event)\b': 'INTERVENING_EVENT',

            # Breach
            r'\b(?:material(?:ly)?\s+)?(?:breach|fail(?:ure)?)\s+(?:of|to)\b': 'MATERIAL_BREACH',
            r'\b(?:representations?\s+and\s+)?warranties\b': 'REPS_WARRANTIES',
            r'\b(?:covenants?\s+(?:and\s+)?agreements?)\b': 'COVENANTS',

            # Materiality
            r'\bmaterial\s+adverse\s+(?:effect|change|impact)\b': 'MATERIAL_ADVERSE',

            # Common qualifiers
            r'\b(?:with\s+)?(?:prior\s+)?written\s+notice\b': 'WRITTEN_NOTICE',
            r'\b(?:business\s+days?)\b': 'BUSINESS_DAYS',
            r'\bprovided\s+(?:that|however)\b': 'PROVIDED',
        }

        for pattern, replacement in substitutions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text


class NewDealTerminationClassifier:
    """Classifies new deal termination triggers against existing benchmark"""

    def __init__(self, cohere_key: str):
        self.cohere_client = cohere.ClientV2(cohere_key)
        self.preprocessor = TerminationPreprocessor()
        self.benchmark_data = None
        self.cluster_centroids = None
        self.cluster_analyses = None

    def load_benchmark(self, cache_dir: str):
        """Load the most recent benchmark clustering and analysis"""
        print("\nLoading benchmark data...")

        # Find most recent termination clustering file
        clustering_files = sorted([f for f in os.listdir(cache_dir)
                                  if f.startswith('termination_clustering_') and f.endswith('.json')])
        if not clustering_files:
            raise FileNotFoundError(
                "No termination clustering files found in cache directory")

        # Also check final_results for benchmark file (preferred)
        final_results_dir = cache_dir.replace(
            'embeddings_cache', 'final_results')
        benchmark_files = []
        if os.path.exists(final_results_dir):
            benchmark_files = sorted([f for f in os.listdir(final_results_dir)
                                      if f.startswith('termination_benchmark_') and f.endswith('.json')])

        if benchmark_files:
            # Load from benchmark file (has full annotations)
            benchmark_path = os.path.join(
                final_results_dir, benchmark_files[-1])
            print(f"  Loading from benchmark file: {benchmark_files[-1]}")
            with open(benchmark_path, 'r') as f:
                benchmark_json = json.load(f)

            # Extract cluster thresholds (centroids + metrics)
            self.cluster_centroids = {}
            cluster_thresholds = benchmark_json.get(
                'cluster_metrics', {}).get('cluster_thresholds', {})
            for cluster_id_str, cluster_info in cluster_thresholds.items():
                cid = int(cluster_id_str)
                self.cluster_centroids[cid] = np.array(
                    cluster_info['centroid'])

            # Extract cluster analyses
            raw_analyses = benchmark_json.get('cluster_analyses', {})
            self.cluster_analyses = {}
            for cid_str, info in raw_analyses.items():
                cid = int(cid_str)
                self.cluster_analyses[cid] = {
                    **info,
                    'cluster_id': cid,
                    'cluster_theme': info.get('cluster_label', info.get('primary_theme', 'Unknown'))
                }

            n_valid = sum(int(v.get('size', 0))
                          for v in cluster_thresholds.values())
            print(
                f"  Loaded benchmark: {len(self.cluster_centroids)} clusters, ~{n_valid} clauses")

        else:
            # Fall back to raw clustering file
            clustering_file = clustering_files[-1]
            clustering_path = os.path.join(cache_dir, clustering_file)

            with open(clustering_path, 'r') as f:
                self.benchmark_data = json.load(f)

            print(f"  Loaded clustering: {clustering_file}")
            print(f"    - {self.benchmark_data['n_valid_clauses']} clauses")
            print(
                f"    - {self.benchmark_data['metrics']['n_clusters']} clusters")

            # Extract cluster centroids from metrics
            self.cluster_centroids = {}
            cluster_thresholds = self.benchmark_data['metrics'].get(
                'cluster_thresholds', {})
            for cluster_id, cluster_info in cluster_thresholds.items():
                self.cluster_centroids[int(cluster_id)] = np.array(
                    cluster_info['centroid'])

            # Find corresponding analysis file
            embedding_ts = self.benchmark_data['embedding_timestamp'].replace('_', '')[
                :8]
            analysis_files = sorted([f for f in os.listdir(cache_dir)
                                    if f.startswith('termination_cluster_analysis_')
                                    and embedding_ts in f])

            if analysis_files:
                analysis_file = analysis_files[-1]
                analysis_path = os.path.join(cache_dir, analysis_file)

                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)

                raw = analysis_data['cluster_analyses']
                if isinstance(raw, list):
                    self.cluster_analyses = {a['cluster_id']: a for a in raw}
                else:
                    self.cluster_analyses = {}
                    for cluster_id_str, info in raw.items():
                        cid = int(cluster_id_str)
                        self.cluster_analyses[cid] = {
                            **info,
                            'cluster_id': cid,
                            'cluster_theme': info.get('cluster_label', info.get('primary_theme', 'Unknown'))
                        }
                print(f"  Loaded analysis: {analysis_file}")
            else:
                print("  No cluster analysis found - will use basic classification only")
                self.cluster_analyses = {}

        print(f"  Total clusters loaded: {len(self.cluster_centroids)}")

    def generate_embeddings(self, clauses: List[Dict]) -> np.ndarray:
        """Generate embeddings for new deal termination trigger clauses"""
        print(
            f"\nGenerating embeddings for {len(clauses)} termination clauses...")

        # Prepare texts
        texts = []
        for clause in clauses:
            text = clause.get('original_text') or clause.get(
                'processed_text') or clause.get('text', '')
            processed = self.preprocessor.preprocess(text)
            texts.append(processed)

        # Generate embeddings in batches
        all_embeddings = []
        batch_size = 96

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            # Truncate to 4096 chars (Cohere limit)
            batch = [t[:4096] for t in batch]

            response = self.cohere_client.embed(
                texts=batch,
                model=COHERE_MODEL,
                input_type=COHERE_INPUT_TYPE,
                embedding_types=["float"]
            )

            batch_embeddings = response.embeddings.float_
            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings)
        print(f"  Generated embeddings: {embeddings_array.shape}")

        return embeddings_array

    def classify_clauses(self, embeddings: np.ndarray) -> List[Dict]:
        """Assign each termination trigger clause to nearest cluster"""
        print("\nClassifying clauses against benchmark clusters...")

        classifications = []

        for i, embedding in enumerate(tqdm(embeddings, desc="Classifying")):
            # Calculate cosine similarity to each cluster centroid
            similarities = {}
            for cluster_id, centroid in self.cluster_centroids.items():
                similarity = 1 - cosine(embedding, centroid)
                similarities[cluster_id] = similarity

            # Assign to nearest cluster (highest cosine similarity)
            best_cluster = max(similarities.items(), key=lambda x: x[1])
            cluster_id = best_cluster[0]
            similarity_score = best_cluster[1]

            # Get cluster analysis if available
            cluster_info = self.cluster_analyses.get(cluster_id, {})

            classifications.append({
                'clause_index': i,
                'assigned_cluster': cluster_id,
                'similarity_score': float(similarity_score),
                'cluster_trigger_category': cluster_info.get('trigger_category', 'unknown'),
                'cluster_theme': cluster_info.get('cluster_theme', 'Unknown'),
                'party_favored': cluster_info.get('party_favored', 'unknown'),
                'deal_risk_pattern': cluster_info.get('deal_risk_pattern', 'unknown'),
                'deal_risk_score': cluster_info.get('avg_deal_risk_score',
                                   cluster_info.get('deal_risk_score', None)),
                'fee_implications': cluster_info.get('fee_implications', ''),
            })

        return classifications

    def generate_deal_report(self, deal_data: Dict, clauses: List[Dict],
                             classifications: List[Dict], output_dir: str) -> str:
        """Generate comprehensive report for the new deal's termination triggers.
        Writes to local output_dir. Returns local file path."""
        report, timestamp = self._build_report(
            deal_data, clauses, classifications)

        deal_id = report['deal_id']
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(
            output_dir, f"termination_classification_{deal_id}_{timestamp}.json")

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Report saved: {report_file}")

        self._generate_summary_csv(
            report['classified_clauses'], deal_id, timestamp, output_dir)

        return report_file

    def generate_deal_report_s3(self, deal_data: Dict, clauses: List[Dict],
                                classifications: List[Dict],
                                accession: str, doc_type: str) -> Dict[str, str]:
        """Generate report and upload to S3. Returns dict of S3 URLs."""
        from termination_s3_utils import upload_json, upload_text

        report, timestamp = self._build_report(
            deal_data, clauses, classifications)
        deal_id_for_file = report['deal_id']

        _, classification_url = upload_json(
            report, accession, doc_type, "classification_json.json")
        print(f"  Classification uploaded to S3: {classification_url}")

        csv_content = self._generate_summary_csv_content(
            report['classified_clauses'], deal_id_for_file, timestamp)
        _, summary_url = upload_text(csv_content, accession, doc_type, "summary_csv.csv",
                                     content_type="text/csv; charset=utf-8")
        print(f"  Summary CSV uploaded to S3: {summary_url}")

        return {
            "classification_json": classification_url,
            "summary_csv": summary_url,
            "report": report,
        }

    def _build_report(self, deal_data: Dict, clauses: List[Dict],
                      classifications: List[Dict]) -> tuple:
        """Build the classification report dict. Returns (report, timestamp)."""
        print("\nGenerating deal termination trigger report...")

        deal_id = deal_data.get('deal_id') or deal_data.get(
            'document_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        classified_clauses = []
        for i, clause in enumerate(clauses):
            classification = classifications[i]
            classified_clauses.append({
                **clause,
                **classification
            })

        cluster_distribution = {}
        trigger_category_distribution = {}
        party_distribution = {}

        for cls in classifications:
            cluster_id = cls['assigned_cluster']
            category = cls['cluster_trigger_category']
            party = cls['party_favored']

            cluster_distribution[cluster_id] = cluster_distribution.get(
                cluster_id, 0) + 1
            trigger_category_distribution[category] = trigger_category_distribution.get(
                category, 0) + 1
            party_distribution[party] = party_distribution.get(party, 0) + 1

        avg_similarity = np.mean([c['similarity_score']
                                 for c in classifications])

        fee_bearing_count = sum(1 for cls in classifications
                                if 'fee' in str(cls.get('cluster_trigger_category', '')).lower()
                                or 'fee' in str(cls.get('deal_risk_pattern', '')).lower())

        report = {
            'deal_id': deal_id,
            'analysis_timestamp': timestamp,
            'benchmark_info': {
                'benchmark_clusters': len(self.cluster_centroids),
            },
            'deal_summary': {
                'total_termination_clauses': len(clauses),
                'avg_similarity_to_clusters': float(avg_similarity),
                'clusters_matched': len(cluster_distribution),
            },
            'cluster_distribution': cluster_distribution,
            'trigger_category_distribution': trigger_category_distribution,
            'party_distribution': party_distribution,
            'classified_clauses': classified_clauses,
            'insights': self._generate_insights(classified_clauses, classifications)
        }

        return report, timestamp

    def _generate_insights(self, classified_clauses: List[Dict],
                           classifications: List[Dict]) -> Dict:
        """Generate insights about the deal's termination provisions"""

        # Expected cluster category by trigger_type
        TRIGGER_TO_EXPECTED_CATEGORY = {
            'mutual_consent':        ['mutual_termination'],
            'outside_date':          ['outside_date', 'mutual_termination'],
            'regulatory_block':      ['regulatory_failure', 'mutual_termination'],
            'shareholder_vote_failure': ['mutual_termination', 'regulatory_failure'],
            'target_fiduciary_out':  ['fiduciary_out', 'target_termination'],
            'acquirer_fiduciary_out': ['fiduciary_out', 'acquirer_termination'],
            'target_breach':         ['breach_based', 'target_termination'],
            'acquirer_breach':       ['breach_based', 'acquirer_termination'],
            'financing_failure':     ['breach_based', 'acquirer_termination', 'financing_failure'],
            'other':                 None,  # any category ok
        }

        # Low similarity clauses (potentially unusual or novel termination triggers)
        low_similarity = [
            c for c in classifications if c['similarity_score'] < 0.75]

        # Type-mismatch clauses: trigger_type doesn't align with matched cluster category
        type_mismatches = []
        for clause, cls in zip(classified_clauses, classifications):
            trigger = clause.get('trigger_type', 'other')
            matched_cat = cls.get('cluster_trigger_category', 'unknown')
            expected = TRIGGER_TO_EXPECTED_CATEGORY.get(trigger)
            if expected is not None and matched_cat not in expected:
                type_mismatches.append({
                    'clause_index': cls['clause_index'],
                    'trigger_type': trigger,
                    'matched_category': matched_cat,
                    'expected_categories': expected,
                    'similarity_score': cls['similarity_score'],
                    'cluster_theme': cls.get('cluster_theme', ''),
                })

        # Combine low-similarity and type-mismatch as "flags"
        flagged_indices = set(c['clause_index'] for c in low_similarity) | \
            set(c['clause_index'] for c in type_mismatches)

        # Category breakdown
        categories = {}
        for cls in classifications:
            cat = cls['cluster_trigger_category']
            categories[cat] = categories.get(cat, 0) + 1

        top_categories = sorted(
            categories.items(), key=lambda x: x[1], reverse=True)[:5]

        # Party balance
        party_counts = {}
        for cls in classifications:
            party = cls['party_favored']
            party_counts[party] = party_counts.get(party, 0) + 1

        # Fee-bearing triggers — clause's matched cluster has substantive fee_implications text
        fee_triggers = [c for c in classifications
                        if len(str(c.get('fee_implications', ''))) > 30]

        # High-risk triggers (clusters labeled high_walkaway_risk)
        high_risk_triggers = [c for c in classified_clauses
                              if 'high' in str(c.get('deal_risk_pattern', '')).lower()]

        return {
            'unusual_clauses_count': len(flagged_indices),
            'unusual_clause_indices': sorted(flagged_indices),
            'low_similarity_clauses': [c['clause_index'] for c in low_similarity],
            'type_mismatch_clauses': type_mismatches,
            'top_trigger_categories': [{'category': cat, 'count': count}
                                       for cat, count in top_categories],
            'party_balance': party_counts,
            'fee_bearing_triggers': len(fee_triggers),
            'high_risk_trigger_count': len(high_risk_triggers),
            'coverage_quality': 'good' if len(flagged_indices) < len(classifications) * 0.15 else 'review_needed'
        }

    def _build_summary_rows(self, classified_clauses: List[Dict]) -> List[Dict]:
        rows = []
        for clause in classified_clauses:
            rows.append({
                'clause_id': clause.get('clause_id', ''),
                'trigger_type': clause.get('trigger_type', ''),
                'assigned_cluster': clause['assigned_cluster'],
                'cluster_trigger_category': clause['cluster_trigger_category'],
                'cluster_theme': clause['cluster_theme'],
                'similarity_score': f"{clause['similarity_score']:.3f}",
                'party_favored': clause['party_favored'],
                'deal_risk_pattern': clause['deal_risk_pattern'],
                'text_preview': str(clause.get('original_text', clause.get('text', '')))[:120] + '...'
            })
        return rows

    def _generate_summary_csv(self, classified_clauses: List[Dict], deal_id: str,
                              timestamp: str, output_dir: str):
        """Generate summary CSV for easy review (local file)."""
        rows = self._build_summary_rows(classified_clauses)
        df = pd.DataFrame(rows)
        csv_file = os.path.join(
            output_dir, f"termination_summary_{deal_id}_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"  Summary CSV: {csv_file}")

    def _generate_summary_csv_content(self, classified_clauses: List[Dict],
                                      deal_id: str, timestamp: str) -> str:
        """Generate summary CSV content as a string (for S3 upload)."""
        rows = self._build_summary_rows(classified_clauses)
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def print_summary(self, report_data: Dict):
        """Print summary to console"""
        print("\n" + "="*80)
        print("TERMINATION TRIGGER CLASSIFICATION SUMMARY")
        print("="*80)

        print(f"\nDeal: {report_data['deal_id']}")
        print(
            f"Termination clauses analyzed: {report_data['deal_summary']['total_termination_clauses']}")
        print(
            f"Avg similarity to benchmark: {report_data['deal_summary']['avg_similarity_to_clusters']:.2%}")
        print(
            f"Clusters matched: {report_data['deal_summary']['clusters_matched']}/{report_data['benchmark_info']['benchmark_clusters']}")

        print(f"\nTrigger Categories:")
        for category, count in sorted(report_data['trigger_category_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)[:10]:
            pct = count / \
                report_data['deal_summary']['total_termination_clauses'] * 100
            print(f"  - {category}: {count} clauses ({pct:.1f}%)")

        print(f"\nParty Balance:")
        for party, count in sorted(report_data.get('party_distribution', {}).items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  - {party}: {count} clauses")

        insights = report_data['insights']
        print(f"\nInsights:")
        print(
            f"  - Flagged clauses (low similarity or type mismatch): {insights['unusual_clauses_count']}")
        print(f"  - Fee-bearing triggers: {insights['fee_bearing_triggers']}")
        print(
            f"  - High-risk triggers (walkaway): {insights['high_risk_trigger_count']}")
        print(f"  - Coverage quality: {insights['coverage_quality']}")

        mismatches = insights.get('type_mismatch_clauses', [])
        if mismatches:
            print(f"\n  ⚠️  Type mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(f"     clause {m['clause_index']}: trigger={m['trigger_type']} → "
                      f"matched cluster={m['matched_category']} "
                      f"(expected: {m['expected_categories']}) "
                      f"sim={m['similarity_score']:.3f} | \"{m['cluster_theme']}\"")

        low_sim = insights.get('low_similarity_clauses', [])
        if low_sim:
            print(f"\n  ⚠️  Low similarity clauses: indices {low_sim}")

        if insights['unusual_clauses_count'] > 0:
            print(f"\nReview these clauses (low similarity to benchmark):")
            for idx in insights['unusual_clause_indices'][:5]:
                clause = report_data['classified_clauses'][idx]
                print(f"    - Clause {idx}: trigger_type={clause.get('trigger_type', 'N/A')} "
                      f"(similarity: {clause['similarity_score']:.2%})")


def run_stage6(triggers_s3_url: str, accession: str, doc_type: str) -> Dict:
    """
    New-flow entry point: download triggers from S3, classify, upload results to S3.
    Returns dict with S3 URLs: classification_json, summary_csv, and the report data.
    """
    from termination_s3_utils import download_json

    print("="*80)
    print("STAGE 6: CLASSIFY NEW DEAL TERMINATION TRIGGERS (S3 FLOW)")
    print("="*80)

    deal_data = download_json(triggers_s3_url)
    if not deal_data or 'clauses' not in deal_data or not deal_data['clauses']:
        raise ValueError(
            f"Invalid or empty triggers data from {triggers_s3_url}")

    original_count = len(deal_data['clauses'])
    clauses = [c for c in deal_data['clauses']
               if c.get('trigger_type') != 'preamble']
    preamble_count = original_count - len(clauses)
    if preamble_count > 0:
        print(f"  Skipped {preamble_count} preamble clause(s)")
    if not clauses:
        raise ValueError("No non-preamble clauses found after filtering")

    print(f"  Accession: {accession}")
    print(f"  Clauses to classify: {len(clauses)} (of {original_count} total)")

    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not set")

    classifier = NewDealTerminationClassifier(cohere_key)
    classifier.load_benchmark(CACHE_DIR)
    embeddings = classifier.generate_embeddings(clauses)
    classifications = classifier.classify_clauses(embeddings)

    result = classifier.generate_deal_report_s3(
        deal_data, clauses, classifications,
        accession, doc_type,
    )

    classifier.print_summary(result["report"])

    print("\n" + "="*80)
    print("STAGE 6 COMPLETE!")
    print("="*80)

    return {
        "classification_json": result["classification_json"],
        "summary_csv": result["summary_csv"],
        "report": result["report"],
    }


def main():
    """Main execution"""
    print("="*80)
    print("STAGE 6: CLASSIFY NEW DEAL TERMINATION TRIGGERS AGAINST BENCHMARK")
    print("="*80)

    # Check for input file
    if len(sys.argv) < 2:
        print("\nError: Please provide path to termination triggers file")
        print("\nUsage:")
        print("  python3 6_classify_new_deal_termination.py path/to/termination_response_ACCESSION_triggers.json")
        print("\nExpected input format:")
        print('  {"document_id": "...", "total_clauses": N, "clauses": [')
        print(
            '    {"section_number": ..., "clause_id": ..., "trigger_type": ..., "original_text": ...}')
        print('  ]}')
        return

    deal_file = sys.argv[1]
    print(f"Deal file: {deal_file}")

    if not os.path.exists(deal_file):
        print(f"\nError: File not found: {deal_file}")
        return

    # Load deal data
    print(
        f"\nLoading new deal termination file: {os.path.basename(deal_file)}")
    with open(deal_file, 'r') as f:
        deal_data = json.load(f)

    # Check format
    if 'clauses' not in deal_data:
        print("Error: Invalid file format - missing 'clauses' field")
        return

    if len(deal_data['clauses']) == 0:
        print("Error: No clauses found in deal file")
        return

    # Filter out preamble clauses
    original_count = len(deal_data['clauses'])
    clauses = [c for c in deal_data['clauses']
               if c.get('trigger_type') != 'preamble']
    preamble_count = original_count - len(clauses)

    if preamble_count > 0:
        print(f"  Skipped {preamble_count} preamble clause(s)")

    if len(clauses) == 0:
        print("Error: No non-preamble clauses found after filtering")
        return

    deal_id = deal_data.get('deal_id') or deal_data.get(
        'document_id', 'unknown')
    print(f"  Deal ID: {deal_id}")
    print(
        f"  Termination clauses to classify: {len(clauses)} (of {original_count} total)")

    # Get API key
    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        print("\nERROR: COHERE_API_KEY not set")
        print("   Please add to .env file")
        return

    try:
        # Initialize classifier
        classifier = NewDealTerminationClassifier(cohere_key)

        # Load benchmark
        classifier.load_benchmark(CACHE_DIR)

        # Generate embeddings for new deal
        embeddings = classifier.generate_embeddings(clauses)

        # Classify against benchmark
        classifications = classifier.classify_clauses(embeddings)

        # Generate report
        report_file = classifier.generate_deal_report(
            deal_data, clauses, classifications, OUTPUT_DIR
        )

        # Load and print summary
        with open(report_file, 'r') as f:
            report_data = json.load(f)

        classifier.print_summary(report_data)

        print("\n" + "="*80)
        print("STAGE 6 COMPLETE!")
        print("="*80)
        print(f"\nReport saved to: {report_file}")
        print(f"Estimated cost: ~$0.01 (embeddings only)")
        print(f"\nNext: Run Stage 7 to assess deal risk:")
        print(f"  python3 7_assess_new_deal_termination.py {report_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
