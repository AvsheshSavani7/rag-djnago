#!/usr/bin/env python3
"""
Stage 6: Classify New Deal Against Benchmark

Takes a new deal's covenant clauses and classifies them against the existing
49-cluster benchmark without re-running the entire pipeline.

Usage:
    python3 6_classify_new_deal.py path/to/openai_response_DEALID_individual_clauses.json
"""

import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import cohere
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cosine

# ================================
# CONFIGURATION
# ================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CACHE_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/embeddings_cache"
OUTPUT_DIR = f"{PROJECT_ROOT}/Covenant_Embeddings_v1/new_deal_reports"

# Cohere settings (must match Stage 1)
COHERE_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "search_document"
COHERE_DIMS = 1536


class CovenantPreprocessor:
    """Text preprocessing for covenant clauses (same as Stage 1)"""

    def preprocess(self, text: str) -> str:
        """Clean text with covenant-specific substitutions"""
        import re

        substitutions = {
            r'\b(?:the\s+)?(?:company|target|titanium|kenvue|avidity|axalta|azek|brighthouse|celgene|compass|cvgw|cybr|nathan\'?s|penumbra|sealed\s+air|u\.?s\.?\s+steel|vimeo)(?:\s+and\s+(?:its\s+)?subsidiaries)?\b': 'COMPANY_ENTITY',
            r'\b(?:the\s+)?(?:parent|acquirer|buyer|purchaser|novartis|akzonobel|james\s+hardie|bristol[‐-]?myers\s+squibb|anywhere\s+real\s+estate|mission\s+produce|palo\s+alto\s+networks|johnson\s+&\s+johnson|smithfield|boston\s+scientific|nippon\s+steel|bending\s+spoons)\b': 'PARENT_ENTITY',
            r'\b(?:its\s+)?subsidiaries\b': 'SUBSIDIARIES',
            r'\b(?:the\s+)?(?:merger\s+)?sub(?:sidiary)?\b': 'MERGER_SUB',
            r'\b(?:the\s+)?(?:buyer|purchaser)\b': 'BUYER',
            r'\b(?:the\s+)?seller\b': 'SELLER',
        }

        text = text.lower()
        for pattern, replacement in substitutions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text


class NewDealClassifier:
    """Classifies new deal clauses against existing benchmark"""

    def __init__(self, cohere_key: str):
        self.cohere_client = cohere.ClientV2(cohere_key)
        self.preprocessor = CovenantPreprocessor()
        self.benchmark_data = None
        self.cluster_centroids = None
        self.cluster_analyses = None

    def load_benchmark(self, cache_dir: str):
        """Load the most recent benchmark clustering and analysis"""
        print("\n📂 Loading benchmark data...")

        # Find most recent clustering file
        clustering_files = sorted([f for f in os.listdir(cache_dir)
                                  if f.startswith('covenant_clustering_') and f.endswith('.json')])
        if not clustering_files:
            raise FileNotFoundError(
                "No clustering files found in cache directory")

        clustering_file = clustering_files[-1]
        clustering_path = os.path.join(cache_dir, clustering_file)

        with open(clustering_path, 'r') as f:
            self.benchmark_data = json.load(f)

        print(f"  ✓ Loaded clustering: {clustering_file}")
        print(f"    • {self.benchmark_data['n_valid_clauses']} clauses")
        print(f"    • {self.benchmark_data['metrics']['n_clusters']} clusters")

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
                                if f.startswith('covenant_cluster_analysis_')
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
            print(f"  ✓ Loaded analysis: {analysis_file}")
        else:
            print("  ⚠️  No cluster analysis found - will use basic classification only")
            self.cluster_analyses = {}

    def generate_embeddings(self, clauses: List[Dict]) -> np.ndarray:
        """Generate embeddings for new deal clauses"""
        print(f"\n🔄 Generating embeddings for {len(clauses)} clauses...")

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

            response = self.cohere_client.embed(
                texts=batch,
                model=COHERE_MODEL,
                input_type=COHERE_INPUT_TYPE,
                embedding_types=["float"]
            )

            batch_embeddings = response.embeddings.float_
            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings)
        print(f"  ✓ Generated embeddings: {embeddings_array.shape}")

        return embeddings_array

    def classify_clauses(self, embeddings: np.ndarray) -> List[Dict]:
        """Assign each clause to nearest cluster"""
        print("\n🎯 Classifying clauses against benchmark clusters...")

        classifications = []

        for i, embedding in enumerate(tqdm(embeddings, desc="Classifying")):
            # Calculate cosine similarity to each cluster centroid
            similarities = {}
            for cluster_id, centroid in self.cluster_centroids.items():
                similarity = 1 - cosine(embedding, centroid)
                similarities[cluster_id] = similarity

            # Assign to nearest cluster
            best_cluster = max(similarities.items(), key=lambda x: x[1])
            cluster_id = best_cluster[0]
            similarity_score = best_cluster[1]

            # Get cluster analysis if available
            cluster_info = self.cluster_analyses.get(cluster_id, {})

            classifications.append({
                'clause_index': i,
                'assigned_cluster': cluster_id,
                'similarity_score': float(similarity_score),
                'cluster_category': cluster_info.get('covenant_category', 'unknown'),
                'cluster_theme': cluster_info.get('cluster_theme', 'Unknown'),
                'restriction_type': cluster_info.get('restriction_type', 'unknown'),
                'typical_restrictiveness': cluster_info.get('restrictiveness_pattern', 'unknown')
            })

        return classifications

    def generate_deal_report(self, deal_data: Dict, classifications: List[Dict],
                             output_dir: str) -> str:
        """Generate comprehensive report for the new deal.
        Writes to local output_dir. Returns local file path."""
        report, timestamp = self._build_report(deal_data, classifications)

        deal_id = report['deal_id']
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(
            output_dir, f"deal_classification_{deal_id}_{timestamp}.json")

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Report saved: {report_file}")

        self._generate_summary_csv(
            report['classified_clauses'], deal_id, timestamp, output_dir)

        return report_file

    def generate_deal_report_s3(self, deal_data: Dict, classifications: List[Dict],
                                accession: str, doc_type: str = None) -> Dict[str, str]:
        """Generate report and upload to S3. Returns dict of S3 URLs."""
        from covenant_s3_utils import upload_json, upload_text

        report, timestamp = self._build_report(deal_data, classifications)
        deal_id = report['deal_id']

        _, classification_url = upload_json(
            report, accession, "classification_json.json")
        print(f"  ✓ Classification uploaded to S3: {classification_url}")

        csv_content = self._generate_summary_csv_content(
            report['classified_clauses'], deal_id, timestamp)
        _, summary_url = upload_text(csv_content, accession, "summary_csv.csv",
                                     content_type="text/csv; charset=utf-8")
        print(f"  ✓ Summary CSV uploaded to S3: {summary_url}")

        return {
            "classification_json": classification_url,
            "summary_csv": summary_url,
            "report": report,
        }

    def _build_report(self, deal_data: Dict, classifications: List[Dict]) -> tuple:
        """Build the classification report dict. Returns (report, timestamp)."""
        print("\n📊 Generating deal report...")

        deal_id = deal_data.get('deal_id') or deal_data.get(
            'document_id', 'unknown')
        clauses = deal_data['clauses']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        classified_clauses = []
        for i, clause in enumerate(clauses):
            classification = classifications[i]
            classified_clauses.append({
                **clause,
                **classification
            })

        cluster_distribution = {}
        category_distribution = {}
        for cls in classifications:
            cluster_id = cls['assigned_cluster']
            category = cls['cluster_category']

            cluster_distribution[cluster_id] = cluster_distribution.get(
                cluster_id, 0) + 1
            category_distribution[category] = category_distribution.get(
                category, 0) + 1

        avg_similarity = np.mean([c['similarity_score']
                                 for c in classifications])

        report = {
            'deal_id': deal_id,
            'analysis_timestamp': timestamp,
            'benchmark_info': {
                'benchmark_clusters': len(self.cluster_centroids),
                'benchmark_clauses': self.benchmark_data['n_valid_clauses'],
            },
            'deal_summary': {
                'total_clauses': len(clauses),
                'avg_similarity_to_clusters': float(avg_similarity),
                'clusters_matched': len(cluster_distribution),
                'section_titles': list(set([c['section_title'] for c in clauses]))
            },
            'cluster_distribution': cluster_distribution,
            'category_distribution': category_distribution,
            'classified_clauses': classified_clauses,
            'insights': self._generate_insights(classified_clauses, classifications)
        }

        return report, timestamp

    def _generate_insights(self, classified_clauses: List[Dict],
                           classifications: List[Dict]) -> Dict:
        """Generate insights about the deal"""

        # Low similarity clauses (potentially unusual)
        low_similarity = [
            c for c in classifications if c['similarity_score'] < 0.7]

        # Category breakdown
        categories = {}
        for cls in classifications:
            cat = cls['cluster_category']
            categories[cat] = categories.get(cat, 0) + 1

        top_categories = sorted(
            categories.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'unusual_clauses_count': len(low_similarity),
            'unusual_clause_indices': [c['clause_index'] for c in low_similarity],
            'top_covenant_categories': [{'category': cat, 'count': count}
                                        for cat, count in top_categories],
            'coverage_quality': 'good' if len(low_similarity) < len(classifications) * 0.1 else 'review_needed'
        }

    def _build_summary_rows(self, classified_clauses: List[Dict]) -> List[Dict]:
        rows = []
        for clause in classified_clauses:
            rows.append({
                'clause_id': clause.get('clause_id', ''),
                'section_title': clause.get('section_title', ''),
                'assigned_cluster': clause['assigned_cluster'],
                'cluster_category': clause['cluster_category'],
                'cluster_theme': clause['cluster_theme'],
                'similarity_score': f"{clause['similarity_score']:.3f}",
                'restriction_type': clause['restriction_type'],
                'text_preview': clause.get('text', '')[:100] + '...'
            })
        return rows

    def _generate_summary_csv(self, classified_clauses: List[Dict], deal_id: str,
                              timestamp: str, output_dir: str):
        """Generate summary CSV for easy review (local file)."""
        rows = self._build_summary_rows(classified_clauses)
        df = pd.DataFrame(rows)
        csv_file = os.path.join(
            output_dir, f"deal_summary_{deal_id}_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"  ✓ Summary CSV: {csv_file}")

    def _generate_summary_csv_content(self, classified_clauses: List[Dict],
                                      deal_id: str, timestamp: str) -> str:
        """Generate summary CSV content as a string (for S3 upload)."""
        rows = self._build_summary_rows(classified_clauses)
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def print_summary(self, report_data: Dict):
        """Print summary to console"""
        print("\n" + "="*80)
        print("📋 DEAL CLASSIFICATION SUMMARY")
        print("="*80)

        print(f"\n🆔 Deal: {report_data['deal_id']}")
        print(
            f"📊 Clauses analyzed: {report_data['deal_summary']['total_clauses']}")
        print(
            f"🎯 Avg similarity to benchmark: {report_data['deal_summary']['avg_similarity_to_clusters']:.2%}")
        print(
            f"📦 Clusters matched: {report_data['deal_summary']['clusters_matched']}/{report_data['benchmark_info']['benchmark_clusters']}")

        print(f"\n📂 Covenant Categories:")
        for category, count in sorted(report_data['category_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)[:10]:
            pct = count / report_data['deal_summary']['total_clauses'] * 100
            print(f"  • {category}: {count} clauses ({pct:.1f}%)")

        insights = report_data['insights']
        print(f"\n💡 Insights:")
        print(
            f"  • Unusual clauses (low similarity): {insights['unusual_clauses_count']}")
        print(f"  • Coverage quality: {insights['coverage_quality']}")

        if insights['unusual_clauses_count'] > 0:
            print(f"\n⚠️  Review these clauses (low similarity to benchmark):")
            for idx in insights['unusual_clause_indices'][:5]:
                clause = report_data['classified_clauses'][idx]
                print(f"    • Clause {idx}: {clause.get('section_title', 'N/A')} "
                      f"(similarity: {clause['similarity_score']:.2%})")


def run_stage6(clauses_s3_url: str, accession: str, doc_type: str = None) -> Dict:
    """
    S3-flow entry point: download clauses from S3, classify, upload results to S3.
    Returns dict with S3 URLs: classification_json, summary_csv, and the report data.
    """
    from covenant_s3_utils import download_json

    print("="*80)
    print("STAGE 6: CLASSIFY NEW DEAL COVENANTS (S3 FLOW)")
    print("="*80)

    deal_data = download_json(clauses_s3_url)
    if not deal_data or 'clauses' not in deal_data or not deal_data['clauses']:
        raise ValueError(
            f"Invalid or empty clauses data from {clauses_s3_url}")

    print(f"  Accession: {accession}")
    print(f"  Clauses to classify: {len(deal_data['clauses'])}")

    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not set")

    classifier = NewDealClassifier(cohere_key)
    classifier.load_benchmark(CACHE_DIR)
    embeddings = classifier.generate_embeddings(deal_data['clauses'])
    classifications = classifier.classify_clauses(embeddings)

    result = classifier.generate_deal_report_s3(
        deal_data, classifications,
        accession, doc_type,
    )

    classifier.print_summary(result["report"])

    print("\n" + "="*80)
    print("✅ STAGE 6 COMPLETE!")
    print("="*80)

    return {
        "classification_json": result["classification_json"],
        "summary_csv": result["summary_csv"],
        "report": result["report"],
    }


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()

    print("="*80)
    print("STAGE 6: CLASSIFY NEW DEAL AGAINST BENCHMARK")
    print("="*80)

    # Check for input file
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide path to deal file")
        print("\nUsage:")
        print("  python3 6_classify_new_deal.py path/to/openai_response_DEALID_individual_clauses.json")
        print("\nExample:")
        print("  python3 6_classify_new_deal.py ../openai_response_d12345dex21_individual_clauses.json")
        return

    deal_file = sys.argv[1]

    if not os.path.exists(deal_file):
        print(f"\n❌ Error: File not found: {deal_file}")
        return

    # Load deal data
    print(f"\n📂 Loading new deal: {os.path.basename(deal_file)}")
    with open(deal_file, 'r') as f:
        deal_data = json.load(f)

    # Check if it's in the correct format
    if 'clauses' not in deal_data:
        print("❌ Error: Invalid file format - missing 'clauses' field")
        return

    if len(deal_data['clauses']) == 0:
        print("❌ Error: No clauses found in deal file")
        return

    print(f"  ✓ Loaded {len(deal_data['clauses'])} clauses")
    print(
        f"  ✓ Deal ID: {deal_data.get('deal_id') or deal_data.get('document_id', 'unknown')}")

    # Get API key
    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        print("\n❌ ERROR: COHERE_API_KEY not set")
        print("   Please add to .env file")
        return

    try:
        # Initialize classifier
        classifier = NewDealClassifier(cohere_key)

        # Load benchmark
        classifier.load_benchmark(CACHE_DIR)

        # Generate embeddings for new deal
        embeddings = classifier.generate_embeddings(deal_data['clauses'])

        # Classify against benchmark
        classifications = classifier.classify_clauses(embeddings)

        # Generate report
        report_file = classifier.generate_deal_report(
            deal_data, classifications, OUTPUT_DIR
        )

        # Load and print summary
        with open(report_file, 'r') as f:
            report_data = json.load(f)

        classifier.print_summary(report_data)

        print("\n" + "="*80)
        print("✅ STAGE 6 COMPLETE!")
        print("="*80)
        print(f"\n📁 Report saved to: {report_file}")
        print(f"💰 Estimated cost: ~$0.01 (embeddings only)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
