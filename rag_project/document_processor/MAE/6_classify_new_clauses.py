
#!/usr/bin/env python3
"""
Stage 6: Classify New MAE Clauses (Clean Version)
- Matches Stage 1 preprocessing exactly
- Uses full 1536 dimensions (no truncation)
- Guaranteed consistency with training embeddings
"""

import json
import numpy as np
import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import cohere

# ================================
# CONFIGURATION - paths relative to script/project root
# ================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent

INPUT_FILE = str(_SCRIPT_DIR / "UHG_clauses.json")
# Use AWS S3 URL for benchmark (can also use local path)
BENCHMARK_FILE = "https://rag-mna-doc.s3.eu-north-1.amazonaws.com/MAE_BenchMark/benchmark_MAE.json"
OUTPUT_DIR = str(_BASE_DIR / "classification_output")

# ================================
# Classification thresholds
# ================================
TYPICAL_THRESHOLD = 1.0    # ratio <= 1.0 = inside cluster
ATYPICAL_THRESHOLD = 1.25  # ratio > 1.25 = outlier

# ================================
# Simple Preprocessor - MATCHES STAGE 1 EXACTLY
# ================================
class SimplePreprocessor:
    """MUST match Stage 1 preprocessing exactly"""
    
    def preprocess(self, text: str) -> str:
        """Clean text - SAME as Stage 1"""
        
        # Remove leading boilerplate
        text = re.sub(
            r'^(?:except\s+)?(?:as\s+)?(?:set\s+forth|provided)\s+(?:in|below)[:,]?\s*',
            '', text, flags=re.IGNORECASE
        )
        
        # Standardize entities (SAME as Stage 1)
        substitutions = {
            r'\b(?:the\s+)?(?:company|acquired\s+compan(?:y|ies)|target)\s+and\s+(?:its\s+)?subsidiaries\b': 'COMPANY_ENTITY',
            r'\b(?:the\s+)?(?:company|acquired\s+compan(?:y|ies)|target)(?:\s+and\s+its\s+subsidiaries)?\b': 'COMPANY_ENTITY',
            r'\bmaterial\s+adverse\s+effect\b': 'MAE',
            r'\bgeneral(?:ly)?\s+affecting\b': 'GENERALLY_AFFECTING',
            r'\bdisproportionate(?:ly)?\s+(?:manner|affect)\b': 'DISPROPORTIONATE_EFFECT'
        }
        
        for pattern, replacement in substitutions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# ================================
# Embedding Generator - MATCHES STAGE 1 EXACTLY
# ================================
class EmbeddingGenerator:
    """Generate embeddings - MUST match Stage 1 exactly"""
    
    def __init__(self, expected_dims: int, cohere_key: str):
        self.client = cohere.ClientV2(cohere_key)
        self.preprocessor = SimplePreprocessor()
        self.expected_dims = expected_dims
        print(f"✅ Cohere client initialized")
        print(f"   Model: embed-v4.0")
        print(f"   Dimensions: {expected_dims} (full, matching Stage 1)")
    
    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for single clause"""
        
        # Preprocess (SAME as Stage 1)
        cleaned = self.preprocessor.preprocess(text)
        
        try:
            # Generate embedding (SAME params as Stage 1)
            response = self.client.embed(
                texts=[cleaned[:4096]],  # Same text limit as Stage 1
                model="embed-v4.0",  # Same model as Stage 1
                input_type="search_document",  # Same input type as Stage 1
                embedding_types=['float']  # Same as Stage 1
            )
            
            # Get full embedding (no truncation, SAME as Stage 1)
            embedding = np.array(response.embeddings.float[0], dtype=float)
            
            # Verify dimensions
            if embedding.shape[0] != self.expected_dims:
                raise ValueError(
                    f"Dimension mismatch: got {embedding.shape[0]}, expected {self.expected_dims}"
                )
            
            return embedding
            
        except Exception as e:
            print(f"\n❌ Embedding error: {e}")
            raise


# ================================
# Classifier
# ================================
class ClauseClassifier:
    """Classify new clauses against benchmark clusters"""
    
    def __init__(self, benchmark_file: str, cohere_key: str):
        print(f"📂 Loading benchmark: {benchmark_file}")
        
        # Check if benchmark_file is a URL or local path
        if benchmark_file.startswith('http://') or benchmark_file.startswith('https://'):
            import urllib.request
            print(f"   Fetching from URL...")
            with urllib.request.urlopen(benchmark_file) as response:
                self.benchmark = json.loads(response.read().decode('utf-8'))
        else:
            with open(benchmark_file, 'r') as f:
                self.benchmark = json.load(f)
        
        # Extract clusters from cluster_metrics.cluster_thresholds
        cluster_data = self.benchmark['cluster_metrics']['cluster_thresholds']
        
        # Get distance metric from benchmark
        self.metric = self.benchmark['cluster_metrics'].get('metric', 'cosine')
        
        # Build clusters dict in expected format
        self.clusters = {}
        self.centroids = {}
        
        for cluster_id, data in cluster_data.items():
            cluster_id_int = int(cluster_id)
            
            # Extract centroid
            self.centroids[cluster_id_int] = np.array(data['centroid'])
            
            # Build cluster info
            self.clusters[cluster_id] = {
                'label': data.get('cluster_label', f'Cluster {cluster_id}'),
                'category': data.get('legal_category', 'unknown'),
                'size': data.get('size', 0),
                'threshold_95th': data.get('distance_threshold', 0.5)
            }
        
        self.dims = len(next(iter(self.centroids.values())))
        
        print(f"✅ Loaded {len(self.clusters)} clusters")
        print(f"   Dimensions: {self.dims}")
        print(f"   Distance metric: {self.metric}")
        
        # Initialize embedder
        self.embedder = EmbeddingGenerator(self.dims, cohere_key)
    
    def classify_clause(self, text: str, label: str) -> Dict:
        """Classify a single clause"""
        
        # Generate embedding
        embedding = self.embedder.generate(text)
        
        # Calculate distances to all centroids using the correct metric
        distances = {}
        for cluster_id, centroid in self.centroids.items():
            if self.metric == 'cosine':
                # Cosine distance = 1 - cosine similarity
                similarity = np.dot(embedding, centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(centroid)
                )
                dist = float(1 - similarity)
            else:  # euclidean
                dist = float(np.linalg.norm(embedding - centroid))
            
            distances[cluster_id] = dist
        
        # Build results for each cluster (avoid inf/nan for JSON and sort order)
        MIN_THRESHOLD = 1e-10
        MAX_RATIO = 999.0
        results = []
        for cluster_id, distance in distances.items():
            cluster_info = self.clusters[str(cluster_id)]
            threshold = float(cluster_info['threshold_95th'])
            if threshold < MIN_THRESHOLD:
                threshold = MIN_THRESHOLD
            ratio = distance / threshold
            if not (ratio <= MAX_RATIO):  # catch inf/nan
                ratio = MAX_RATIO
            ratio = float(ratio)
            results.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_info['label'],
                'category': cluster_info.get('category', 'unknown'),
                'distance': float(distance),
                'threshold': float(threshold),
                'ratio': ratio,
                'cluster_size': int(cluster_info['size']) if cluster_info['size'] is not None else 0
            })
        
        # Sort by ratio (best match first)
        results.sort(key=lambda x: x['ratio'])
        
        # Best match
        best = results[0]
        
        # Determine zone
        if best['ratio'] <= TYPICAL_THRESHOLD:
            zone = 'typical'
            icon = '🟢'
        elif best['ratio'] <= ATYPICAL_THRESHOLD:
            zone = 'atypical'
            icon = '🟡'
        else:
            zone = 'outlier'
            icon = '🔴'
        
        # Find all matches (ratio <= 1.0)
        matches = [r for r in results if r['ratio'] <= TYPICAL_THRESHOLD]
        
        return {
            'label': label,
            'text': text,
            'best_match': best,
            'all_matches': matches,
            'zone': zone,
            'icon': icon,
            'all_clusters': results
        }
    
    def process_file(self, input_file: str, output_dir: str):
        """Process a file of new clauses"""
        
        print(f"\n📂 Loading: {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle both formats
        if isinstance(data, list):
            clauses = data
            deal_name = clauses[0].get('deal', 'Unknown') if clauses else 'Unknown'
        else:
            deal_name = data.get('deal_name', 'Unknown')
            clauses = data.get('clauses', [])
        
        print(f"🎯 Processing {len(clauses)} clauses from: {deal_name}\n")
        
        # Classify each clause
        results = []
        for i, clause in enumerate(clauses, 1):
            label = clause.get('label', clause.get('clause_id', f'clause_{i}'))
            text = clause.get('text', '')
            
            if not text:
                print(f"  ⚠️ Skipping {label}: no text")
                continue
            
            result = self.classify_clause(text, label)
            results.append(result)
            
            best = result['best_match']
            match_count = len(result['all_matches'])
            match_text = f"{match_count} matches" if match_count > 1 else "single match"
            
            print(f"  [{i:3d}/{len(clauses)}] {result['icon']} {label}: "
                  f"{best['cluster_name']} (ratio {best['ratio']:.2f}, {match_text})")
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full results
        output_file = f"{output_dir}/classification_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'deal_name': deal_name,
                'classification_date': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        print(f"\n✅ Saved: {output_file}")
        
        # Summary CSV
        summary_data = []
        for r in results:
            best = r['best_match']
            summary_data.append({
                'label': r['label'],
                'zone': r['zone'],
                'best_cluster': best['cluster_name'],
                'category': best['category'],
                'ratio': round(best['ratio'], 2),
                'distance': round(best['distance'], 4),
                'threshold': round(best['threshold'], 4),
                'num_matches': len(r['all_matches']),
                'text_preview': r['text'][:100]
            })
        
        summary_file = f"{output_dir}/summary_{timestamp}.csv"
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        print(f"✅ Saved: {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"📊 CLASSIFICATION SUMMARY: {deal_name}")
        print("="*80)
        
        zones = {'typical': 0, 'atypical': 0, 'outlier': 0}
        for r in results:
            zones[r['zone']] += 1
        
        total = len(results)
        print(f"✅ {zones['typical']} clauses ({zones['typical']/total*100:.1f}%): 🟢 TYPICAL - Auto-classify")
        print(f"⚠️  {zones['atypical']} clauses ({zones['atypical']/total*100:.1f}%): 🟡 ATYPICAL - Quick review")
        print(f"🚨 {zones['outlier']} clauses ({zones['outlier']/total*100:.1f}%): 🔴 OUTLIER - Flag for analysis")
        
        multi_label = sum(1 for r in results if len(r['all_matches']) > 1)
        print(f"📋 Multi-label clauses: {multi_label} ({multi_label/total*100:.1f}%)")
        
        # Category distribution
        categories = {}
        for r in results:
            cat = r['best_match']['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"📂 Category distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            print(f"   {cat}: {count} clauses")
        
        print("="*80)
        
        return results

    def process_clauses(self, clauses: list, deal_name: str) -> dict:
        """
        Classify clauses from in-memory list. No file I/O.
        clauses: list of {label, text, deal?}
        Returns: {deal_name, classification_date, results}
        """
        print(f"\n🎯 Processing {len(clauses)} clauses from: {deal_name}\n")
        results = []
        for i, clause in enumerate(clauses, 1):
            label = clause.get('label', clause.get('clause_id', f'clause_{i}'))
            text = clause.get('text', '')
            if not text:
                print(f"  ⚠️ Skipping {label}: no text")
                continue
            result = self.classify_clause(text, label)
            results.append(result)
            best = result['best_match']
            match_count = len(result['all_matches'])
            match_text = f"{match_count} matches" if match_count > 1 else "single match"
            print(f"  [{i:3d}/{len(clauses)}] {result['icon']} {label}: "
                  f"{best['cluster_name']} (ratio {best['ratio']:.2f}, {match_text})")
        total = len(results)
        if total:
            zones = {'typical': 0, 'atypical': 0, 'outlier': 0}
            for r in results:
                zones[r['zone']] += 1
            print(f"\n📊 Summary: 🟢 {zones['typical']} typical  🟡 {zones['atypical']} atypical  🔴 {zones['outlier']} outlier")
        return {
            'deal_name': deal_name,
            'classification_date': datetime.now().isoformat(),
            'results': results
        }


def main():
    """Main execution"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("="*80)
    print("STAGE 6: CLAUSE CLASSIFICATION (CLEAN VERSION)")
    print("="*80)
    
    # Get API key
    cohere_key = os.getenv('COHERE_API_KEY')
    if not cohere_key:
        print("❌ ERROR: COHERE_API_KEY not set")
        return
    
    # Check files
    if not Path(INPUT_FILE).exists():
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        print(f"   Edit INPUT_FILE on line 24")
        return
    
    if not Path(BENCHMARK_FILE).exists():
        print(f"❌ ERROR: Benchmark file not found: {BENCHMARK_FILE}")
        print(f"   Edit BENCHMARK_FILE on line 25")
        print(f"   Use the benchmark file from Stage 5")
        return
    
    print(f"Input: {INPUT_FILE}")
    print(f"Benchmark: {BENCHMARK_FILE}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Classify
    try:
        classifier = ClauseClassifier(BENCHMARK_FILE, cohere_key)
        results = classifier.process_file(INPUT_FILE, OUTPUT_DIR)
        
        print(f"\n✅ Classification complete! Processed {len(results)} clauses.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()