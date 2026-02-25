"""
MAE (Material Adverse Effect) Analysis Pipeline

This package provides tools for analyzing Material Adverse Effect clauses
in merger agreements using machine learning and LLM-based analysis.

Main Components:
- Clause Extraction: Extract MAE exclusions from deal documents
- Classification: Classify clauses against benchmark using embeddings
- Risk Assessment: LLM-based risk analysis of flagged clauses
- Compliance Checks: Specific risk checks (cybersecurity, disclosure, etc.)

Usage:
    # Via Django management command (recommended)
    python manage.py run_mae_pipeline --input-file path/to/input.json
    
    # Or run full pipeline directly
    python run_full_pipeline.py
"""

__version__ = "1.0.0"
__author__ = "MAE Analysis Team"

# Package metadata
__all__ = [
    'run_full_pipeline',
    'ClauseClassifier',
    'NewDealRiskAnalyzer',
    'ComplianceChecker',
]
