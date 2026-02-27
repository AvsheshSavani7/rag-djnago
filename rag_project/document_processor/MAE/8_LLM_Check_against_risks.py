#!/usr/bin/env python3
"""
Stage 9: Compliance Checklist (OPTIMIZED)
Runs client-specific compliance checks on ALL clauses using single Opus call per clause.
Catches specific risks (cybersecurity, tariffs, disclosure schedules, etc.) that embeddings might miss.

CRITICAL: This runs on ALL clauses, not just outliers, to ensure no compliance issues are missed.

Cost: ~$0.33 per deal (was $0.88 with multiple calls)
Time: ~1 minute for 11 clauses
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic
from tqdm import tqdm


class ComplianceChecker:
    """Run compliance checks on all MAE clauses"""
    
    def __init__(self, anthropic_key: str):
        """
        Initialize compliance checker
        
        Args:
            anthropic_key: Anthropic API key for Claude Opus
        """
        self.client = Anthropic(api_key=anthropic_key)
        
    def analyze_deal(self, classification_file: str, output_dir: str) -> Dict:
        """
        Run compliance checks on all clauses
        
        Args:
            classification_file: Output from Stage 6
            output_dir: Directory to save results
        
        Returns:
            Comprehensive compliance analysis
        """
        
        # Load classification results
        print(f"📂 Loading classification from: {classification_file}")
        with open(classification_file, 'r') as f:
            classification = json.load(f)
        
        deal_name = classification['deal_name']
        clauses = classification['results']  # NEW: 'results' instead of old format
        
        print(f"\n📄 Running compliance checks: {deal_name}")
        print(f"   Total clauses: {len(clauses)}")
        
        estimated_cost = len(clauses) * 0.03  # Opus cost per clause
        print(f"   Estimated cost: ${estimated_cost:.2f}")
        
        # Run compliance checks on all clauses
        compliance_results = []
        
        print(f"\n🔍 Analyzing all clauses with Claude Opus...")
        for i, clause in enumerate(tqdm(clauses, desc="Compliance checks"), 1):
            
            # Run single compliance check with all questions
            compliance = self._check_compliance(
                clause['text'],
                clause.get('label', f"clause_{i}")
            )
            
            # Build result
            result = {
                'clause_id': clause.get('label', f"clause_{i}"),
                'text': clause['text'],
                'text_preview': clause['text'][:100] + "..." if len(clause['text']) > 100 else clause['text'],
                'zone': clause['zone'],
                'cluster_match': clause['best_match']['cluster_name'],
                'category': clause['best_match']['category'],
                'compliance': compliance
            }
            
            compliance_results.append(result)
        
        # Generate summary
        summary = self._generate_summary(compliance_results)
        
        # Prepare comprehensive results
        results = {
            'deal_name': deal_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clauses': len(clauses),
            'compliance_summary': summary,
            'detailed_results': compliance_results,
            'classification_file': os.path.basename(classification_file)
        }
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main compliance file
        output_file = f"{output_dir}/compliance_{deal_name}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Compliance analysis saved: {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results

    def analyze_deal_from_data(self, classification: dict) -> Dict:
        """
        Run compliance checks from in-memory classification dict. No file I/O.
        classification: {deal_name, results} from Stage 6.
        Returns: compliance results dict (same as analyze_deal).
        """
        deal_name = classification['deal_name']
        clauses = classification['results']

        print(f"\n📄 Running compliance checks: {deal_name}")
        print(f"   Total clauses: {len(clauses)}")
        estimated_cost = len(clauses) * 0.03
        print(f"   Estimated cost: ${estimated_cost:.2f}")

        compliance_results = []
        print(f"\n🔍 Analyzing all clauses with Claude Opus...")
        for i, clause in enumerate(tqdm(clauses, desc="Compliance checks"), 1):
            compliance = self._check_compliance(
                clause['text'],
                clause.get('label', f"clause_{i}")
            )
            result = {
                'clause_id': clause.get('label', f"clause_{i}"),
                'text': clause['text'],
                'text_preview': clause['text'][:100] + "..." if len(clause['text']) > 100 else clause['text'],
                'zone': clause['zone'],
                'cluster_match': clause['best_match']['cluster_name'],
                'category': clause['best_match']['category'],
                'compliance': compliance
            }
            compliance_results.append(result)

        summary = self._generate_summary(compliance_results)
        results = {
            'deal_name': deal_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clauses': len(clauses),
            'compliance_summary': summary,
            'detailed_results': compliance_results,
            'classification_file': None
        }
        self._print_summary(results)
        return results

    def _check_compliance(self, text: str, clause_id: str) -> Dict:
        """
        Run all compliance checks on a single clause using ONE Opus call
        
        This is the CRITICAL optimization: all 7 questions in one call instead of 7 separate calls
        """
        
        prompt = f"""Analyze this Material Adverse Effect (MAE) exclusion clause for specific compliance risks.

CLAUSE ID: {clause_id}

CLAUSE TEXT:
{text}

Answer ALL of the following questions. This is a compliance checklist - be precise and literal.

Return your answers in this EXACT JSON format:
{{
    "cybersecurity_mentioned": "Yes|No",
    "cybersecurity_details": "Brief quote if yes, otherwise null",
    
    "tariffs_trade_mentioned": "Yes|No",
    "tariffs_trade_details": "Brief quote if yes, otherwise null",
    
    "countries_regions_mentioned": "Yes|No",
    "countries_list": ["country1", "country2"] or null,
    "countries_details": "How they were referenced, or null",
    
    "geographic_changes_mentioned": "Yes|No",
    "geographic_changes_details": "Brief description if yes, otherwise null",
    
    "disclosure_schedules_referenced": "Yes|No",
    "disclosure_schedules_details": "What schedules/sections referenced, or null",
    
    "financing_mentioned": "Yes|No",
    "financing_details": "Brief quote if yes, otherwise null",
    
    "regulatory_products_mentioned": "Yes|No",
    "regulatory_products_details": "Brief description if yes, otherwise null"
}}

IMPORTANT GUIDANCE:

1. CYBERSECURITY/HACKS:
   - Look for: cyber attacks, cyber terrorism, cyber espionage, cyber war, data breaches, hacking, ransomware
   - Answer "Yes" only if specifically mentioned

2. TARIFFS/TRADE DISPUTES:
   - Look for: tariffs, trade disputes, trade wars, import/export restrictions, customs duties
   - Answer "Yes" only if specifically mentioned

3. COUNTRIES/REGIONS:
   - EXCLUDE any references to "United States", "U.S.", "US", "USA"
   - Only list OTHER countries or regions if specifically named
   - Generic terms like "any country" without specific names = "No"

4. GEOGRAPHIC CHANGES:
   - Changes to geographical areas where company operates
   - Particularly important for insurance deals
   - Answer "Yes" only if operational geography is mentioned

5. DISCLOSURE SCHEDULES:
   - References to "Company Disclosure Schedule", "Section X.X of Disclosure Letter", etc.
   - Answer "Yes" only if specific schedules/sections are referenced

6. FINANCING AVAILABILITY:
   - Availability and cost of financing for M&A entities
   - Credit markets, lending conditions, debt financing
   - Answer "Yes" only if financing conditions mentioned

7. REGULATORY CHANGES ON PRODUCTS:
   - Regulatory approvals, FDA decisions, product clearances
   - Changes affecting company's specific products/services
   - Answer "Yes" only if product-specific regulations mentioned

Be literal and precise. Only answer "Yes" if the specific item is clearly present in the text."""

        try:
            response = self.client.messages.create(
                model="claude-opus-4-20250514",  # Using Opus for accuracy
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
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
            response_text = ''.join(filter(lambda x: x in printable, response_text))
            response_text = response_text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
            
            result = json.loads(response_text)
            
            # Ensure all required fields exist
            required_fields = {
                'cybersecurity_mentioned': 'No',
                'cybersecurity_details': None,
                'tariffs_trade_mentioned': 'No',
                'tariffs_trade_details': None,
                'countries_regions_mentioned': 'No',
                'countries_list': None,
                'countries_details': None,
                'geographic_changes_mentioned': 'No',
                'geographic_changes_details': None,
                'disclosure_schedules_referenced': 'No',
                'disclosure_schedules_details': None,
                'financing_mentioned': 'No',
                'financing_details': None,
                'regulatory_products_mentioned': 'No',
                'regulatory_products_details': None
            }
            
            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error checking compliance for {clause_id}: {e}")
            return {
                'cybersecurity_mentioned': 'Error',
                'cybersecurity_details': f'Analysis failed: {str(e)[:100]}',
                'tariffs_trade_mentioned': 'Error',
                'tariffs_trade_details': None,
                'countries_regions_mentioned': 'Error',
                'countries_list': None,
                'countries_details': None,
                'geographic_changes_mentioned': 'Error',
                'geographic_changes_details': None,
                'disclosure_schedules_referenced': 'Error',
                'disclosure_schedules_details': None,
                'financing_mentioned': 'Error',
                'financing_details': None,
                'regulatory_products_mentioned': 'Error',
                'regulatory_products_details': None
            }
    
    def _generate_summary(self, compliance_results: List[Dict]) -> Dict:
        """Generate summary statistics from compliance results"""
        
        summary = {
            'cybersecurity': {
                'count': 0,
                'clauses': []
            },
            'tariffs_trade': {
                'count': 0,
                'clauses': []
            },
            'countries_regions': {
                'count': 0,
                'clauses': [],
                'countries_found': []
            },
            'geographic_changes': {
                'count': 0,
                'clauses': []
            },
            'disclosure_schedules': {
                'count': 0,
                'clauses': []
            },
            'financing': {
                'count': 0,
                'clauses': []
            },
            'regulatory_products': {
                'count': 0,
                'clauses': []
            }
        }
        
        for result in compliance_results:
            comp = result['compliance']
            clause_id = result['clause_id']
            
            # Cybersecurity
            if comp.get('cybersecurity_mentioned', '').lower() == 'yes':
                summary['cybersecurity']['count'] += 1
                summary['cybersecurity']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('cybersecurity_details')
                })
            
            # Tariffs/Trade
            if comp.get('tariffs_trade_mentioned', '').lower() == 'yes':
                summary['tariffs_trade']['count'] += 1
                summary['tariffs_trade']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('tariffs_trade_details')
                })
            
            # Countries/Regions
            if comp.get('countries_regions_mentioned', '').lower() == 'yes':
                summary['countries_regions']['count'] += 1
                summary['countries_regions']['clauses'].append({
                    'clause_id': clause_id,
                    'countries': comp.get('countries_list', []),
                    'details': comp.get('countries_details')
                })
                if comp.get('countries_list'):
                    summary['countries_regions']['countries_found'].extend(comp['countries_list'])
            
            # Geographic Changes
            if comp.get('geographic_changes_mentioned', '').lower() == 'yes':
                summary['geographic_changes']['count'] += 1
                summary['geographic_changes']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('geographic_changes_details')
                })
            
            # Disclosure Schedules
            if comp.get('disclosure_schedules_referenced', '').lower() == 'yes':
                summary['disclosure_schedules']['count'] += 1
                summary['disclosure_schedules']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('disclosure_schedules_details')
                })
            
            # Financing
            if comp.get('financing_mentioned', '').lower() == 'yes':
                summary['financing']['count'] += 1
                summary['financing']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('financing_details')
                })
            
            # Regulatory Products
            if comp.get('regulatory_products_mentioned', '').lower() == 'yes':
                summary['regulatory_products']['count'] += 1
                summary['regulatory_products']['clauses'].append({
                    'clause_id': clause_id,
                    'details': comp.get('regulatory_products_details')
                })
        
        # Deduplicate countries list
        if summary['countries_regions']['countries_found']:
            summary['countries_regions']['countries_found'] = list(set(
                summary['countries_regions']['countries_found']
            ))
        
        return summary
    
    def _print_summary(self, results: Dict):
        """Print compliance summary to console"""
        
        summary = results['compliance_summary']
        
        print("\n" + "="*80)
        print(f"📋 COMPLIANCE CHECKLIST SUMMARY: {results['deal_name']}")
        print("="*80)
        
        print(f"\n📊 Total clauses analyzed: {results['total_clauses']}")
        
        # Cybersecurity
        cyber = summary['cybersecurity']
        if cyber['count'] > 0:
            print(f"\n🔐 Cybersecurity Mentions: {cyber['count']}")
            for clause in cyber['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n🔐 Cybersecurity: None found ✓")
        
        # Tariffs/Trade
        tariffs = summary['tariffs_trade']
        if tariffs['count'] > 0:
            print(f"\n📦 Tariffs/Trade Mentions: {tariffs['count']}")
            for clause in tariffs['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n📦 Tariffs/Trade: None found ✓")
        
        # Countries
        countries = summary['countries_regions']
        if countries['count'] > 0:
            print(f"\n🌍 Countries/Regions Mentioned: {countries['count']}")
            if countries['countries_found']:
                print(f"   Countries: {', '.join(countries['countries_found'])}")
            for clause in countries['clauses']:
                print(f"   • {clause['clause_id']}: {', '.join(clause.get('countries', []))}")
        else:
            print(f"\n🌍 Countries/Regions: None found ✓")
        
        # Disclosure Schedules (CRITICAL)
        disclosure = summary['disclosure_schedules']
        if disclosure['count'] > 0:
            print(f"\n⚠️  DISCLOSURE SCHEDULES REFERENCED: {disclosure['count']} ⚠️")
            for clause in disclosure['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n📄 Disclosure Schedules: None found ✓")
        
        # Geographic Changes
        geo = summary['geographic_changes']
        if geo['count'] > 0:
            print(f"\n🗺️  Geographic Changes: {geo['count']}")
            for clause in geo['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n🗺️  Geographic Changes: None found ✓")
        
        # Financing
        fin = summary['financing']
        if fin['count'] > 0:
            print(f"\n💰 Financing Mentions: {fin['count']}")
            for clause in fin['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n💰 Financing: None found ✓")
        
        # Regulatory Products
        reg = summary['regulatory_products']
        if reg['count'] > 0:
            print(f"\n⚖️  Regulatory Product Changes: {reg['count']}")
            for clause in reg['clauses']:
                print(f"   • {clause['clause_id']}: {clause['details']}")
        else:
            print(f"\n⚖️  Regulatory Products: None found ✓")
        
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
    _classification_files = sorted(_classification_dir.glob("classification_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    CLASSIFICATION_FILE = str(_classification_files[0]) if _classification_files else ""
    OUTPUT_DIR = str(_base_dir / "compliance_output")

    # Check files exist
    if not CLASSIFICATION_FILE or not Path(CLASSIFICATION_FILE).exists():
        print("❌ No classification file found in classification_output/")
        print(f"   Looked in: {_classification_dir}")
        print("\n💡 Run Stage 6 first to generate classification")
        return
    
    # Initialize checker
    checker = ComplianceChecker(ANTHROPIC_KEY)
    
    # Run compliance checks
    print("\n🚀 Starting Stage 9: Compliance Checklist")
    print("="*80)
    print("⚠️  CRITICAL: This checks ALL clauses for client-specific risks")
    print("="*80)
    
    results = checker.analyze_deal(
        classification_file=CLASSIFICATION_FILE,
        output_dir=OUTPUT_DIR
    )
    
    if results:
        print(f"\n✅ Stage 9 complete!")
        print(f"\n💡 Next: Run Stage 10 to generate complete analysis package")
        print(f"   (or run Stage 8 for Excel reports)")


if __name__ == "__main__":
    main()