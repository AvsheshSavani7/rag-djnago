"""
Django management command to run the MAE (Material Adverse Effect) analysis pipeline.

Usage:
    python manage.py run_mae_pipeline --deal-name "Company Name" --mae-file path/to/mae.json
    python manage.py run_mae_pipeline --deal-name "Company Name" --mae-text "MAE clause text..."
    python manage.py run_mae_pipeline --input-file path/to/deal_input.json

Examples:
    # From JSON file with MAE clauses
    python manage.py run_mae_pipeline --input-file document_processor/MAE/UHG.json
    
    # With deal name and text directly
    python manage.py run_mae_pipeline --deal-name "United Homes Group" --mae-text "Material Adverse Effect means..."
"""

import json
import os
import sys
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = 'Run the MAE (Material Adverse Effect) analysis pipeline'

    def add_arguments(self, parser):
        parser.add_argument(
            '--deal-id',
            type=str,
            help='Deal ID to process (fetches MAE text from Pinecone)'
        )
        parser.add_argument(
            '--input-file',
            type=str,
            help='Path to input JSON file containing MAE clauses'
        )
        parser.add_argument(
            '--deal-name',
            type=str,
            help='Name of the deal (required if --mae-text is used)'
        )
        parser.add_argument(
            '--mae-text',
            type=str,
            help='MAE clause text (required if --deal-name is used)'
        )
        parser.add_argument(
            '--mae-file',
            type=str,
            help='Path to text file containing MAE clause'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default=None,
            help='Output directory for results (default: MAE folder)'
        )
        parser.add_argument(
            '--analyze-all',
            action='store_true',
            help='Analyze all clauses (default: only outliers/atypical)'
        )

    def handle(self, *args, **options):
        # Check required packages
        self._check_requirements()
        
        # Get MAE script directory
        mae_dir = Path(settings.BASE_DIR) / 'document_processor' / 'MAE'
        if not mae_dir.exists():
            raise CommandError(f"MAE directory not found: {mae_dir}")
        
        # Add MAE directory to Python path
        sys.path.insert(0, str(mae_dir))
        
        try:
            # Check if deal-id is provided
            deal_id = options.get('deal_id')
            
            if deal_id:
                # Use Pinecone-based pipeline for deal_id
                self.stdout.write(self.style.SUCCESS("="*80))
                self.stdout.write(self.style.SUCCESS("MAE ANALYSIS PIPELINE - Pinecone Mode"))
                self.stdout.write(self.style.SUCCESS("="*80))
                self.stdout.write(f"\n📊 Processing deal_id: {deal_id}")
                
                # Import pipeline module
                from run_full_pipeline import run_pipeline_for_deal_id
                
                # Run pipeline with deal_id
                results = run_pipeline_for_deal_id(deal_id)
                
                if results:
                    self.stdout.write("\n" + "="*80)
                    self.stdout.write(self.style.SUCCESS("✅ Pipeline complete!"))
                    self.stdout.write("="*80)
                    
                    # Print summary
                    classification = results.get('classification', {})
                    risk_assessment = results.get('risk_assessment', {})
                    compliance = results.get('compliance', {})
                    self._print_summary(classification, risk_assessment, compliance)
                else:
                    self.stdout.write("\n" + "="*80)
                    self.stdout.write(self.style.ERROR("❌ Pipeline failed: No MAE text found"))
                    self.stdout.write("="*80)
                    raise CommandError("No MAE text found in Pinecone for this deal_id")
                
                return
            
            # Standard file-based pipeline
            # Import pipeline module
            from run_full_pipeline import (
                _run_step1, _run_step2, _run_step3, _run_step4, 
                _sanitize_for_json
            )
            
            self.stdout.write(self.style.SUCCESS("="*80))
            self.stdout.write(self.style.SUCCESS("MAE ANALYSIS PIPELINE"))
            self.stdout.write(self.style.SUCCESS("="*80))
            
            # Prepare deal input
            deal_input = self._prepare_deal_input(options, mae_dir)
            
            # Verify API keys
            self._verify_api_keys()
            
            # Run pipeline steps
            self.stdout.write("\n📌 Step 1: Extract MAE clauses")
            self.stdout.write("-" * 40)
            deal_name, clauses = _run_step1(deal_input)
            self.stdout.write(self.style.SUCCESS(
                f"   ✓ Extracted {len(clauses)} clauses from: {deal_name}"
            ))
            
            self.stdout.write("\n📌 Step 2: Classify clauses")
            self.stdout.write("-" * 40)
            classification = _run_step2(clauses, deal_name)
            self.stdout.write(self.style.SUCCESS(
                f"   ✓ Classified {len(classification.get('results', []))} clauses"
            ))
            
            self.stdout.write("\n📌 Step 3: Risk assessment")
            self.stdout.write("-" * 40)
            risk_assessment = _run_step3(classification)
            self.stdout.write(self.style.SUCCESS("   ✓ Risk assessment complete"))
            
            self.stdout.write("\n📌 Step 4: Compliance checks")
            self.stdout.write("-" * 40)
            compliance = _run_step4(classification)
            self.stdout.write(self.style.SUCCESS("   ✓ Compliance checks complete"))
            
            # Save results
            output_dir = options.get('output_dir') or mae_dir
            output_path = self._save_results(
                output_dir, deal_name, classification, 
                risk_assessment, compliance, _sanitize_for_json
            )
            
            self.stdout.write("\n" + "="*80)
            self.stdout.write(self.style.SUCCESS("✅ Pipeline complete!"))
            self.stdout.write(self.style.SUCCESS(f"📄 Results saved to: {output_path}"))
            self.stdout.write("="*80)
            
            # Print summary
            self._print_summary(classification, risk_assessment, compliance)
            
        except ImportError as e:
            raise CommandError(f"Error importing MAE modules: {e}")
        except Exception as e:
            raise CommandError(f"Pipeline failed: {e}")
        finally:
            # Clean up sys.path
            if str(mae_dir) in sys.path:
                sys.path.remove(str(mae_dir))

    def _check_requirements(self):
        """Check if required packages are installed"""
        missing = []
        
        try:
            import anthropic
        except ImportError:
            missing.append('anthropic')
        
        try:
            import cohere
        except ImportError:
            missing.append('cohere')
        
        try:
            import pandas
        except ImportError:
            missing.append('pandas')
        
        try:
            import numpy
        except ImportError:
            missing.append('numpy')
        
        if missing:
            raise CommandError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

    def _verify_api_keys(self):
        """Verify that required API keys are set"""
        missing_keys = []
        
        if not os.getenv('ANTHROPIC_API_KEY'):
            missing_keys.append('ANTHROPIC_API_KEY')
        
        if not os.getenv('COHERE_API_KEY'):
            missing_keys.append('COHERE_API_KEY')
        
        if missing_keys:
            raise CommandError(
                f"Missing required API keys in environment: {', '.join(missing_keys)}\n"
                f"Add them to your .env file"
            )

    def _prepare_deal_input(self, options, mae_dir):
        """Prepare deal input from command line options"""
        input_file = options.get('input_file')
        deal_name = options.get('deal_name')
        mae_text = options.get('mae_text')
        mae_file = options.get('mae_file')
        
        # Option 1: Input file
        if input_file:
            input_path = Path(input_file)
            if not input_path.is_absolute():
                input_path = Path(settings.BASE_DIR) / input_path
            
            if not input_path.exists():
                raise CommandError(f"Input file not found: {input_path}")
            
            self.stdout.write(f"📂 Loading from: {input_path}")
            with open(input_path, 'r') as f:
                return json.load(f)
        
        # Option 2: Deal name + MAE text
        elif deal_name and mae_text:
            self.stdout.write(f"📂 Using deal: {deal_name}")
            return {
                "mae_clauses": [{
                    "dealName": deal_name,
                    "text": mae_text
                }]
            }
        
        # Option 3: Deal name + MAE file
        elif deal_name and mae_file:
            mae_path = Path(mae_file)
            if not mae_path.is_absolute():
                mae_path = Path(settings.BASE_DIR) / mae_path
            
            if not mae_path.exists():
                raise CommandError(f"MAE file not found: {mae_path}")
            
            with open(mae_path, 'r') as f:
                mae_text = f.read()
            
            self.stdout.write(f"📂 Using deal: {deal_name}")
            return {
                "mae_clauses": [{
                    "dealName": deal_name,
                    "text": mae_text
                }]
            }
        
        # Option 4: Check for default UHG.json
        else:
            default_input = mae_dir / "UHG.json"
            if default_input.exists():
                self.stdout.write(f"📂 Using default: {default_input}")
                with open(default_input, 'r') as f:
                    return json.load(f)
            
            raise CommandError(
                "No input provided. Use one of:\n"
                "  --input-file path/to/input.json\n"
                "  --deal-name 'Name' --mae-text 'text...'\n"
                "  --deal-name 'Name' --mae-file path/to/mae.txt\n"
                "Or create document_processor/MAE/UHG.json"
            )

    def _save_results(self, output_dir, deal_name, classification, 
                     risk_assessment, compliance, sanitize_func):
        """Save pipeline results to JSON file and MongoDB"""
        from datetime import datetime
        import re
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        deal_safe = re.sub(r'[^\w\-]', '_', deal_name)[:80]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "deal_name": deal_name,
            "pipeline_timestamp": datetime.now().isoformat(),
            "classification": classification,
            "risk_assessment": risk_assessment,
            "compliance": compliance,
        }
        
        output_path = output_dir / f"final_analysis_{deal_safe}_{timestamp}.json"
        results_clean = sanitize_func(results)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        # Save to MongoDB
        try:
            from document_processor.models import MAEAnalysis
            
            # Use sanitized deal_name as unique identifier
            unique_deal_id = deal_safe
            
            # Prepare data for MongoDB
            mongo_data = {
                'deal_name': deal_name,
                'pipeline_timestamp': results['pipeline_timestamp'],
                'classification': results.get('classification'),
                'risk_assessment': results.get('risk_assessment'),
                'compliance': results.get('compliance')
            }
            
            # Save or update in MongoDB
            mae_record = MAEAnalysis.save_or_update(unique_deal_id, mongo_data)
            self.stdout.write(self.style.SUCCESS(
                f"💾 Saved to MongoDB: Collection 'mae_analyses', Deal ID: {unique_deal_id}"
            ))
            self.stdout.write(self.style.SUCCESS(
                f"   MongoDB Document ID: {mae_record.id}"
            ))
        except Exception as e:
            self.stdout.write(self.style.WARNING(
                f"⚠️  Warning: Failed to save to MongoDB: {e}"
            ))
            self.stdout.write("   Pipeline results are still available in JSON file")
        
        return output_path

    def _print_summary(self, classification, risk_assessment, compliance):
        """Print analysis summary"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("📊 ANALYSIS SUMMARY"))
        self.stdout.write("="*80)
        
        # Classification summary
        results = classification.get('results', [])
        if results:
            typical = sum(1 for c in results if c.get('zone') == 'typical')
            atypical = sum(1 for c in results if c.get('zone') == 'atypical')
            outlier = sum(1 for c in results if c.get('zone') == 'outlier')
            
            self.stdout.write(f"\n🔍 Classification:")
            self.stdout.write(f"   🟢 Typical: {typical} ({typical/len(results)*100:.1f}%)")
            self.stdout.write(f"   🟡 Atypical: {atypical} ({atypical/len(results)*100:.1f}%)")
            self.stdout.write(f"   🔴 Outliers: {outlier} ({outlier/len(results)*100:.1f}%)")
        
        # Risk summary
        if risk_assessment:
            risk_summary = risk_assessment.get('risk_summary', {})
            self.stdout.write(f"\n⚠️  Risk Assessment:")
            self.stdout.write(f"   Overall Risk: {risk_summary.get('final_risk_level', 'unknown').upper()}")
            self.stdout.write(f"   Risk Score: {risk_summary.get('risk_score', 0):.1f}/100")
            self.stdout.write(f"   🔴 High risk: {risk_summary.get('high_risk_count', 0)}")
            self.stdout.write(f"   🟡 Medium risk: {risk_summary.get('medium_risk_count', 0)}")
            self.stdout.write(f"   🟢 Low risk: {risk_summary.get('low_risk_count', 0)}")
        
        # Compliance summary
        if compliance:
            comp_summary = compliance.get('compliance_summary', {})
            flags = []
            if comp_summary.get('cybersecurity', {}).get('count', 0) > 0:
                flags.append(f"🔐 Cybersecurity: {comp_summary['cybersecurity']['count']}")
            if comp_summary.get('disclosure_schedules', {}).get('count', 0) > 0:
                flags.append(f"📄 Disclosure Schedules: {comp_summary['disclosure_schedules']['count']}")
            if comp_summary.get('tariffs_trade', {}).get('count', 0) > 0:
                flags.append(f"📦 Tariffs/Trade: {comp_summary['tariffs_trade']['count']}")
            
            if flags:
                self.stdout.write(f"\n📋 Compliance Flags:")
                for flag in flags:
                    self.stdout.write(f"   {flag}")
            else:
                self.stdout.write(f"\n📋 Compliance: ✅ No special flags")
        
        self.stdout.write("="*80)
