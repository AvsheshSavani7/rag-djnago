#!/usr/bin/env python3
"""
MAE Clause Extractor - Frozen Version for Consistency
Uses Claude Sonnet 4.5 with simple, consistent extraction methodology

This extractor is FROZEN - do not modify the prompt or model.
Use this exact script for both training data AND new deals.
Django-compatible version
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic


try:
    from dotenv import load_dotenv
except ImportError:
    # Django projects might not use dotenv directly
    def load_dotenv():
        pass

# ================================
# CONFIGURATION
# ================================
INPUT_FILE = str(Path(__file__).resolve().parent / "UHG.json")
OUTPUT_FILE = ""  # Leave empty to auto-generate
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls

# ================================
# FROZEN EXTRACTION PROMPT - DO NOT MODIFY
# ================================
EXTRACTION_PROMPT = """You are extracting Material Adverse Effect (MAE) exclusions from a merger agreement.

Your task: Extract ONLY the main numbered exclusions from the MAE clause.

CRITICAL RULES:
1. Extract ONLY the primary exclusion items: (i), (ii), (iii), (iv), etc. or (a), (b), (c), etc.
2. Copy the text AS WRITTEN from the agreement - do NOT rewrite or interpret
3. IGNORE nested sub-clauses: (A), (B), (C) or (x), (y), (z) - these are part of the main clause
4. STOP before the disproportionality qualifier (the "except" or "provided" at the end)
5. Remove leading "any" if it makes the text cleaner, but otherwise keep original wording
6. Each exclusion should be standalone text

Example transformation:
Input: "(i) any changes in general economic conditions... (A) including interest rates (B) exchange rates..."
Output: "changes in general economic conditions, including interest rates and exchange rates"

Return ONLY a JSON array with this exact structure:
[
  {{
    "label": "(i)",
    "text": "extracted clause text as written",
    "deal": "{deal_name}"
  }},
  {{
    "label": "(ii)", 
    "text": "extracted clause text as written",
    "deal": "{deal_name}"
  }}
]

Deal: {deal_name}

MAE Clause:
{mae_text}

Return ONLY the JSON array, no other text."""

# ================================
# (No need to edit below this line)
# ================================

load_dotenv()


def extract_exclusions(api_key: str, deal_name: str, mae_text: str) -> list:
    """Extract MAE exclusions using frozen Sonnet 4.5 prompt"""

    client = Anthropic(api_key=api_key)

    try:
        # Use replace to avoid curly brace issues
        prompt = EXTRACTION_PROMPT.replace(
            "{deal_name}", deal_name).replace("{mae_text}", mae_text)

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # FROZEN - DO NOT CHANGE
            max_tokens=4096,
            temperature=0,  # Deterministic
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract response text
        if not response.content or len(response.content) == 0:
            print(f"  ⚠️ Empty response")
            return []

        response_text = response.content[0].text.strip()

        # Extract JSON
        import re
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            response_text = json_match.group()
        else:
            print(f"  ⚠️ No JSON array found")
            return []

        # Clean up JSON
        # Remove trailing commas
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)

        # Parse JSON
        try:
            exclusions = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON parse error: {e}")
            # Save debug output
            debug_file = f"debug_{deal_name.replace(' ', '_')}.txt"
            with open(debug_file, 'w') as f:
                f.write(response_text)
            print(f"  Debug saved to: {debug_file}")
            return []

        # Validate structure
        if not isinstance(exclusions, list):
            print(f"  ⚠️ Response is not a list")
            return []

        # Clean and validate each exclusion
        valid_exclusions = []
        for exc in exclusions:
            if isinstance(exc, dict) and 'text' in exc and 'label' in exc:
                exc['deal'] = deal_name  # Ensure deal name is correct
                exc['text'] = exc['text'].strip('"')
                valid_exclusions.append(exc)

        return valid_exclusions

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []


def process_deals_data(data, api_key: str):
    """
    Process deal(s) from in-memory data. No file I/O.
    data: dict with key 'mae_clauses' (list of {dealName, text}) or list of such dicts.
    Returns: (deal_name, all_clauses).
    """
    deals = data.get('mae_clauses', []) if isinstance(data, dict) else data
    if not deals:
        return 'Unknown', []

    print("="*80)
    print("MAE CLAUSE EXTRACTOR - FROZEN VERSION (in-memory)")
    print("="*80)
    print(f"Model: claude-sonnet-4-5-20250929")
    print(f"📊 Found {len(deals)} deals to process\n")

    all_clauses = []
    failed_deals = []

    for i, deal in enumerate(deals, 1):
        deal_name = deal.get('dealName', 'Unknown')
        mae_text = deal.get('text', '')

        print(f"[{i:3d}/{len(deals)}] Processing: {deal_name[:50]}...")

        if not (mae_text or '').strip():
            print(f"  ⚠️ Empty MAE text, skipping")
            failed_deals.append(deal_name)
            continue

        exclusions = extract_exclusions(api_key, deal_name, mae_text)

        if exclusions:
            print(f"  ✅ Extracted {len(exclusions)} clauses")
            all_clauses.extend(exclusions)
        else:
            print(f"  ❌ No clauses extracted")
            failed_deals.append(deal_name)

        if i < len(deals):
            time.sleep(RATE_LIMIT_DELAY)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"✅ Successfully extracted: {len(all_clauses)} total clauses")
    print(f"✅ From {len(deals) - len(failed_deals)}/{len(deals)} deals")
    if failed_deals:
        print(f"\n⚠️ Failed deals ({len(failed_deals)}):")
        for name in failed_deals[:10]:
            print(f"  - {name}")
        if len(failed_deals) > 10:
            print(f"  ... and {len(failed_deals) - 10} more")

    out_deal_name = all_clauses[0].get('deal', deals[0].get('dealName', 'Unknown')) if all_clauses else (
        deals[0].get('dealName', 'Unknown') if deals else 'Unknown')
    return out_deal_name, all_clauses


def process_all_deals(input_file: str, output_file: str, api_key: str):
    """Process all deals from input JSON (reads file, writes output)."""

    print("="*80)
    print("MAE CLAUSE EXTRACTOR - FROZEN VERSION")
    print("="*80)
    print(f"Model: claude-sonnet-4-5-20250929")
    print(f"Input: {input_file}")
    print()

    with open(input_file, 'r') as f:
        data = json.load(f)

    deal_name, all_clauses = process_deals_data(data, api_key)

    with open(output_file, 'w') as f:
        json.dump(all_clauses, f, indent=2)
    print(f"\n💾 Saved to: {output_file}")

    summary = {
        "extraction_date": datetime.now().isoformat(),
        "model": "claude-sonnet-4-5-20250929",
        "total_deals": len(data.get('mae_clauses', [])),
        "successful_deals": len(data.get('mae_clauses', [])) - (len(data.get('mae_clauses', [])) - len(all_clauses)),
        "failed_deals": [],
        "total_clauses": len(all_clauses),
        "input_file": input_file,
        "output_file": output_file
    }
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Summary: {summary_file}")
    print(
        f"\n💰 Estimated cost: ~${len(data.get('mae_clauses', [])) * 0.001:.2f}")
    print(f"\n✅ Ready for Stage 1!")


def main():
    """Main execution"""

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not set")
        print("   Set in environment or .env file")
        return

    # Check input file
    if not Path(INPUT_FILE).exists():
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        print("   Edit INPUT_FILE on line 18")
        return

    # Determine output file
    output_file = OUTPUT_FILE
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"extracted_clauses_{timestamp}.json"

    # Process
    process_all_deals(INPUT_FILE, output_file, api_key)


if __name__ == "__main__":
    main()
