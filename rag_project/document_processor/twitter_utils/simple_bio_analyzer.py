#!/usr/bin/env python3
"""
Simple Bio Analyzer
Analyzes a single bio against a company name using GPT to determine high-value potential.

Usage:
# python simple_bio_analyzer.py "Company Name" "Bio text here"
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load environment variables
load_dotenv()


class SimpleBioAnalyzer:
    """Simple class for analyzing a single bio against a company name"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            openai_api_key: OpenAI API key
        """
        # Setup configuration
        self.config = {
            'gpt_model': 'gpt-4o-mini',
            'gpt_max_tokens': 1000,
            'gpt_temperature': 0.1,
        }

        # Setup OpenAI client
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.openai_api_key)

        # Setup logger
        self.logger = logging.getLogger(__name__)

    def create_analysis_prompt(self, bio: str, company_name: str) -> str:
        """Create a detailed prompt for GPT analysis"""
        prompt = f"""
Score this person's likelihood to tweet valuable competitive intelligence about {company_name} or its industry.

BIO: {bio}

CORE QUESTION: Will this person likely tweet insider knowledge, technical realities, or competitive insights that corporate PR wouldn't share?

HIGH VALUE (8-10) - The "truth-tellers":
- Technical practitioners who build/implement/fix products
- People with direct product knowledge who share opinions
- Cross-company expertise (knows multiple competitors)
- Signs they actively share views ("opinions mine", personal blog mentioned)
Example: "Senior SRE at Datadog. Ex-Splunk. Thoughts on observability. Views mine." = 9

MEDIUM VALUE (5-7) - Potential insights:
- Clear industry role but less likely to share freely
- Customers/partners with implementation experience
- Executives (have knowledge but filter it)
- Analysts/journalists covering the space
Example: "CTO at FinTech startup" = 5

LOW VALUE (3-4) - Weak connection:
- Generic professional description
- Industry-adjacent but not directly relevant
- Marketing/PR (share corporate messages)
Example: "Software developer. Love tech." = 3

NO VALUE (0-2) - Ignore:
- No professional context
- Unrelated interests only
- Bot/spam/organizational accounts
Example: "XX" = 0

The sweet spot: Someone with KNOWLEDGE (technical role, insider position) + WILLINGNESS TO SHARE (personal account, disclaimer, active voice).

JSON Response:
{{
 "overall_score": <0-10>,
 "has_insider_knowledge": <true/false>,
 "likely_to_share": <true/false>,
 "reason": "<why this score in 20 words>",
 "key_indicators": ["<relevant phrases>"]
}}

Remember: Executives score lower than engineers. Corporate accounts score near zero. Vague bios are worthless.
"""
        return prompt

    def analyze_bio(self, company_name: str, bio: str) -> Optional[Dict[str, Any]]:
        """Analyze a single bio using GPT"""
        try:
            self.logger.info(f"Analyzing bio for company: {company_name}")
            self.logger.info(f"Bio: {bio}")
            self.logger.info("-" * 50)

            prompt = self.create_analysis_prompt(bio, company_name)

            response = self.client.chat.completions.create(
                model=self.config['gpt_model'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst specializing in business intelligence, antitrust law, and corporate affairs. Provide accurate, objective assessments based on publicly available profile information."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['gpt_max_tokens'],
                temperature=self.config['gpt_temperature'],
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)

            # Add metadata to analysis
            analysis['company_name'] = company_name
            analysis['bio'] = bio
            analysis['analyzed_at'] = datetime.now().isoformat()
            analysis['token_usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            # Log the results
            self.logger.info("=== ANALYSIS RESULTS ===")
            self.logger.info(
                f"Overall Score: {analysis.get('overall_score', 'N/A')}/10")
            self.logger.info(
                f"Has Insider Knowledge: {analysis.get('has_insider_knowledge', 'N/A')}")
            self.logger.info(
                f"Likely to Share: {analysis.get('likely_to_share', 'N/A')}")
            self.logger.info(f"Reason: {analysis.get('reason', 'N/A')}")
            self.logger.info(
                f"Key Indicators: {analysis.get('key_indicators', [])}")
            self.logger.info(
                f"Tokens Used: {analysis.get('token_usage', {}).get('total_tokens', 'N/A')}")
            self.logger.info("=" * 50)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing bio: {str(e)}")
            return None


def main():
    """Main function to run the script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    if len(sys.argv) < 3:
        logger.error(
            "Usage: python simple_bio_analyzer.py \"Company Name\" \"Bio text here\"")
        logger.error("Examples:")
        logger.error(
            '  python simple_bio_analyzer.py "Datadog" "Senior SRE at Datadog. Ex-Splunk. Thoughts on observability. Views mine."')
        logger.error(
            '  python simple_bio_analyzer.py "Microsoft" "Software engineer. Love coding and coffee."')
        sys.exit(1)

    company_name = sys.argv[1]
    bio = sys.argv[2]

    try:
        # Initialize analyzer
        analyzer = SimpleBioAnalyzer()

        # Run analysis
        result = analyzer.analyze_bio(company_name, bio)

        if result:
            logger.info("Analysis completed successfully!")
        else:
            logger.error("Analysis failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
