#!/usr/bin/env python3
"""
Twitter Handle Finder
Service for finding company Twitter handles using GPT and storing them in the deal table
"""

import os
import logging
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from document_processor.models import ProcessingJob


class TwitterHandleFinder:
    """Service for finding company Twitter handles using GPT"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        """
        Initialize Twitter Handle Finder

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: GPT model to use (default: gpt-4.1-mini)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY_SEC_FILING')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY_SEC_FILING environment variable.")

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    def find_company_twitter_handles(self, company_name: str, company_type: str = "main") -> Dict[str, Any]:
        """
        Find Twitter handles for a company and its subsidiaries using GPT

        Args:
            company_name: Name of the company
            company_type: Type of company ('main', 'subsidiary', etc.)

        Returns:
            Dictionary containing Twitter handle information
        """
        try:
            prompt = f"""
You are a social media research expert. I need you to find the official Twitter handles for {company_name}.

Company: {company_name}
Company Type: {company_type}

Please provide:
1. The main company's official Twitter handle

Focus on:
- Official corporate accounts (usually verified with blue checkmark)

Format your response strictly as JSON with the following schema:
{{
  "company_name": "{company_name}",
  "company_type": "{company_type}",
  "main_twitter_handle": "@company_handle",
  "search_notes": "Any notes about the search or verification status"
}}

Important:
- Only include verified or clearly official accounts
- Use @ symbol for all handles
- If no Twitter handle is found, use null for the handle field
- Be specific about account types and descriptions
- Only return JSON, no additional text
"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a social media research expert specializing in finding official company Twitter handles and verified corporate accounts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            self.logger.info(f"GPT prompt for {company_name}: {prompt}")
            self.logger.info(f"GPT response for {company_name}: {content}")

            # Parse JSON response
            try:
                # Check if the response is wrapped in markdown code block
                markdown_match = re.search(
                    r"```(?:json)?\s*([\s\S]+?)\s*```", content)
                if markdown_match:
                    # Extract the JSON content from the markdown code block
                    json_content = markdown_match.group(1).strip()
                    parsed_response = json.loads(json_content)
                else:
                    # Try parsing directly if not in markdown format
                    parsed_response = json.loads(content)

                self.logger.info(
                    f"Parsed Twitter handles for {company_name}: {parsed_response}")
                return parsed_response

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse GPT response as JSON for {company_name}: {e}")
                self.logger.debug(f"GPT response content: {content}")
                return {
                    "company_name": company_name,
                    "company_type": company_type,
                    "main_twitter_handle": None,
                    "subsidiaries": [],
                    "additional_handles": [],
                    "search_notes": f"Error parsing response: {str(e)}"
                }

        except Exception as e:
            self.logger.error(
                f"Error finding Twitter handles for {company_name}: {e}")
            return {
                "company_name": company_name,
                "company_type": company_type,
                "main_twitter_handle": None,
                "subsidiaries": [],
                "additional_handles": [],
                "search_notes": f"Error during search: {str(e)}"
            }

    def process_deal_twitter_handles(self, deal_id: str, unique_companies: List[str] = None) -> Dict[str, Any]:
        """
        Find Twitter handles for specific companies in a deal

        Args:
            deal_id: Deal ID to process
            unique_companies: List of unique company names to find handles for

        Returns:
            Dictionary containing Twitter handle information for all companies
        """
        try:
            # Get deal data
            deal = ProcessingJob.objects.get(id=deal_id)

            # If no unique_companies provided, use default target and acquire
            if not unique_companies:
                target_company = deal.target_name
                acquire_company = deal.acquire_name
                unique_companies = [
                    target_company, acquire_company] if target_company and acquire_company else []

            if not unique_companies:
                self.logger.error(
                    f"No companies to process for deal {deal_id}")
                return {}

            self.logger.info(
                f"Starting Twitter handle search for deal {deal_id} with {len(unique_companies)} companies: {unique_companies}")

            # Find Twitter handles for each unique company
            company_twitter_info = {}
            for company_name in unique_companies:
                if company_name and company_name.strip():
                    self.logger.info(
                        f"Finding Twitter handles for: {company_name}")
                    twitter_info = self.find_company_twitter_handles(
                        company_name, "company")
                    company_twitter_info[company_name] = twitter_info

            # Combine all Twitter handle information
            all_twitter_details = {
                "deal_id": deal_id,
                "search_timestamp": datetime.utcnow().isoformat(),
                "unique_companies": unique_companies,
                "company_handles": company_twitter_info,
            }

            # Save to deal table
            deal.twitter_details = all_twitter_details
            deal.updatedAt = datetime.utcnow()
            deal.save()

            self.logger.info(
                f"Twitter handle search completed for deal {deal_id}")
            self.logger.info(
                f"Processed {len(unique_companies)} companies")

            return all_twitter_details

        except ProcessingJob.DoesNotExist:
            self.logger.error(f"Deal with ID {deal_id} not found")
            return {}
        except Exception as e:
            self.logger.error(
                f"Error processing Twitter handles for deal {deal_id}: {e}")
            return {}

    def get_twitter_handles_for_deal(self, deal_id: str) -> Dict[str, Any]:
        """
        Get stored Twitter handles for a deal

        Args:
            deal_id: Deal ID

        Returns:
            Dictionary containing Twitter handle information
        """
        try:
            deal = ProcessingJob.objects.get(id=deal_id)
            return deal.twitter_details or {}
        except ProcessingJob.DoesNotExist:
            self.logger.error(f"Deal with ID {deal_id} not found")
            return {}
        except Exception as e:
            self.logger.error(
                f"Error getting Twitter handles for deal {deal_id}: {e}")
            return {}


def main():
    """Main function to run the script"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Find Twitter handles for companies in a deal')
    parser.add_argument('deal_id', help='Deal ID to process')
    parser.add_argument(
        '--company', help='Single company name to search (optional)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        finder = TwitterHandleFinder()

        if args.company:
            # Search for single company
            result = finder.find_company_twitter_handles(args.company)
            print(f"Twitter handles for {args.company}:")
            print(json.dumps(result, indent=2))
        else:
            # Process entire deal
            result = finder.process_deal_twitter_handles(args.deal_id)
            print(f"Twitter handles for deal {args.deal_id}:")
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
