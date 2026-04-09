#!/usr/bin/env python3
"""
GPT Product Analyzer
Service for extracting company products and analyzing competitive relationships using GPT
"""

import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai
from document_processor.models import ProcessingJob, CompanyProducts, CompetitiveAnalysis


class GPTProductAnalyzer:
    """Service for analyzing company products and competitive relationships using GPT"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        """
        Initialize GPT Product Analyzer

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: GPT model to use (default: gpt-4.1-mini)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client (matching services.py pattern)
        self.openai_client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    def extract_company_products(self, company_name: str, company_type: str, deal_context: str = "") -> List[str]:
        """
        Extract products/services offered by a company using GPT

        Args:
            company_name: Name of the company
            company_type: 'target' or 'acquire'
            deal_context: Additional context about the deal (optional)

        Returns:
            List of products/services
        """
        try:
            # Use raw string to avoid f-string formatting issues with JSON template
            prompt = f"""
You are a business analyst expert. I need you to identify the main products and services offered by {company_name}.

Company: {company_name}
{f"Deal Context: {deal_context}" if deal_context else ""}

Please provide a comprehensive list of the main products, services, and business segments offered by this company.

Focus on:
1. Core products and services
2. Major business divisions or segments
3. Key offerings that generate significant revenue
4. Any specialized or niche products 
5. Organize the output into exactly these categories: Software products, Hardware products, Engineering services, Services, Other products.
6. Do not stop at portfolio or suite names. Always expand them into their major **individual sub-products and applications**. Include not only current offerings but also **specialized or legacy tools that remain relevant**. Each sub-product should appear as a separate entry with its own name and description. Only include the suite name itself if it provides unique functionality beyond its sub-products.


Format your response strictly as JSON with the following schema:
{{
                "products": [
    {{
                    "product_category": "Software products | Hardware products | Engineering services | Services | Other products",
      "products": [
        {{
                        "name": "string",
          "description": "string" # Give the description of the product in 100 words.
        }}
      ]
    }}
  ]
}}

Only return JSON, no additional text.
"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business analyst expert specializing in company analysis and product identification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            content = response.choices[0].message.content.strip()

            self.logger.info(f"GPT prompt: {prompt}")

            self.logger.info(f"GPT response: {content}")

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

                self.logger.info(f"Parsed response: {parsed_response}")

                # Extract products from the new structured format
                products = []
                if isinstance(parsed_response, dict) and 'products' in parsed_response:
                    # Return the complete structured data
                    products = parsed_response['products']
                    self.logger.info(
                        f"Extracted {len(products)} product categories for {company_name}")
                    return products
                elif isinstance(parsed_response, list):
                    # Fallback: handle old format if GPT still returns a simple list
                    self.logger.info(
                        f"Extracted {len(parsed_response)} products for {company_name} (legacy format)")
                    return parsed_response
                else:
                    self.logger.error(
                        f"GPT response does not match expected schema for {company_name}")
                    return []
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse GPT response as JSON for {company_name}: {e}")
                self.logger.debug(f"GPT response content: {content}")
                return []

        except Exception as e:
            self.logger.error(
                f"Error extracting products for {company_name}: {e}")
            return []

    def analyze_competitive_products(self, target_products: List[str], target_company: str,
                                     acquire_products: List[str], acquire_company: str) -> List[Dict[str, Any]]:
        """
        Analyze competitive relationships between products of two companies

        Args:
            target_products: List of target company products
            target_company: Name of target company
            acquire_products: List of acquire company products
            acquire_company: Name of acquire company

        Returns:
            List of competitive product pairs with analysis
        """
        try:
            prompt = f"""
            You are a competitive analysis expert. I need you to identify competitive product relationships between two companies involved in an M&A deal.
            
            Target Company: {target_company}
            Target Products: {json.dumps(target_products, indent=2)}
            
            Acquire Company: {acquire_company}
            Acquire Products: {json.dumps(acquire_products, indent=2)}
            
            Note: The products are structured with categories and detailed descriptions. Focus on the product names and descriptions when identifying competitive relationships.
            
            Please analyze which products from each company compete with each other. For each competitive pair, provide:
            1. The target company product
            2. The competing acquire company product
            3. A competition score from 0.0 to 1.0 (1.0 = direct competitors, 0.0 = no competition)
            4. A brief analysis explaining the competitive relationship
            
            Focus on:
            - Direct product competitors (same market, similar functionality)
            - Indirect competitors (different approach to same customer need)
            - Market overlap and customer base similarity
            - Feature/capability overlap
            
            Only include pairs with competition score >= 0.5 (moderate to high competition).
            
            Format your response as a JSON array of objects:
            [
              {{
                "target_product": "Product A",
                "acquire_product": "Product 1",
                "competition_score": 0.85,
                "analysis": "Both products serve the same market segment with similar core functionality...(very short analysis)"
              }}
            ]
            
            Only return the JSON array, no additional text.
            """

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a competitive analysis expert specializing in product competition and market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            content = response.choices[0].message.content.strip()

            self.logger.info(f"GPT competitive analysis response: {content}")

            # Parse JSON response
            try:
                # Check if the response is wrapped in markdown code block
                markdown_match = re.search(
                    r"```(?:json)?\s*([\s\S]+?)\s*```", content)
                if markdown_match:
                    # Extract the JSON content from the markdown code block
                    json_content = markdown_match.group(1).strip()
                    competitive_pairs = json.loads(json_content)
                else:
                    # Try parsing directly if not in markdown format
                    competitive_pairs = json.loads(content)

                if isinstance(competitive_pairs, list):
                    self.logger.info(
                        f"Identified {len(competitive_pairs)} competitive pairs between {target_company} and {acquire_company}")
                    return competitive_pairs
                else:
                    self.logger.error(
                        f"GPT response is not a list for competitive analysis")
                    return []
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse GPT competitive analysis response as JSON: {e}")
                self.logger.debug(f"GPT response content: {content}")
                return []

        except Exception as e:
            self.logger.error(f"Error analyzing competitive products: {e}")
            return []

    def process_deal_product_analysis(self, deal_id: str) -> Optional[CompetitiveAnalysis]:
        """
        Complete product analysis workflow for a deal

        Args:
            deal_id: Deal ID to analyze

        Returns:
            CompetitiveAnalysis object if successful, None otherwise
        """
        try:
            # Get deal data
            deal = ProcessingJob.objects.get(id=deal_id)

            target_company = deal.target_name
            acquire_company = deal.acquire_name

            if not target_company or not acquire_company:
                self.logger.error(f"Missing company names for deal {deal_id}")
                return None

            self.logger.info(
                f"Starting product analysis for deal {deal_id}: {acquire_company} acquiring {target_company}")

            # Extract products for target company
            target_products = self.extract_company_products(
                target_company, "target")
            if not target_products:
                self.logger.warning(
                    f"No products extracted for target company {target_company}")

            # Save target company products
            target_company_products = CompanyProducts(
                deal_id=deal_id,
                company=target_company,
                company_type="target",
                products=target_products,
                gpt_model_used=self.model,
                processing_status="completed" if target_products else "failed"
            )
            target_company_products.save()

            # Extract products for acquire company
            acquire_products = self.extract_company_products(
                acquire_company, "acquire")
            if not acquire_products:
                self.logger.warning(
                    f"No products extracted for acquire company {acquire_company}")

            # Save acquire company products
            acquire_company_products = CompanyProducts(
                deal_id=deal_id,
                company=acquire_company,
                company_type="acquire",
                products=acquire_products,
                gpt_model_used=self.model,
                processing_status="completed" if acquire_products else "failed"
            )
            acquire_company_products.save()

            # Analyze competitive relationships if both companies have products
            competitive_pairs = []
            if target_products and acquire_products:
                competitive_pairs = self.analyze_competitive_products(
                    target_products, target_company,
                    acquire_products, acquire_company
                )

            # Save competitive analysis
            competitive_analysis = CompetitiveAnalysis(
                deal_id=deal_id,
                target_company_products=target_company_products,
                acquire_company_products=acquire_company_products,
                competitive_pairs=competitive_pairs,
                gpt_model_used=self.model,
                processing_status="completed" if competitive_pairs else "failed"
            )
            competitive_analysis.save()

            self.logger.info(
                f"Product analysis completed for deal {deal_id}: {len(competitive_pairs)} competitive pairs identified")

            return competitive_analysis

        except ProcessingJob.DoesNotExist:
            self.logger.error(f"Deal with ID {deal_id} not found")
            return None
        except Exception as e:
            self.logger.error(
                f"Error processing deal product analysis for {deal_id}: {e}")
            return None
