import json
import anthropic
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def parse_analysis(analysis_text):
    """
    Parse the analysis text into structured fields.
    First tries to parse as JSON, falls back to regex parsing for backward compatibility.
    """
    import re

    parsed = {
        "relevance_score": None,
        "risk_score": None,
        "bullet_summary": [],
        "verdict": None,
        "regulatory_flags": [],
        "sentiment": None,
        "full_analysis": analysis_text
    }

    # First, try to parse as JSON
    try:
        # Look for JSON content in the response - try multiple patterns
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            json_data = json.loads(json_str)

            # Validate that we have the expected structure
            if isinstance(json_data, dict):
                # Map JSON fields to our expected structure
                parsed["relevance_score"] = json_data.get("relevance_score")
                # Note: different field name in JSON
                parsed["risk_score"] = json_data.get("regulatory_risk_score")
                parsed["bullet_summary"] = json_data.get("bullet_summary", [])
                parsed["verdict"] = json_data.get("one_line_verdict")

                # Handle regulatory flags with better logic
                raw_flags = json_data.get("key_regulatory_flags", [])
                logger.debug(f"Raw regulatory flags from JSON: {raw_flags}")

                if isinstance(raw_flags, list):
                    # Filter out empty strings and "none" values
                    filtered_flags = [
                        flag.strip() for flag in raw_flags
                        if flag and isinstance(flag, str) and
                        not re.match(r'^none\s*[-–—]?\s*',
                                     flag.strip(), re.IGNORECASE)
                    ]
                    parsed["regulatory_flags"] = filtered_flags

                    # If all flags were filtered out, add a default message
                    if not parsed["regulatory_flags"]:
                        parsed["regulatory_flags"] = [
                            "No merger-specific regulatory concerns identified"]
                        logger.debug(
                            "No valid regulatory flags found, using default message")
                    else:
                        logger.debug(
                            f"Processed regulatory flags: {parsed['regulatory_flags']}")
                else:
                    parsed["regulatory_flags"] = [
                        "No merger-specific regulatory concerns identified"]
                    logger.debug(
                        f"Regulatory flags not a list, using default message. Type: {type(raw_flags)}")

                parsed["sentiment"] = json_data.get("sentiment")

                # Validate that we got meaningful data
                if parsed["relevance_score"] is not None or parsed["risk_score"] is not None:
                    logger.info(
                        f"Successfully parsed JSON response with scores: relevance={parsed['relevance_score']}, risk={parsed['risk_score']}")
                    return parsed
                else:
                    logger.warning(
                        "JSON parsed but no scores found, falling back to regex")
            else:
                logger.warning(
                    "JSON structure is not a dict, falling back to regex")

    except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON, falling back to regex: {e}")
        # Fall through to regex parsing
    except Exception as e:
        logger.error(
            f"Unexpected error during JSON parsing, falling back to regex: {e}")
        # Fall through to regex parsing

    # Fallback: Extract relevance score - try multiple patterns
    # Pattern 1: "RELEVANCE: 75" or "RELEVANCE SCORE: 75"
    relevance_match = re.search(
        r'(?:2\.\s+)?RELEVANCE(?:\s+SCORE)?:\s*(\d+)', analysis_text, re.IGNORECASE)
    if relevance_match:
        parsed["relevance_score"] = int(relevance_match.group(1))
    else:
        # Pattern 2: Look for "## 2. RELEVANCE SCORE" followed by a number on next line
        relevance_match = re.search(
            r'##\s*2\.\s*RELEVANCE\s+SCORE[:\s]*\n?\s*(\d+)', analysis_text, re.IGNORECASE | re.MULTILINE)
        if relevance_match:
            parsed["relevance_score"] = int(relevance_match.group(1))
        else:
            # Pattern 3: Look for "RELEVANCE SCORE" section followed by a number
            relevance_match = re.search(
                r'RELEVANCE\s+SCORE[:\s]*\n?\s*(\d+)', analysis_text, re.IGNORECASE | re.MULTILINE)
            if relevance_match:
                parsed["relevance_score"] = int(relevance_match.group(1))
            else:
                # Pattern 4: Look for section header followed by just a number (like in your log)
                relevance_match = re.search(
                    r'##\s*2\.\s*RELEVANCE\s+SCORE[:\s]*\n?\s*(\d+)(?:\n|$)', analysis_text, re.IGNORECASE | re.MULTILINE)
                if relevance_match:
                    parsed["relevance_score"] = int(relevance_match.group(1))

    # Extract risk score - try multiple patterns
    # Pattern 1: "RISK: 45" or "REGULATORY RISK SCORE: 45"
    risk_match = re.search(
        r'(?:3\.\s+)?(?:REGULATORY\s+)?RISK(?:\s+SCORE)?:\s*(\d+)', analysis_text, re.IGNORECASE)
    if risk_match:
        parsed["risk_score"] = int(risk_match.group(1))
    else:
        # Pattern 2: Look for "## 3. RISK SCORE" or similar followed by a number
        risk_match = re.search(
            r'##\s*3\.\s*(?:REGULATORY\s+)?RISK\s+SCORE[:\s]*\n?\s*(\d+)', analysis_text, re.IGNORECASE | re.MULTILINE)
        if risk_match:
            parsed["risk_score"] = int(risk_match.group(1))
        else:
            # Pattern 3: Look for "RISK SCORE" section followed by a number
            risk_match = re.search(
                r'(?:REGULATORY\s+)?RISK\s+SCORE[:\s]*\n?\s*(\d+)', analysis_text, re.IGNORECASE | re.MULTILINE)
            if risk_match:
                parsed["risk_score"] = int(risk_match.group(1))
            else:
                # Pattern 4: Look for section header followed by just a number
                risk_match = re.search(
                    r'##\s*3\.\s*(?:REGULATORY\s+)?RISK\s+SCORE[:\s]*\n?\s*(\d+)(?:\n|$)', analysis_text, re.IGNORECASE | re.MULTILINE)
                if risk_match:
                    parsed["risk_score"] = int(risk_match.group(1))

    # Extract bullet summary - more flexible pattern
    summary_match = re.search(
        r'(?:1\.\s+)?(?:##\s+)?BULLET SUMMARY[:\s]*(.+?)(?=(?:\n\n|\n)(?:##\s+)?(?:2\.|RELEVANCE))', analysis_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary_text = summary_match.group(1)
        # Look for both bullet styles: "- text" and "• text"
        bullets = re.findall(
            r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\n|\Z)', summary_text, re.DOTALL)
        parsed["bullet_summary"] = [b.strip().replace('\n', ' ')
                                    for b in bullets if b.strip()]

    # Extract verdict - more flexible
    verdict_match = re.search(
        r'(?:4\.\s+)?(?:##\s+)?ONE-LINE VERDICT[:\s]*(.+?)(?=(?:\n\n|\n)(?:##\s+)?(?:5\.|KEY REGULATORY))', analysis_text, re.DOTALL | re.IGNORECASE)
    if verdict_match:
        parsed["verdict"] = verdict_match.group(1).strip().replace('\n', ' ')

    # Extract regulatory flags - more flexible
    flags_match = re.search(
        r'(?:5\.\s+)?(?:##\s+)?KEY REGULATORY FLAGS[:\s]*(?:\(if any\)[:\s]*)?(.+?)(?=(?:\n\n|\n)(?:##\s+)?(?:6\.|SENTIMENT))', analysis_text, re.DOTALL | re.IGNORECASE)
    if flags_match:
        flags_text = flags_match.group(1)
        # Try to find bulleted flags
        flags = re.findall(
            r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\n|\Z)', flags_text, re.DOTALL)
        if flags:
            parsed["regulatory_flags"] = [f.strip().replace('\n', ' ')
                                          for f in flags if f.strip()]
        else:
            # If no bullets, take the whole text if it's not just "None"
            cleaned = flags_text.strip().replace('\n', ' ')
            if cleaned and not re.match(r'^none\s*[-–—]?\s*', cleaned, re.IGNORECASE):
                parsed["regulatory_flags"] = [cleaned]
            else:
                parsed["regulatory_flags"] = [
                    "None - no merger-specific concerns identified"]

    # Extract sentiment - more flexible
    sentiment_match = re.search(
        r'(?:6\.\s+)?(?:##\s+)?SENTIMENT[:\s]*(.+?)(?=\n\n|RELEVANCE:|RISK:|\*\*RELEVANCE|\Z)', analysis_text, re.DOTALL | re.IGNORECASE)
    if sentiment_match:
        parsed["sentiment"] = sentiment_match.group(
            1).strip().replace('\n', ' ').replace('**', '')

    return parsed


def get_competition_context(competitive_pair):
    """
    Extract detailed competition context from the competitive_pair data.
    """
    if not competitive_pair:
        return "No competitive analysis available"

    return f"""
Target Product: {competitive_pair['target_product']}
Acquiring Product: {competitive_pair['acquire_product']}
Competition Score: {competitive_pair['competition_score']} (0-1 scale, where 1.0 = direct competitors)
Competitive Analysis: {competitive_pair['analysis']}

REGULATORY CONTEXT: These products have a competition score of {competitive_pair['competition_score']}, indicating {"very high" if competitive_pair['competition_score'] > 0.85 else "high" if competitive_pair['competition_score'] > 0.7 else "moderate"} overlap. A merger combining these products could raise antitrust concerns if it eliminates alternatives or creates market concentration.
"""


def analyze_post(post_data, competition_context):
    """
    Analyze a single Reddit post for merger arb insights using Claude Sonnet.
    """

    # Build context from post and top comments
    post_content = f"""
Title: {post_data['title']}
Author: u/{post_data['author']}
Score: {post_data['score']}
Content: {post_data['selftext']}

Top Comments:
"""

    # Include top 7 comments by score
    comments = sorted(post_data.get('comments', []),
                      key=lambda x: x['score'], reverse=True)[:7]

    # Also try to find controversial comments (high replies but moderate score)
    # Controversy indicator: replies exist and score is relatively low
    all_comments = post_data.get('comments', [])
    controversial = []
    for comment in all_comments:
        # Check if comment has replies (nested structure) or just count as proxy
        num_replies = len(comment.get('replies', []))
        if num_replies > 3 and comment['score'] < 100 and comment not in comments:
            controversial.append(comment)

    # Add up to 2 controversial comments
    if controversial:
        controversial_sorted = sorted(controversial, key=lambda x: len(
            x.get('replies', [])), reverse=True)[:2]
        comments.extend(controversial_sorted)

    for comment in comments:
        post_content += f"\n- u/{comment['author']} (score: {comment['score']}): {comment['body']}\n"

    # Store the post content for reference (full text, not truncated)
    post_text_summary = {
        "title": post_data['title'],
        "selftext": post_data['selftext'],
        "top_comments": [
            {
                "author": c['author'],
                "score": c['score'],
                "text": c['body']
            } for c in comments[:3]  # Save top 3 comments with full text
        ]
    }

    # Create the analysis prompt
    prompt = f"""You are analyzing Reddit discussions for merger arbitrage research. Think like an MD at an investment bank evaluating deal risk.

CRITICAL: Focus ONLY on issues specific to THIS MERGER between these two companies. Ignore generic industry complaints that apply to all vendors.

MERGER CONTEXT:
{competition_context}

REDDIT POST TO ANALYZE:
{post_content}

IMPORTANT DISTINCTIONS:
- RELEVANT: "Seed Pay fees are higher than competitors" or "365 Retail has a monopoly in our region" or "these two merging would eliminate our only alternative"
- NOT RELEVANT: "All credit card processors charge high fees" or "payment systems are unreliable generally"

Focus on competitive dynamics, market concentration, and whether THIS specific merger would harm competition or consumers.

Please provide your analysis in the following format:

1. BULLET SUMMARY (3-5 bullets): Key points from the post and top comments
   - Use concise bullet points
   - Focus on actionable insights
   - Include both concerns and positive mentions

2. RELEVANCE SCORE (1-100): How directly relevant is this to the specific products/companies in the merger?
   - 1-20: Generic industry discussion, applies to all vendors
   - 21-40: Industry discussion, mentions similar products but not these specific ones
   - 41-60: Mentions these product categories, some relevance to market dynamics
   - 61-80: Directly discusses one of these specific products/companies
   - 81-100: Discusses both products, competitive dynamics, or how this merger would affect the market
   
   Provide ONLY the number (e.g., "75")

3. REGULATORY RISK SCORE (1-100): If regulators read this, how concerned would they be about THIS SPECIFIC MERGER?
   - 1-20: No merger-specific concern (generic complaints that exist regardless of merger)
   - 21-40: Minimal merger concern (operational issues common to all vendors)
   - 41-60: Moderate merger concern (suggests these companies have market power or reduced alternatives)
   - 61-80: High merger concern (indicates lack of alternatives, market concentration, or that merger would reduce competition)
   - 81-100: Critical merger concern (merger would create monopoly, eliminate only alternative, enable price increases)
   
   Provide ONLY the number (e.g., "45")

4. ONE-LINE VERDICT: In one sentence, should an MD spend time on this? Why or why not?

5. KEY REGULATORY FLAGS:
   - ONLY list concerns specific to this merger reducing competition
   - NOT generic industry issues
   - If no merger-specific concerns exist, use empty array: []

6. SENTIMENT: Positive/Neutral/Negative toward the products/services discussed

Format your scores clearly as:

IMPORTANT:
- Return your response as valid JSON with the following structure:  
{{
  "bullet_summary": ["point1", "point2", "point3"],
  "relevance_score": <number>,
  "regulatory_risk_score": <number>,
  "one_line_verdict": "<string>",
  "key_regulatory_flags": ["specific merger concern 1", "specific merger concern 2"] or [],
  "sentiment": "<Positive|Neutral|Negative>"
}}

EXAMPLES:
- If there are merger-specific concerns: "key_regulatory_flags": ["These two companies are the only alternatives in our region", "Merger would eliminate competition for enterprise customers"]
- If no merger-specific concerns: "key_regulatory_flags": []

CRITICAL: Ensure your response contains ONLY the JSON object above. Do not include any additional text, explanations, or markdown formatting around the JSON.

"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        analysis_text = message.content[0].text

        logger.info(f"Analysis text: {analysis_text}")
        parsed = parse_analysis(analysis_text)

        return {
            "post_id": post_data['id'],
            "post_title": post_data['title'],
            "post_url": post_data['url'],
            "reddit_score": post_data['score'],
            "num_comments": post_data.get('num_comments', 0),
            "competition": None,  # Will be added later

            # Parsed analysis fields
            "relevance_score": parsed["relevance_score"],
            "risk_score": parsed["risk_score"],
            "bullet_summary": parsed["bullet_summary"],
            "verdict": parsed["verdict"],
            "regulatory_flags": parsed["regulatory_flags"],
            "sentiment": parsed["sentiment"],

            # Reference materials
            "post_content": post_text_summary,
            "claude_analysis": analysis_text,

            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "post_id": post_data['id'],
            "post_title": post_data.get('title', 'Unknown'),
            "error": str(e)
        }


def analyze_reddit_data(json_file_path, test_mode=False, test_limit=5):
    """
    Main function to process the Reddit JSON file and analyze all posts.

    Args:
        json_file_path: Path to the JSON file
        test_mode: If True, only analyze first test_limit posts
        test_limit: Number of posts to analyze in test mode
    """

    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Competitive pair data is now embedded in each result
    print("✓ Using competitive pair data from individual results\n")

    print(f"Deal ID: {data.get('deal_id', 'N/A')}")
    print(f"Total competitions: {data.get('total_competitions', 'N/A')}")
    total_posts = data['deduplication_stats']['deduplicated_total_posts']

    if test_mode:
        print(
            f"🧪 TEST MODE: Analyzing only first {test_limit} posts (out of {total_posts} total)\n")
    else:
        print(f"Total posts to analyze: {total_posts}\n")

    all_analyses = []
    posts_processed = 0

    # Process each competition search result
    for result in data['results']:
        if test_mode and posts_processed >= test_limit:
            print(f"\n🧪 Test limit reached ({test_limit} posts). Stopping.")
            break

        competition = result['competition']
        posts = result['posts']
        competitive_pair = result.get('competitive_pair')

        if not posts:
            print(f"⚠ No posts for: {competition}")
            continue

        print(f"\nAnalyzing {len(posts)} posts for: {competition}")

        # Get enhanced competition context from embedded competitive_pair
        competition_context = get_competition_context(competitive_pair)
        print(f"Competition context: {competition_context}")

        for i, post in enumerate(posts, 1):
            if test_mode and posts_processed >= test_limit:
                break

            print(
                f"  Processing post {i}/{len(posts)}: {post['title'][:50]}...")

            analysis = analyze_post(post, competition_context)
            analysis['competition'] = competition
            all_analyses.append(analysis)
            posts_processed += 1

    # Save results
    output_folder = "Reddit LLM Review"
    filename_prefix = "test_" if test_mode else "merger_analysis_"
    output_file = os.path.join(
        output_folder, f"{filename_prefix}{data['deal_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "deal_id": data['deal_id'],
            "test_mode": test_mode,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_posts_analyzed": len(all_analyses),
            "analyses": all_analyses
        }, f, indent=2)

    print(f"\n✓ Analysis complete! Results saved to: {output_file}")

    # Create Excel export
    excel_file = output_file.replace('.json', '.xlsx')
    print(f"✓ Excel export saved to: {excel_file}")

    print(f"Total posts analyzed: {len(all_analyses)}")

    return all_analyses


# Example usage
if __name__ == "__main__":
    json_file = "reddit_input.json"

    print(f"Loading Reddit data from: {json_file}\n")

    # Set test_mode=True to analyze only first 10 posts
    # Set test_mode=False to analyze all posts
    analyses = analyze_reddit_data(json_file, test_mode=False)

    # Optional: Print summary of first analysis
    if analyses and 'error' not in analyses[0]:
        print("\n" + "="*60)
        print("SAMPLE ANALYSIS")
        print("="*60)
        a = analyses[0]
        print(f"\nPost: {a['post_title']}")
        print(f"Competition: {a['competition']}")
        print(f"Relevance Score: {a['relevance_score']}")
        print(f"Risk Score: {a['risk_score']}")
        print(f"\nBullet Summary:")
        for bullet in a['bullet_summary']:
            print(f"  • {bullet}")
        print(f"\nVerdict: {a['verdict']}")
        print(f"Sentiment: {a['sentiment']}")

    # Print summary statistics
    if analyses:
        relevance_scores = [a['relevance_score']
                            for a in analyses if a.get('relevance_score')]
        risk_scores = [a['risk_score']
                       for a in analyses if a.get('risk_score')]

        if relevance_scores and risk_scores:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            print(
                f"Average Relevance Score: {sum(relevance_scores)/len(relevance_scores):.1f}")
            print(
                f"Average Risk Score: {sum(risk_scores)/len(risk_scores):.1f}")
            print(f"Highest Risk Score: {max(risk_scores)}")
            print(
                f"High Risk Posts (>60): {sum(1 for r in risk_scores if r > 60)}")
            print(
                f"High Relevance Posts (>60): {sum(1 for r in relevance_scores if r > 60)}")
            print(
                f"Priority Posts (Relevance>60 AND Risk>60): {sum(1 for a in analyses if a.get('relevance_score') and a.get('risk_score') and a['relevance_score'] > 60 and a['risk_score'] > 60)}")
