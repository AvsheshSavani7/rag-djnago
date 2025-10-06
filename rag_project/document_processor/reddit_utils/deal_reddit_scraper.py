from document_processor.models import ProcessingJob, CompetitiveAnalysis, CompanyProducts, RedditPost
from mongoengine import connect
import json
import logging
import os
import re
import time
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import serpapi
import praw
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the LLM analysis functions
from .reddit_llm_review_new import get_competition_context, analyze_post

# Django and MongoDB imports
import django
from django.conf import settings
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

# Ensure MongoDB connection is established
MONGODB_URI = os.getenv("MONGODB_CONNECTION_STRING")
MONGODB_NAME = os.getenv("MONGODB_NAME", "Deal_DB")

if MONGODB_URI:
    print("✅ MongoDB config detected in deal scraper.")
    connect(
        db=MONGODB_NAME,
        host=MONGODB_URI,
        alias="default"
    )
else:
    print("❌ No MongoDB connection string found in environment for deal scraper.")
    raise ConnectionError("MongoDB connection string not found")

# Import Django models after MongoDB connection is established


# Load environment variables
load_dotenv()

# ---------- CONFIGURATION ----------
SERPAPI_KEY = "05c6f59846a3e01cda9c1e47e99d2cabdc3981a0566d792c8ee0185bd0ac5147"

# josh
# SERPAPI_KEY = "4ef39c2711bef0264e5bde45ff957422dc569eccf8f651b2fa82613d4d5ce038"

# Reddit API credentials
REDDIT_CLIENT_ID = "nprXSCFbiDcmTS4H50ajqQ"
REDDIT_CLIENT_SECRET = "Bitkg99FFa1WQvkoeyKDv7LWKL0_0w"
REDDIT_USER_AGENT = "Deal Reddit Scraper by /u/Fun-Consequence-9402"

# Output Configuration
OUTPUT_DIR = "deal_reddit_analysis"
FINAL_OUTPUT_FILE = "deal_reddit_analysis_results.json"

# Worker Configuration
MAX_WORKERS = 4  # Maximum number of parallel workers for LLM analysis

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('deal_reddit_scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------- GLOBAL VARIABLES FOR TOOLS ----------
reddit_client = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global storage for results
reddit_results = []

# ---------- TOOL FUNCTIONS ----------

# Thread-safe lock for MongoDB operations
mongodb_lock = threading.Lock()


def process_single_post(post_data, competition_context, deal_id, post_id, search_query, competition):
    """
    Process a single Reddit post with LLM analysis (worker function).

    Args:
        post_data: Reddit post data
        competition_context: Competition context for analysis
        deal_id: Deal ID
        post_id: Reddit post ID
        search_query: Search query used
        competition: Competition name

    Returns:
        dict: Processed post data with analysis
    """
    try:
        logger.info(f"Worker processing post: {post_data['title'][:50]}...")

        # Run LLM analysis on the post
        analysis_result = analyze_post(post_data, competition_context)

        # Create enhanced post info with analysis
        post_info = {
            **post_data,  # Include all original post data
            "post_id": post_id,  # Add post_id field for consistency
            "relevance_score": analysis_result.get("relevance_score"),
            "risk_score": analysis_result.get("risk_score"),
            "bullet_summary": analysis_result.get("bullet_summary"),
            "verdict": analysis_result.get("verdict"),
            "regulatory_flags": analysis_result.get("regulatory_flags"),
            "sentiment": analysis_result.get("sentiment"),
            "post_content": analysis_result.get("post_content"),
            "claude_analysis": analysis_result.get("claude_analysis"),
            "timestamp": analysis_result.get("timestamp")
        }

        # Save to MongoDB with thread-safe lock
        with mongodb_lock:
            saved_post, is_new = RedditPost.save_unique_post(
                deal_id=deal_id,
                reddit_id=post_id,
                search_query=search_query,
                post_data=post_info,
                competition=competition,
                approach="REDDIT_SCRAPER"
            )

            if is_new:
                logger.info(
                    f"✅ New post saved to MongoDB: '{post_data['title']}'")
                return {"post_info": post_info, "is_new": True, "success": True}
            else:
                # Check if existing post needs LLM analysis update
                existing_post = RedditPost.objects(
                    deal_id=deal_id, reddit_id=post_id).first()
                has_llm_analysis = (
                    existing_post and
                    existing_post.post and
                    isinstance(existing_post.post, dict) and
                    'relevance_score' in existing_post.post
                )

                if not has_llm_analysis:
                    # Update existing post with LLM analysis
                    existing_post.post = post_info
                    existing_post.updated_at = datetime.now(timezone.utc)
                    existing_post.save()
                    logger.info(
                        f"🔄 Updated existing post with LLM analysis: '{post_data['title']}'")
                    return {"post_info": post_info, "is_new": True, "success": True}
                else:
                    logger.info(
                        f"⏭️ Post already exists with LLM analysis: '{post_data['title']}'")
                    return {"post_info": post_info, "is_new": False, "success": True}

    except Exception as e:
        logger.error(f"Error processing post {post_id}: {e}")
        return {"post_info": post_data, "is_new": False, "success": False, "error": str(e)}


def extract_comments(comment_forest):
    """
    Recursively extract comments and their replies from a Reddit comment forest.

    Args:
        comment_forest: PRAW comment forest object

    Returns:
        List[Dict]: List of comment dictionaries with nested replies
    """
    comments = []
    for comment in comment_forest:
        if isinstance(comment, praw.models.MoreComments):
            continue
        comment_data = {
            "id": comment.id,
            "author": str(comment.author),
            "body": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "replies": extract_comments(comment.replies) if comment.replies else []
        }
        comments.append(comment_data)
    return comments


def fetch_deal_data(deal_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch deal data from database using deal_id

    Args:
        deal_id: MongoDB ObjectId string

    Returns:
        Deal data dictionary or None if not found
    """
    try:
        deal = ProcessingJob.objects.get(id=deal_id)

        # Convert to dictionary format
        deal_data = {
            'id': str(deal.id),
            'cik': deal.cik,
            'acquire_name': deal.acquire_name,
            'target_name': deal.target_name,
            'announce_date': deal.announce_date.strftime('%Y-%m-%d') if deal.announce_date else None,
            'schema_results': deal.schema_results
        }

        logger.info(f"Successfully fetched deal data for {deal_id}")
        return deal_data

    except ProcessingJob.DoesNotExist:
        logger.error(f"Deal with ID {deal_id} not found")
        return None
    except Exception as e:
        logger.error(f"Error fetching deal data: {e}")
        return None


def fetch_competitive_products(deal_id: str) -> List[Dict[str, Any]]:
    """
    Fetch competitive products from database

    Args:
        deal_id: Deal ID

    Returns:
        List of competitive product pairs
    """
    competitive_pairs = []

    try:
        # Get competitive analysis for this deal
        competitive_analysis = CompetitiveAnalysis.objects(
            deal_id=deal_id).first()

        if competitive_analysis and competitive_analysis.competitive_pairs:
            competitive_pairs = competitive_analysis.competitive_pairs
            logger.info(
                f"Found {len(competitive_pairs)} competitive product pairs for deal {deal_id}")
        else:
            logger.warning(f"No competitive analysis found for deal {deal_id}")

    except Exception as e:
        logger.error(f"Error fetching competitive products: {e}")

    return competitive_pairs


def scrape_reddit_competition(competition: str, deal_id: str, pair: Dict[str, Any]) -> str:
    """
    Scrape Reddit discussions for a specific product competition.

    Args:
        competition (str): Competition pair (e.g., "HyperMesh vs NX Nastran")
        deal_id (str): Deal ID for tracking

    Returns:
        str: JSON string with scraped Reddit data
    """

    logger.info(f"Scraping Reddit for competition: {competition}")

    try:
        search_query = f"{competition} site:reddit.com"
        logger.info(f"Search query: {search_query}")

        # Use SerpAPI to search Google for Reddit links
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERPAPI_KEY,
            "num": 20
        }

        logger.info("Making SerpAPI call")
        results = serpapi.search(params)

        # Filter Reddit links
        reddit_links = []
        for i, res in enumerate(results.get("organic_results", [])):
            url = res.get("link", "")
            title = res.get("title", "")

            if "reddit.com/r/" in url and "/comments/" in url:
                clean_url = url.split("?")[0]
                reddit_links.append(clean_url)
                logger.info(f"Found Reddit link: {clean_url}")

        logger.info(f"Found {len(reddit_links)} Reddit links")

        # Extract post IDs
        reddit_ids = []
        for url in reddit_links:
            match = re.search(r'/comments/([a-z0-9]+)/', url)
            if match:
                post_id = match.group(1)
                reddit_ids.append(post_id)
                logger.info(f"Extracted post ID: {post_id}")

        # Initialize a set for already scraped post IDs
        already_scraped_posts = set()

        # Get competition context from the competitive pair
        competition_context = get_competition_context(pair)
        logger.info(f"Competition context: {competition_context}")

        # Fetch posts and comments
        posts_data = []
        new_posts_saved = 0
        existing_posts_skipped = 0

        # Prepare posts for parallel processing
        posts_to_process = []

        for i, post_id in enumerate(reddit_ids):
            if post_id in already_scraped_posts:
                logger.info(f"Post {post_id} already scraped, skipping.")
                continue  # Skip fetching this post

            try:
                logger.info(
                    f"Fetching post {i+1}/{len(reddit_ids)}: {post_id}")
                submission = reddit_client.submission(id=post_id)
                submission.comments.replace_more(limit=0)

                # Extract comments using the enhanced extract_comments function
                comments = extract_comments(submission.comments)

                # Create basic post info
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "author": str(submission.author),
                    "score": submission.score,
                    "url": submission.url,
                    "selftext": submission.selftext,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "comments": comments,  # Limit comments for efficiency
                    "pair": pair  # Add competitive pair information to each post
                }

                posts_to_process.append({
                    "post_data": post_data,
                    "post_id": post_id
                })
                already_scraped_posts.add(post_id)
                logger.info(f"Successfully fetched: '{submission.title}'")

            except Exception as e:
                logger.error(f"Error fetching post {post_id}: {e}")

        # Process posts in parallel with workers
        logger.info(
            f"Processing {len(posts_to_process)} posts with workers...")

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(MAX_WORKERS, len(posts_to_process)
                          )  # Limit workers based on config
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_post = {
                executor.submit(
                    process_single_post,
                    post_info["post_data"],
                    competition_context,
                    deal_id,
                    post_info["post_id"],
                    search_query,
                    competition
                ): post_info for post_info in posts_to_process
            }

            # Process completed tasks
            for future in as_completed(future_to_post):
                post_info = future_to_post[future]
                try:
                    result = future.result()
                    if result["success"]:
                        posts_data.append(result["post_info"])
                        if result["is_new"]:
                            new_posts_saved += 1
                        else:
                            existing_posts_skipped += 1
                    else:
                        logger.error(
                            f"Failed to process post {post_info['post_id']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(
                        f"Exception processing post {post_info['post_id']}: {e}")

        competition_result = {
            "competition": competition,
            "deal_id": deal_id,
            "search_query": search_query,
            "total_posts_found": len(posts_data),
            "posts": posts_data,
            "mongodb_stats": {
                "new_posts_saved": new_posts_saved,
                "existing_posts_skipped": existing_posts_skipped,
                "total_posts_processed": new_posts_saved + existing_posts_skipped
            },
            "timestamp": datetime.now().isoformat(),
            "pair": pair
        }

        # Save individual result
        safe_filename = competition.replace(
            ' vs ', '_vs_').replace(' ', '_').replace('/', '_')
        filename = f"{OUTPUT_DIR}/{deal_id}_{safe_filename}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(competition_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result to {filename}")
        logger.info(f"Found {len(posts_data)} posts for '{competition}'")

        # Store in global results
        reddit_results.append(competition_result)

        return json.dumps({
            "success": True,
            "competition": competition,
            "posts_found": len(posts_data),
            "file_saved": filename,
            "mongodb_stats": {
                "new_posts_saved": new_posts_saved,
                "existing_posts_skipped": existing_posts_skipped,
                "total_posts_processed": new_posts_saved + existing_posts_skipped
            }
        })

    except Exception as e:
        error_msg = f"Error scraping Reddit for '{competition}': {e}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "competition": competition})


def save_final_analysis(deal_id: str) -> str:
    """
    Save final consolidated analysis results with duplicate post removal.

    Args:
        deal_id (str): Deal ID for the analysis

    Returns:
        str: Status message with file path
    """

    logger.info(f"Saving final analysis for deal {deal_id}")

    try:
        # Remove duplicate posts across all competitions
        seen_post_ids = set()
        deduplicated_results = []
        total_original_posts = 0
        total_deduplicated_posts = 0

        for competition_result in reddit_results:
            original_posts = competition_result.get('posts', [])
            total_original_posts += len(original_posts)

            # Filter out duplicate posts by ID
            unique_posts = []
            for post in original_posts:
                post_id = post.get('id')
                if post_id and post_id not in seen_post_ids:
                    seen_post_ids.add(post_id)
                    unique_posts.append(post)
                elif post_id:
                    logger.info(
                        f"Removed duplicate post ID: {post_id} from competition: {competition_result.get('competition')}")

            # Update the competition result with deduplicated posts
            deduplicated_competition = competition_result.copy()
            deduplicated_competition['posts'] = unique_posts
            deduplicated_competition['total_posts_found'] = len(unique_posts)
            deduplicated_competition['original_posts_count'] = len(
                original_posts)
            deduplicated_competition['duplicates_removed'] = len(
                original_posts) - len(unique_posts)

            deduplicated_results.append(deduplicated_competition)
            total_deduplicated_posts += len(unique_posts)

        # Calculate MongoDB statistics
        total_new_posts_saved = sum(
            result.get('mongodb_stats', {}).get('new_posts_saved', 0)
            for result in deduplicated_results
        )
        total_existing_posts_skipped = sum(
            result.get('mongodb_stats', {}).get('existing_posts_skipped', 0)
            for result in deduplicated_results
        )

        # Create final data with deduplication stats
        final_data = {
            "deal_id": deal_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_competitions": len(deduplicated_results),
            "deduplication_stats": {
                "original_total_posts": total_original_posts,
                "deduplicated_total_posts": total_deduplicated_posts,
                "duplicates_removed": total_original_posts - total_deduplicated_posts,
                "unique_post_ids": len(seen_post_ids)
            },
            "mongodb_stats": {
                "total_new_posts_saved": total_new_posts_saved,
                "total_existing_posts_skipped": total_existing_posts_skipped,
                "total_posts_processed": total_new_posts_saved + total_existing_posts_skipped
            },
            "results": deduplicated_results
        }

        final_filename = f"{OUTPUT_DIR}/deal_{deal_id}_reddit_analysis.json"
        with open(final_filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Final results saved to {final_filename}")
        logger.info(
            f"Deduplication complete - {total_original_posts} original posts -> {total_deduplicated_posts} unique posts")
        logger.info(
            f"Removed {total_original_posts - total_deduplicated_posts} duplicate posts")

        return json.dumps({
            "success": True,
            "file": final_filename,
            "competitions_analyzed": len(deduplicated_results),
            "original_posts": total_original_posts,
            "unique_posts": total_deduplicated_posts,
            "duplicates_removed": total_original_posts - total_deduplicated_posts,
            "mongodb_stats": {
                "total_new_posts_saved": total_new_posts_saved,
                "total_existing_posts_skipped": total_existing_posts_skipped,
                "total_posts_processed": total_new_posts_saved + total_existing_posts_skipped
            }
        })

    except Exception as e:
        error_msg = f"Error saving final results: {e}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def run_deal_reddit_analysis(deal_id: str):
    """
    Run the complete Reddit analysis for a specific deal.

    Args:
        deal_id (str): Deal ID to analyze
    """

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("DEAL REDDIT SCRAPER STARTING")
    logger.info("=" * 80)
    logger.info(f"ANALYSIS: Target deal ID: {deal_id}")
    logger.info(f"ANALYSIS: Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Clear previous results
    global reddit_results
    reddit_results = []

    try:
        # Step 1: Fetch deal data
        logger.info("Step 1: Fetching deal data...")
        deal_data = fetch_deal_data(deal_id)
        if not deal_data:
            logger.error(f"Could not fetch deal data for {deal_id}")
            return None

        logger.info(
            f"Deal: {deal_data['acquire_name']} acquiring {deal_data['target_name']}")

        # Step 2: Fetch competitive products
        logger.info("Step 2: Fetching competitive products...")
        competitive_pairs = fetch_competitive_products(deal_id)

        if not competitive_pairs:
            logger.warning(f"No competitive products found for deal {deal_id}")
            return None

        logger.info(
            f"Found {len(competitive_pairs)} competitive product pairs")

        logger.info(f"Competitive pairs: {competitive_pairs}")

        # Step 3: Scrape Reddit for each competition
        logger.info("Step 3: Scraping Reddit discussions...")
        for i, pair in enumerate(competitive_pairs):
            try:
                # Extract product names from the competitive pair
                target_product = pair.get('target_product', '')
                acquire_product = pair.get('acquire_product', '')

                if target_product and acquire_product:
                    competition = f"{target_product} vs {acquire_product}"
                    logger.info(
                        f"Processing competition {i+1}/{len(competitive_pairs)}: {competition}")

                    result = scrape_reddit_competition(
                        competition, deal_id, pair)
                    logger.info(f"Competition {i+1} completed")
                else:
                    logger.warning(
                        f"Skipping pair {i+1} - missing product names: {pair}")

            except Exception as e:
                logger.error(f"Error processing competition pair {i+1}: {e}")
                continue

        # Step 4: Save final analysis
        logger.info("Step 4: Saving final analysis...")
        final_result = save_final_analysis(deal_id)

        # Log results
        total_time = time.time() - start_time
        total_posts = sum(r.get('total_posts_found', 0)
                          for r in reddit_results)

        # Calculate MongoDB statistics
        total_new_posts_saved = sum(
            r.get('mongodb_stats', {}).get('new_posts_saved', 0)
            for r in reddit_results
        )
        total_existing_posts_skipped = sum(
            r.get('mongodb_stats', {}).get('existing_posts_skipped', 0)
            for r in reddit_results
        )

        logger.info("ANALYSIS: ✅ REDDIT ANALYSIS COMPLETED!")
        logger.info(
            f"ANALYSIS: Total execution time: {total_time:.2f} seconds")
        logger.info(
            f"ANALYSIS: Total competitions analyzed: {len(reddit_results)}")
        logger.info(f"ANALYSIS: Total Reddit posts scraped: {total_posts}")
        logger.info(
            f"ANALYSIS: MongoDB - New posts saved: {total_new_posts_saved}")
        logger.info(
            f"ANALYSIS: MongoDB - Existing posts skipped: {total_existing_posts_skipped}")
        logger.info("=" * 80)

        # Print summary
        print("\n" + "="*60)
        print("DEAL REDDIT ANALYSIS COMPLETED")
        print("="*60)
        print(f"🆔 Deal ID: {deal_id}")
        print(
            f"🏢 Deal: {deal_data['acquire_name']} acquiring {deal_data['target_name']}")
        print(f"🥊 Competitions Analyzed: {len(reddit_results)}")
        print(f"📊 Total Posts Scraped: {total_posts}")
        print(f"💾 MongoDB - New Posts Saved: {total_new_posts_saved}")
        print(
            f"⏭️ MongoDB - Existing Posts Skipped: {total_existing_posts_skipped}")
        print(f"⏱️ Execution Time: {total_time:.2f} seconds")
        print(f"📁 Results Directory: {OUTPUT_DIR}/")
        print("="*60)

        return final_result

    except Exception as e:
        logger.error(f"ANALYSIS: ❌ Analysis failed: {e}")
        raise


def main():
    """Main function to run the deal Reddit scraper."""

    if len(sys.argv) != 2:
        print("Usage: python deal_reddit_scraper.py <deal_id>")
        print("Example: python deal_reddit_scraper.py 507f1f77bcf86cd799439011")
        sys.exit(1)

    deal_id = sys.argv[1]

    try:
        result = run_deal_reddit_analysis(deal_id)
        if result:
            print(f"\n🤖 Analysis Result:\n{result}")
        else:
            print(f"\n❌ Analysis failed for deal {deal_id}")

    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        logger.error(f"Main execution error: {e}")


if __name__ == "__main__":
    main()
