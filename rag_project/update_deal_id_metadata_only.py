#!/usr/bin/env python
import os
import pinecone
import time
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def update_deal_id_metadata_only(old_deal_id, new_deal_id, batch_size=120):
    """
    Update the deal_id in Pinecone metadata for all vectors that have the old deal_id.
    This script only changes the metadata, not the vector IDs.

    Args:
        old_deal_id (str): The current deal_id in the vectors
        new_deal_id (str): The new deal_id to replace with
        batch_size (int): Number of vectors to update in each batch
    """
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME", "contract-chunks")

    try:
        # Connect to the index
        index = pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

        # Use a dummy vector for metadata-only search
        dummy_vector = [0.0] * 3072  # Dimension for text-embedding-3-small

        # Prepare filter to get vectors with the old deal_id
        filter_dict = {"deal_id": str(old_deal_id)}

        # Query Pinecone with filter for the old deal_id
        logger.info(f"Fetching vectors with deal_id: {old_deal_id}")
        query_response = index.query(
            vector=dummy_vector,
            top_k=10000,  # Use maximum allowed to get all vectors
            include_metadata=True,
            filter=filter_dict
        )

        # Get all vector IDs with old deal_id
        vector_ids = [match.id for match in query_response.matches]
        total_vectors = len(vector_ids)

        if total_vectors == 0:
            logger.info(f"No vectors found with deal_id: {old_deal_id}")
            return

        logger.info(
            f"Found {total_vectors} vectors with deal_id: {old_deal_id}")

        # Process vectors in batches to update deal_id
        batches = [vector_ids[i:i + batch_size]
                   for i in range(0, len(vector_ids), batch_size)]

        # For each vector, update the metadata to have the new deal_id
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            logger.info(
                f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} vectors")

            # Process each vector in the current batch
            for vector_id in tqdm(batch, desc=f"Batch {batch_idx+1}", leave=False):
                try:
                    # Update metadata to have new deal_id
                    index.update(
                        id=vector_id,
                        set_metadata={"deal_id": str(new_deal_id)}
                    )
                    logger.debug(
                        f"Updated metadata for vector ID: {vector_id}")

                except Exception as e:
                    logger.error(
                        f"Error updating vector {vector_id}: {str(e)}")

            # Add a small delay between batches to avoid rate limits
            if batch_idx < len(batches) - 1:
                time.sleep(1)

        logger.info(
            f"Successfully updated deal_id from {old_deal_id} to {new_deal_id} for {total_vectors} vectors")

    except Exception as e:
        logger.error(f"Error updating deal_id in Pinecone: {str(e)}")
        raise


if __name__ == "__main__":
    # Replace these with your actual deal IDs
    OLD_DEAL_ID = "6814bfdb478abf06ec1a1e96"  # Replace with your old deal_id
    NEW_DEAL_ID = "68184d52478abf06ec1a28ec"  # Your new deal_id

    update_deal_id_metadata_only(OLD_DEAL_ID, NEW_DEAL_ID)
