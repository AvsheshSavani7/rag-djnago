# pinecone_utils.py
import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone
from openai import OpenAI
# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pinecone_chunks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _extract_section_core_ids(raw_refs: List[str]) -> List[str]:
    """
    Normalize strings like 'Section 6.1 (d)', 'Section 5.4', '§ 5.4' to ['6.1', '5.4'].
    For non-section references like 'Definition > Confidentiality Agreement', keep as-is.
    """
    sections = []
    for r in raw_refs or []:
        if not r:
            continue
        # Check if this is a section reference (contains section number pattern)
        m = re.search(r'(\d+(?:\.\d+)+|\d+)', r)
        if m:
            # Extract just the section number
            sections.append(m.group(1))
        else:
            # Not a section reference, keep the original string
            sections.append(r.strip())
    # de-dupe preserving order
    seen = OrderedDict()
    for s in sections:
        seen.setdefault(s, True)
    return list(seen.keys())


def _expand_section_variants(core_ids: List[str]) -> List[str]:
    """
    For each section number like '8.6' produce both '8.6' and 'Section 8.6' to match how metadata is stored.
    For non-section references, keep as-is without adding 'Section' prefix.
    """
    out = []
    for s in core_ids:
        out.append(s)
        # Only add "Section " prefix if this looks like a section number (contains digits)
        if re.search(r'\d', s):
            out.append(f"Section {s}")
            out.append(f"SECTION {s}")

    # de-dupe preserving order
    seen = OrderedDict()
    for v in out:
        seen.setdefault(v, True)
    return list(seen.keys())


def _dedupe_texts(texts: List[Tuple[int, str, str]]) -> List[str]:
    """
    Accepts (rank_key, text). De-duplicates by normalized text while preserving rank order.
    """
    seen = set()
    out: List[str] = []
    for _, t, s in texts:
        if not t:
            continue
        key = " ".join(t.split())  # collapse whitespace
        if key not in seen:
            seen.add(key)
            out.append(s + " : " + t)
    return out


class PineconeSectionFetcher:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        # e.g. 'ma-contracts'
        index_name = os.getenv("PINECONE_INDEX_NAME")
        # optional for serverless
        host = os.getenv("PINECONE_HOST")
        if not api_key or not index_name:
            raise RuntimeError(
                "Missing PINECONE_API_KEY or PINECONE_INDEX env vars")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(
            index_name, host=host) if host else self.pc.Index(index_name)

        # For metadata-only queries we supply a zero vector with the index dim
        # set to your index dimension
        self.dim = int(os.getenv("PINECONE_DIM", "3072"))
        self.default_top_k = int(os.getenv("PINECONE_TOP_K", "300"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Optional namespace if you use it
        self.namespace = os.getenv("PINECONE_NAMESPACE", None)

    def get_context_for_references(
        self,
        raw_reference_strings: List[str],
        top_k: int | None = None,
        additional_filter: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        1) Convert references (e.g., 'Section 8.6') into core ids ['8.6'] and variants ['8.6','Section 8.6'].
        2) Query Pinecone with metadata filter: section: {$in: variants}
        3) Prefer 'combined_text' for chunk text; sort by 'chunk_index' to maintain order per section.
        """
        core_ids = _extract_section_core_ids(raw_reference_strings)
        if not core_ids:
            return {"sections": [], "context_chunks": [], "context_text": ""}

        variants = _expand_section_variants(core_ids)

        # Use $in for array metadata fields
        flt: Dict[str, Any] = {"Section": {"$in": variants}}

        # Merge any additional filters (e.g., deal_id scoping)
        # print(f"additional_filter {additional_filter}")
        if additional_filter:
            for k, v in additional_filter.items():
                if isinstance(v, dict):
                    flt[k] = v
                else:
                    flt[k] = {"$eq": v}

        # Zero-vector to enable metadata-only retrieval
        vec = [0.0] * self.dim
        tk = top_k or self.default_top_k

        # --- Log query details:

        # logger.info(f"[PINECONE] Query filter: {flt}")
        # logger.info(f"[PINECONE] Query top_k: {tk}")

        try:
            res = self.index.query(
                vector=vec,
                top_k=tk,
                filter=flt,
                include_values=False,
                include_metadata=True,

            )

            # logger.info(
            # f"[PINECONE] Query successful, found {len(getattr(res, 'matches', []) or [])} matches")
            # logger.info(
            # f"[PINECONE] Query successful Result {res}")

        except Exception as e:
            # Fail soft: surface empty context but keep the run going
            # logger.error(f"[PINECONE] Query error: {e}")
            return {"sections": core_ids, "context_chunks": [], "context_text": ""}

        # Collect tuples for sorting: (rank_key, text)
        tuples: List[Tuple[int, str, str]] = []
        matches = getattr(res, "matches", None) or []

        # logger.info(f"[PINECONE] Processing {len(matches)} matches...")

        for i, match in enumerate(matches):
            md = getattr(match, "metadata", None) or {}

            text = (
                md.get("combined_text")
                or md.get("text")
                or md.get("chunk_text")
                or md.get("original_text")
                or ""
            )
            sec = md.get("Section")[0]
            rank = md.get("chunk_index")
            try:
                rank_key = int(rank) if rank is not None else 10_000_000
            except Exception:
                rank_key = 10_000_000

            tuples.append((rank_key, text, sec))

        # Sort then de-duplicate
        tuples.sort(key=lambda x: x[0])

        # logger.info(f"[PINECONE] tuples {tuples}")
        chunk_texts = _dedupe_texts(tuples)

        # Always define joined, even if there are zero chunks
        # joined = ""
        # if chunk_texts:
        #     max_chars = int(os.getenv("PINECONE_CONTEXT_CHAR_BUDGET", "8000"))
        #     for t in chunk_texts:
        #         t = (t or "").strip()
        #         if not t:
        #             continue
        #         if len(joined) + len(t) + 2 > max_chars:
        #             break
        #         joined += (("\n\n" if joined else "") + t)

        return {
            "sections": core_ids,        # normalized section ids like ['8.6']
            "context_chunks": chunk_texts,
            "context_text": chunk_texts,

        }

    def get_all_chunks_for_deal(self, deal_id: str):
        """Get all chunks for a specific deal from Pinecone"""
        try:
            # Create a dummy query to get all chunks for the deal
            vec = [0.0] * self.dim

            search_response = self.index.query(
                vector=vec,
                top_k=500,  # Large number to get all chunks
                include_metadata=True,
                filter={
                    "deal_id": deal_id
                }
            )

            results = []
            for match in search_response.matches:
                result = {
                    'label': match.metadata['label'],
                    'text': match.metadata['combined_text'],
                    'score': match.score
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"Error getting all chunks for deal: {str(e)}")

            return []

    def semantic_search_chunks_for_deal(
        self,
        deal_id: str,
        query_text: str,
        top_k: int = 5,
        *,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search within a deal using query embedding.

        Returns list of dicts with keys: label, text, score.
        """
        if not query_text:
            return []

        embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        try:
            emb = self.openai_client.embeddings.create(
                model=embedding_model,
                input=query_text,
            )
            vector = emb.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding for semantic search: {e}")
            return []

        try:
            res = self.index.query(
                vector=vector,
                top_k=top_k,
                include_values=False,
                include_metadata=include_metadata,
                filter={"deal_id": deal_id},
            )

        except Exception as e:
            logger.error(f"Error querying Pinecone (semantic search): {e}")
            return []

        matches = getattr(res, "matches", None) or []
        results: List[Dict[str, Any]] = []
        for match in matches:
            md = getattr(match, "metadata", None) or {}
            results.append(
                {
                    "label": md.get("label", ""),
                    "text": md.get("combined_text")
                    or md.get("text")
                    or md.get("chunk_text")
                    or md.get("original_text")
                    or "",
                    "score": getattr(match, "score", None),
                }
            )
        return results

    def process_definition_chunk(self, chunk: Dict) -> Dict:
        """Process a single definition chunk with GPT - worker method"""
        try:
            text = chunk.get('text', '')
            raw_label = chunk.get('label', '')

            # Extract actual label by splitting on ">" and taking the last part
            actual_label = raw_label.split(
                '>')[-1].strip() if '>' in raw_label else raw_label

            # Use GPT to extract definition
            definition_obj = {"label": actual_label, "means": text}
            if definition_obj:
                # logger.info(
                #     f"Worker extracted definition: {definition_obj.get('label', 'N/A')} -> {definition_obj.get('means', '')[:50]}...")
                # print(f"definition_obj: {definition_obj}")
                return definition_obj
            else:
                # logger.warning(
                #     f"Worker failed to extract definition for: {actual_label}")
                return None

        except Exception as e:
            logger.error(
                f"Error in worker processing chunk {chunk.get('label', 'N/A')}: {str(e)}")
            return None

    def extract_preamble_from_chunks(self, chunks: List[Dict]) -> Dict:
        """Extract preamble from chunks with 'Preamble' in label"""
        try:
            preamble_chunks = []

            # Filter chunks with "Preamble" in label (case insensitive)
            for chunk in chunks:
                label = chunk.get('label', '').lower()
                if 'preamble' in label:
                    preamble_chunks.append(chunk)
                    # logger.info(
                    #     f"Found preamble chunk: {chunk.get('label', 'N/A')}")

            # logger.info(f"Found {len(preamble_chunks)} preamble chunks")

            if not preamble_chunks:
                return {
                    "preamble_text": "",
                    "preamble_chunks": [],
                    "found": False
                }

            # Extract text from preamble chunks
            preamble_texts = []
            for chunk in preamble_chunks:
                text = chunk.get('text', '')
                if text:
                    preamble_texts.append({
                        'label': chunk.get('label', ''),
                        'text': text
                    })

            # Combine all preamble text
            combined_preamble = "\n\n".join(
                [p['text'] for p in preamble_texts])

            # logger.info(
            #     f"Extracted preamble with {len(combined_preamble)} characters from {len(preamble_texts)} chunks")

            return {
                "preamble_text": combined_preamble,
                "preamble_chunks": preamble_texts,
                "found": True
            }

        except Exception as e:
            logger.error(f"Error extracting preamble: {str(e)}")
            return {
                "preamble_text": "",
                "preamble_chunks": [],
                "found": False
            }

    def extract_definitions_from_chunks(self, chunks: List[Dict], max_workers: int = 5) -> List[Dict]:
        """Extract definitions from chunks with 'Definition' or 'Definitions' in label using GPT with workers"""
        try:
            definition_chunks = []

            # Filter chunks with "Definition" or "Definitions" in label (case insensitive)
            for chunk in chunks:
                label = chunk.get('label', '').lower()
                if 'definitions' in label or "definition" in label:  # Match both singular and plural
                    definition_chunks.append(chunk)
                    # logger.info(
                    #     f"Found definition chunk: {chunk.get('label', 'N/A')}")

            # logger.info(f"Found {len(definition_chunks)} definition chunks")

            if not definition_chunks:
                return []

            # Process chunks in parallel using workers
            definitions_array = []
            # logger.info(
            #     f"Starting parallel processing with {max_workers} workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(self.process_definition_chunk, chunk): chunk
                    for chunk in definition_chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        definition_obj = future.result()

                        if definition_obj:
                            definitions_array.append(definition_obj)

                    except Exception as e:
                        logger.error(
                            f"Worker failed for chunk {chunk.get('label', 'N/A')}: {str(e)}")

            # logger.info(
            #     f"Extracted {len(definitions_array)} definitions using {max_workers} workers")
            return definitions_array

        except Exception as e:
            logger.error(f"Error extracting definitions: {str(e)}")
            return []
