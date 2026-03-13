"""
classifier.py — Content-based block classification using Haiku LLM batches.
"""

import json
import re
from typing import List

from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Block, CanonicalDocument
from .config import (
    BLOCK_CLASSIFY_PROMPT, MODEL_CLASSIFY,
    CLASSIFY_BATCH_SIZE, CLASSIFY_MAX_WORKERS, CLASSIFY_MIN_WORDS,
    _TOPIC_PRIORITY,
)


def _classify_batch(batch: List[Block], client: Anthropic) -> List[str]:
    """Classify a batch of blocks by content topic. Returns list of topic strings."""
    para_block = ""
    for i, b in enumerate(batch):
        # Truncate to 800 chars -- enough to identify topic, cheap on tokens
        text_preview = b.text[:800]
        para_block += f"\n--- P{i+1} ---\n{text_preview}\n"

    try:
        response = client.messages.create(
            model=MODEL_CLASSIFY,
            max_tokens=1024,
            messages=[{"role": "user", "content": BLOCK_CLASSIFY_PROMPT + "\n\nPARAGRAPHS:\n" + para_block}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        results = json.loads(raw)
        if isinstance(results, list):
            topics = []
            for r in results:
                topic = r.get("topic", "general") if isinstance(r, dict) else "general"
                topics.append(topic)
            # Pad if LLM returned fewer than batch size
            while len(topics) < len(batch):
                topics.append("general")
            return topics[:len(batch)]
    except Exception as e:
        print(f"    Warning: classify batch failed: {e}")

    return ["general"] * len(batch)


def classify_blocks(doc: CanonicalDocument, client: Anthropic) -> None:
    """Tag every substantive block with a content-based topic. Modifies blocks in place."""
    # Only classify paragraphs and list items with enough words
    classifiable = [b for b in doc.blocks
                    if b.type in ("paragraph", "list_item")
                    and len(b.text.split()) >= CLASSIFY_MIN_WORDS]

    if not classifiable:
        print(f"    No classifiable blocks.")
        return

    batches = [classifiable[i:i + CLASSIFY_BATCH_SIZE]
               for i in range(0, len(classifiable), CLASSIFY_BATCH_SIZE)]

    print(f"    Classifying {len(classifiable)} blocks in {len(batches)} batches "
          f"({CLASSIFY_BATCH_SIZE}/batch, {CLASSIFY_MAX_WORKERS} workers)...")

    results_map = {}  # batch_idx -> list of topics

    with ThreadPoolExecutor(max_workers=CLASSIFY_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_classify_batch, batch, client): idx
            for idx, batch in enumerate(batches)
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            topics = future.result()
            results_map[idx] = topics
            done_count += 1
            if done_count % 20 == 0 or done_count == len(batches):
                print(f"    Classified {done_count}/{len(batches)} batches...")

    # Apply topics to blocks
    topic_counts = {}
    for idx in range(len(batches)):
        batch = batches[idx]
        topics = results_map.get(idx, ["general"] * len(batch))
        for block, topic in zip(batch, topics):
            block.topic = topic
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    # Tag headings with the topic of the first subsequent paragraph
    for i, block in enumerate(doc.blocks):
        if block.type == "heading" and not block.topic:
            for j in range(i + 1, min(i + 5, len(doc.blocks))):
                if doc.blocks[j].topic:
                    block.topic = doc.blocks[j].topic
                    break

    print(f"    Topic distribution: {json.dumps(topic_counts, indent=None)}")


def _get_blocks_by_topic(doc: CanonicalDocument, topics: List[str],
                          max_chars: int = 60000) -> str:
    """Get combined text from blocks tagged with the given topics, grouped by topic.

    Topics are ordered by priority (minority topics first) so that if
    truncation occurs, HSR/termination/financing are preserved.
    """
    by_topic = {}
    for b in doc.blocks:
        if b.topic in topics and b.text.strip():
            by_topic.setdefault(b.topic, []).append(b.text)

    # Sort topics by priority order (minority first)
    ordered = sorted(topics, key=lambda t: (
        _TOPIC_PRIORITY.index(t) if t in _TOPIC_PRIORITY else len(_TOPIC_PRIORITY)
    ))

    parts = []
    for topic in ordered:
        texts = by_topic.get(topic, [])
        if texts:
            parts.append(f"[Topic: {topic}]\n" + "\n\n".join(texts))

    text = "\n\n---\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"
    return text
