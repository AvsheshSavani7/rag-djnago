from collections import Counter
from typing import Dict, List

from bs4 import BeautifulSoup, Tag


def _element_signature(el: Tag) -> str:
    classes = el.get("class") or []
    class_part = ".".join(sorted(classes[:3])) if classes else ""
    return f"{el.name}.{class_part}" if class_part else el.name


def _to_css_selector(el: Tag) -> str:
    if el.get("id"):
        return f"#{el['id']}"

    # Exclude classes that are invalid in CSS selectors — Tailwind responsive/state
    # prefixes use colons (sm:flex, lg:hidden) and arbitrary-value slashes (w-1/2).
    classes = [
        c for c in (el.get("class") or [])
        if c and not c.startswith("_") and ":" not in c and "/" not in c
    ]
    if classes:
        return f"{el.name}.{'.'.join(classes[:4])}"
    return el.name


def _safe_select_one(el: Tag, selector: str) -> bool:
    try:
        return bool(el.select_one(selector))
    except Exception:
        return False


def _score_container(el: Tag, count: int) -> int:
    score = count
    if el.get("class"):
        score += 5
    if _safe_select_one(el, "h1, h2, h3, h4, h5"):
        score += 4
    if _safe_select_one(el, "a[href]"):
        score += 3
    if _safe_select_one(el, "img[src]"):
        score += 2
    if _safe_select_one(el, "time, small, .date"):
        score += 2
    class_text = " ".join(el.get("class") or []).lower()
    for token in ("news", "article", "card", "story", "release", "item", "post"):
        if token in class_text:
            score += 3
    return score


def suggest_container_selectors(html: str, limit: int = 8) -> List[Dict]:
    """
    Suggest repeated DOM blocks that likely represent article cards.
    Returns list of {selector, signature, count, sample_text, score}.
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates: Counter = Counter()
    sample_by_sig: Dict[str, Tag] = {}

    for el in soup.find_all(["article", "li", "div", "section", "tr"]):
        if not isinstance(el, Tag):
            continue
        if not el.find("a", href=True):
            continue

        sig = _element_signature(el)
        if sig in {"div", "li", "section"} and not el.get("class"):
            continue

        candidates[sig] += 1
        if sig not in sample_by_sig:
            sample_by_sig[sig] = el

    ranked = []
    for sig, count in candidates.items():
        if count < 2:
            continue
        el = sample_by_sig[sig]
        ranked.append(
            {
                "selector": _to_css_selector(el),
                "signature": sig,
                "count": count,
                "sample_text": el.get_text(" ", strip=True)[:120],
                "score": _score_container(el, count),
            }
        )

    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:limit]
