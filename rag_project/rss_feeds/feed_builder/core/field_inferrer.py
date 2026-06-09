import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag


def _class_selector(el: Tag) -> str:
    classes = [c for c in (el.get("class") or []) if c and not c.startswith("_")]
    if classes:
        return f"{el.name}.{'.'.join(classes[:4])}"
    if el.get("id"):
        return f"#{el['id']}"
    return el.name


def _minimal_unique_selector(container: Tag, el: Tag) -> str:
    """Shortest selector that matches exactly one element inside container."""
    tag = el.name
    classes = [c for c in (el.get("class") or []) if c and not c.startswith("_")]

    candidates: List[str] = []
    if classes:
        candidates.append(f"{tag}.{classes[0]}")
        candidates.append(f"{tag}.{'.'.join(classes[:2])}")
        candidates.append(f".{classes[0]}")
    candidates.append(tag)

    for sel in candidates:
        try:
            matched = container.select_one(sel)
            if matched is el:
                return sel
        except Exception:
            continue

    return _class_selector(el)


def _is_byline_link(link: Tag) -> bool:
    classes = " ".join(link.get("class") or []).lower()
    if "byline" in classes:
        return True
    parent = link.parent
    if parent:
        parent_classes = " ".join(parent.get("class") or []).lower()
        if "byline" in parent_classes:
            return True
    href = (link.get("href") or "").lower()
    return "/writers/" in href or "/author" in href


def _unique_selector_if_one(container: Tag, sel: str) -> Optional[str]:
    try:
        if len(container.select(sel)) == 1:
            return sel
    except Exception:
        pass
    return None


def _best_link_element(container: Tag) -> Optional[Tag]:
    for sel in (".headline a[href]", "[class*='headline'] a[href]"):
        try:
            link = container.select_one(sel)
            if link and link.get("href"):
                return link
        except Exception:
            continue

    best_link = None
    best_score = -999

    for link in container.select("a[href]"):
        href = link.get("href", "")
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        score = 0
        if link.select_one("h1,h2,h3,h4,h5"):
            score += 25
        parent_classes = " ".join((link.parent.get("class") or []) if link.parent else []).lower()
        if "headline" in parent_classes:
            score += 35
        if "underline" in parent_classes:
            score += 5
        text = link.get_text(" ", strip=True)
        if len(text) >= 20:
            score += 15
        elif len(text) <= 2:
            score -= 10
        if link.select("img") and len(text) <= 3:
            score -= 12
        classes = " ".join(link.get("class") or [])
        if "item-image" in classes or "thumbnail" in classes:
            score -= 15
        if _is_byline_link(link):
            score -= 40
        if "/article/" in href or "/news/" in href or "/ctechnews/" in href:
            score += 5

        if score > best_score:
            best_score = score
            best_link = link

    return best_link


def _best_link_selector(container: Tag) -> str:
    for sel in (
        ".headline a",
        ".headline > a",
        "div.headline a",
        "a:not(.item-image):not(.byline-link)",
        "a:has(h2)",
        "a:has(.item-title)",
    ):
        unique = _unique_selector_if_one(container, sel)
        if unique:
            return unique

    link = _best_link_element(container)
    if not link:
        return "a"

    return _minimal_unique_selector(container, link)


def _best_title_selector(container: Tag) -> str:
    for sel in (
        ".headline a",
        ".headline",
        "[class*='headline'] a",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
    ):
        try:
            el = container.select_one(sel)
        except Exception:
            continue
        if not el:
            continue
        text = el.get_text(" ", strip=True)
        if len(text) < 5:
            continue
        unique = _unique_selector_if_one(container, sel)
        if unique:
            return unique
        return sel
    return ""


def _best_date_selector(container: Tag, title_selector: str) -> str:
    for sel in (
        ".search-index__header time",
        "time[datetime]",
        "time",
        ".item-date",
        ".date",
        ".timestamp",
        "small",
    ):
        el = container.select_one(sel)
        if el and el.get_text(strip=True):
            unique = _minimal_unique_selector(container, el)
            if len(container.select(unique)) == 1:
                return unique
            return sel.lstrip(".")

    title_el = container.select_one(title_selector) if title_selector else None
    if title_el:
        small = title_el.select_one("small")
        if small and small.get_text(strip=True):
            return f"{title_selector} small"

    return ""


def _best_description_selector(container: Tag) -> str:
    for sel in (
        ".underline a",
        ".underline",
        ".item-sub-title",
        ".summary",
        ".description",
        ".excerpt",
        ".deck",
        ".standfirst",
        "p.remove-outline",
        "p",
    ):
        try:
            el = container.select_one(sel)
        except Exception:
            continue
        if not el:
            continue
        text = el.get_text(" ", strip=True)
        if len(text) < 15:
            continue
        unique = _minimal_unique_selector(container, el)
        if container.select_one(unique) is el:
            return unique
        return sel
    return ""


def _best_image_selector(container: Tag) -> str:
    for img in container.select("img[src]"):
        src = img.get("src", "")
        if src and not src.startswith("data:"):
            unique = _minimal_unique_selector(container, img)
            if len(container.select(unique)) == 1:
                return unique
            return "img"
    return ""


def infer_field_selectors(html: str, container_selector: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(container_selector)
    if not container:
        return {
            "container": container_selector,
            "title": "",
            "detail_url": "a",
            "published_at": "",
            "description": "",
            "author": "",
            "image": "",
        }

    title_sel = _best_title_selector(container)
    return {
        "container": container_selector,
        "title": title_sel,
        "detail_url": _best_link_selector(container),
        "published_at": _best_date_selector(container, title_sel),
        "description": _best_description_selector(container),
        "author": "",
        "image": _best_image_selector(container),
    }


def clean_title_text(raw: Optional[str], date_text: Optional[str] = None) -> Optional[str]:
    if not raw:
        return None
    title = raw.strip()
    if date_text:
        title = title.replace(date_text.strip(), "", 1).strip()
    title = re.sub(r"^\d{1,2}:\d{2}\s*(ET|PT|CT|UTC|GMT)\s*", "", title, flags=re.I)
    return title or None
