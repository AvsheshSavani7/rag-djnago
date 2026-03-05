"""HTML paragraph parser: clean text, detect sections, parse SEC filings."""

import re
from typing import List

from bs4 import BeautifulSoup

from .models import ParsedParagraph


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def wc(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text or ""))


def is_table_content(text: str) -> bool:
    if not text:
        return False
    nums = len(re.findall(r'\d', text))
    words = wc(text)
    if words > 0 and nums / max(words, 1) > 2:
        return True
    if text.count('|') > 5 or text.count('$') > 10:
        return True
    return False


def has_text_children(element) -> bool:
    text_tags = {'p', 'div', 'span', 'li', 'td'}
    for child in element.find_all(text_tags, recursive=False):
        if len(child.get_text(strip=True)) > 50:
            return True
    return False


SECTION_HEADER_PATTERNS = [
    r"^item\s+\d", r"^part\s+[iv]+", r"^note\s+\d",
    r"^(\(?\d+\)?\.?\s*)?(note|subsequent events)",
    r"^(financial statements|management.s discussion|risk factors)",
    r"^(quantitative|controls and procedures|legal proceedings)",
    r"^exhibit",
]


def detect_section(text: str, current_section: str) -> tuple:
    text_lower = text.lower().strip()
    if wc(text) < 20:
        for pattern in SECTION_HEADER_PATTERNS:
            if re.match(pattern, text_lower):
                return text[:120], True
    return current_section, False


def merge_mid_sentence_paragraphs(
    paragraphs: List[ParsedParagraph], max_merged_words: int = 500,
) -> List[ParsedParagraph]:
    if not paragraphs:
        return paragraphs
    merged = []
    for p in paragraphs:
        text = p.text.strip()
        if (text and text[0].islower() and merged
                and merged[-1].section == p.section
                and not merged[-1].is_header
                and (merged[-1].word_count + p.word_count) <= max_merged_words):
            merged[-1].text = merged[-1].text.rstrip() + ' ' + text
            merged[-1].word_count += p.word_count
        else:
            merged.append(p)
    for i, p in enumerate(merged):
        p.index = i
    return merged


def parse_html_to_paragraphs(html: str, min_words: int = 20) -> List[ParsedParagraph]:
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup.find_all(['script', 'style', 'noscript']):
        tag.decompose()

    paragraphs = []
    seen_texts = set()
    current_section = "Document Start"
    idx = 0

    for element in soup.find_all(['p', 'div', 'span', 'td', 'li',
                                   'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        if element.find_parent('thead'):
            continue
        if element.name in ('div', 'td') and has_text_children(element):
            continue

        text = element.get_text(separator=' ', strip=True)
        text = clean_text(text)
        if not text:
            continue

        new_section, is_header = detect_section(text, current_section)
        if is_header:
            current_section = new_section
            if wc(text) >= 3:
                sig = text[:100].lower()
                if sig not in seen_texts:
                    seen_texts.add(sig)
                    paragraphs.append(ParsedParagraph(
                        index=idx, text=text, section=current_section,
                        word_count=wc(text), is_header=True,
                    ))
                    idx += 1
            continue

        if wc(text) < min_words or is_table_content(text):
            continue

        sig = re.sub(r'\s+', ' ', text[:120].lower())
        if sig in seen_texts:
            continue
        seen_texts.add(sig)

        paragraphs.append(ParsedParagraph(
            index=idx, text=text, section=current_section, word_count=wc(text),
        ))
        idx += 1

    paragraphs = merge_mid_sentence_paragraphs(paragraphs)
    return paragraphs
