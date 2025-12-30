import json
import logging
import sys
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import html2text
from difflib import SequenceMatcher
import pdb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extract_sections_html_class.log',
                            mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class SECDocumentProcessor:
    def __init__(self, url: str, toc_path: str):
        """
        Initialize the SEC document processor.

        Args:
            url: URL of the SEC document
            toc_path: Path to the table of contents JSON file
        """
        self.url = url
        self.toc_path = toc_path
        self.entries = []
        self.all_text = ""
        self.any_entry_has_page_reference = False

    def remove_page_references(self, text: str) -> str:
        """
        Remove page references from text like "(see page 25)", "(page 25)", "(pg. 25)", etc.
        """
     # Remove patterns like (see page X), (page X), (pg. X), (p. X), etc.
        text = re.sub(r'\s*\([^)]*(?:page|pg|p\.)\s*\d+[^)]*\)',
                      '', text, flags=re.IGNORECASE)
        # Also remove any remaining standalone parenthetical references at the end
        text = re.sub(r'\s*\([^)]*\)\s*$', '', text)
        return text.strip()

    def preprocess_title(self, title: str) -> str:
        """
        Preprocess a title for comparison.

        Args:
            title: The title to preprocess

        Returns:
            Preprocessed title
        """

        # Remove punctuation like ; , . :
        title = re.sub(r'[;,:.]', '', title)
        # Lowercase and strip extra whitespace
        title = title.lower().strip()
        # Collapse multiple spaces
        title = re.sub(r'\s+', ' ', title)
        return title

    def is_similar(self, title1: str, title2: str, threshold=0.95) -> bool:
        """
        Check if two titles are similar.

        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (default: 0.95)

        Returns:
            True if titles are similar, False otherwise
        """
        t1 = self.preprocess_title(title1)
        t2 = self.preprocess_title(title2)
        ratio = SequenceMatcher(None, t1, t2).ratio()
        if title2.lower() == "background of the merger":
            logger.info(f"t1: {t1}")
            logger.info(f"t2: {t2}")
            logger.info(f"ratio: {ratio}")
        return ratio >= threshold

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing invisible characters and normalizing whitespace.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove zero-width spaces and similar invisible chars
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        # Strip whitespace and convert to lowercase
        return text.strip().lower()

    def extract_annex_title(self, title: str) -> str:
        """
        Extract annex title from a full title.

        Args:
            title: Full title

        Returns:
            Extracted annex title
        """
        match = re.match(
            r'(Annex\s+[A-Z]+(?:-\d+)?)(?:\s|:|-|—|–|$)', title, re.IGNORECASE)
        return match.group(1).strip() if match else title.strip()

    def is_pipe_separated(self, line: str) -> bool:
        """
        Check if a line contains pipe separators.

        Args:
            line: Line to check

        Returns:
            True if the line contains pipe separators, False otherwise
        """
        return "|" in line

    def find_title_in_text(self, lines: list[str], title: str, currentEntry: int = 0, skip_first_n: int = 0) -> Optional[int]:
        """
        Find the exact line number where a title appears.

        Args:
            lines: List of text lines
            title: Title to find
            currentEntry: Current entry index for context
            skip_first_n: Number of initial matches to skip (default: 0)

        Returns:
            Line number (0-based) or None if not found
        """
        if not title:
            logger.warning("Received None as title in find_title_in_text")
            return None

        title_clean = title.strip().lower()
        title_clean = self.remove_page_references(title_clean)
        title_clean = title_clean.replace(":", "")
        title_clean = title_clean.replace("-", " ")

        if "annex" in title_clean:
            logger.info(f"title_clean annex before: {title_clean}")
            title_clean = self.extract_annex_title(title_clean)
            logger.info(f"title_clean annex after: {title_clean}")

        matches_found = 0

        for i, line in enumerate(lines):
            if self.is_pipe_separated(line):
                logger.info(f"line is pipe_separated: {line}")
                continue

            line_text = line.strip()
            line_text = self.remove_page_references(line_text)
            line_text = line_text.replace(":", "")
            line_text = line_text.replace("-", " ")
            logger.info(f"line_text: {line_text} {i}")
            if not self.any_entry_has_page_reference:
                logger.info(
                    f"line_has_page_reference: {line_text},{self.any_entry_has_page_reference}")
                line_text = re.sub(r'\s*\([^)]*\)', '', line_text)
            line_text = line_text.replace(":", "")
            if not line_text:
                continue

            # Track if this line matched (even if we skip it)
            line_matched = False

            # Case 1: Single-line match
            if "annex" in title_clean:
                line_text_clean = self.clean_text(line_text)
                logger.info(f"title_clean annex2: {title_clean}")
                logger.info(f"line_text annex: {line_text_clean}")

                # Use exact match after cleaning
                if line_text_clean == title_clean:
                    matches_found += 1
                    line_matched = True
                    if matches_found > skip_first_n:
                        logger.info(f"Found title annex '{title}' at line {i}")
                        return i+1
                    else:
                        logger.info(
                            f"Skipping first occurrence of '{title}' at line {i} (match {matches_found})")
                        continue  # Move to next line after skipping

            if not line_matched and line_text.lower() in title_clean and line_text.lower() == title_clean:
                matches_found += 1
                line_matched = True
                if matches_found > skip_first_n:
                    logger.info(f"Found title1 '{title}' at line {i}")
                    return i+1
                else:
                    logger.info(
                        f"Skipping first occurrence of '{title}' at line {i} (match {matches_found})")
                    continue  # Move to next line after skipping

            # Case 2: Multi-line match up to 6 lines
            if not line_matched:
                combined = line_text

                for j in range(1, 14):  # combine up to 5 more lines (total 6)
                    if i + j >= len(lines):
                        break
                    next_line = lines[i + j].strip()
                    combined += f" {next_line}" if len(next_line) > 0 else ""
                    combined_clean = combined.strip().lower()
                    logger.info(f"combined_clean: {combined_clean}")

                    if combined_clean in title_clean:
                        if combined_clean == title_clean:
                            matches_found += 1
                            line_matched = True
                            if matches_found > skip_first_n:
                                logger.info(
                                    f"Found title2 '{title}' at line {i}")
                                return i
                            else:
                                logger.info(
                                    f"Skipping first occurrence of '{title}' at line {i} (match {matches_found})")
                                break  # Break inner loop, then continue outer loop
                    else:
                        logger.info(
                            f" combined_clean : '{combined_clean}' is not in title_clean : '{title_clean}' at line {i}")

                        break

                # If we found a match in multi-line but skipped it, move to next line
                if line_matched:
                    continue

            if not line_matched and self.is_similar(line_text, title_clean):
                matches_found += 1
                line_matched = True
                if matches_found > skip_first_n:
                    logger.info(f"Found title4 '{title}' at line {i}")
                    return i+1
                else:
                    logger.info(
                        f"Skipping first occurrence of '{title}' at line {i} (match {matches_found})")
                    continue  # Move to next line after skipping

            if not line_matched and currentEntry != 0:
                logger.info(f"line_text in is_similar: {line_text}")
                logger.info(f"title_clean in is_similar: {title_clean}")
                if self.preprocess_title(line_text) in self.preprocess_title(title_clean):
                    test_combined = line_text
                    best_ratio = 0
                    best_index = None

                    for j in range(1, 14):  # combine up to 14 more lines
                        if i + j >= len(lines):
                            break
                        next_line = lines[i + j].strip()
                        test_combined += f" {next_line}" if len(
                            next_line) > 0 else ""
                        test_combined_clean = test_combined.strip().lower()

                        # Calculate similarity ratio
                        ratio = SequenceMatcher(
                            None, test_combined_clean, title_clean).ratio()
                        logger.info(f"Testing combination at index {j}")
                        logger.info(f"Combined text: {test_combined_clean}")
                        logger.info(f"Similarity ratio: {ratio}")
                        logger.info(
                            f"test_combined_clean: {test_combined_clean}")
                        logger.info(f"title_clean: {title_clean}")

                        if ratio >= 0.95 and ratio > best_ratio:
                            best_ratio = ratio
                            best_index = i
                            logger.info(
                                f"New best match found at index {best_index} with ratio {best_ratio}")

                    if best_index is not None:
                        matches_found += 1
                        line_matched = True
                        if matches_found > skip_first_n:
                            logger.info(
                                f"Returning best match at index {best_index} with ratio {best_ratio}")
                            return best_index
                        else:
                            logger.info(
                                f"Skipping first occurrence of '{title}' at line {best_index} (match {matches_found})")
                            continue  # Move to next line after skipping

        logger.info(f"Not found title: '{title}'")
        return None

    def roman_to_int(self, roman):
        """
        Convert a Roman numeral to an integer.

        Args:
            roman: Roman numeral

        Returns:
            Integer value or None if not a valid Roman numeral
        """
        roman_numerals = {
            'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10,
            'XL': 40, 'L': 50, 'XC': 90, 'C': 100,
            'CD': 400, 'D': 500, 'CM': 900, 'M': 1000
        }
        i = 0
        num = 0
        roman = str(roman).upper()
        while i < len(roman):
            if i+1 < len(roman) and roman[i:i+2] in roman_numerals:
                num += roman_numerals[roman[i:i+2]]
                i += 2
            elif roman[i] in roman_numerals:
                num += roman_numerals[roman[i]]
                i += 1
            else:
                # not a roman numeral
                return None
        return num

    def page_to_int(self, page):
        """
        Convert a page number (numeric or Roman) to an integer.

        Args:
            page: Page number

        Returns:
            Integer page number or None if conversion fails
        """
        if page is None:
            return None
        page = str(page).strip()
        if page.isdigit():
            return int(page)
        roman = self.roman_to_int(page)
        if roman is not None:
            return roman
        return None

    def extract_content_between_titles(self, text: str, current_title: str, next_title: Optional[str] = None,
                                       next_of_next_title: Optional[str] = None, currentEntry: int = 0,
                                       current_page: Optional[str] = None, next_page: Optional[str] = None,
                                       next_of_next_page: Optional[str] = None) -> Optional[Tuple[str, str, Optional[str]]]:
        """
        Extract content between two titles in the text.

        Args:
            text: Full document text
            current_title: Current title to find
            next_title: Next title (boundary)
            next_of_next_title: Title after next title (fallback boundary)
            currentEntry: Current entry index
            current_page: Current page number
            next_page: Next page number
            next_of_next_page: Page number after next page

        Returns:
            Tuple of (extracted content, remaining text, actual_end_title_used) or None if title not found.
            actual_end_title_used indicates which title was actually used as the boundary (next_title or next_of_next_title).
            If None, it means next_title was used or there was no next title.
        """
        logger.info(f"current_title: {current_title}")
        logger.info(f"next_title: {next_title}")
        logger.info(f"current_page: {current_page}")
        logger.info(f"next_page: {next_page}")
        logger.info(f"next_of_next_page: {next_of_next_page}")

        lines = text.split('\n')

        # Skip first occurrence for the first entry (index 0) to avoid table of contents
        skip_first = 1 if currentEntry == 0 else 0
        start_line = self.find_title_in_text(
            lines, current_title, currentEntry, skip_first_n=skip_first)

        logger.info(
            f"start_line: {start_line}, current_title: {current_title}, skip_first: {skip_first}")

        if start_line is None:
            return None

        content, remaining_text = "", ""
        end_line = start_line
        actual_end_title_used = None  # Track which title was actually used as boundary

        # If there's a next title, find its position
        if next_title:

            # if next_title == "The Go-Shop Period — Solicitation of Other Offers":
            #     pdb.set_trace()
            # Search for next title starting from after the current title to avoid finding earlier occurrences
            remaining_lines_for_next = lines[start_line +
                                             1:] if start_line is not None else lines
            next_title_line = self.find_title_in_text(
                remaining_lines_for_next, next_title, currentEntry)
            end_line = (
                start_line + 1 + next_title_line) if next_title_line is not None else None
            if end_line is not None:
                actual_end_title_used = next_title  # next_title was found and used

            cp_int = self.page_to_int(current_page)
            np_int = self.page_to_int(next_page)
            if cp_int is not None and np_int is not None:
                diff = np_int - cp_int
                content_aprox_length_max = diff * 1500 if diff >= 1 else 1500
                content_aprox_length_min = diff * 300 if diff >= 1 else 300
            else:
                diff = 1
                content_aprox_length_max = None
                content_aprox_length_min = None

            logger.info(
                f"content_aprox_length_max: {content_aprox_length_max}, diff: {diff}, min length: {content_aprox_length_min if content_aprox_length_min else 'N/A'}")

            # Try to find a better end_line if initial one is not satisfactory
            for _ in range(2):  # Limit attempts to avoid infinite loops
                if end_line is not None:
                    content = '\n'.join(lines[start_line+1:end_line-1]).strip()
                    content_length = len(content.split())
                    logger.info(
                        f"content_length: {content_length}, max-min: {content_aprox_length_max}, {content_aprox_length_min}")
                    logger.info(f"next_title find: {next_title}")

                    # If content is too long, try next_of_next_title
                    if content_aprox_length_max is not None and content_length > content_aprox_length_max:
                        logger.info(
                            f"Content too long ({content_length} > {content_aprox_length_max}), trying next_of_next_title")
                        remaining_lines_for_next_of_next = lines[start_line +
                                                                 1:] if start_line is not None else lines
                        next_of_next_title_line = self.find_title_in_text(
                            remaining_lines_for_next_of_next, next_of_next_title, currentEntry)
                        end_line = (
                            start_line + 1 + next_of_next_title_line) if next_of_next_title_line is not None else None
                        if end_line is not None:
                            # next_of_next_title was used instead
                            actual_end_title_used = next_of_next_title
                        continue

                    # If content is too short and pages are far apart, search further
                    if content_aprox_length_min is not None and diff > 3 and content_length < content_aprox_length_min:
                        logger.info(
                            f"Content too short ({content_length} < {content_aprox_length_min}) and pages far apart (diff={diff}), likely false match")
                        remaining_lines = lines[end_line:]
                        next_occurrence = self.find_title_in_text(
                            remaining_lines, next_title, currentEntry)
                        if next_occurrence is not None:
                            end_line += next_occurrence
                            logger.info(
                                f"Found better match for next title at line {end_line}")
                            continue
                        else:
                            logger.info(
                                f"No better match found, trying next_of_next_title")
                            remaining_lines_for_next_of_next = lines[start_line +
                                                                     1:] if start_line is not None else lines
                            next_of_next_title_line = self.find_title_in_text(
                                remaining_lines_for_next_of_next, next_of_next_title, currentEntry)
                            end_line = (
                                start_line + 1 + next_of_next_title_line) if next_of_next_title_line is not None else None
                            if end_line is not None:
                                # next_of_next_title was used instead
                                actual_end_title_used = next_of_next_title
                            continue
                    # Good match found
                    break
                else:
                    # Try next_of_next_title if not found
                    remaining_lines_for_next_of_next = lines[start_line +
                                                             1:] if start_line is not None else lines
                    next_of_next_title_line = self.find_title_in_text(
                        remaining_lines_for_next_of_next, next_of_next_title, currentEntry)
                    end_line = (
                        start_line + 1 + next_of_next_title_line) if next_of_next_title_line is not None else None
                    if end_line is not None:
                        # next_of_next_title was used instead
                        actual_end_title_used = next_of_next_title

            # Final assignment of content and text after, once only
            if end_line is not None:
                content = '\n'.join(lines[start_line+1:end_line-1]).strip()
                remaining_text = '\n'.join(lines[end_line-1:]).strip()
            else:
                content = '\n'.join(lines[start_line+1:]).strip()
                remaining_text = ""

        else:
            content = '\n'.join(lines[start_line+1:]).strip()
            remaining_text = ""

        return content, remaining_text, actual_end_title_used

    def fetch_sec_document(self) -> str:
        """
        Fetch SEC document from URL and convert HTML to clean text.

        Returns:
            Cleaned document text
        """
        try:
            # Headers to mimic a browser request
            headers = {
                "User-Agent":
                "MNA-Finder/1.0 (https://teqnodux.com; contact: ashish.kachadiya@teqnodux.com)",
                'Accept': "application/json",
            }

            # Fetch the document
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "hr"]):
                script.decompose()

            # Remove inline formatting (e.g., <b>, <u>, <i>, <span>)
            for inline_tag in soup(["b", "strong", "u", "i", "em", "span"]):
                inline_tag.unwrap()

            # First convert HTML to markdown-style text
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_tables = False
            h.body_width = 0  # Don't wrap text

            markdown_text = h.handle(str(soup))

            # Clean up the text
            # Replace multiple newlines with double newlines
            text = re.sub(r'\n{3,}', '\n\n', markdown_text)

            lines = text.splitlines()
            cleaned_lines = []

            for line in lines:
                stripped = line.strip()

                # Skip empty lines
                if not stripped:
                    continue

                # Normalize for keyword comparison
                normalized = re.sub(r"\s+", " ", stripped).upper()

                # Remove known page markers like 'Page 3'
                if re.fullmatch(r'Page\s+\d{1,4}', stripped, re.IGNORECASE):
                    continue

                # Remove visual dividers (e.g., '***', '-----')
                if re.fullmatch(r'[*\-_=~]{3,}', stripped):
                    continue

                # Remove visual dividers (e.g., '* * *', '- - - - -')
                if re.fullmatch(r'([*\-_=~] ?){3,}', stripped):
                    continue

                # Remove hidden characters
                normalized = stripped.replace(
                    '\u200b', '').replace('\u00a0', '')
                normalized = re.sub(
                    r"\s+", " ", normalized).upper().strip(' .:-')

                # Now compare
                if any(normalized.startswith(header) for header in {
                    "TABLE OF CONTENTS", "INDEX", "EXHIBIT INDEX"
                }):
                    continue

                cleaned_lines.append(stripped)

            return "\n\n".join(cleaned_lines).strip()

        except Exception as e:
            logger.error(f"Error fetching SEC document: {e}")
            raise

    def load_toc(self) -> None:
        """
        Load table of contents and create flat list of entries.
        """
        # Load table of contents
        with open(self.toc_path, 'r') as f:
            toc = json.load(f)

        # Create flat list of entries
        self.entries = []
        for section in toc:
            # Handle missing page-no key
            # Default to page 1 if missing
            page_no = section.get('page-no', '1')
            if 'page-no' not in section:
                logger.warning(
                    f"Section '{section['title']}' is missing page-no, defaulting to page 1")
            self.entries.append({
                'title': section['title'],
                'page': page_no,
                'content': '',
                'is_main_section': True,
                'original_title': section['title']
            })
            if 'subsection' in section:
                for subsection in section['subsection']:
                    # Handle missing page-no key in subsections
                    # Default to parent page if missing
                    sub_page_no = subsection.get('page-no', page_no)
                    if 'page-no' not in subsection:
                        logger.warning(
                            f"Subsection '{subsection['title']}' is missing page-no, defaulting to parent page {page_no}")
                    self.entries.append({
                        'title': subsection['title'],
                        'page': sub_page_no,
                        'content': '',
                        'is_main_section': False,
                        'parent_title': section['title']
                    })

    def process_document(self) -> List[Dict[str, Any]]:
        """
        Process SEC document by extracting text and finding exact title matches.

        Returns:
            List of sections with content
        """
        try:
            # Load table of contents
            self.load_toc()

            # Check if any entry has a page reference in the title like "(page 20)"
            self.any_entry_has_page_reference = any(
                re.search(r'\([^)]+\)\s*$', entry.get('title', ''),
                          re.IGNORECASE) is not None
                for entry in self.entries)
            # self.any_entry_has_page_reference = any(
            #     re.search(r'\(page\s+\d+\)', entry.get('title', ''),
            #               re.IGNORECASE) is not None
            #     for entry in self.entries)
            logger.info(
                f"Any entry has page reference in title: {self.any_entry_has_page_reference}")

            # Fetch and extract text from SEC document
            self.all_text = self.fetch_sec_document()
            logger.info(f"all_text: {self.all_text}")

            logger.info(f"Successfully fetched and converted SEC document")

            # Track which entries have been processed or skipped
            processed_entries = set()
            i = 0

            # Process each entry
            while i < len(self.entries):
                # Skip entries that were already processed or bypassed
                if i in processed_entries:
                    i += 1
                    continue

                entry = self.entries[i]
                logger.info(f"Processing entry: {entry['title']}")

                # Get next entry for boundary (skip already processed entries)
                next_entry_idx = i + 1
                while next_entry_idx < len(self.entries) and next_entry_idx in processed_entries:
                    next_entry_idx += 1
                next_entry = self.entries[next_entry_idx] if next_entry_idx < len(
                    self.entries) else None
                next_title = next_entry['title'] if next_entry else None
                next_page = next_entry['page'] if next_entry else None

                # Get next of next entry (skip already processed entries)
                next_of_next_entry_idx = next_entry_idx + 1
                while next_of_next_entry_idx < len(self.entries) and next_of_next_entry_idx in processed_entries:
                    next_of_next_entry_idx += 1
                next_of_next_entry = self.entries[next_of_next_entry_idx] if next_of_next_entry_idx < len(
                    self.entries) else None
                next_of_next_title = next_of_next_entry['title'] if next_of_next_entry else None
                next_of_next_page = next_of_next_entry['page'] if next_of_next_entry else None

                # Extract content between current title and next title
                result = self.extract_content_between_titles(
                    self.all_text, entry['title'], next_title, next_of_next_title, i,
                    entry['page'], next_page, next_of_next_page)

                # logger.info(f"result: {result}")

                if result:
                    content, text, actual_end_title_used = result
                    self.all_text = text
                    entry['content'] = content
                    processed_entries.add(i)  # Mark current entry as processed

                    # If we used next_of_next_title instead of next_title, skip the next entry
                    if actual_end_title_used == next_of_next_title and next_entry:
                        logger.info(
                            f"Skipping entry '{next_entry['title']}' because it was not found and we used '{next_of_next_title}' instead")
                        # Mark next entry as skipped
                        processed_entries.add(next_entry_idx)
                        # Set empty content for skipped entry
                        next_entry['content'] = ''
                else:
                    logger.warning(f"No content found for {entry['title']}")
                    # Mark as processed even if no content found
                    processed_entries.add(i)

                i += 1

            # Convert back to hierarchical structure
            result = []
            current_main_section = None

            for entry in self.entries:
                if entry['is_main_section']:
                    current_main_section = {
                        'title': entry['original_title'],
                        'page-no': entry['page'],
                        'content': entry['content'],
                        'subsection': []
                    }
                    result.append(current_main_section)
                else:
                    if current_main_section:
                        current_main_section['subsection'].append({
                            'title': entry['title'],
                            'page-no': entry['page'],
                            'content': entry['content']
                        })

            # Save results
            # output_filename = f"sections_with_content_html_{os.path.basename(self.toc_path).replace('table_of_contents_new_', '').replace('.json', '')}.json"
            # with open(output_filename, 'w', encoding='utf-8') as f:
            #     json.dump(result, f, indent=2, ensure_ascii=False)

            return result

        except Exception as e:
            logger.error(f"Error processing SEC document: {e}")
            raise


if __name__ == "__main__":
    # Example SEC document URL
    sec_url = "https://www.sec.gov/Archives/edgar/data/718877/000110465922025210/tm225196-3_prem14a.htm#tP1AO"
    toc_path = 'new_table_of_content/table_of_contents_new_Activision.json'

    try:
        processor = SECDocumentProcessor(sec_url, toc_path)
        sections = processor.process_document()
        print("\nExtracted sections with content:")
        print(json.dumps(sections, indent=2))
        print(
            f"\nSections with content have been saved to sections_with_content_html_{os.path.basename(toc_path).replace('table_of_contents_new_', '').replace('.json', '')}.json")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
