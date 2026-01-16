from typing import List, Dict
import cloudscraper
from .base import BaseCrawler
import re


class ThuvienhoasenCrawler(BaseCrawler):
    def __init__(self):
        super().__init__("Thuvienhoasen")
        self.scraper = cloudscraper.create_scraper()

    def fetch(self, url: str) -> str:
        try:
            response = self.scraper.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Cloudscraper failed: {e}")
            raise

    def parse(self, html_content: str) -> List[Dict[str, str]]:
        soup = self._get_soup(html_content)
        data = []

        # Specific logic for Heart Sutra page on Thuvienhoasen varies,
        # but often text is in paragraphs.
        # This is a heuristic based on common layouts.
        # We look for lines that might be pairs.
        # Given the irregularity, we might need manual alignment later.
        # For this demo, we will try to find the main content div.

        # Note: This is a simplified extraction.
        # A robust one would need specific selectors for each page layout.

        main_content = soup.find("div", class_="pd-body")  # Common class in Joomla/CMS
        if not main_content:
            main_content = soup.find("div", id="content")

        if not main_content:
            print("Could not find main content div.")
            return []

        # Simple extraction: splitting by lines and filtering
        # Ideally, we look for patterns like "Sanskrit:" or similar,
        # but often it's just raw text.

        # Strategy: Extract all text, cleanup, and return raw chunks for now.
        # The user can then manually align or we can improve logic if specific structure exists.

        text = main_content.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Heuristic: try to identify Sanskrit (Latin script) vs Vietnamese
        # This is very rough.

        for line in lines:
            # Skip very short lines
            if len(line) < 10:
                continue

            data.append({"raw_text": line, "source_url": "thuvienhoasen.org"})

        return data
