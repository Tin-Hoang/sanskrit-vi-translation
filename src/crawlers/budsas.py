from typing import List, Dict
import requests
from .base import BaseCrawler
from bs4 import BeautifulSoup
import re


class BudsasCrawler(BaseCrawler):
    def __init__(self):
        super().__init__("Budsas")

    def fetch(self, url: str) -> str:
        # Budsas.net is simple, requests is enough.
        # It uses encoding windows-1252 or utf-8, need to check.
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Auto-detect encoding, usually utf-8 for budsas.net now (based on meta tag)
            response.encoding = response.apparent_encoding
            return response.text
        except Exception as e:
            print(f"Request failed: {e}")
            raise

    def parse(self, html_content: str) -> List[Dict[str, str]]:
        soup = self._get_soup(html_content)
        data = []

        # Budsas structure helps: mostly text in <p> or <font> tags.
        # For Heart Sutra specifically:
        # Typically looks like: block of Sanskrit (sometimes), block of Vietnamese.
        # Or interleaved.

        # Let's extract all semantic text blocks and cleanup.
        # We target the main body.

        # Heuristic: Remove navigation tables.
        for table in soup.find_all("table"):
            table.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            if len(line) < 20:
                continue  # Skip short headers/nav junk

            data.append({"raw_text": line, "source_url": "budsas.net"})

        return data
