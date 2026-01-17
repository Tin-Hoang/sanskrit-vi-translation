from abc import ABC, abstractmethod
from typing import List, Dict
import requests
from bs4 import BeautifulSoup


class BaseCrawler(ABC):
    def __init__(self, source_name: str):
        self.source_name = source_name

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Downloads the content from the URL."""
        pass

    @abstractmethod
    def parse(self, html_content: str) -> List[Dict[str, str]]:
        """Parses the HTML content and returns a list of data dictionaries."""
        pass

    def run(self, url: str) -> List[Dict[str, str]]:
        """Executes the crawl process for a given URL."""
        print(f"[{self.source_name}] Fetching {url}...")
        try:
            content = self.fetch(url)
            data = self.parse(content)
            print(f"[{self.source_name}] Extracted {len(data)} items.")
            return data
        except Exception as e:
            print(f"[{self.source_name}] Error: {e}")
            return []

    def _get_soup(self, html_content: str) -> BeautifulSoup:
        return BeautifulSoup(html_content, "html.parser")
