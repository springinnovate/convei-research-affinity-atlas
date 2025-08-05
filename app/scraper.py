# src/webmapper/scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    " AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/136.0.0.0 Safari/537.36"
}


class WebScraper:
    def __init__(self, base_url: str, max_depth: int = 1):
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.max_depth = max_depth
        self.base_domain = urlparse(base_url).netloc

    def scrape(self) -> List[dict]:
        results = []
        self._scrape_recursive(self.base_url, 0, results)
        return results

    def _scrape_recursive(self, url: str, depth: int, results: List[dict]):
        if depth > self.max_depth or url in self.visited_urls:
            return

        self.visited_urls.add(url)
        print(f"Scraping: {url} (Depth: {depth})")

        try:
            response = requests.get(url, timeout=10, headers=HEADERS)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {url} with error: {e}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.get_text(separator="\n", strip=True)

        results.append(
            {
                "url": url,
                "content": content,
            }
        )

        links = self._extract_links(soup, url)

        for link in links:
            self._scrape_recursive(link, depth + 1, results)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        links = set()
        for anchor in soup.find_all("a", href=True):
            link = urljoin(base_url, anchor["href"])
            if self._is_valid_link(link):
                links.add(link)
        return links

    def _is_valid_link(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme.startswith("http")
            and parsed.netloc == self.base_domain
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape a website recursively."
    )
    parser.add_argument("url", help="The URL to scrape")
    parser.add_argument(
        "--depth", type=int, default=1, help="Depth of recursive scraping"
    )
    args = parser.parse_args()

    scraper = WebScraper(args.url, max_depth=args.depth)
    scraped_data = scraper.scrape()

    for page in scraped_data:
        print(
            f'\n{"="*80}\nURL: {page["url"]}\n{"-"*80}\n{page["content"][:1000]}...\n'
        )
