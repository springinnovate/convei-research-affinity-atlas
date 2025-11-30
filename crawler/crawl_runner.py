"""
This script crawls a set of urls storing the information in an sqlite database
based off of the input YAML configuration provided. The YAML configuration
example can be seen at `example_crawler.yaml` in this directory.

To run this inside the Docker environment from the current directory:

  docker build -t crawl_env .
  docker run --rm -it -v %CD%:/app crawl_env

Then, inside the container:

  python crawl_runner.py path/to/configuration.yaml
"""

from urllib.parse import urlparse, urljoin
import argparse
import logging
import os

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests
import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def parse_crawler_config(path):
    """
    Parse a crawler YAML configuration file.

    The YAML file is expected to have a single top-level key whose name matches
    the stem of the YAML filename. For example, for ``example_crawler.yaml``,
    the top-level key must be ``example_crawler``. Within that section, this
    function reads crawl settings including start URLs, allowed scopes, page
    limits, and output database path.

    Args:
        path (str): Filesystem path to the YAML configuration file.

    Returns:
        dict: A dictionary containing:
            - name (str): The configuration section name (filename stem).
            - start_urls (list[str]): Seed URLs for the crawl.
            - allowed_scopes (list[str]): Allowed domains/hosts/URL prefixes.
            - max_pages (int | None): Maximum number of pages to crawl, or
              ``None`` if the configuration uses ``'no_limit'``.
            - sqlite_path (str): Path to the SQLite database file.

    Raises:
        ValueError: If the YAML file does not contain a top-level key matching
            the filename stem.
        KeyError: If required keys (e.g., ``start_urls``, ``allowed_scopes``,
            ``output.sqlite_path``) are missing from the configuration.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem not in data:
        raise ValueError(
            f"Expected a section named the filename stem '{stem}' but none was found in {path}"
        )
    section = data[stem]
    max_pages_raw = section.get("max_pages")
    if isinstance(max_pages_raw, str) and max_pages_raw.lower() == "no_limit":
        max_pages = None
    else:
        max_pages = max_pages_raw
    return {
        "name": stem,
        "start_urls": section["start_urls"],
        "allowed_scopes": section["allowed_scopes"],
        "max_pages": max_pages,
        "sqlite_path": section["output"]["sqlite_path"],
    }


def fetch_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html


def crawl_from_config(config):
    flaresolverr_url = os.environ.get("FLARESOLVERR_URL")
    frontier = list(config["start_urls"])
    visited = set(frontier)
    allowed_scopes = config["allowed_scopes"]
    max_pages = config["max_pages"]
    if max_pages is None:
        max_pages = float("inf")
    pages_crawled = 0
    while frontier and pages_crawled < max_pages:
        url = frontier.pop()
        html = fetch_rendered_html(url)
        pages_crawled += 1
        logging.debug(html)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if link in visited:
                continue
            parsed = urlparse(link)
            netloc = parsed.netloc
            allowed = False
            for scope in allowed_scopes:
                if scope.startswith("http://") or scope.startswith("https://"):
                    if link.startswith(scope):
                        allowed = True
                        break
                else:
                    if netloc.endswith(scope):
                        allowed = True
                        break
            if not allowed:
                continue
            visited.add(link)
            frontier.append(link)
    logging.debug(frontier)
    return visited


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse a crawler YAML configuration and print the "
            "resolved settings."
        )
    )
    parser.add_argument(
        "config_path",
        help=(
            "Path to the YAML configuration file defining the "
            "crawler settings."
        ),
    )
    args = parser.parse_args()
    config = parse_crawler_config(args.config_path)
    logging.debug(config)
    crawl_from_config(config)


if __name__ == "__main__":
    main()
