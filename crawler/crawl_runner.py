"""Runner for web crawls.

This script crawls a set of urls storing the information in an sqlite database
based off of the input YAML configuration provided. The YAML configuration
example can be seen at `example_crawler.yaml` in this directory.

To run this inside the Docker environment from the current directory:

  docker build -t crawl_env .
  docker run --rm -it -v %CD%:/app crawl_env

Then, inside the container:

  python crawl_runner.py path/to/configuration.yaml
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import urlparse, urljoin, quote, urlsplit, urlunsplit
import argparse
import logging
import os
import threading
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml

from models import Base, Page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def parse_crawler_config(yaml_config_path):
    """Parse a crawler YAML configuration file.

    The YAML file is expected to have a single top-level key whose name matches
    the stem of the YAML filename. For example, for ``example_crawler.yaml``,
    the top-level key must be ``example_crawler``. Within that section, this
    function reads crawl settings including start URLs, allowed scopes, page
    limits, and output database path.

    Args:
        yaml_config_path (str): Filesystem path to the YAML configuration file.

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
    with open(yaml_config_path, "r") as f:
        data = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(yaml_config_path))[0]
    if stem not in data:
        raise ValueError(
            f"Expected a section named the filename stem '{stem}' but none "
            f"was found in {yaml_config_path}"
        )
    section = data[stem]
    max_pages_raw = section.get("max_pages")
    if isinstance(max_pages_raw, str) and max_pages_raw.lower() == "no_limit":
        max_pages = None
    else:
        max_pages = max_pages_raw

    raw_sqlite_path = section["sqlite_path"]
    if os.path.isabs(raw_sqlite_path):
        sqlite_path = raw_sqlite_path
    else:
        sqlite_path = str(Path(yaml_config_path).parent / raw_sqlite_path)

    return {
        "name": stem,
        "start_urls": section["start_urls"],
        "allowed_scopes": section["allowed_scopes"],
        "max_pages": max_pages,
        "sqlite_path": sqlite_path,
        "content_sections": section["content_sections"],
        "num_workers": section["num_workers"],
        "requests_per_second": section["requests_per_second"],
        "drop_elements": section["drop_elements"],
        "max_fetch_retries": section["max_fetch_retries"],
    }


def fetch_and_filter_rendered_html(
    browser, url, content_sections, drop_elements
):
    """Fetch rendered HTML for a URL and optionally extract and clean sections.

    Uses Playwright to load the page with JavaScript executed, then either:
    - returns the full rendered HTML if no content_sections are provided, or
    - returns a concatenation of only the elements matching the given CSS
      selectors in content_sections, with any sub-elements matching
      drop_elements removed.

    Args:
        browser: A Playwright :class:`Browser` instance to use for creating
            pages.
        url: Page URL to fetch.
        content_sections: Iterable of CSS selectors for sections to keep.
            If empty or falsy, the full rendered HTML is returned.
        drop_elements: Iterable of CSS selectors for elements to remove
            from within each selected content section.

    Returns:
        A string containing the rendered HTML or the concatenated, cleaned
        subset of the HTML defined by content_sections and drop_elements.
    """
    page = browser.new_page()
    try:
        page.goto(url, wait_until="networkidle")
        html = page.content()
        if not content_sections:
            return html
        soup = BeautifulSoup(html, "lxml")
        parts = []
        base_tag = soup.find("base", href=True)
        if base_tag:
            parts.append(str(base_tag))
        for selector in content_sections:
            for el in soup.select(selector):
                for drop_selector in drop_elements:
                    for d in el.select(drop_selector):
                        d.decompose()
                parts.append(str(el))
        return "\n".join(parts)
    finally:
        page.close()


def get_session(sqlite_path):
    """Create and return a SQLAlchemy session for the given SQLite database.

    Ensures that the directory for the SQLite file exists, initializes the
    database engine, creates all tables defined on the Base metadata, and
    returns a new session bound to that engine.

    Args:
        sqlite_path (str | pathlib.Path): Filesystem path to the SQLite
            database file.

    Returns:
        sqlalchemy.orm.Session: A new SQLAlchemy session bound to the
            initialized SQLite engine.
    """
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path}")
    Session = sessionmaker(bind=engine)
    return Session()


def normalize_url(url):
    scheme, netloc, path, query, fragment = urlsplit(url)

    path = quote(path, safe="/%")  # keep / and existing % encodings
    query = quote(query, safe="=&?/%")  # keep query separators and %
    # fragment is never sent to the server, but encoding anyway for consistency
    # like https://example.com/page#section1
    fragment = quote(fragment, safe="=%")

    return urlunsplit((scheme, netloc, path, query, fragment))


def crawl_from_config(config):
    """Crawl a small site frontier based on a configuration dict.

    Starts from the URLs in config["start_urls"], respects allowed scopes,
    and limits the number of pages crawled. For each page, rendered content
    is fetched via fetch_and_filter_rendered_html, cleaned according to the config, and
    stored. Links found in the cleaned HTML are filtered by allowed_scopes
    and added to the frontier.

    Config keys:
        start_urls (Iterable[str]): Initial URLs to seed the crawl.
        allowed_scopes (Iterable[str]): Domain or URL scope filters. Each
            entry can be:
            - a full URL prefix (starting with "http://" or "https://"),
              matched via startswith, or
            - a bare domain suffix (e.g. "example.org"), matched via
              netloc.endswith.
        max_pages (int or None): Maximum number of pages to crawl. If None,
            the crawl is unbounded.
        content_sections (Iterable[str]): CSS selectors passed to
            fetch_and_filter_rendered_html to extract desired page sections.
        drop_elements (Iterable[str]): CSS selectors passed to
            fetch_and_filter_rendered_html to remove unwanted sub-elements.

    Args:
        config: Configuration dictionary as described above.

    Returns:
        A tuple (visited, contents) where:
            visited: A set of all URLs that were popped from the frontier
                and attempted (whether or not they yielded links).
            contents: A dict mapping each visited URL to the corresponding
                cleaned HTML string returned by fetch_and_filter_rendered_html.
    """
    session_factory_path = config["sqlite_path"]
    allowed_scopes = config["allowed_scopes"]
    max_pages = config["max_pages"]
    if max_pages is None:
        max_pages = float("inf")
    num_workers = config["num_workers"]

    # this is the graph term `frontier` for the front nodes to be visited
    frontier = Queue()
    for start_url in config["start_urls"]:
        frontier.put(start_url)

    # here, `visited` is the graph terminology visited rather than url visited
    # consistent with `frontier`
    visited = set()
    pages_crawled = 0

    visited_lock = threading.Lock()
    pages_lock = threading.Lock()

    rate_lock = threading.Lock()
    recent_requests = deque()

    def _worker():
        """Run a single crawl worker loop.

        The worker repeatedly pulls URLs from the shared frontier queue, normalizes
        and de-duplicates them, fetches and renders HTML with Playwright, persists
        pages to the database, updates crawl status, and discovers and enqueues new
        in-scope URLs. It stops when the frontier is empty for a timeout or when
        the global max_pages limit is reached.

        This function relies on shared, nonlocal state such as the frontier queue,
        visitation tracking, rate-limiting structures, a SQLAlchemy session
        factory, and configuration values, and is intended to be executed
        concurrently in multiple threads or processes.

        Returns:
            None: The function runs until a termination condition is met.
        """
        nonlocal pages_crawled
        session = get_session(session_factory_path)
        with sync_playwright() as p:
            playwright_browser = p.chromium.launch(headless=True)
            while True:
                try:
                    page_url = frontier.get(timeout=5.0)
                except Empty:
                    logging.info("frontier is empty, quitting")
                    break
                with pages_lock:
                    if pages_crawled >= max_pages:
                        logging.info("max_pages reached quitting worker")
                        return
                page_url = normalize_url(page_url)
                logging.debug(page_url)
                with visited_lock:
                    if page_url in visited:
                        logging.debug("already visited page_url: %s", page_url)
                        continue
                    visited.add(page_url)
                logging.debug("visiting this page_url: %s", page_url)
                page = session.query(Page).filter_by(url=page_url).one_or_none()
                if page is None:
                    page = Page(
                        url=page_url,
                        html=None,
                        crawled_at=datetime.utcnow(),
                        status=Page.IN_PROGRESS,
                    )
                    session.add(page)
                    session.commit()
                if page.html is None:
                    attempts = 0
                    delay = config.get("initial_backoff", 1.0)
                    max_attempts = config["max_fetch_retries"]
                    html = None
                    error_list = []
                    while attempts < max_attempts:
                        attempts += 1

                        while True:
                            with rate_lock:
                                now = time.monotonic()
                                # drop all requests older than 1 second ago
                                while (
                                    recent_requests
                                    and now - recent_requests[0] >= 1.0
                                ):
                                    recent_requests.popleft()
                                # now recent_requests just has the number
                                # of requests < 1 second old
                                if (
                                    len(recent_requests)
                                    < config["requests_per_second"]
                                ):
                                    recent_requests.append(now)
                                    sleep_for = 0.0
                                    break
                                sleep_for = 1.0 - (now - recent_requests[0])
                            if sleep_for > 0:
                                logging.info(f"sleeping for {sleep_for}s")
                                time.sleep(sleep_for)

                        try:
                            html_candidate = fetch_and_filter_rendered_html(
                                playwright_browser,
                                page_url,
                                config["content_sections"] + ["base"],
                                config["drop_elements"],
                            )
                            if html_candidate is None or html_candidate == "":
                                raise ValueError("empty html")
                            html = html_candidate
                            break
                        except Exception as e:
                            msg = str(e).lower()
                            error_list.append(msg)
                            over_ping = (
                                "429" in msg
                                or "too many requests" in msg
                                or "rate limit" in msg
                                or "retry later" in msg
                            )
                            logging.exception(
                                f"error fetching {page_url} on attempt {attempts}: {e}",
                            )
                            if over_ping and attempts < max_attempts:
                                time.sleep(delay)
                                delay *= 2
                                continue
                            else:
                                break
                    if html is not None:
                        page.html = html
                        page.status = Page.SUCCESS
                    else:
                        page.status = f"{Page.ERROR}: " + "\n".join(error_list)
                        logging.error(page.status)
                    session.commit()
                    with pages_lock:
                        pages_crawled += 1
                else:
                    html = page.html

                if not html:
                    error_msg = (
                        f"{Page.ERROR}: no html stored for page_url: {page_url}"
                    )
                    page.status = error_msg
                    logging.error(error_msg)
                    continue

                soup = BeautifulSoup(html, "lxml")
                base_tag = soup.find("base")
                if base_tag and base_tag.has_attr("href"):
                    base_url = urljoin(page_url, base_tag["href"])
                else:
                    base_url = page_url

                for a in soup.find_all("a", href=True):
                    next_url = urljoin(base_url, a["href"])
                    parsed = urlparse(next_url)
                    host = parsed.hostname or ""
                    allowed = False
                    for scope in allowed_scopes:
                        if scope.startswith("http://") or scope.startswith(
                            "https://"
                        ):
                            if next_url.startswith(scope):
                                allowed = True
                                break
                        else:
                            if host == scope or host.endswith("." + scope):
                                allowed = True
                                break
                    if not allowed:
                        continue
                    with visited_lock:
                        if next_url in visited:
                            continue
                    with pages_lock:
                        if pages_crawled >= max_pages:
                            break
                    frontier.put(next_url)

    crawl_done = False

    def _monitor_frontier():
        """Periodically log crawl progress and an ETA based on frontier size and page rate.

        This function is intended to run in a separate thread. It samples the size of
        the shared frontier queue and the number of pages crawled so far, then logs
        the current rate (pages per second) and an estimated time to completion.
        The estimate is based on the recent rate and the remaining page budget.

        The function relies on external shared state:
          - frontier: queue of URLs pending crawl (must support qsize()).
          - pages_crawled: integer count of pages completed so far.
          - max_pages: maximum number of pages to crawl.
          - pages_lock: lock protecting pages_crawled.
          - crawl_done: boolean flag indicating that crawling should stop.

        The loop terminates when either crawl_done is set to True, or when the
        number of crawled pages reaches max_pages and the frontier is empty.

        Returns:
            None
        """
        last_pages = 0
        last_time = time.time()
        while not crawl_done:
            time.sleep(5.0)
            qsize = frontier.qsize()
            with pages_lock:
                done = pages_crawled
            now = time.time()
            delta_p = done - last_pages
            delta_t = now - last_time
            rate = delta_p / delta_t if delta_t > 0 else 0.0
            if max_pages == float("inf"):
                pages_total = qsize
                remaining = qsize
            else:
                pages_total = max_pages
                remaining = pages_total - done
            eta = remaining / rate if rate > 0 else None
            if eta is None:
                logging.info(
                    f"frontier size={qsize}, pages_crawled={done}/{pages_total}, rate={rate:.3f} pages/s, eta=unknown"
                )
            else:
                logging.info(
                    f"frontier size={qsize}, pages_crawled={done}/{pages_total}, rate={rate:.3f} pages/s, eta={eta:.1f}s"
                )
            last_pages = done
            last_time = now
            if done >= max_pages and qsize == 0:
                break

    monitor_thread = threading.Thread(target=_monitor_frontier, daemon=True)
    monitor_thread.start()

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            futures.append(executor.submit(_worker))
        for f in futures:
            f.result()
        crawl_done = True


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
    crawl_from_config(config)


if __name__ == "__main__":
    main()
