"""Webcrawler framework."""

import logging
import sys
import time
import asyncio
import re
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from database import SessionLocal
from models import WebpageContent
from llm_analyzer import analyze_entity_context

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)


async def extract_links(content, base_url, domain):
    href_links = re.findall(r'href=["\'](.*?)["\']', content, re.IGNORECASE)
    links = set()
    for link in href_links:
        abs_url = urljoin(base_url, link)
        if urlparse(abs_url).netloc == domain:
            links.add(abs_url)
    return links


CURRENTLY_PROCESSING = {}
CURRENTLY_PROCESSING_LOCK = asyncio.Lock()


async def fetch_page(url, page, db):
    webpage_record = (
        db.query(WebpageContent).filter(WebpageContent.url == url).first()
    )

    if not webpage_record:
        start = time.time()
        LOGGER.debug(f"attempting to fetch {url}")
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            LOGGER.warning(f"Timed out fetching {url}. Retrying with 'load'.")
            await page.goto(url, wait_until="load", timeout=15000)
        html_content = await page.content()
        text_content = await page.evaluate("document.body.innerText")
        title = await page.title()
        LOGGER.debug(f"took {time.time()-start:.2f}s to fetch {url}")
        webpage_record = WebpageContent(
            url=url,
            html_content=html_content,
            text_content=text_content,
            title=title,
            analyzed=False,
        )
        db.add(webpage_record)
        db.commit()
    else:
        LOGGER.debug(f"we already know this record: {url}")
    return webpage_record


async def safe_fetch_url(url, page, db):
    async with CURRENTLY_PROCESSING_LOCK:
        if url in CURRENTLY_PROCESSING:
            future = CURRENTLY_PROCESSING[url]
        else:
            future = asyncio.get_event_loop().create_future()
            CURRENTLY_PROCESSING[url] = future

    if not future.done():
        try:
            result = await fetch_page(url, page, db)
            future.set_result(result)
        except Exception as e:
            LOGGER.exception(f"Error fetching URL {url}: {e}")
            # Set result to None to avoid propagating the exception
            future.set_result(None)
        finally:
            async with CURRENTLY_PROCESSING_LOCK:
                del CURRENTLY_PROCESSING[url]

    return await future


async def crawl_domain(
    start_url, url_pattern, required_text, max_pages, progress_store, crawl_id
):
    LOGGER.debug(
        f"about to start crawling {start_url} (max pages to check {max_pages})"
    )
    db = SessionLocal()
    domain = urlparse(start_url).netloc
    links_to_crawl = set([start_url])
    processed_links = set(links_to_crawl)

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            while (
                links_to_crawl
                and progress_store[crawl_id]["fetched"] < max_pages
            ):
                LOGGER.debug(
                    f"about to pull from the queue, we are pulling {max_pages}"
                )
                url = links_to_crawl.pop()
                processed_links.add(url)
                LOGGER.debug(f"about to crawl {url}")
                webpage_record = await safe_fetch_url(url, page, db)

                if webpage_record is None:
                    LOGGER.warning(f"Oops, couldn't fetch {url}. Skipping.")
                    continue

                if (
                    required_text.lower()
                    not in webpage_record.text_content.lower()
                ):
                    LOGGER.warning(
                        f"could not find required text {required_text} at {url}, skipping"
                    )
                    continue

                if not webpage_record.analyzed:
                    progress_store[crawl_id]["fetched"] += 1
                    analyze_entity_task = asyncio.create_task(
                        analyze_entity_context(
                            webpage_record.webpage_content_id,
                            progress_store,
                            crawl_id,
                            db,
                        )
                    )
                    LOGGER.debug(
                        f"this is the analyze task: {analyze_entity_task}"
                    )
                html_content = webpage_record.html_content
                LOGGER.debug(f"extracting links from {url}")
                raw_links_found_in_page = await extract_links(
                    html_content, url, domain
                )
                filtered_links_found_in_page = set(
                    [
                        x
                        for x in raw_links_found_in_page
                        if url_pattern.lower() in x.lower()
                    ]
                )

                links_to_crawl.update(
                    set(filtered_links_found_in_page) - processed_links
                )
                progress_store[crawl_id]["discovered"] = len(processed_links)

        except:
            LOGGER.exception("something bad happened")
            raise
        await browser.close()

    db.close()
    progress_store[crawl_id]["completed"] = True
