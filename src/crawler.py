"""Webcrawler framework."""

import logging
import sys
import time
import asyncio
import re
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

from .database import SessionLocal
from .models import WebpageContent
from .llm_analyzer import analyze_entity_context

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)


async def fetch_page(url, page):
    await page.goto(url, wait_until="networkidle")
    html_content = await page.content()
    text_content = await page.evaluate("document.body.innerText")
    title = await page.title()
    return html_content, text_content, title


async def extract_links(content, base_url, domain):
    href_links = re.findall(r'href=["\'](.*?)["\']', content, re.IGNORECASE)
    links = set()
    for link in href_links:
        abs_url = urljoin(base_url, link)
        if urlparse(abs_url).netloc == domain:
            links.add(abs_url)
    return links


async def crawl_domain(start_url, max_pages, progress_store, crawl_id):
    db = SessionLocal()
    domain = urlparse(start_url).netloc
    queue = asyncio.Queue()
    visited = set()

    await queue.put(start_url)
    visited.add(start_url)

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            while (
                not queue.empty()
                and progress_store[crawl_id]["discovered"] < max_pages
            ):
                LOGGER.debug(
                    f"about to pull from the queue, we are pulling {max_pages}"
                )
                url = await queue.get()
                LOGGER.debug(f"about to crawl {url}")
                page_record = (
                    db.query(WebpageContent)
                    .filter(WebpageContent.url == url)
                    .first()
                )

                try_another = True
                if not page_record:
                    try_another = False  # this url needs processing
                    start = time.time()
                    LOGGER.debug(f"attempting to fetch {url}")
                    html_content, text_content, title = await fetch_page(
                        url, page
                    )
                    LOGGER.debug(
                        f"took {time.time()-start:.2f}s to fetch {url}"
                    )
                    page_record = WebpageContent(
                        url=url,
                        html_content=html_content,
                        text_content=text_content,
                        title=title,
                        analyzed=False,
                    )
                    db.add(page_record)
                    db.commit()
                if not page_record.analyzed:
                    try_another = False  # this url needs processing
                    analyze_entity_task = asyncio.create_task(
                        analyze_entity_context(
                            page_record.webpage_content_id,
                            progress_store,
                            crawl_id,
                        )
                    )
                    LOGGER.debug(
                        f"this is the analyze task: {analyze_entity_task}"
                    )

                if try_another:
                    LOGGER.debug(
                        f"already visited {url} so not counting this one as a search"
                    )
                    max_pages += 1  # we didn't search it, so do one more

                html_content = page_record.html_content
                LOGGER.debug(f"extracting links from {url}")
                links = await extract_links(html_content, url, domain)
                LOGGER.debug(f"extracted {len(links)} links")
                for link in links:
                    # TODO: this should kick out if there are too many pages to be visited because then the loop does the same thing above.
                    LOGGER.debug(f"attempting to add {link} to search")
                    if link not in visited and len(visited) < max_pages:
                        visited.add(link)
                        await queue.put(link)
                        LOGGER.debug(f"ADDED to add {link} to search")
                    else:
                        LOGGER.debug(
                            f"did not add  {link} to search. visited? {link in visited}; too many visted? {len(visited) >= max_pages}"
                        )
                LOGGER.debug(f"about to try again with these: {queue}")
                progress_store[crawl_id]["fetched"] += 1
                progress_store[crawl_id]["discovered"] = len(visited)
        except:
            LOGGER.exception("something bad happened")
            raise
        await browser.close()

    db.close()
    progress_store[crawl_id]["completed"] = True
