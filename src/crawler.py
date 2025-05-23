"""Webcrawler framework."""

import time
import asyncio
import re
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

from .database import SessionLocal
from .models import URLContent


async def fetch_page(url, page):
    await page.goto(url)
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


async def crawl_domain(start_url, max_pages, progress_dict, crawl_id):
    db = SessionLocal()
    domain = urlparse(start_url).netloc
    queue = asyncio.Queue()
    visited = set()

    progress_dict[crawl_id] = {"analyzed": 0, "discovered": 1}

    await queue.put(start_url)
    visited.add(start_url)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        while (
            not queue.empty()
            and progress_dict[crawl_id]["analyzed"] < max_pages
        ):
            url = await queue.get()

            page_record = (
                db.query(URLContent).filter(URLContent.url == url).first()
            )

            if not page_record:
                start = time.time()
                html_content, text_content, title = await fetch_page(url, page)
                print(f"took {time.time()-start:.2f}s to fetch {url}")
                page_record = URLContent(
                    url=url,
                    html_content=html_content,
                    text_content=text_content,
                    title=title,
                    analyzed=True,
                )
                db.add(page_record)
            else:
                html_content = page_record.html_content
                max_pages += 1  # we didn't search it, so do one more

            db.commit()

            links = await extract_links(html_content, url, domain)
            for link in links:
                if link not in visited and len(visited) < max_pages:
                    visited.add(link)
                    await queue.put(link)

            progress_dict[crawl_id]["analyzed"] += 1
            progress_dict[crawl_id]["discovered"] = len(visited)
        await browser.close()

    db.close()
    progress_dict[crawl_id]["completed"] = True
