import asyncio
import re
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

from .database import SessionLocal
from .models import URLContent


async def fetch_page(url, browser):
    page = await browser.new_page()
    await page.goto(url)
    content = await page.content()
    title = await page.title()
    return content, title


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

    await queue.put(start_url)
    visited.add(start_url)

    progress_dict[crawl_id] = {"analyzed": 0, "discovered": 1}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        while (
            not queue.empty()
            and progress_dict[crawl_id]["analyzed"] < max_pages
        ):
            url = await queue.get()
            content, title = await fetch_page(url, browser)

            page_record = URLContent(
                url=url, content=content, title=title, analyzed=True
            )
            db.merge(page_record)
            db.commit()

            links = await extract_links(content, url, domain)
            for link in links:
                if link not in visited and len(visited) < max_pages:
                    visited.add(link)
                    await queue.put(link)

            progress_dict[crawl_id]["analyzed"] += 1
            progress_dict[crawl_id]["discovered"] = len(visited)
        await browser.close()

    db.close()
    progress_dict[crawl_id]["completed"] = True
