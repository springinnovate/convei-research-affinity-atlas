"""Parse a webpage."""

import asyncio

from playwright.async_api import async_playwright


async def fetch_page_content(url):
    """Fetch url to parsable text."""
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            browser_context = await browser.new_context(
                accept_downloads=False,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            content = None
            page = await browser_context.new_page()
            await page.goto(url)
            content = await asyncio.wait_for(
                page.content(), timeout=5000  # timeout in ms
            )
            content = await asyncio.wait_for(page.content(), timeout=5000)
            title = await page.title()

            if content is not None:
                content = content.encode("ascii", errors="ignore").decode(
                    "ascii", errors="ignore"
                )
            print(f"the title is: {title}")
            return {"content": content, "title": title}
    except Exception as e:
        print(f"ERROR: {e}")
        raise
