from playwright.async_api import async_playwright
from database import SessionLocal, init_db
from crawler import safe_fetch_url

init_db()


async def main():
    db = SessionLocal()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        url = "https://events.rdmobile.com/Speakers/Index/19095"
        result = safe_fetch_url(url, page, db)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
