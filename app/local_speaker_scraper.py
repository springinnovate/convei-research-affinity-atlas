import logging
import json
import sys
import asyncio
from collections import defaultdict
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from httpx import AsyncClient

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://events.rdmobile.com"

MAX_VISITED = 10

speaker_data = defaultdict(list)
visited = set()


async def fetch(client, url):
    LOGGER.info(f"attempting to fetch {url}")
    if url in visited and len(visited) < MAX_VISITED:
        return None
    visited.add(url)
    try:
        r = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        r.raise_for_status()
        return r.text
    except Exception:
        LOGGER.exception(f"something failed on {url}")
        return None


async def parse_listing(client, url, queue):
    LOGGER.info(f"attempting to parse listing {url}")
    html = await fetch(client, url)
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select(
        'a.block-list__disclosure[href*="/Speakers/Details/"]'
    ):
        href = urljoin(BASE, a["href"])
        await queue.put(("speaker", href))


async def parse_speaker(client, url, queue):
    LOGGER.info(f"attempting to parse speaker {url}")
    html = await fetch(client, url)
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    title_block = soup.select_one("div.title-block__content")
    if not title_block:
        return
    name = title_block.select_one("h1").get_text(strip=True)
    affiliation = title_block.select_one("p").get_text(strip=True)
    content = f"{name}\n{affiliation}"
    LOGGER.info(f"got {content}")
    speaker_data[name].append({"url": url, "content": content})

    for a in soup.select('a[href*="/Sessions/Details/"]'):
        href = urljoin(BASE, a["href"])
        await queue.put(("session", href, name))


async def parse_session(client, url, name):
    LOGGER.info(f"attempting to scrape session {url} for {name}")
    html = await fetch(client, url)
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    card = soup.select_one("div.content__card")
    if not card:
        return
    title = card.select_one("h1").get_text(strip=True)
    desc_el = card.select_one("div.session__description .user-content")
    desc = desc_el.get_text(separator=" ", strip=True) if desc_el else ""
    speaker_data[name].append({"url": url, "content": f"{title}\n\n{desc}"})


async def worker(queue):
    async with AsyncClient(base_url=BASE) as client:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                return
            kind = item[0]
            if kind == "listing":
                _, url = item
                await parse_listing(client, url, queue)
            elif kind == "speaker":
                _, url = item
                await parse_speaker(client, url, queue)
            elif kind == "session":
                _, url, name = item
                await parse_session(client, url, name)
            queue.task_done()


async def scrape(url):
    queue = asyncio.Queue()
    await queue.put(("listing", url))
    workers = [asyncio.create_task(worker(queue)) for _ in range(10)]
    await queue.join()
    for w in workers:
        w.cancel()
    return speaker_data


URL_LIST = [
    "https://events.rdmobile.com/Speakers/Index/19095",
    "https://events.rdmobile.com/Speakers/Index/19095?Sort=LastName&PageNumber=1",
    "https://events.rdmobile.com/Speakers/Index/19095?Sort=LastName&PageNumber=2",
    "https://events.rdmobile.com/Speakers/Index/19095?Sort=LastName&PageNumber=3",
    "https://events.rdmobile.com/Speakers/Index/19095?Sort=LastName&PageNumber=4",
]


if __name__ == "__main__":
    for url in URL_LIST:
        results = asyncio.run(scrape(url))
        print(json.dumps(results, indent=2))
        break
