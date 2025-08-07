from __future__ import annotations

import asyncio
from collections import defaultdict
from urllib.parse import urljoin

from apify import Actor
from bs4 import BeautifulSoup
from httpx import AsyncClient, AsyncHTTPTransport

BASE = "https://events.rdmobile.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_VISITED = None
CONCURRENCY = None

SPEAKER_DATA: dict[str, list[dict]] = defaultdict(list)
VISITED: set[str] = set()


URL_CACHE: dict[str, asyncio.Task] = {}
URL_LOCK = asyncio.Lock()


async def fetch(client: AsyncClient, url: str) -> str | None:
    async with URL_LOCK:
        if url in URL_CACHE:
            Actor.log.info(f"Awaiting cached fetch for {url}")
            task = URL_CACHE[url]
        else:
            Actor.log.info(f"Starting new fetch for {url}")
            task = asyncio.create_task(fetch_url(client, url))
            VISITED.add(url)
            URL_CACHE[url] = task

    try:
        return await task
    except Exception as e:
        Actor.log.warning(f"Fetch task failed for {url}: {e}")
        return None


async def fetch_url(client: AsyncClient, url: str) -> str | None:
    try:
        Actor.log.info(f"Fetching URL: {url}")
        response = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        Actor.log.warning(f"Fetch failed for {url}: {e}")
        raise


async def parse_listing(client: AsyncClient, url: str, q: asyncio.Queue):
    if len(VISITED) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(VISITED)} >= {MAX_VISITED}")
        return
    Actor.log.info(f"Parsing listing {url}")
    html = await fetch(client, url)
    Actor.log.info(f"got some html")
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select(
        'a.block-list__disclosure[href*="/Speakers/Details/"]'
    ):
        await q.put((1, ("speaker", urljoin(BASE, a["href"]))))


async def parse_speaker(client: AsyncClient, url: str, q: asyncio.Queue):
    if len(VISITED) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(VISITED)} >= {MAX_VISITED}")
        return
    Actor.log.info(f"Parsing listing {url} for speaker")
    html = await fetch(client, url)
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")
    block = soup.select_one("div.title-block__content")
    if not block:
        return
    name_el = block.select_one("h1")
    aff_el = block.select_one("p")

    if not name_el or not name_el.get_text(strip=True):
        Actor.log.warning(f"No valid name found in {url}, skipping.")
        return  # Toss the whole thing if no name

    name = name_el.get_text(strip=True)
    aff = (
        aff_el.get_text(strip=True)
        if aff_el and aff_el.get_text(strip=True)
        else ""
    )  # okay to have empty affiliation

    SPEAKER_DATA[name].append({"url": url, "content": f"{name}\n{aff}"})
    await Actor.push_data(
        {"speaker": name, "url": url, "content": f"{name}\n{aff}"}
    )

    for a in soup.select(
        'a[href*="/Sessions/Details/"], a[href*="/Lists/Details/"]'
    ):
        await q.put((0, ("session", urljoin(BASE, a["href"]), name)))


async def parse_session(
    client: AsyncClient, url: str, name: str, q: asyncio.Queue
):
    if len(VISITED) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(VISITED)} >= {MAX_VISITED}")
        return

    Actor.log.info(f"Parsing session {url} for {name}")
    html = await fetch(client, url)
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")

    # Single card (first content__card)
    card = soup.select_one("div.content__card")
    if not card:
        Actor.log.info(f"No content__card found for {url}")
        return

    # Title
    title_el = card.select_one("h1.title-block__title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Description / Abstract – try session__description first, fall back to user-content
    desc_el = card.select_one(
        "div.session__description .user-content"
    ) or card.select_one("div.user-content.space-b-200")
    desc = desc_el.get_text(separator=" ", strip=True) if desc_el else ""

    content = f"{title}\n\n{desc}"
    SPEAKER_DATA[name].append({"url": url, "content": content})
    await Actor.push_data({"speaker": name, "url": url, "content": content})

    # Follow any "Abstracts" links in same page
    for card in soup.select("div.content__card"):
        header = card.select_one("h2.section-title__title")
        if header and "Abstracts" in header.get_text(strip=True):
            Actor.log.info('Found "Abstracts" section in card.')
            for a in card.select('a[href*="/Lists/Details/"]'):
                await q.put((0, ("session", urljoin(BASE, a["href"]), name)))


async def worker(q: asyncio.Queue, client: AsyncClient):
    while True:
        try:
            _, item = await asyncio.wait_for(q.get(), timeout=9999)
        except asyncio.TimeoutError:
            Actor.log.warning("timeout error on waiting for queue")
            return
        try:
            if len(VISITED) >= MAX_VISITED:
                # don't process anything else
                continue
            kind = item[0]
            if kind == "listing":
                _, url = item
                await parse_listing(client, url, q)
            elif kind == "speaker":
                _, url = item
                await parse_speaker(client, url, q)
            elif kind == "session":
                _, url, name = item
                await parse_session(client, url, name, q)
        except Exception as e:
            Actor.log.warning(f"Worker error processing {item}: {e}")
        finally:
            q.task_done()


async def main() -> None:
    global MAX_VISITED, CONCURRENCY
    async with Actor:
        inp = await Actor.get_input() or {}
        urls = inp.get("urls", [])
        MAX_VISITED = inp.get("max_VISITED", 10_000)
        CONCURRENCY = inp.get("concurrency", 1)

        if not urls:
            raise ValueError('Input must contain a "urls" array.')

        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for u in urls:
            await queue.put((10, ("listing", u)))

        proxy_cfg = await Actor.create_proxy_configuration(groups=["auto"])
        proxy_url = await proxy_cfg.new_url()

        transport = AsyncHTTPTransport(proxy=proxy_url)
        async with AsyncClient(base_url=BASE, transport=transport) as client:
            tasks = [
                asyncio.create_task(worker(queue, client))
                for _ in range(CONCURRENCY)
            ]
            await queue.join()
            for t in tasks:
                t.cancel()

        await Actor.set_value("OUTPUT", SPEAKER_DATA)
