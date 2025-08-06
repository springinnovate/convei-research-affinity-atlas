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

speaker_data: dict[str, list[dict]] = defaultdict(list)
visited: set[str] = set()


async def fetch(client: AsyncClient, url: str) -> str | None:
    if url in visited or len(visited) >= MAX_VISITED:
        Actor.log.warning(
            f"skipping {url} because {url in visited} or {len(visited)} >= {MAX_VISITED}"
        )
        return None
    visited.add(url)
    try:
        Actor.log.info(f"in the fetch, getting {url}")
        r = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        r.raise_for_status()
        Actor.log.info(f"done with fetching {url}")
        return r.text
    except Exception as e:
        Actor.log.warning(f"Fetch failed for {url}: {e}")
        return None


async def parse_listing(client: AsyncClient, url: str, q: asyncio.Queue):
    if len(visited) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(visited)} >= {MAX_VISITED}")
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
        await q.put(("speaker", urljoin(BASE, a["href"])))


async def parse_speaker(client: AsyncClient, url: str, q: asyncio.Queue):
    if len(visited) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(visited)} >= {MAX_VISITED}")
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

    speaker_data[name].append({"url": url, "content": f"{name}\n{aff}"})
    await Actor.push_data(
        {"speaker": name, "url": url, "content": f"{name}\n{aff}"}
    )

    for a in soup.select('a[href*="/Sessions/Details/"]'):
        await q.put(("session", urljoin(BASE, a["href"]), name))


async def parse_session(client: AsyncClient, url: str, name: str):
    if len(visited) >= MAX_VISITED:
        Actor.log.warning(f"skipping because {len(visited)} >= {MAX_VISITED}")
        return
    Actor.log.info(f"Parsing session {url} for {name}")
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
    await Actor.push_data(
        {"speaker": name, "url": url, "content": f"{title}\n\n{desc}"}
    )


async def worker(q: asyncio.Queue, client: AsyncClient):
    while True:
        if len(visited) >= MAX_VISITED:
            return
        try:
            item = await asyncio.wait_for(q.get(), timeout=5)
        except asyncio.TimeoutError:
            Actor.log.warning(f"timeout error on waiting for queue")
            return
        try:
            kind = item[0]
            if kind == "listing":
                _, url = item
                await parse_listing(client, url, q)
            elif kind == "speaker":
                _, url = item
                await parse_speaker(client, url, q)
            elif kind == "session":
                _, url, name = item
                await parse_session(client, url, name)
        except Exception as e:
            Actor.log.warning(f"Worker error processing {item}: {e}")
        finally:
            q.task_done()


async def main() -> None:
    global MAX_VISITED, CONCURRENCY
    async with Actor:
        inp = await Actor.get_input() or {}
        urls = inp.get("urls", [])
        MAX_VISITED = inp.get("max_visited", 10_000)
        CONCURRENCY = inp.get("concurrency", 1)

        if not urls:
            raise ValueError('Input must contain a "urls" array.')

        queue: asyncio.Queue = asyncio.Queue()
        for u in urls:
            await queue.put(("listing", u))

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

        await Actor.set_value("OUTPUT", speaker_data)
