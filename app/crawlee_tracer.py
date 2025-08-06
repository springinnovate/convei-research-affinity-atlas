from apify import Actor
from bs4 import BeautifulSoup
from httpx import AsyncClient
from urllib.parse import urljoin, urlparse
import asyncio

REQUIRED_TEXT = None
MAX_PAGES = None
MAX_CONCURRENCY = None

visited_urls = set()


"""
* https://events.rdmobile.com/Speakers/Details/* -- this is the speaker page
    it contains links to presentations/participation that the person is interested in
    when you encounter a speaker page ->
        * scrape everything out of it
        * follow each of the https://events.rdmobile.com/Sessions/Details/* associted with it
            * scrape the text there and associate it with that speaker

I want to scrape all the links that look like this:

<a class="block-list__disclosure" href="/Speakers/Details/2820732">
    <span class="block-list__thumbnail block-list__thumbnail--square pfp">
        <span class="pfp-initials brand-bg" aria-hidden="true">
          <span class="pfp-initials__text brand-fg">FA</span>
        </span>
    </span>
    <div class="block-list__content">
      <strong class="block-list__title">Fred Abbott</strong>
              <p class="block-list__meta">Ecological Society of America</p>
    </div>
  </a>

Then for each of those links I want to follow them, then on each page there will be a header like this:

<div class="title-block__content">
        <h1 class="title-block__title brand-fg">Peter Adebayo</h1>
          <p class="title-block__meta">
                        Community InfoSource
          </p>
              </div>

From that i want to extract the name and reference, basically the H1 and p content in that div

Then there are any number of content cards that look like this:

<div class="content__card">
    <div class="section-title">
      <h2 class="section-title__title">Session Participation</h2>
    </div>
    <ul class="block-list">

<li class="block-list__item " data-start-time="1755106200000">

  <div class="block-list__content">
    <p class="block-list__meta emphasize brand-fg">

<time datetime="2025-08-13T13:30:00" data-unix-timestamp-ms="1755106200000" data-format="weekday,day,year,time" class="" data-original="Wednesday, August 13, 2025, 1:30 PM" style="cursor: pointer;">Wednesday, August 13, 2025 at 10:30 AM</time>

      –

<time datetime="15:00:00" data-unix-timestamp-ms="1755111600000" data-format="time,zone" class="" data-original="3:00 PM EDT" style="cursor: pointer;">12:00 PM PDT</time>

    </p>
    <strong class="block-list__title">
      <a href="/Sessions/Details/2976200">
        OOS 30 - Modeling Variation of Physico-Chemical Soil Properties of Cashew Plantation under Different Management Systems in the Guinea Savanna of Southwestern Nigeria
      </a>
    </strong>
      <p class="block-list__meta">
                          Hilton Key 11-12
      </p>

      <p class="block-list__tags">
        <span class="screenreaders-only">Tags:</span>
                            <span class="pill">
                Session Type:
              Organized Oral Session
            </span>
      </p>

  </div>

</li>
    </ul>
  </div>

  each of those content cards contains links like this:

  <a href="/Sessions/Details/2976200">
        OOS 30 - Modeling Variation of Physico-Chemical Soil Properties of Cashew Plantation under Different Management Systems in the Guinea Savanna of Southwestern Nigeria
      </a>

that i want to follow, then on THAT page i want to extract the content of the top content__card, in this case the anme of the session and the abstract, specifically the title in h1 and the abstract in session__description

      <div class="content__card">
  <div class="title-block">
    <div class="title-block__content">
      <h1 class="title-block__title brand-fg">OOS 30 - Modeling Variation of Physico-Chemical Soil Properties of Cashew Plantation under Different Management Systems in the Guinea Savanna of Southwestern Nigeria</h1>
      <p class="title-block__meta">

<time datetime="2025-08-13" data-unix-timestamp-ms="1755106200000" data-format="weekday,day" class="" data-original="Wednesday, August 13" style="cursor: pointer;">Wednesday, August 13</time>


<time datetime="13:30:00" data-unix-timestamp-ms="1755106200000" data-format="time" class="" data-original="1:30 PM" style="cursor: pointer;">10:30 AM</time>

        –

<time datetime="15:00:00" data-unix-timestamp-ms="1755111600000" data-format="time,zone" class="" data-original="3:00 PM EDT" style="cursor: pointer;">12:00 PM PDT</time>

          </p><div class="text-size deemphasize space-t-25">
            Hilton Key 11-12
          </div>
              <p></p>
        <div class="title-block__tags">
          <div data-truncate-to="4 items" data-truncate-class="pill pill--deemphasize" data-truncate-style="tags">
      <span class="pill">
        Session Type:
      Organized Oral Session
    </span>
</div>


        </div>
    </div>


  </div>


  <div class="session__description">
      <div data-truncate-to="10 lines" data-truncate-leeway="2" style="overflow: hidden;">
          <div class="user-content">
            The study aims to understand the variations in soil properties under cashew plantations of different ages (0–9 years, 10–20 years, and over 20 years) and compare these to fallow lands (which serve as a control). It also examines the effect of management systems (subsistence individual versus institution-managed plantations) on these soil properties. By analyzing soil samples from various plantation ages, the study seeks to identify how long-term cashew cultivation influences factors like soil texture, bulk density, pH, and nutrient content (Ca++, Mg++, K+, nitrogen, phosphorus, etc.).<a href="theme:The" target="_blank">Theme:The</a> study explores:    Soil Degradation and Nutrient Loss: Younger cashew plantations (0–9 years old) showed significant variations in soil properties compared to older plantations and fallow lands, with effects on soil pH, bulk density, and nutrient content (like Ca++ and Mg++).    Restoration of Soil Nutrients: As plantations age (above 20 years), there is some indication of nutrient restoration, with higher concentrations of certain nutrients (like potassium and calcium) compared to younger plantations and fallow lands.    Differences Between Management Systems: The study also highlights how management systems influence soil properties, with institution-managed plantations potentially better at maintaining or restoring soil fertility over time.
          </div>

      </div>

  </div>
</div>

"""

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://events.rdmobile.com"


async def fetch(client, url):
    if url in visited:
        return None
    visited.add(url)
    try:
        r = await client.get(
            url, headers=HEADERS, follow_redirects=True, timeout=30
        )
        r.raise_for_status()
        return r.text
    except Exception:
        return None


async def fetch_and_process(url, url_queue, client, base_domain):
    if url in visited_urls or len(visited_urls) >= MAX_PAGES:
        return

    visited_urls.add(url)
    Actor.log.info(f"Crawling: {url}")

    try:
        response = await client.get(url, follow_redirects=True, timeout=30)
    except Exception as e:
        Actor.log.warning(f"Failed to fetch {url}: {e}")
        return

    page_text = response.text

    if REQUIRED_TEXT in page_text:
        soup = BeautifulSoup(response.content, "lxml")
        paragraphs = [
            section.get_text(strip=True)
            for section in soup.find_all(
                [
                    "h1",
                    "div",
                    "p",
                    "strong",
                ]
            )
            if section.get_text(strip=True)
        ]

        # Optionally, join into one block of full text
        full_text = "\n\n".join(paragraphs)

        await Actor.push_data({"url": url, "text": full_text})
        Actor.log.info(f'"{REQUIRED_TEXT}" found at {url}. Data saved.')

        put_tasks = []
        for link in soup.find_all("a", href=True):
            absolute_url = urljoin(url, link["href"])
            parsed_url = urlparse(absolute_url)

            if (
                parsed_url.netloc == base_domain
                and absolute_url not in visited_urls
            ):
                put_tasks.append(url_queue.put(absolute_url))

        await asyncio.gather(*put_tasks)
    else:
        Actor.log.info(f'"{REQUIRED_TEXT}" NOT found at {url}. Skipped.')


async def worker(url_queue, client, base_domain):
    while True:
        try:
            url = await asyncio.wait_for(url_queue.get(), timeout=30)
        except asyncio.TimeoutError:
            return

        await fetch_and_process(url, url_queue, client, base_domain)
        url_queue.task_done()


async def main():
    global MAX_PAGES
    global MAX_CONCURRENCY
    global REQUIRED_TEXT
    async with Actor:
        actor_input = await Actor.get_input() or {"url": None}
        start_url = actor_input.get("url")
        MAX_PAGES = actor_input.get("max_pages")
        MAX_CONCURRENCY = actor_input.get("max_concurrency")
        REQUIRED_TEXT = actor_input.get("required_text")
        if not start_url:
            raise ValueError('Missing "url" attribute in input!')

        base_domain = urlparse(start_url).netloc
        url_queue = asyncio.Queue()
        await url_queue.put(start_url)

        async with AsyncClient() as client:
            tasks = [
                worker(url_queue, client, base_domain)
                for _ in range(MAX_CONCURRENCY)
            ]
            await asyncio.gather(*tasks)
            await url_queue.join()
