"""Entrypoint for CONVEI research affinity atlas app."""

import os
from urllib.parse import urlparse
from collections import Counter
import asyncio
import logging
import sys

from pathlib import Path
import uvicorn
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from parser import fetch_page_content
from database import SessionLocal, init_db
from models import Entity
from llm_analyzer import generate_bios

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

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.auto_reload = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    init_db()
    input_json_path = os.environ.get("INPUT_JSON_PATH", None)
    if input_json_path is None:
        raise ValueError(f"undefined INPUT_JSON_PATH env variable")
    db = SessionLocal()
    await generate_bios(input_json_path, db)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin")
async def read_root(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.post("/analyze/")
async def analyze_url(request: Request, url: str = Form(...)):
    result = await fetch_page_content(url)

    db = SessionLocal()
    url_content = WebpageContent(
        url=url, content=result["content"], title=result["title"]
    )
    db.add(url_content)
    db.commit()
    db.close()
    return RedirectResponse("/", status_code=303)


@app.get("/urls/")
async def list_urls():
    db = SessionLocal()
    urls = db.query(WebpageContent.url).all()
    db.close()

    # Extract the top-level domains
    top_domains = [
        f"{urlparse(u.url).scheme}://{urlparse(u.url).netloc}" for u in urls
    ]

    domain_counts = Counter(top_domains)

    return {
        "top_level_domains": [
            {"domain": domain, "url_count": count}
            for domain, count in domain_counts.items()
        ]
    }


@app.get("/entities/")
async def list_entities():
    db = SessionLocal()
    entities = db.query(Entity).all()
    db.close()
    return {"entities": set(p.name for p in entities)}


@app.get("/WebpageContent/{url_id}")
async def url_content(url_id: int):
    LOGGER.debug(f"fetching content for {url_id}")
    db = SessionLocal()
    url_content = (
        db.query(WebpageContent)
        .filter(WebpageContent.webpage_content_id == url_id)
        .first()
    )
    db.close()

    if not url_content:
        raise HTTPException(status_code=404, detail="URL content not found")

    return {
        "id": url_content.webpage_content_id,
        "url": url_content.url,
        "title": url_content.title,
        "text_content": url_content.text_content,
    }


class PersonRequest(BaseModel):
    person_name: str


@app.post("/person/bio")
async def get_bio(req: PersonRequest):
    db = SessionLocal()
    entity = db.query(Entity).filter(Entity.name == req.person_name).first()

    if not entity:
        raise HTTPException(
            status_code=404, detail=f"Person '{req.person_name}' not found."
        )
    LOGGER.debug(f"generate bio for person {req.person_name}")
    bio = await generate_bio(entity.entity_id, db)
    LOGGER.debug(f"here's the bio: {bio}")
    db.close()
    return {"name": req.person_name, "bio": bio}


class CrawlRequest(BaseModel):
    url: str
    max_pages: int
    url_pattern: str
    required_text: str


PROGRESS_STORE = {}


@app.post("/start_crawl/")
async def start_crawl(request: CrawlRequest):
    crawl_id = request.url
    max_pages = request.max_pages
    url_pattern = request.url_pattern
    required_text = request.required_text
    existing_progress = PROGRESS_STORE.get(crawl_id)
    LOGGER.debug(f"EXISTING PROGRESS: {existing_progress}")
    if existing_progress and not existing_progress["completed"]:
        return {
            "crawl_id": crawl_id,
            "status": "already in progress",
        }

    PROGRESS_STORE[crawl_id] = {
        "processed": 0,
        "fetched": 0,
        "discovered": 0,
        "completed": False,
    }

    asyncio.create_task(
        crawl_domain(
            request.url,
            url_pattern,
            required_text,
            max_pages,
            PROGRESS_STORE,
            crawl_id,
        )
    )
    return {"crawl_id": crawl_id, "status": "started"}


@app.post("/crawl_status")
async def crawl_status(request: Request):
    data = await request.json()
    crawl_id = data.get("crawl_id")
    status = PROGRESS_STORE.get(crawl_id)
    if not status:
        return {"error": "Invalid crawl ID"}
    return status


# add after PROGRESS_STORE definition
@app.get("/active_crawls/")
async def active_crawls():
    return PROGRESS_STORE


@app.post("/match_people")
async def match_people(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_interest_text = data.get("user_interest_text")
    result = await llm_match_people(user_interest_text, db)
    return result


@app.post("/generate_all_bios")
async def refresh_all_entity_analyses():
    """
    Ensure every entity has an up‑to‑date LLM analysis for its current snippet set.
    """
    db = SessionLocal()
    entities = db.execute(select(Entity.entity_id, Entity.name)).all()

    total = len(entities)

    for idx, (entity_id, entity_name) in enumerate(entities, 1):
        LOGGER.debug(
            f"Processing {entity_name} ({idx}/{total}) – {total - idx} remaining"
        )
        await generate_bio(entity_id, db)

    LOGGER.debug("ALL DONE refreshing summaries")
    db.close()
    return {"status": "success", "entities_processed": len(entities)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
