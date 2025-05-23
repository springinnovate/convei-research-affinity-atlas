"""Entrypoint for AAA app."""

from urllib.parse import urlparse, urljoin
import asyncio
import re
import uuid

from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi import HTTPException
from pydantic import BaseModel

from ..parser import fetch_page_content
from ..database import SessionLocal, init_db
from ..models import URLContent, PersonContext
from ..crawler import crawl_domain
from ..llm_analyzer import generate_bio

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

init_db()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyse/")
async def analyse_url(request: Request, url: str = Form(...)):
    result = await fetch_page_content(url)

    db = SessionLocal()
    url_content = URLContent(
        url=url, content=result["content"], title=result["title"]
    )
    db.add(url_content)
    db.commit()
    db.close()
    return RedirectResponse("/", status_code=303)


@app.get("/urls/")
async def list_urls():
    db = SessionLocal()
    urls = db.query(URLContent).all()
    db.close()

    return {
        "urls": [
            {
                "id": u.id,
                "title": u.title,
                "url": u.url,
                "has_content": bool(u.text_content),
            }
            for u in urls
        ]
    }


@app.get("/people/")
async def list_people():
    db = SessionLocal()
    people = db.query(PersonContext).all()
    db.close()
    return {
        "people": [
            {"id": p.id, "name": p.name, "context": p.context} for p in people
        ]
    }


@app.get("/urlcontent/{url_id}")
async def url_content(url_id: int):
    print(f"fetching content for {url_id}")
    db = SessionLocal()
    url_content = db.query(URLContent).filter(URLContent.id == url_id).first()
    db.close()

    if not url_content:
        raise HTTPException(status_code=404, detail="URL content not found")

    return {
        "id": url_content.id,
        "url": url_content.url,
        "title": url_content.title,
        "text_content": url_content.text_content,
    }


@app.get("/person/{person_id}/bio")
async def get_bio(person_id: int):
    print(f"generate bio for person {person_id}")
    bio = await generate_bio(person_id)
    print(f"here's the bio: {bio}")
    return bio


class CrawlRequest(BaseModel):
    url: str
    max_pages: int


# Simple in-memory storage (move to DB/Redis later)
progress_store = {}


@app.post("/start_crawl/")
async def start_crawl(request: CrawlRequest):
    crawl_id = str(uuid.uuid4())
    progress_store[crawl_id] = {
        "analyzed": 0,
        "discovered": 0,
        "completed": False,
    }

    asyncio.create_task(
        crawl_domain(request.url, request.max_pages, progress_store, crawl_id)
    )
    return {"crawl_id": crawl_id}


@app.get("/crawl_status/{crawl_id}")
async def crawl_status(crawl_id: str):
    status = progress_store.get(crawl_id)
    if not status:
        return {"error": "Invalid crawl ID"}
    return status


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
