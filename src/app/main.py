"""Entrypoint for AAA app."""

import re
from urllib.parse import urlparse, urljoin

from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi import HTTPException

from ..parser import fetch_page_content
from ..database import SessionLocal, init_db
from ..models import URLContent, Entity


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

init_db()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyse/")
async def analyse_url(request: Request, url: str = Form(...)):
    content = await fetch_page_content(url)

    db = SessionLocal()
    url_content = URLContent(url=url, content=content)
    db.add(url_content)
    db.commit()
    db.refresh(url_content)

    # for entity_name in entities:
    #     entity_embedding = embed_text(entity_name)
    #     entity = Entity(
    #         name=entity_name,
    #         embedding=entity_embedding,
    #         url_content_id=url_content.id,
    #     )
    #     db.add(entity)

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
            {"id": u.id, "url": u.url, "has_content": bool(u.content)}
            for u in urls
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
        "content": url_content.content,
    }


@app.get("/entities/")
async def list_entities():
    db = SessionLocal()
    entities = db.query(Entity).all()
    db.close()
    return {"entities": [entity.name for entity in entities]}


@app.post("/urlcontent/{url_id}/extract-links/")
async def extract_links(url_id: int):
    db = SessionLocal()
    url_content = db.query(URLContent).filter(URLContent.id == url_id).first()

    if not url_content or not url_content.content:
        db.close()
        raise HTTPException(404, "Content not found or empty")

    original_url = url_content.url
    original_domain = urlparse(original_url).netloc

    href_links = re.findall(
        r'href=["\'](.*?)["\']', url_content.content, re.IGNORECASE
    )

    urls_found = set()
    for link in href_links:
        absolute_url = urljoin(original_url, link)
        parsed_url = urlparse(absolute_url)
        if (
            parsed_url.scheme in {"http", "https"}
            and parsed_url.netloc == original_domain
        ):
            urls_found.add(absolute_url)
    new_entries = []
    for url in urls_found:
        existing = db.query(URLContent).filter(URLContent.url == url).first()
        if not existing:
            new_entry = URLContent(url=url, content=None)
            db.add(new_entry)
            new_entries.append(url)

    db.commit()
    db.close()

    return {"added_urls": new_entries}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
