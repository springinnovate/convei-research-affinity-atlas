"""Entrypoint for AAA app."""

from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from ..parser import fetch_page_content
from ..database import SessionLocal, init_db
from ..models import URLContent, Entity
from ..vectorizer import embed_text, search_similar


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
async def list_entities():
    db = SessionLocal()
    url_list = db.query(URLContent).all()
    db.close()
    return {"urls": [{"id": _url.id, "url": _url.url} for _url in url_list]}


@app.get("/urlcontent/{url_id}")
async def url_content(url_id: int):
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


@app.get("/similar/{entity_id}")
async def find_similar(entity_id: int):
    db = SessionLocal()
    target = db.query(Entity).filter(Entity.id == entity_id).first()
    all_entities = db.query(Entity).filter(Entity.id != entity_id).all()
    similar_entities = search_similar(target.embedding, all_entities)
    db.close()
    return {
        "target": target.name,
        "similar": [e.name for e in similar_entities],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
