"""Entrypoint for CONVEI research affinity atlas app."""

import os
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import sys
import uuid

from pathlib import Path
import uvicorn
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import Query


from database import SessionLocal, init_db
from models import Entity, ProcessedFile
from llm_analyzer import (
    generate_bios,
    _cached_llm_chunk_match,
    _merge_matches,
    _enc,
)

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
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.auto_reload = True

PROGRESS_STORE: Dict[str, Dict[str, Any]] = {}
RESULT_STORE: Dict[str, Dict[str, Any]] = {}
STORE_LOCK = asyncio.Lock()

TUPLES_PER_CHUNK = 250  # simple tuple-based chunking


class StartRequest(BaseModel):
    query: str


class ProgressResponse(BaseModel):
    job_id: str
    done: int
    total: int
    status: str  # running|done|error
    message: Optional[str] = None


class ResultResponse(BaseModel):
    query: str
    matches: List[Dict[str, Any]]
    notes: Optional[str] = None


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
        raise ValueError("undefined INPUT_JSON_PATH env variable")
    db = SessionLocal()
    await generate_bios(input_json_path, db)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_info/")
def get_info(db: Session = Depends(get_db)):
    pf = db.query(ProcessedFile).order_by(ProcessedFile.processed_at.desc()).first()
    if not pf:
        return {"dbInfo": "Database not initialized yet."}

    ts = pf.processed_at
    # fall back if naive
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return {"dbInfo": f"Database created from {pf.filename} at {ts.isoformat()}"}


@app.get("/entities/")
async def list_entities():
    db = SessionLocal()
    try:
        entities = db.query(Entity).all()
        names = [p.name for p in entities if p.name]

        # sort by last name (case-insensitive)
        sorted_names = sorted(set(names), key=lambda n: n.strip().split()[-1].lower())

        return {"entities": sorted_names}
    finally:
        db.close()


class PersonRequest(BaseModel):
    person_name: str


class CrawlRequest(BaseModel):
    url: str
    max_pages: int
    url_pattern: str
    required_text: str


@app.post("/person/bio")
def get_person_bio(payload: PersonRequest, db: Session = Depends(get_db)):
    entity = db.query(Entity).filter(Entity.name == payload.person_name).first()
    if not entity:
        raise HTTPException(status_code=404, detail="Person not found")

    return {
        "name": entity.name,
        "bio": entity.bio,
        "url_list": entity.url_list or [],
    }


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    matches: List[Dict[str, Any]]
    notes: Optional[str] = None


MAX_TOKENS_PER_CHUNK = 9000


@app.post("/people/search", response_model=SearchResponse)
async def people_search(req: SearchRequest, db: Session = Depends(get_db)):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    rows: List[Entity] = db.query(Entity).all()
    # TODO: limit 10 for debugging
    # rows: List[Entity] = db.query(Entity).limit(10).all()
    candidates: List[Tuple[str, str]] = [(r.name, r.bio or "") for r in rows if r.name]

    current_tokens = 0
    current_chunk = []
    chunks = []

    for name, bio in candidates:
        pair_text = f"Name: {name}\nBio:\n{bio}\n"
        LOGGER.debug(pair_text)
        tokens = len(_enc.encode(pair_text))  # using tiktoken encoder

        if current_tokens + tokens > MAX_TOKENS_PER_CHUNK and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append((name, bio))
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    # run chunks with bounded concurrency
    sem = asyncio.Semaphore(8)

    async def run_chunk(ch: List[Tuple[str, str]]) -> Optional[Dict[str, Any]]:
        async with sem:
            try:
                LOGGER.info(f"running chunk that's {len(ch)} long")
                return await _cached_llm_chunk_match(query, ch)
            except Exception as e:
                LOGGER.warning(f"failure on {query} {ch} {e}")
                return None

    tasks = [run_chunk(ch) for ch in chunks]
    payloads = await asyncio.gather(*tasks)

    merged = _merge_matches([p for p in payloads if p], query)
    return merged


async def _run_search_job(job_id: str, query: str):
    # each job uses its own DB session
    db = SessionLocal()
    try:
        rows: List[Entity] = db.query(Entity).order_by(Entity.name).all()
        candidates: List[Tuple[str, str]] = [
            (r.name, r.bio or "") for r in rows if r.name
        ]

        # build chunks
        chunks: List[List[Tuple[str, str]]] = []
        buf: List[Tuple[str, str]] = []
        for pair in candidates:
            buf.append(pair)
            if len(buf) >= TUPLES_PER_CHUNK:
                chunks.append(buf)
                buf = []
        if buf:
            chunks.append(buf)

        async with STORE_LOCK:
            PROGRESS_STORE[job_id] = {
                "job_id": job_id,
                "done": 0,
                "total": len(chunks),
                "status": "running",
                "message": None,
            }

        sem = asyncio.Semaphore(8)

        async def run_chunk(ch):
            async with sem:
                try:
                    payload = await _cached_llm_chunk_match(db, query, ch)
                except Exception as e:
                    LOGGER.exception(f"error on {ch}")
                    payload = None
                finally:
                    async with STORE_LOCK:
                        st = PROGRESS_STORE.get(job_id)
                        if st:
                            st["done"] = min(st["done"] + 1, st["total"])
                return payload

        payloads = await asyncio.gather(*[run_chunk(ch) for ch in chunks])
        merged = _merge_matches([p for p in payloads if p], query)

        async with STORE_LOCK:
            RESULT_STORE[job_id] = merged
            st = PROGRESS_STORE.get(job_id)
            if st:
                st["status"] = "done"
                st["message"] = None

    except Exception as e:
        async with STORE_LOCK:
            st = PROGRESS_STORE.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "done": 0,
                    "total": 0,
                    "status": "error",
                    "message": str(e),
                },
            )
            st["status"] = "error"
            st["message"] = str(e)
    finally:
        db.close()


@app.post("/people/search/start")
async def start_people_search(req: StartRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    job_id = uuid.uuid4().hex
    async with STORE_LOCK:
        PROGRESS_STORE[job_id] = {
            "job_id": job_id,
            "done": 0,
            "total": 0,
            "status": "running",
            "message": None,
        }
    asyncio.create_task(_run_search_job(job_id, q))
    return {"job_id": job_id}


@app.get("/people/search/progress", response_model=ProgressResponse)
async def people_search_progress(job_id: str = Query(...)):
    async with STORE_LOCK:
        st = PROGRESS_STORE.get(job_id)
        if not st:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return st


@app.get("/people/search/result", response_model=ResultResponse)
async def people_search_result(job_id: str = Query(...)):
    async with STORE_LOCK:
        res = RESULT_STORE.get(job_id)
        st = PROGRESS_STORE.get(job_id)
        if not st:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if st["status"] == "error":
            raise HTTPException(
                status_code=500, detail=st.get("message") or "Search failed"
            )
        if st["status"] != "done" or not res:
            raise HTTPException(status_code=202, detail="Not ready")
        return res


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
