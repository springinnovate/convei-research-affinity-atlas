import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import logging

import faiss
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker
import uvicorn
from dotenv import load_dotenv


from crawler.models import EntityBio, Page, Entity
from crawler.utils import (
    parse_crawler_config,
    load_embedding_index,
    build_index,
    build_entity_context_for_query,
    analyze_results_by_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DB_URL: Optional[str] = None
ENGINE = None
SessionLocal = None
CONFIG: Dict[str, Any] = {}
EMBEDDING_INDEX = None
ENTITY_IDS: List[int] = []

INDEX_PATH = BASE_DIR / "entity_index.faiss"

executor = ThreadPoolExecutor(max_workers=4)
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.auto_reload = True


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    matches: List[Dict[str, Any]]
    notes: Optional[str] = None


class SearchJobResponse(BaseModel):
    job_id: str


class SearchProgressResponse(BaseModel):
    job_id: str
    done: int
    total: int
    status: str
    message: Optional[str] = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    global DB_URL, ENGINE, SessionLocal, CONFIG, EMBEDDING_INDEX, ENTITY_IDS, MODEL_NAME

    MODEL_NAME = os.environ.get("APP_LLM_MODEL")
    config_path = os.environ.get("CONFIG_PATH")
    db_path = os.environ.get("DB_PATH")
    if not config_path or not db_path:
        raise RuntimeError("CONFIG_PATH and DB_PATH environment variables must be set")

    CONFIG = parse_crawler_config(Path(config_path))

    DB_URL = f"sqlite:///{db_path}"
    ENGINE = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)

    ENTITY_IDS, embedding_vectors = load_embedding_index(DB_URL)

    if not INDEX_PATH.exists():
        raise RuntimeError(f"{INDEX_PATH} seems to not exist, build it first.")
    else:
        EMBEDDING_INDEX = faiss.read_index(str(INDEX_PATH))


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/entities/")
async def list_entities(db: Session = Depends(get_db)):
    rows = (
        db.query(EntityBio.name, EntityBio.type)
        .filter(EntityBio.name.isnot(None))
        .all()
    )

    sorted_pairs = sorted(
        rows,
        key=lambda p: p[0].strip().split()[-1].lower(),
    )

    entities = [{"name": name, "type": etype} for name, etype in sorted_pairs]

    return {"entities": entities}


class EntityBioRequest(BaseModel):
    type: Optional[str] = None
    name: str


class EntityBioResponse(BaseModel):
    type: str
    name: str
    bio: str
    url_list: List[str]


@app.post("/entity_bio", response_model=EntityBioResponse)
async def entity_bio(req: EntityBioRequest, db: Session = Depends(get_db)):
    logging.info(f"querying for {req.name}")
    q = db.query(EntityBio).filter(EntityBio.name == req.name)

    if req.type is not None:
        q = q.filter(EntityBio.type == req.type)

    bio_row = q.one_or_none()
    logging.info(f"got this result {bio_row} for {req.name}")
    if not bio_row:
        raise HTTPException(status_code=404, detail="Bio not found")

    urls_q = (
        db.query(Page.url)
        .join(Entity, Entity.page_id == Page.id)
        .filter(Entity.name == req.name)
    )

    if req.type is not None:
        urls_q = urls_q.filter(Entity.type == req.type)

    urls = urls_q.distinct().all()
    url_list = [u[0] for u in urls]

    return EntityBioResponse(
        type=bio_row.type,
        name=bio_row.name,
        bio=bio_row.bio,
        url_list=url_list,
    )


@app.post("/search", response_model=SearchJobResponse)
async def search(req: SearchRequest, db: Session = Depends(get_db)):
    user_query = (req.query or "").strip()
    if not user_query:
        logging.info("Search request with empty query")
        raise HTTPException(status_code=400, detail="Empty query")

    job_id = str(uuid4())
    logging.info("Created search job %s for query: %s", job_id, user_query)

    JOBS[job_id] = {
        "done": 0,
        "total": 0,
        "status": "queued",
        "message": "Job queued",
        "result": None,
        "query": user_query,
    }

    def run_job(job_id: str):
        job = JOBS[job_id]
        user_query = job["query"]
        logging.info("Starting job %s for query: %s", job_id, user_query)
        try:
            job["status"] = "building_context"
            job["message"] = "Building entity context"
            logging.info("Job %s: building entity context", job_id)
            result_by_name = build_entity_context_for_query(
                user_query,
                DB_URL,
                EMBEDDING_INDEX,
                ENTITY_IDS,
                CONFIG,
            )
            job["total"] = max(len(result_by_name), 1)
            job["done"] = 1
            logging.info(
                "Job %s: built context for %d entities",
                job_id,
                len(result_by_name),
            )

            job["status"] = "analyzing"
            job["message"] = "Analyzing entities with LLM"
            logging.info("Job %s: analyzing entities with model %s", job_id, MODEL_NAME)
            analyses = analyze_results_by_name(
                result_by_name,
                user_query,
                MODEL_NAME,
                CONFIG["max_fetch_retries"],
                max_workers=len(result_by_name),
            )
            for name, value in analyses.items():
                urls_q = (
                    db.query(Page.url)
                    .join(Entity, Entity.page_id == Page.id)
                    .filter(Entity.name == name)
                )
                urls = urls_q.distinct().all()
                url_list = [u[0] for u in urls]
                value["url_list"] = url_list
            logging.info(
                "Job %s: analysis complete, %d matches",
                job_id,
                len(analyses),
            )

            resp = SearchResponse(
                query=user_query,
                matches=list(analyses.values()),
                notes=None,
            )
            job["result"] = resp.dict()
            job["done"] = job["total"]
            job["status"] = "done"
            job["message"] = "Job completed"
            logging.info("Job %s: completed successfully", job_id)
        except Exception as exc:
            job["status"] = "error"
            job["message"] = f"Job failed: {exc}"
            job["result"] = None
            logging.exception("Job %s: failed with exception", job_id)

    executor.submit(run_job, job_id)
    logging.info("Submitted job %s to executor", job_id)
    return SearchJobResponse(job_id=job_id)


@app.get("/search_progress", response_model=SearchProgressResponse)
async def search_progress(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        logging.info("Progress requested for unknown job_id %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    logging.debug(
        "Progress for job %s: done=%s total=%s status=%s",
        job_id,
        job.get("done", 0),
        job.get("total", 0),
        job.get("status", "unknown"),
    )
    return SearchProgressResponse(
        job_id=job_id,
        done=job.get("done", 0),
        total=job.get("total", 0),
        status=job.get("status", "unknown"),
        message=job.get("message"),
    )


@app.get("/search_result", response_model=SearchResponse)
async def search_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        logging.info("Result requested for unknown job_id %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done" or job.get("result") is None:
        logging.info(
            "Result requested for incomplete job %s (status=%s)",
            job_id,
            job.get("status"),
        )
        raise HTTPException(status_code=202, detail="Job not finished")
    logging.info("Returning result for job %s", job_id)
    return SearchResponse(**job["result"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
