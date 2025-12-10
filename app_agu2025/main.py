from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rapidfuzz import fuzz
from sqlalchemy import create_engine, or_, and_
from sqlalchemy.orm import Session, sessionmaker
from tqdm.auto import tqdm
import faiss
import uvicorn

from crawler.models import RawEntity, Page, EntityBio, CombinedEntity
from crawler.utils import (
    parse_crawler_config,
    load_entity_ids,
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

INDEX_PATH = BASE_DIR / "data" / "entity_index.faiss"

executor = ThreadPoolExecutor(max_workers=5)
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI()
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.auto_reload = True


class SearchRequest(BaseModel):
    query: str
    types: list[str]


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


class FindPeopleRequest(BaseModel):
    names: List[str]


class FindPeopleMatch(BaseModel):
    score: float
    base_name: str
    matched_name: Optional[str]
    match_type: str
    url_list: List[str]


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
        raise RuntimeError(
            "CONFIG_PATH and DB_PATH environment variables must be set"
        )

    CONFIG = parse_crawler_config(Path(config_path))

    DB_URL = f"sqlite:///{db_path}"
    ENGINE = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)

    ENTITY_IDS = load_entity_ids(DB_URL)

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


@app.post("/search", response_model=SearchJobResponse)
async def search(req: SearchRequest, db: Session = Depends(get_db)):
    user_query = (req.query or "").strip()
    if not user_query:
        logging.info("Search request with empty query")
        raise HTTPException(status_code=400, detail="Empty query")

    types = req.types or []
    if not types:
        logging.info("Search request with empty types")
        raise HTTPException(status_code=400, detail="No entity types selected")

    job_id = str(uuid4())
    logging.info(
        "Created search job %s for query: %s; types: %s",
        job_id,
        user_query,
        types,
    )

    JOBS[job_id] = {
        "done": 0,
        "total": 0,
        "status": "queued",
        "message": "Job queued",
        "result": None,
        "query": user_query,
        "job_type": "search",
        "types": types,
    }

    def run_job(job_id: str):
        job = JOBS[job_id]
        user_query = job["query"]
        types = job["types"]
        logging.info(
            "Starting job %s for query: %s; types: %s",
            job_id,
            user_query,
            types,
        )
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
                types,
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
            logging.info(
                "Job %s: analyzing entities with model %s", job_id, MODEL_NAME
            )
            analyses = analyze_results_by_name(
                result_by_name,
                user_query,
                MODEL_NAME,
                CONFIG["max_fetch_retries"],
                max_workers=len(result_by_name),
            )
            for name, value in analyses.items():
                logging.info(f"analysis result: {name}: {value}")
                urls_q = (
                    db.query(Page.url)
                    .join(RawEntity, RawEntity.page_id == Page.id)
                    .filter(RawEntity.name == name)
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
    logging.info("Submitted search job %s to executor", job_id)
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


def _norm_name(s: str) -> str:
    return " ".join(
        "".join(
            ch.lower() if ch.isalnum() or ch in "- " else " " for ch in s
        ).split()
    )


def _normalize_name(s):
    _name_stopwords = {
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "dr",
        "prof",
        "phd",
        "ph",
        "dphil",
        "md",
        "dvm",
        "esq",
        "jr",
        "sr",
        "ii",
        "iii",
        "iv",
        "cpa",
        "she",
        "her",
        "him",
        "he",
        "they",
        "them",
    }

    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    parts = sorted(p for p in s.split() if p not in _name_stopwords)
    return " ".join(parts)


def score_similar_names(base_name, test_name):
    if base_name == test_name:
        return 100
    base_norm = _normalize_name(base_name)
    test_norm = _normalize_name(test_name)
    return fuzz.token_set_ratio(base_norm, test_norm)


def _process_one_person(
    base: str, db: Session
) -> Optional[List[FindPeopleMatch]]:
    base_norm = _norm_name(base)
    base_parts = base_norm.split()
    if not base_parts:
        return None

    exact = (
        db.query(CombinedEntity)
        .filter(
            CombinedEntity.type == "Person",
            CombinedEntity.name.ilike(base),
        )
        .first()
    )
    if exact:
        urls_q = (
            db.query(Page.url)
            .join(RawEntity, RawEntity.page_id == Page.id)
            .filter(RawEntity.combined_entity_id == exact.id)
            .distinct()
            .all()
        )
        url_list = [u[0] for u in urls_q]

        return [
            FindPeopleMatch(
                score=1,
                base_name=base,
                matched_name=exact.name,
                match_type="exact",
                url_list=url_list,
            )
        ]

    def norm_last(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z\- ]+", " ", s)
        parts = s.split()
        if not parts:
            return ""
        last = parts[-1]
        return last.replace("-", "")

    base_last_norm = norm_last(base)

    candidates = (
        db.query(CombinedEntity)
        .filter(CombinedEntity.type == "Person")
        .filter(
            or_(
                CombinedEntity.last_name_norm == base_last_norm,
                and_(
                    CombinedEntity.last_name_norm.is_(None),
                    CombinedEntity.name.ilike(f"%{base_last_norm}%"),
                ),
            )
        )
        .all()
    )

    matches = []

    for c in candidates:
        score = score_similar_names(base, c.name)

        if score == 0:
            # nothing matched
            continue

        urls_q = (
            db.query(Page.url)
            .join(RawEntity, RawEntity.page_id == Page.id)
            .filter(RawEntity.combined_entity_id == c.id)
            .distinct()
            .all()
        )
        url_list = [u[0] for u in urls_q]

        matches.append(
            FindPeopleMatch(
                score=score,
                base_name=base,
                matched_name=c.name,
                match_type="partial",
                url_list=url_list,
            )
        )

    if matches:
        return matches

    return [
        FindPeopleMatch(
            score=0,
            base_name=base,
            matched_name=None,
            match_type="not matched",
            url_list=[],
        )
    ]


def run_find_people_job(job_id: str):
    job = JOBS[job_id]
    names: List[str] = job["names"]
    logging.info("Starting find_people job %s for %d names", job_id, len(names))
    db = SessionLocal()
    try:
        job["status"] = "running"
        job["message"] = "Finding people"
        job["done"] = 0
        job["total"] = len(names)

        results: List[List[FindPeopleMatch]] = []

        for base in tqdm(names, desc=f"find_people {job_id}"):
            match_list = _process_one_person(base, db)
            if match_list is not None:
                results.append(match_list)
            job["done"] += 1

        job["result"] = [
            [m.dict() for m in match_list] for match_list in results
        ]
        job["status"] = "done"
        job["message"] = "Job completed"
        logging.info(
            "find_people job %s completed (%d of %d names matched)",
            job_id,
            len(results),
            len(names),
        )
    except Exception as exc:
        job["status"] = "error"
        job["message"] = f"Job failed: {exc}"
        job["result"] = None
        logging.exception("find_people job %s failed", job_id)
    finally:
        db.close()


@app.post("/find_people", response_model=SearchJobResponse)
async def find_people(req: FindPeopleRequest):
    names: List[str] = []
    for raw_name in req.names:
        base = " ".join((raw_name or "").split())
        if base:
            names.append(base)

    if not names:
        raise HTTPException(status_code=400, detail="No names provided")

    job_id = str(uuid4())
    logging.info("Created find_people job %s for %d names", job_id, len(names))

    JOBS[job_id] = {
        "done": 0,
        "total": len(names),
        "status": "queued",
        "message": "Job queued",
        "result": None,
        "names": names,
        "job_type": "find_people",
    }

    executor.submit(run_find_people_job, job_id)
    logging.info("Submitted find_people job %s to executor", job_id)
    return SearchJobResponse(job_id=job_id)


@app.get("/find_people_status", response_model=SearchProgressResponse)
async def find_people_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        logging.info(
            "find_people status requested for unknown job_id %s", job_id
        )
        raise HTTPException(status_code=404, detail="Job not found")
    logging.debug(
        "find_people status for job %s: done=%s total=%s status=%s",
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


@app.get("/find_people_result", response_model=list[list[FindPeopleMatch]])
async def find_people_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        logging.info(
            "find_people result requested for unknown job_id %s", job_id
        )
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done" or job.get("result") is None:
        logging.info(
            "find_people result requested for incomplete job %s (status=%s)",
            job_id,
            job.get("status"),
        )
        raise HTTPException(status_code=202, detail="Job not finished")
    logging.info("Returning find_people result for job %s", job_id)
    return [
        [FindPeopleMatch(**m) for m in match_list]
        for match_list in job["result"]
    ]


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
