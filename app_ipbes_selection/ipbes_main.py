"""Entrypoint for CONVEI research affinity atlas app."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import json
import logging
import os
import sys
import uuid

from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
import uvicorn


from pdf_miner import (
    SELECTION_COMMITTEE_BASE_MESSAGE,
    chunk_people_into_batches,
)
from llm_analyzer import (
    estimate_tokens_for_messages,
    safe_openai_completion,
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
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.auto_reload = True

PROGRESS_STORE: Dict[str, Dict[str, Any]] = {}
RESULT_STORE: Dict[str, Dict[str, Any]] = {}
STORE_LOCK = asyncio.Lock()


MAX_TOKENS_PER_CHUNK = 100000

MERGED_OUTPUT_JSON_PATH = "merged_output.json"


@app.on_event("startup")
async def startup_event():
    global NAME_TO_CONTEXT
    NAME_TO_CONTEXT = json.loads(
        open(MERGED_OUTPUT_JSON_PATH, "r", encoding="utf-8").read()
    )


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_info/")
def get_info():
    timestamp = os.path.getmtime(MERGED_OUTPUT_JSON_PATH)
    formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return {
        "Number of people": len(NAME_TO_CONTEXT),
        "Timestamp": formatted_timestamp,
    }


class SearchRequest(BaseModel):
    query: str


class SearchSubmitResponse(BaseModel):
    job_id: str
    total_batches: int


class SearchProgressResponse(BaseModel):
    job_id: str
    status: str
    total_batches: int
    submitted: int
    completed: int
    percent: float
    started_at: str
    finished_at: Optional[str] = None
    errors: List[str] = []


class SearchResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = []


async def _process_job(job_id: str):
    async with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    try:
        user_question = job.query

        batches = chunk_people_into_batches(
            system_content=SELECTION_COMMITTEE_BASE_MESSAGE,
            user_question=user_question,
            name_to_context=NAME_TO_CONTEXT,
            token_limit=MAX_TOKENS_PER_CHUNK,
            estimate_tokens_for_messages=estimate_tokens_for_messages,
            reserved_for_response=1500,
        )

        async with _JOBS_LOCK:
            job.total_batches = len(batches)

        sem = asyncio.Semaphore(_CONCURRENCY)

        async def _run_one(batch_msgs):
            async with sem:
                try:
                    payload = await safe_openai_completion(
                        batch_msgs, "gpt-5-mini"
                    )
                except Exception as e:
                    async with _JOBS_LOCK:
                        job.errors.append(repr(e))
                        job.submitted += 1
                    return None
                else:
                    async with _JOBS_LOCK:
                        job.submitted += 1
                    return payload

        tasks = [asyncio.create_task(_run_one(m)) for m in batches]

        results: List[Any] = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res is not None:
                results.append(res)
            async with _JOBS_LOCK:
                job.completed += 1

        merged = _merge_matches([p for p in results if p], job.query)

        async with _JOBS_LOCK:
            job.results = [merged]
            job.status = "done"
            job.finished_at = datetime.utcnow().isoformat() + "Z"

    except Exception as e:
        async with _JOBS_LOCK:
            job.status = "error"
            job.errors.append(repr(e))
            job.finished_at = datetime.utcnow().isoformat() + "Z"


def _merge_matches(payloads, query):
    def _parse(p):
        if p is None:
            return {}
        if isinstance(p, str):
            try:
                return json.loads(p)
            except Exception:
                return {}
        if isinstance(p, dict):
            # already parsed JSON
            return p
        # best-effort for SDK-like objects
        try:
            content = (
                p.choices[0].message["content"]
                if isinstance(p.choices[0].message, dict)
                else p.choices[0].message.content
            )
            return json.loads(content)
        except Exception:
            return {}

    def _merge_evidence(a, b):
        a = a or []
        b = b or []
        seen = {(e.get("source", ""), e.get("snippet", "")) for e in a}
        out = list(a)
        for e in b:
            key = (e.get("source", ""), e.get("snippet", ""))
            if key not in seen:
                out.append(
                    {
                        "source": e.get("source", ""),
                        "snippet": e.get("snippet", ""),
                    }
                )
                seen.add(key)
        return out

    shortlist_map = {}  # key: lower(name) -> record
    near_map = {}  # key: lower(name) -> record
    unknown = set()
    notes = []

    for p in payloads:
        data = _parse(p)
        if not data:
            continue

        if isinstance(data.get("notes"), str) and data["notes"].strip():
            notes.append(data["notes"].strip())

        for item in data.get("shortlist", []) or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in shortlist_map:
                shortlist_map[key] = {
                    "name": name,
                    "score": int(item.get("score", 0) or 0),
                    "summary": item.get("summary", "") or "",
                    "evidence": item.get("evidence") or [],
                    "fit_tags": sorted(set(item.get("fit_tags") or [])),
                    "confidence": float(item.get("confidence", 0) or 0.0),
                }
            else:
                cur = shortlist_map[key]
                cur["score"] = max(
                    cur.get("score", 0), int(item.get("score", 0) or 0)
                )
                cur["confidence"] = max(
                    float(cur.get("confidence", 0) or 0.0),
                    float(item.get("confidence", 0) or 0.0),
                )
                # keep longer summary
                cand_sum = item.get("summary", "") or ""
                if len(cand_sum) > len(cur.get("summary", "") or ""):
                    cur["summary"] = cand_sum
                # merge evidence and tags
                cur["evidence"] = _merge_evidence(
                    cur.get("evidence"), item.get("evidence")
                )
                cur["fit_tags"] = sorted(
                    set(
                        (cur.get("fit_tags") or [])
                        + (item.get("fit_tags") or [])
                    )
                )

        for nm in data.get("near_misses", []) or []:
            name = (nm.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in shortlist_map:
                # already shortlisted elsewhere; fold in any extra evidence
                shortlist_map[key]["evidence"] = _merge_evidence(
                    shortlist_map[key].get("evidence"), nm.get("evidence")
                )
                continue
            if key not in near_map:
                near_map[key] = {
                    "name": name,
                    "reason": nm.get("reason", "") or "",
                    "evidence": nm.get("evidence") or [],
                }
            else:
                cur = near_map[key]
                if len(nm.get("reason", "") or "") > len(
                    cur.get("reason", "") or ""
                ):
                    cur["reason"] = nm.get("reason", "") or ""
                cur["evidence"] = _merge_evidence(
                    cur.get("evidence"), nm.get("evidence")
                )

        for unk in data.get("unknown_or_insufficient", []) or []:
            if isinstance(unk, str) and unk.strip():
                unknown.add(unk.strip())

    # remove unknowns that are shortlisted or near-miss elsewhere
    unknown = sorted(
        n
        for n in unknown
        if n.lower() not in shortlist_map and n.lower() not in near_map
    )

    merged = {
        "query": query,
        "shortlist": sorted(
            shortlist_map.values(),
            key=lambda x: (
                -x.get("score", 0),
                -float(x.get("confidence", 0) or 0.0),
                x.get("name", ""),
            ),
        ),
        "near_misses": sorted(
            near_map.values(), key=lambda x: x.get("name", "")
        ),
        "unknown_or_insufficient": unknown,
        "notes": " | ".join(dict.fromkeys(n for n in notes if n))[:2000],
    }
    return merged


@dataclass
class JobState:
    job_id: str
    query: str
    total_batches: int
    submitted: int = 0
    completed: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    finished_at: Optional[str] = None
    status: str = "running"  # running|done|error|cancelled
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


JOBS: Dict[str, JobState] = {}
_JOBS_LOCK = asyncio.Lock()
_CONCURRENCY = 32  # tune as needed


class PersonRequest(BaseModel):
    person_name: str


@app.post("/person/bio")
def get_person_bio(payload: PersonRequest, request: Request):
    return {
        "name": payload.person_name,
        "bio": "Bio implementation under construction",
        "url_list": [
            (
                request.url_for("static", path=file_path)
                for file_path in NAME_TO_CONTEXT[payload.person_name]["files"]
            )
        ],
    }


@app.post("/people/search/start", response_model=SearchSubmitResponse)
async def people_search_start(req: SearchRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    job_id = str(uuid.uuid4())
    job = JobState(job_id=job_id, query=query, total_batches=0)

    async with _JOBS_LOCK:
        JOBS[job_id] = job

    asyncio.create_task(_process_job(job_id))
    return SearchSubmitResponse(job_id=job_id, total_batches=0)


@app.get("/people/search/result/{job_id}", response_model=JobState)
async def people_search_result(job_id: str):
    async with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


@app.get("/people/search/progress/{job_id}", response_model=JobState)
async def people_search_progress(job_id: str):
    async with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    LOGGER.info(job)
    return job


@app.get("/entities/")
async def list_entities():
    sorted_names = list(sorted(NAME_TO_CONTEXT))
    return {"entities": sorted_names}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
