# cache_utils.py
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy.orm import Session

from models import LlmChunkCache

SYSTEM_PROMPT_VERSION = "v1"  # bump this if you change the matching prompt
CACHE_TTL = timedelta(days=14)  # set None to disable expiry


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def chunk_fingerprint(candidates: List[Tuple[str, str]]) -> str:
    # deterministic fingerprint for a chunk of (name, bio)
    parts = []
    for name, bio in candidates:
        parts.append(normalize_text(name))
        parts.append("\n")
        parts.append(normalize_text(bio))
        parts.append("\n---\n")
    return _sha256("".join(parts))


def query_fingerprint(query: str) -> str:
    return _sha256(normalize_text(query))


def make_cache_key(
    model: str, system_version: str, query_hash: str, chunk_hash: str
) -> str:
    return f"{model}:{system_version}:{query_hash}:{chunk_hash}"


def cache_get(db: Session, cache_key: str) -> Optional[Dict[str, Any]]:
    row = db.query(LlmChunkCache).filter_by(cache_key=cache_key).first()
    if not row:
        return None
    if CACHE_TTL is not None:
        if row.created_at < datetime.now(timezone.utc) - CACHE_TTL:
            return None
    return row.response_json


def cache_put(
    db: Session,
    *,
    cache_key: str,
    query_hash: str,
    chunk_hash: str,
    model: str,
    system_version: str,
    response_json: Dict[str, Any],
) -> None:
    row = LlmChunkCache(
        cache_key=cache_key,
        query_hash=query_hash,
        chunk_hash=chunk_hash,
        model=model,
        system_version=system_version,
        response_json=response_json,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
