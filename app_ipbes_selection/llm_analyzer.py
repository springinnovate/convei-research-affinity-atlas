"""Module to handle all LLM analysis."""

from typing import Any, Dict, List
import time
import httpx
import hashlib
import asyncio
from pathlib import Path
import logging
import random
import json
import sys


from openai import (
    BadRequestError,
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tiktoken import get_encoding


_TOKENS_USED_LAST_MINUTE = []
_REQUESTS_LAST_MINUTE = []

# OpenAI account/model limits
TPM_LIMIT = 2_000_000
RPM_LIMIT = 10_000

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("hpack").setLevel(logging.WARNING)

load_dotenv()

_ENCODER = get_encoding("cl100k_base")


CONCURRENCY = 64  # adjust to your sustainable QPS

limits = httpx.Limits(
    max_connections=CONCURRENCY * 2,
    max_keepalive_connections=CONCURRENCY * 2,
    keepalive_expiry=30.0,  # <-- belongs here, not on transport
)

transport = httpx.AsyncHTTPTransport(
    http2=True,  # ok to keep
    # no 'keepalive_expiry' or 'retries' here
)

http_client = httpx.AsyncClient(
    transport=transport,
    limits=limits,
    timeout=httpx.Timeout(
        connect=5.0,
        read=180.0,  # bump if generations are long
        write=30.0,
        pool=30.0,
    ),
)

OPENAI_CLIENT = AsyncOpenAI(
    http_client=http_client,
    timeout=180.0,
    max_retries=0,  # keep your own backoff logic
)

BIO_GEN_SEMAPHORE = asyncio.Semaphore(CONCURRENCY)

CACHE_FILE = Path("openai_cache.json")
_cache: dict[str, str] = {}


def _make_cache_key(messages: list[dict[str, Any]], openai_model: str) -> str:
    """Create a stable hash key for messages + model."""
    m_json = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    key_input = f"{openai_model}:{m_json}"
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


CACHE_LOCK = asyncio.Lock()


async def _load_cache():
    global _cache
    async with CACHE_LOCK:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
            except json.JSONDecodeError:
                _cache = {}
        else:
            _cache = {}


async def _save_cache():
    async with CACHE_LOCK:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)


async def safe_openai_completion(*args, **kwargs):
    async with BIO_GEN_SEMAPHORE:
        return await _safe_openai_completion(*args, **kwargs)


def estimate_tokens_for_messages(messages: list[Dict]) -> int:
    """Rough token count for messages (system+user+assistant roles)."""
    # Each message has some overhead in ChatML format; +3 per message, +3 overall
    tokens = 3
    for m in messages:
        tokens += 3  # per message overhead
        tokens += len(_ENCODER.encode(m.get("content", "")))
    return tokens


async def _wait_for_rate_limit(tokens_needed: int):
    """Wait until sending this many tokens won't exceed TPM/RPM."""
    global _TOKENS_USED_LAST_MINUTE, _REQUESTS_LAST_MINUTE

    while True:
        now = time.time()
        _TOKENS_USED_LAST_MINUTE = [
            (t, ts) for t, ts in _TOKENS_USED_LAST_MINUTE if now - ts < 60
        ]
        _REQUESTS_LAST_MINUTE = [
            ts for ts in _REQUESTS_LAST_MINUTE if now - ts < 60
        ]

        used_tokens = sum(t for t, _ in _TOKENS_USED_LAST_MINUTE)
        used_requests = len(_REQUESTS_LAST_MINUTE)

        if (used_tokens + tokens_needed <= TPM_LIMIT) and (
            used_requests + 1 <= RPM_LIMIT
        ):
            _TOKENS_USED_LAST_MINUTE.append((tokens_needed, now))
            _REQUESTS_LAST_MINUTE.append(now)
            return

        await asyncio.sleep(0.05)


async def _safe_openai_completion(
    messages: List[Dict[str, Any]],
    openai_model: str,
    timeout: float = 900.0,
    max_attempts: int = 3,
    base_backoff: float = 0.75,
    max_backoff: float = 8.0,
    job_id: str = None,
) -> Any:

    if not _cache:
        await _load_cache()

    cache_key = _make_cache_key(messages, openai_model)
    if cache_key in _cache:
        return _cache[cache_key]

    attempt = 0
    # Avoid mutating caller-provided lists/dicts
    msgs = [dict(m) for m in messages]

    while True:
        attempt += 1
        try:
            kwargs = {
                "model": openai_model,
                "messages": msgs,
                "timeout": timeout,
            }

            result = await OPENAI_CLIENT.chat.completions.create(**kwargs)
            content = result.choices[0].message.content

            _cache[cache_key] = content
            await _save_cache()
            return content

        except BadRequestError as e:
            LOGGER.exception(
                f"{job_id}: bad request error forasyncio cancelled error: {e}"
            )
            # Caller likely sent an invalid request (schema/content/too long). Do not retry.
            raise

        except (
            RateLimitError,
            APIError,
            APIConnectionError,
            APITimeoutError,
        ) as e:
            LOGGER.warning(f"{job_id}: some sort of rate limit error: {e}")
            if attempt >= max_attempts:
                raise
            # Exponential backoff with jitter
            sleep_for = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            sleep_for += random.uniform(0, 0.25 * sleep_for)
            LOGGER.warning(
                "OpenAI call failed (attempt %d/%d): %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                str(e),
                sleep_for,
            )
            await asyncio.sleep(sleep_for)

        except asyncio.CancelledError as e:
            LOGGER.warning(f"{job_id}: asyncio cancelled error: {e}")
            # Preserve task cancellation
            raise

        except Exception as e:
            if attempt >= max_attempts:
                raise
            sleep_for = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            sleep_for += random.uniform(0, 0.25 * sleep_for)
            LOGGER.warning(
                "Unexpected error during OpenAI call (attempt %d/%d): %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                str(e),
                sleep_for,
            )
            await asyncio.sleep(sleep_for)
