"""Module to handle all LLM analysis."""

from typing import Any, Dict, List
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


BIO_GEN_SEMAPHORE = asyncio.Semaphore(50)
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

load_dotenv()


OPENAI_CLIENT = AsyncOpenAI(timeout=30)

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


async def _safe_openai_completion(
    messages: List[Dict[str, Any]],
    openai_model: str,
    timeout: float = 900.0,
    max_attempts: int = 3,
    base_backoff: float = 0.75,
    max_backoff: float = 8.0,
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

        except BadRequestError:
            # Caller likely sent an invalid request (schema/content/too long). Do not retry.
            raise

        except (
            RateLimitError,
            APIError,
            APIConnectionError,
            APITimeoutError,
        ) as e:
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

        except asyncio.CancelledError:
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
