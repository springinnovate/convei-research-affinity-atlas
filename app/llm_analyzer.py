"""Module to handle all LLM analysis."""

from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import json
import logging
import random
import sys
import time


from openai import (
    BadRequestError,
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tiktoken import encoding_for_model, get_encoding
from tqdm import tqdm

from models import Entity, ProcessedFile


BIO_GEN_SEMAPHORE = asyncio.Semaphore(50)
_TOKENS_USED_LAST_MINUTE = []
_REQUESTS_LAST_MINUTE = []

OPENAI_MODEL = "gpt-5-nano"
try:
    _enc = encoding_for_model(OPENAI_MODEL)
except KeyError:
    _enc = get_encoding("cl100k_base")


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


async def safe_openai_completion(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    *,
    timeout: float = 900.0,
    max_attempts: int = 3,
    base_backoff: float = 0.75,
    max_backoff: float = 8.0,
) -> Any:
    attempt = 0
    # Avoid mutating caller-provided lists/dicts
    msgs = [dict(m) for m in messages]
    tools_payload = [dict(t) for t in tools] if tools else None

    while True:
        attempt += 1
        try:
            kwargs = {
                "model": OPENAI_MODEL,
                "messages": msgs,
                "timeout": timeout,
            }
            if tools_payload is not None:
                kwargs["tools"] = tools_payload
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            return await OPENAI_CLIENT.chat.completions.create(**kwargs)

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


def estimate_tokens_for_messages(messages: list[Dict]) -> int:
    """Rough token count for messages (system+user+assistant roles)."""
    # Each message has some overhead in ChatML format; +3 per message, +3 overall
    tokens = 3
    for m in messages:
        tokens += 3  # per message overhead
        tokens += len(_enc.encode(m.get("content", "")))
    return tokens


async def _wait_for_rate_limit(tokens_needed: int):
    """Wait until sending this many tokens won't exceed TPM/RPM."""
    global _TOKENS_USED_LAST_MINUTE, _REQUESTS_LAST_MINUTE

    while True:
        now = time.time()
        _TOKENS_USED_LAST_MINUTE = [
            (t, ts) for t, ts in _TOKENS_USED_LAST_MINUTE if now - ts < 60
        ]
        _REQUESTS_LAST_MINUTE = [ts for ts in _REQUESTS_LAST_MINUTE if now - ts < 60]

        used_tokens = sum(t for t, _ in _TOKENS_USED_LAST_MINUTE)
        used_requests = len(_REQUESTS_LAST_MINUTE)

        if (used_tokens + tokens_needed <= TPM_LIMIT) and (
            used_requests + 1 <= RPM_LIMIT
        ):
            _TOKENS_USED_LAST_MINUTE.append((tokens_needed, now))
            _REQUESTS_LAST_MINUTE.append(now)
            return

        await asyncio.sleep(0.05)


async def _generate_single_bio(name, entity_bio_context, url_list, db):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful, precise biographer.\n\n"
                "You will be given free-form context text gathered from conference materials, schedules, websites, or documents. "
                "The text may include affiliations, session or presentation titles, abstract excerpts, or other mentions of the person — "
                "but the information will not be pre-labeled.\n\n"
                "Your job is to:\n"
                "1. Identify and extract any verifiable facts about the person (affiliations, roles, research areas, activities, methods, locations, sectors, taxa, etc.).\n"
                "2. Synthesize these into a professional biography that is as long as needed to capture all grounded details.\n"
                '3. Include well-supported inferences about their expertise or focus, using cautious language such as "appears to", "likely", or "suggests work on".\n'
                "4. If context is sparse or ambiguous, produce a shorter bio and clearly note uncertainty or assumptions.\n\n"
                "Output style:\n"
                "- Neutral, professional; no hype.\n"
                "- 1st paragraph: who/where/what (focus + context).\n"
                "- 2nd paragraph (optional): key study or activity, brief method clause, main finding(s), significance.\n"
                "Rules:\n"
                "- Lead with a direct, grounded first sentence (name + affiliation if present + domain focus).\n"
                "- Always include organism/system and geography when available.\n"
                "- Summarize methods and analysis in ≤1 short clause; avoid parameter lists or instrument catalogs.\n"
                "- Mention collaborations by institutions (e.g., 'with Iowa Lakeside Laboratory and Cary Institute'), not full author lists.\n"
                "- Add one sentence on significance/implications when supported by the context.\n"
                "- Do not repeat titles verbatim if already summarized; de-duplicate overlapping phrases.\n"
                "- Use cautious wording only when evidence is thin; otherwise state grounded facts directly.\n"
                "- Never fabricate degrees, titles, institutions, dates, locations, funders, or quantitative results.\n"
                "- If making an assumption, end with a single sentence beginning 'Assumptions:'.\n"
                "- Length is unlimited but expand only when adding grounded, meaningful content.\n"
                "- Do not infer employment or institutional affiliation from session hosts or venues. Only state an affiliation if it appears explicitly next to the name (e.g., 'Name, Organization') or in an 'Affiliation' field. Otherwise say 'associated with' or omit.\n"
                "- Do not assert roles like 'participated', 'led', 'organized', 'panelist', or 'presented' unless the context uses those words (e.g., 'Presenting Author', 'Moderator', 'Organizer'). If unclear, use neutral phrasing: 'listed in the program for…', 'appears in session materials for…'.\n"
                "- When citing frameworks or initiatives (e.g., Global Biodiversity Framework, Science-Based Targets), only include them if they are explicitly named in the context. If you infer alignment, mark it as an assumption in one brief sentence at the end.\n"
                "- Keep methods to ≤1 short clause; avoid long parameter or instrument lists. Prefer the take-home finding and significance.\n"
                "- De-duplicate repeated titles/snippets; summarize once.\n"
            ),
        },
        {
            "role": "user",
            "content": f"Name: {name}\n\nContext:\n{entity_bio_context}\n\nEnd of context.",
        },
    ]

    try:
        tokens_needed = estimate_tokens_for_messages(messages)
        LOGGER.info(f"{tokens_needed} tokens needed for {name}, waiting our turn.")
        await _wait_for_rate_limit(tokens_needed)
        LOGGER.info(f"{name}, OK GO")
        completion = await safe_openai_completion(messages)

        if isinstance(completion, str):
            LOGGER.error(f"Bio generation failed for {name}: {completion}")
            return None

        result = completion.choices[0].message.content.strip()
        LOGGER.info(f"{name} is DONE! Passing back up the chain.")
        return Entity(
            name=name,
            bio=result,
            bio_source=entity_bio_context,
            url_list=url_list,
        )

    except Exception as e:
        LOGGER.error(f"Error generating bio for {name}: {e}")
        return None


async def sem_task(*args, **kwargs):
    async with BIO_GEN_SEMAPHORE:
        return await _generate_single_bio(*args, **kwargs)


async def generate_bios(input_json_path: str, db: Session):
    if db.query(ProcessedFile).filter_by(filename=input_json_path).first() is not None:
        LOGGER.info(f"{input_json_path} is already processed!")
        return

    raw_entities = json.loads(open(input_json_path, "r", encoding="utf-8").read())
    raw_entities.sort(key=itemgetter("speaker"))

    tasks = []

    for index, (name, group) in enumerate(
        groupby(raw_entities, key=itemgetter("speaker"))
    ):
        if db.query(Entity).filter_by(name=name).first() is not None:
            LOGGER.info(f"already processed {name}, skipping")
            continue
        LOGGER.info(f"queuing up {name}")
        group_list = list(group)
        entity_bio_context = " ".join(e["content"] for e in group_list)
        tasks.append(
            sem_task(name, entity_bio_context, [e["url"] for e in group_list], db)
        )

    LOGGER.info(f"waiting for the results of {len(tasks)} tasks")
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generating bios"
    ):
        try:
            entity = await coro
        except Exception as e:
            LOGGER.error(f"Task failed: {e}")
            continue

        if entity:
            db.add(entity)
            db.commit()

    db.add(ProcessedFile(filename=input_json_path))
    db.commit()


CHUNK_SIZE = 250  # tune based on average bio size and model context


def _format_chunk_user_content(query: str, candidates: List[Tuple[str, str]]) -> str:
    parts = [f"Interest query:\n{query}\n", "Candidates:"]
    for name, bio in candidates:
        bio = (bio or "").strip()
        parts.append("=== CANDIDATE START ===")
        parts.append(f"Name: {name}")
        parts.append("Bio:")
        parts.append(bio if bio else "(no bio)")
        parts.append("=== CANDIDATE END ===")
    return "\n".join(parts)


async def _llm_chunk_match(query: str, chunk: List[Tuple[str, str]]) -> Dict[str, Any]:
    system_prompt = (
        "Your task is to read an interest query and a set of bios, and return all people whose bios show direct, adjacent, or emerging relevance to the query.\n\n"
        "Given (A) a free-form interest query and (B) a set of candidate people (each with a name and a bio), identify everyone whose work plausibly aligns with the query and explain why. Return all grounded matches—do not limit the count.\n\n"
        "Match Types\n"
        '- "direct": clear topical/keyword/framework/method overlap with the query\n'
        '- "adjacent": neighboring themes, methods, systems, regions, or policy contexts likely useful to the query\n'
        '- "emerging": weak but intriguing overlap; include only when a specific phrase in the bio suggests a connection\n\n'
        "Inclusion rule\n"
        "- Include a person if you can point to at least one explicit phrase in the bio that supports the relation. Otherwise, exclude.\n\n"
        "Output (valid JSON):\n"
        "{\n"
        '  "query": string,\n'
        '  "matches": [\n'
        "    {\n"
        '      "name": string,\n'
        '      "relation": "direct" | "adjacent" | "emerging",\n'
        '      "why": string,\n'
        '      "matched_signals": string[],\n'
        '      "snippet": string,\n'
        '      "confidence": "low" | "medium" | "high"\n'
        "    }\n"
        "  ],\n"
        '  "notes": string\n'
        "}\n\n"
        "Ordering\n"
        "- Sort matches by usefulness for the query: direct first, then adjacent, then emerging."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_chunk_user_content(query, chunk)},
    ]
    completion = await safe_openai_completion(messages, timeout=90.0)
    content = completion.choices[0].message.content
    try:
        payload = json.loads(content)
    except Exception:
        # try to salvage by locating the first/last braces
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(content[start : end + 1])
        else:
            raise
    return payload


def _relation_rank(rel: str) -> int:
    table = {"direct": 0, "adjacent": 1, "emerging": 2}
    return table.get(str(rel).lower(), 3)


def _conf_rank(conf: str) -> int:
    table = {"high": 0, "medium": 1, "low": 2}
    return table.get(str(conf).lower(), 3)


def _merge_matches(all_payloads: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    by_name: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    for p in all_payloads:
        if not p:
            continue
        if isinstance(p.get("notes"), str) and p["notes"].strip():
            notes.append(p["notes"].strip())
        for m in p.get("matches", []) or []:
            name = m.get("name")
            if not name:
                continue
            cur = by_name.get(name)
            if not cur:
                by_name[name] = {
                    "name": name,
                    "relation": m.get("relation", "adjacent"),
                    "why": m.get("why", ""),
                    "matched_signals": list(
                        dict.fromkeys((m.get("matched_signals") or []))
                    ),
                    "snippet": m.get("snippet", ""),
                    "confidence": m.get("confidence", "medium"),
                }
                continue
            # prefer better relation, then higher confidence; otherwise keep existing but extend signals
            incoming_rel = m.get("relation", cur["relation"])
            incoming_conf = m.get("confidence", cur["confidence"])
            better = (
                _relation_rank(incoming_rel),
                _conf_rank(incoming_conf),
            ) < (_relation_rank(cur["relation"]), _conf_rank(cur["confidence"]))
            if better:
                cur.update(
                    {
                        "relation": incoming_rel,
                        "why": m.get("why", cur["why"]),
                        "snippet": m.get("snippet", cur["snippet"]),
                        "confidence": incoming_conf,
                    }
                )
            # merge matched signals de-duplicated
            cur["matched_signals"] = list(
                dict.fromkeys(cur["matched_signals"] + (m.get("matched_signals") or []))
            )

    merged = list(by_name.values())
    merged.sort(
        key=lambda x: (
            _relation_rank(x["relation"]),
            _conf_rank(x["confidence"]),
            x["name"].lower(),
        )
    )
    return {
        "query": query,
        "matches": merged,
        "notes": " | ".join(notes)[:2000] if notes else None,
    }
