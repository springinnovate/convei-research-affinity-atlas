"""Module to handle all LLM analysis."""

from contextlib import asynccontextmanager
from datetime import datetime
import collections
import asyncio
import logging
import hashlib
import json
import sys

from sqlalchemy import func, exists, select
from openai import OpenAI
from dotenv import load_dotenv
from openai import AsyncOpenAI, APITimeoutError

from .models import (
    WebpageContent,
    EntityLLMAnalysis,
    Entity,
    EntityWebpageSnippet,
    EntityWebpageContentAssociation,
)
from .database import SessionLocal


MATCH_PEOPLE_PROMPT_TEMPLATE = """
You are an assistant that matches researchers to a user's interests.

Use ONLY the data below.

### User_Interest
{user_interest_text}

### Candidate_Bios  (name -> biography)
{bios_json}

Task:
• Carefully read the user's interests and each candidate bio.
• Identify all candidates whose biographies meaningfully align with the user's interests.
• Include only candidates who have clear relevance; do not add individuals whose relevance is questionable or weak.
• For each selected candidate, provide:
    – name
    – relevance_score (1‑100, higher = better match)
    – rationale (one concise sentence)
    – bio_quote (<= 20 words copied verbatim from the bio)

Do not invent new individuals or facts.
Return your answer by calling the **match_people** function tool.
"""

MATCH_PEOPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relevance_score": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "bio_quote": {"type": "string"},
                },
                "required": [
                    "name",
                    "relevance_score",
                    "rationale",
                    "bio_quote",
                ],
            },
        }
    },
    "required": ["matches"],
}

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

EXTRACT_PEOPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["name", "context"],
            },
        }
    },
    "required": ["people"],
}

OPENAI_CLIENT = AsyncOpenAI(timeout=30)

OPENAI_MODEL = "gpt-4o"


async def safe_openai_completion(messages, tools=None, tool_choice=None):
    attempt = 0
    max_attempts = 3
    while True:
        try:
            attempt += 1
            return await OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                timeout=30,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            LOGGER.warning(
                f"Encountered this error during openaiapi call: {e!s}"
            )
            if attempt == max_attempts:
                return f"***OpenAI error***: {e!s}"


# global lock registry  {(entity_id, ctx_hash): asyncio.Lock()}
_LOCKS = collections.defaultdict(asyncio.Lock)


@asynccontextmanager
async def keyed_lock(entity_id: int, ctx_hash: str):
    key = (entity_id, ctx_hash)
    lock = _LOCKS[key]
    async with lock:
        yield


def get_all_snippets(entity_id: int):
    db = SessionLocal()
    snippets = (
        (
            db.execute(
                select(EntityWebpageSnippet.snippet_text).where(
                    EntityWebpageSnippet.entity_id == entity_id
                )
            )
        )
        .scalars()
        .all()
    )
    context_text = " ".join(snippets)
    context_hash = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
    return context_text, context_hash


async def generate_bio(entity_id: int):
    db = SessionLocal()

    # 1) get any entityLLManalysis that was done before
    # 2) get all the entity_webpage snippet texts for that entity
    context_text, context_hash = get_all_snippets(entity_id)
    if not context_text:
        return "No relevant context found."

    async with keyed_lock(entity_id, context_hash):
        cached = db.execute(
            select(EntityLLMAnalysis.summary).where(
                EntityLLMAnalysis.entity_id == entity_id,
                EntityLLMAnalysis.context_hash == context_hash,
            )
        ).scalar_one_or_none()
        summary = cached
        if cached is not None and summary:
            return cached

        cached = (
            db.query(EntityLLMAnalysis)
            .filter_by(entity_id=entity_id, context_hash=context_hash)
            .first()
        )
        if cached:
            LOGGER.debug(
                f"cached result found, returning that:\n\n{cached.summary}"
            )
            return cached.summary

        entity_name = db.execute(
            select(Entity.name).where(Entity.entity_id == entity_id)
        ).scalar_one_or_none()
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a concise, professional biography based solely on the provided context. "
                    "Clearly summarize the individual's research interests, professional background, affiliations, "
                    "and relevant achievements. "
                    "If information is incomplete or unclear, you may state assumptions explicitly as assumptions. "
                    "Do not invent details not supported by the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"""
                Name: {entity_name}

                Context:
                {context_text}
                """,
            },
        ]

        LOGGER.debug(f"about to ask this question {messages}")
        response = await safe_openai_completion(messages)
        LOGGER.debug(f"got this response: {response}")

        summary = response.choices[0].message.content.strip()

        # get the next llm analysis version
        result = db.execute(
            select(func.coalesce(func.max(EntityLLMAnalysis.version), 0)).where(
                EntityLLMAnalysis.entity_id == entity_id
            )
        )
        next_version: int = result.scalar_one() + 1

        analysis = EntityLLMAnalysis(
            entity_id=entity_id,
            version=next_version,
            context_hash=context_hash,
            summary=summary,
            created_at=datetime.utcnow(),
        )
        db.add(analysis)
        db.commit()
        db.close()

    return summary


async def analyze_entity_context(webpage_content_id, progress_store, crawl_id):
    LOGGER.debug(f"analyzing people content for {webpage_content_id}")
    db = SessionLocal()

    try:
        # TODO: make sure there isn't a race condition here where two queries might try to process the same page
        url_content = db.query(WebpageContent).get(webpage_content_id)
        if not url_content or not url_content.text_content:
            LOGGER.error(
                f"couldn't find  WebpageContent:{webpage_content_id}, this was the result {url_content}"
            )
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract all individuals mentioned explicitly in the provided "
                    "snippet. For each individual, include ALL relevant surrounding "
                    "context and directly related text. Return only the function call."
                ),
            },
            {"role": "user", "content": url_content.text_content},
        ]

        OPENAI_MODEL = "gpt-4o"
        LOGGER.debug(
            f"about to query {OPENAI_MODEL} with {len(url_content.text_content)} chars"
        )
        response = await safe_openai_completion(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "extract_people",
                        "parameters": EXTRACT_PEOPLE_SCHEMA,
                    },
                }
            ],
            tool_choice="auto",
        )
        message = response.choices[0].message

        if message.tool_calls:
            result_json = json.loads(message.tool_calls[0].function.arguments)
        elif message.content:
            LOGGER.warning(
                f"LLM response (no function call): {message.content}"
            )
            return
        else:
            raise ValueError(
                f"Unexpected OpenAI response (no tool call, no explanation). {messages} \n\n {response}"
            )

        for person in result_json.get("people", []):
            name = person.get("name", "").strip()
            snippet_text = person.get("context", "").strip()

            if not name or not snippet_text:
                continue

            snippet_hash = hashlib.sha256(snippet_text.encode()).hexdigest()
            LOGGER.debug(f" - found {name!r} on page {webpage_content_id}")

            # grab the entity associated with the person, or make it
            entity = (
                db.query(Entity)
                .filter(func.lower(Entity.name) == name.lower())
                .first()
            )
            if not entity:
                entity = Entity(name=name)
                db.add(entity)
                db.flush()

            # check if the webpage snippet associated with that entity exists
            # if not, create it
            if not db.query(
                exists().where(
                    EntityWebpageSnippet.entity_id == entity.entity_id,
                    EntityWebpageSnippet.snippet_hash == snippet_hash,
                )
            ).scalar():
                db.add(
                    EntityWebpageSnippet(
                        entity_id=entity.entity_id,
                        snippet_text=snippet_text,
                        snippet_hash=snippet_hash,
                    )
                )

            # link the entity to the webpage where it was referenced
            if not db.query(
                exists().where(
                    EntityWebpageContentAssociation.entity_id
                    == entity.entity_id,
                    EntityWebpageContentAssociation.webpage_content_id
                    == webpage_content_id,
                )
            ).scalar():
                db.add(
                    EntityWebpageContentAssociation(
                        entity_id=entity.entity_id,
                        webpage_content_id=webpage_content_id,
                    )
                )

        url_content.analyzed = True
        LOGGER.debug(f"webpage content:{webpage_content_id} is analyzed")
        db.commit()
        progress_store[crawl_id]["processed"] += 1

    except Exception:
        db.rollback()
        LOGGER.exception(f"problem on page {webpage_content_id}")
    finally:
        db.close()


async def llm_match_people(
    user_interest_text,
):
    latest = (
        select(
            EntityLLMAnalysis.entity_id,
            func.max(EntityLLMAnalysis.version).label("max_version"),
        )
        .group_by(EntityLLMAnalysis.entity_id)
        .subquery()
    )

    stmt = (
        select(Entity.entity_id, Entity.name, EntityLLMAnalysis.summary)
        .join(
            EntityLLMAnalysis,
            (EntityLLMAnalysis.entity_id == Entity.entity_id)
            & (EntityLLMAnalysis.version == latest.c.max_version),
        )
        .join(latest, latest.c.entity_id == Entity.entity_id)
    )

    # fetch rows -> dict
    db = SessionLocal()
    rows = db.execute(stmt).all()
    entity_id_to_name = {entity_id: name for entity_id, name, _ in rows}
    name_to_bio_dict = {name: summary for _, name, summary in rows}

    prompt = MATCH_PEOPLE_PROMPT_TEMPLATE.format(
        user_interest_text=user_interest_text,
        bios_json=json.dumps(name_to_bio_dict, indent=2),
    )

    response = await safe_openai_completion(
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "match_people",
                    "description": "Return the top people matching the user's research interests.",
                    "parameters": MATCH_PEOPLE_SCHEMA,
                },
            }
        ],
        tool_choice="auto",
    )
    LOGGER.debug(response)
    try:
        call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(call.function.arguments)
        matches = arguments["matches"]

        matched_names = [m["name"] for m in matches]
        matched_entity_ids = [
            eid
            for eid, name in entity_id_to_name.items()
            if name in matched_names
        ]

        urls_stmt = (
            select(Entity.entity_id, WebpageContent.url)
            .join(
                EntityWebpageContentAssociation,
                Entity.entity_id == EntityWebpageContentAssociation.entity_id,
            )
            .join(
                WebpageContent,
                WebpageContent.webpage_content_id
                == EntityWebpageContentAssociation.webpage_content_id,
            )
            .where(Entity.entity_id.in_(matched_entity_ids))
        )

        url_rows = db.execute(urls_stmt).all()
        db.close()

        # Map entity_id to URLs
        entity_id_to_urls = {}
        for entity_id, url in url_rows:
            entity_id_to_urls.setdefault(entity_id, set()).add(url)

        # Update matches with URLs
        for match in matches:
            for entity_id, name in entity_id_to_name.items():
                if match["name"] == name:
                    match["urls"] = list(entity_id_to_urls.get(entity_id, []))
                    break

        return matches
    except Exception:
        LOGGER.exception(f"{response} could not return a tool")
        return []
