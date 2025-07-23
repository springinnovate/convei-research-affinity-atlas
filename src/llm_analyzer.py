"""Module to handle all LLM analysis."""

from datetime import datetime
import logging
import hashlib
import json
import sys

from sqlalchemy import func, exists, select
from openai import OpenAI
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .models import (
    WebpageContent,
    EntityLLMAnalysis,
    Entity,
    EntityWebpageSnippet,
    EntityWebpageContentAssociation,
)
from .database import SessionLocal


OPENAI_CLIENT = AsyncOpenAI()

MATCH_PEOPLE_PROMPT_TEMPLATE = """
You are an assistant that matches researchers to a user's interests.

Use ONLY the data below.

### User_Interest
{user_interest_text}

### Candidate_Bios  (name -> biography)
{bios_json}

Task:
• Read the user's interests and every candidate bio.
• Choose up to five people whose bio best aligns with the user's interests.
• For each selected person provide:
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
MODEL = "gpt-4o"

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

    existing = db.execute(
        select(EntityLLMAnalysis.summary).where(
            EntityLLMAnalysis.entity_id == entity_id,
            EntityLLMAnalysis.context_hash == context_hash,
        )
    )
    cached = existing.scalar_one_or_none()
    if cached is not None:
        return cached

    if not context_text:
        return "No relevant context found."

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
    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )
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
        response = await OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
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

        args = response.choices[0].message.tool_calls[0].function.arguments
        result_json = json.loads(args)

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
        select(Entity.name, EntityLLMAnalysis.summary)
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
    name_to_bio_dict = {name: summary for name, summary in rows}

    prompt = MATCH_PEOPLE_PROMPT_TEMPLATE.format(
        user_interest_text=user_interest_text,
        bios_json=json.dumps(name_to_bio_dict, indent=2),
    )

    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
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
    call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(call.function.arguments)
    matches = arguments["matches"]
    return matches
