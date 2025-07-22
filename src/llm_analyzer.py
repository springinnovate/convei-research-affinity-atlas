"""Module to handle all LLM analysis."""

import logging
import hashlib
import json
import sys

from sqlalchemy import func, exists
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
MODEL = "gpt-4o-mini"

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


async def generate_bio(entity_id: int):
    db = SessionLocal()
    entity_context = (
        db.query(EntityLLMAnalysis)
        .filter(EntityLLMAnalysis.entity_id == entity_id)
        .order_by(EntityLLMAnalysis.created_at.desc())
        .first()
    )
    entity_name = entity_context.entity.name
    context_text = entity_context.context_text
    db.close()

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

    OPENAI_CLIENT = OpenAI()
    LOGGER.debug(f"about to ask this question {messages}")
    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )
    LOGGER.debug(f"got this response: {response}")

    bio_text = response.choices[0].message.content.strip()
    return {"bio": bio_text}


async def analyze_entity_context(webpage_content_id, progress_store, crawl_id):
    LOGGER.debug(f"analyzing people content for {webpage_content_id}")
    db = SessionLocal()

    try:
        url_content = db.query(WebpageContent).get(webpage_content_id)
        if not url_content or not url_content.text_content:
            LOGGER.error(f"couldn't find  WebpageContent:{webpage_content_id}")
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

        OPENAI_MODEL = "gpt-4o-mini"
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
