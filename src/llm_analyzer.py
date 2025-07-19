"""Module to handle all LLM analysis."""

import logging
import hashlib
import json

from sqlalchemy import func
from openai import OpenAI
from dotenv import load_dotenv

from .models import (
    URLContent,
    EntityContext,
    Entity,
    EntityPage,
    EntityContextPage,
)
from .models import EntityPage
from .database import SessionLocal

load_dotenv()
MODEL = "gpt-4o-mini"

LOGGER = logging.getLogger(__name__)

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
    entity = db.query(EntityContext).get(entity_id)
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
            Name: {entity.name}

            Context:
            {entity.context}
            """,
        },
    ]

    client = OpenAI()
    print(f"about to ask this question {messages}")
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )
    print(f"got this response: {response}")

    bio_text = response.choices[0].message.content.strip()
    return {"bio": bio_text}


async def analyze_entity_context(url_content_id):
    print(f"analyzing people content for {url_content_id}")
    db = SessionLocal()

    try:
        url_content = db.query(URLContent).get(url_content_id)
        if not url_content or not url_content.text_content:
            print("found nothing, closing")
            return

        from openai import OpenAI

        client = OpenAI()

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

        print(
            f"about to query GPT‑4o-mini with {len(url_content.text_content)} chars"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            context = person.get("context", "").strip()

            if not name or not context:
                continue

            print(f" - found {name!r} on page {url_content_id}")

            entity = (
                db.query(Entity)
                .filter(func.lower(Entity.name) == name.lower())
                .first()
            )
            if not entity:
                entity = Entity(name=name)
                db.add(entity)
                db.flush()

            if (
                not db.query(EntityPage)
                .filter_by(
                    entity_id=entity.entities_id,
                    url_content_id=url_content_id,
                )
                .first()
            ):
                db.add(
                    EntityPage(
                        entity_id=entity.entities_id,
                        url_content_id=url_content_id,
                    )
                )

            context_hash = hashlib.sha256(context.encode()).hexdigest()

            entity_context = (
                db.query(EntityContext)
                .filter_by(
                    entity_id=entity.entities_id, context_hash=context_hash
                )
                .first()
            )
            if not entity_context:
                entity_context = EntityContext(
                    entity_id=entity.entities_id,
                    context_text=context,
                    context_hash=context_hash,
                )
                db.add(entity_context)
                db.flush()

            if (
                not db.query(EntityContextPage)
                .filter_by(
                    entity_context_id=entity_context.entity_contexts_id,
                    url_content_id=url_content_id,
                )
                .first()
            ):
                db.add(
                    EntityContextPage(
                        entity_context_id=entity_context.entity_contexts_id,
                        url_content_id=url_content_id,
                    )
                )

        url_content.analyzed = True
        db.commit()

    except Exception:
        db.rollback()
        LOGGER.exception(f"problem on page {url_content_id}")
    finally:
        db.close()
