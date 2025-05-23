"""Module to handle all LLM analysis."""

import logging
import os
import json

from sqlalchemy import func
from openai import OpenAI
from dotenv import load_dotenv

from .models import URLContent, PersonContext
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


async def generate_bio(person_id: int):
    db = SessionLocal()
    person = db.query(PersonContext).get(person_id)
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
            Name: {person.name}

            Context:
            {person.context}
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


async def analyze_people_context(url_content_id: int):
    try:
        print(f"analyzing people content for {url_content_id}")
        db = SessionLocal()
        url_content = db.query(URLContent).get(url_content_id)
        if not url_content or not url_content.text_content:
            print("found nothing, closing")
            db.close()
            return

        client = OpenAI()

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract all individuals mentioned explicitly in the provided snippet. "
                    "For each individual, include ALL relevant surrounding context "
                    "and directly related text. "
                    "(such as presentation title, paper title, affiliation, session name, research interests, academic/professional title, or any other directly related text). "
                    "Be comprehensive and include all directly associated information."
                    "Return only the function call."
                ),
            },
            {"role": "user", "content": url_content.text_content},
        ]

        print(
            f"about to query {MODEL} with {len(url_content.text_content)} characters"
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
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

        result = response.choices[0].message.tool_calls[0].function.arguments
        try:
            result = (
                response.choices[0].message.tool_calls[0].function.arguments
            )
            result_json = json.loads(result)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            print(f"JSON parsing error: {e}")
            db.close()
            return

        for person in result_json.get("people", []):
            name = person.get("name").strip()
            context = person.get("context").strip()
            print(f"found {name} for {url_content_id}")
            existing_person = (
                db.query(PersonContext)
                .filter(
                    func.lower(PersonContext.name) == name.lower(),
                    PersonContext.url_content_id == url_content_id,
                )
                .first()
            )

            if existing_person:
                if context not in existing_person.context:
                    existing_person.context += f"\n\n{context}"
            else:
                new_person = PersonContext(
                    url_content_id=url_content_id, name=name, context=context
                )
                db.add(new_person)
        db.commit()
    except Exception:
        LOGGER.exception(f"problem on page {url_content_id}")
