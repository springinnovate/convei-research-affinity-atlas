import json
import os
from database import SessionLocal
from models import URLContent, PersonContext

from sqlalchemy import func
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
MODEL = "gpt-4o-mini"


SCHEMA = {
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


async def analyze_people_context(url_content_id: int):
    db = SessionLocal()
    url_content = db.query(URLContent).get(url_content_id)
    if not url_content or not url_content.text_content:
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

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {"name": "extract_people", "parameters": SCHEMA},
            }
        ],
        tool_choice="auto",
    )

    result = response.choices[0].message.tool_calls[0].function.arguments
    try:
        result = response.choices[0].message.tool_calls[0].function.arguments
        result_json = json.loads(result)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"JSON parsing error: {e}")
        db.close()
        return

    for person in result_json.get("people", []):
        name = person.get("name").strip()
        context = person.get("context").strip()

        # Check if this person context already exists for this URLContent
        existing_person = (
            db.query(PersonContext)
            .filter(
                func.lower(PersonContext.name) == name.lower(),
                PersonContext.url_content_id == url_content_id,
            )
            .first()
        )

        if existing_person:
            # Append clearly if context is not already present
            if context not in existing_person.context:
                existing_person.context += f"\n\n{context}"
        else:
            # Otherwise, clearly create a new PersonContext
            new_person = PersonContext(
                url_content_id=url_content_id, name=name, context=context
            )
            db.add(new_person)
