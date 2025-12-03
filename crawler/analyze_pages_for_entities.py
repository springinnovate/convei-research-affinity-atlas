from pathlib import Path
import argparse
import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Page, Entity
from utils import parse_crawler_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [line %(lineno)d] %(message)s",
)

load_dotenv()
os.environ["OPENAI_API_KEY"]
logging.info("Initializing OpenAI client")
CLIENT = OpenAI(timeout=600.0)


def extract_entities(db_path: str | Path, config_path: str | Path, model: str):
    config = parse_crawler_config(config_path)
    logging.debug(config)
    entity_types = config["entities"]
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    for page in session.query(Page).all():
        if not page.html:
            continue
        for entity_config in entity_types:
            entity_type = entity_config["type"]
            entity_desc = entity_config.get("description", "")
            if entity_type in page.entities_analyzed:
                continue
            prompt = f"""
You are an information extraction system.

You must extract entities of type "{entity_type}" from the HTML content below.

Entity type description:
{entity_desc}

Use this description to decide what counts as a "{entity_type}" and what
information is important to capture.

Return a JSON object with this exact structure:

{{
  "entities": [
    {{
      "name": "short human-readable label for the {entity_type}",
      "text": "a detailed, self-contained description capturing all relevant information about this {entity_type} found in the HTML. Include as much specific detail as possible, potentially multiple sentences or paragraphs as needed."
    }}
  ]
}}

Requirements:
- The "text" field should include all salient contextual information that could be useful for later semantic search and recommendation, not just a brief summary.
- It is acceptable for "text" to be long if there is a lot of relevant information.
- If you cannot find any {entity_type} entities, return:

{{
  "entities": []
}}

HTML:
{page.html}
""".strip()
            logging.debug(prompt)
            completion = CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are an information extraction system that always returns strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            data = json.loads(completion.choices[0].message.content)
            for item in data.get("entities", []):
                entity = Entity(
                    type=entity_type,
                    name=item["name"],
                    text=item.get("text"),
                    attributes=item.get("attributes"),
                    page_id=page.id,
                )
                session.add(entity)
            page.entities_analyzed.append(entity_type)
            session.commit()
            break
        session.commit()
        break
    session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from pages using OpenAI and store them in the database.",
    )
    parser.add_argument(
        "db_path", type=Path, help="Path to the SQLite database file"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help='Path to YAML config with an "entities" list',
    )
    parser.add_argument(
        "--model", default="gpt-5-mini", help="OpenAI model name"
    )
    args = parser.parse_args()
    extract_entities(args.db_path, args.config_path, model=args.model)


if __name__ == "__main__":
    main()
