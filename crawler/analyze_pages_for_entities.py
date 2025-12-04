"""CLI tool for extracting typed entities from crawled pages using OpenAI.

This script reads a crawler configuration and a SQLite database containing
Page records, uses an OpenAI model to extract entities (such as Person or
Session) from the stored HTML content, and writes Entity rows back into
the same database. Extraction is executed in parallel across pages and
entity types using a ThreadPoolExecutor, with basic retry logic and
exception guarding for robustness in worker threads.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
import argparse
import functools
import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm.auto import tqdm

from models import Page, Entity, EntityBio
from utils import parse_crawler_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [line %(lineno)d] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
os.environ["OPENAI_API_KEY"]
logging.info("Initializing OpenAI client")
CLIENT = OpenAI(timeout=600.0)


def guard_exceptions(fn):
    """Decorator that logs and suppresses unhandled exceptions in a function.

    This is intended for use with functions executed in worker threads or other
    fire-and-forget contexts, where an unhandled exception would otherwise
    terminate the worker silently. Any exception raised by the wrapped function
    is logged with logging.exception and then suppressed.

    Args:
        fn: The function to wrap.

    Returns:
        A wrapped function that behaves like fn but logs and swallows any
        unhandled exceptions.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logging.exception("Unhandled exception in %s", fn.__name__)

    return wrapper


@guard_exceptions
def extract_entities_for_page_and_type(task):
    """Run entity extraction for a single (page, entity_type) task.

    This function is designed to be used with executors like
    concurrent.futures.ThreadPoolExecutor, which expect a single
    argument per mapped call. To keep the executor mapping simple,
    all parameters are bundled into a single tuple rather than
    passing multiple positional arguments.

    Args:
        task: A tuple of
            (db_url, page_id, entity_type, entity_desc, model), where:
                db_url (str): SQLAlchemy database URL for the SQLite DB.
                page_id (int): Primary key of the Page to process.
                entity_type (str): Logical entity type name (e.g. "Person").
                entity_desc (str): Natural language description of the
                    entity type used to guide extraction.
                model (str): OpenAI model name to use for extraction.
                max_fetch_retries (int): number of times to retry a failed
                    OpenAI fetch.
    """
    db_url, page_id, entity_type, entity_desc, model, max_fetch_retries = task
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    page = session.get(Page, page_id)
    if entity_type in page.entities_analyzed:
        session.close()
        return
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
    delay = 1.0
    completion = None
    for attempt in range(max_fetch_retries):
        try:
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
            break
        except Exception as exc:
            logging.error(
                "OpenAI call failed for page %s, entity_type %s, attempt %s/%s: %s",
                page.id,
                entity_type,
                attempt + 1,
                max_fetch_retries,
                exc,
            )
            session.rollback()
            if attempt == max_fetch_retries - 1:
                session.close()
                return
            time.sleep(delay)
            delay *= 2
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
    session.close()


def extract_entities(db_path: Path, config_path: Path, model: str):
    """Run parallel entity extraction for all pages and configured entity types.

    This function reads the crawler configuration, discovers all Page rows in
    the given SQLite database, and enqueues a task for each (page, entity type)
    pair that has not yet been processed. Tasks are executed in parallel using
    a ThreadPoolExecutor, and each task calls
    extract_entities_for_page_and_type to perform the actual extraction and
    database writes.

    Args:
        db_path: Filesystem path to the SQLite database file containing Page
            and Entity tables.
        config_path: Filesystem path to the crawler config YAML, expected to
            contain an "entities" list and concurrency settings such as
            "num_workers" and "max_fetch_retries".
        model: OpenAI model name to use for entity extraction calls.
    """
    config = parse_crawler_config(config_path)
    logging.debug(config)
    entity_configs = config["entities"]
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    pages = session.query(Page).all()
    tasks = []
    for page in pages:
        for entity_config in entity_configs:
            entity_type = entity_config["type"]
            if entity_type in page.entities_analyzed:
                continue
            entity_desc = entity_config["description"]
            tasks.append(
                (
                    db_url,
                    page.id,
                    entity_type,
                    entity_desc,
                    model,
                    config["max_fetch_retries"],
                )
            )
    session.close()
    with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
        list(
            tqdm(
                executor.map(extract_entities_for_page_and_type, tasks),
                total=len(tasks),
            )
        )


def build_entity_bio(db_path: Path, config_path: Path, model: str):

    def _worker(item):
        (entity_name, entity_type), entity_text = item
        prompt = f"""
You are creating a concise but information-rich bio for a single {entity_type}.

Entity type description:
{entity_desc}

Entity name:
{entity_name}

Source snippets about this {entity_type}:
\"\"\"{entity_text}\"\"\"

Write a single coherent bio for this {entity_type} that:
- Clearly describes who or what "{entity_name}" is.
- Captures key topics, domains, methods, regions, or roles mentioned.
- Focuses on information that would help someone understand their work or relevance.
- Avoids repeating identical sentences or boilerplate.
- Is written in natural, fluent prose (a few sentences or a short paragraph).

Return only the bio text, with no extra explanations or formatting.
""".strip()

        delay = 1.0
        completion = None
        for attempt in range(max_fetch_retries):
            try:
                completion = CLIENT.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You write clear, informative bios from provided context.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                break
            except Exception as exc:
                logging.error(
                    "OpenAI bio call failed for %s '%s', attempt %s/%s: %s",
                    entity_type,
                    entity_name,
                    attempt + 1,
                    max_fetch_retries,
                    exc,
                )
                if attempt == max_fetch_retries - 1:
                    completion = None
                    break
                time.sleep(delay)
                delay *= 2

        if completion is None:
            logging.warning(
                "Skipping bio for %s '%s' after repeated failures",
                entity_type,
                entity_name,
            )
            return

        bio_text = completion.choices[0].message.content.strip()
        if not bio_text:
            logging.warning(
                "Empty bio generated for %s '%s', skipping",
                entity_type,
                entity_name,
            )
            return

        s = Session()
        try:
            existing = (
                s.query(EntityBio)
                .filter(
                    EntityBio.type == entity_type,
                    EntityBio.name == entity_name,
                )
                .one_or_none()
            )
            if existing:
                existing.bio = bio_text
                logging.debug(
                    "Updated bio for %s '%s'",
                    entity_type,
                    entity_name,
                )
            else:
                bio = EntityBio(
                    type=entity_type,
                    name=entity_name,
                    bio=bio_text,
                )
                s.add(bio)
                logging.debug(
                    "Created bio for %s '%s'",
                    entity_type,
                    entity_name,
                )
            s.commit()
        finally:
            s.close()

    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    session = Session()
    texts_by_name_type = {}
    config = parse_crawler_config(config_path)
    entity_desc = {}
    for entity_config in tqdm(
        config["entities"], desc="building entity descriptions"
    ):
        entity_desc[entity_config["type"]] = entity_config["description"]
    max_fetch_retries = config["max_fetch_retries"]
    num_workers = config["num_workers"]

    existing_pairs = set(session.query(EntityBio.name, EntityBio.type).all())

    # 2) Get all Entity rows whose (name, type) do NOT have a bio yet
    rows = (
        session.query(Entity.name, Entity.type, Entity.text)
        .filter(Entity.text.isnot(None))
        .all()
    )

    texts_by_name_type = defaultdict(list)

    for entity_name, entity_type, text in rows:
        if (entity_name, entity_type) in existing_pairs:
            continue
        texts_by_name_type[(entity_name, entity_type)].append(text)

    for key, texts in texts_by_name_type.items():
        texts_by_name_type[key] = "\n".join(texts)

    session.close()
    logging.info(
        "Building bios for %d unique entities",
        len(entity_desc),
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(_worker, texts_by_name_type.items()),
                total=len(texts_by_name_type),
                desc="processing openapi calls",
            )
        )


def main():
    """Parse CLI arguments and run entity extraction for a given database.

    This entry point configures the command-line interface, reads the target
    SQLite database path and crawler configuration path, and invokes
    extract_entities with the chosen OpenAI model.

    Expected usage:
        python extract_entities.py DB_PATH CONFIG_PATH [--model MODEL_NAME]

    The config file must define an "entities" list and concurrency settings
    required by extract_entities.
    """
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
        "--model", default="gpt-4o-mini", help="OpenAI model name"
    )
    args = parser.parse_args()
    # extract_entities(args.db_path, args.config_path, model=args.model)
    build_entity_bio(args.db_path, args.config_path, args.model)


if __name__ == "__main__":
    main()
