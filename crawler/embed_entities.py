"""Backfill OpenAI text embeddings for Entity rows.

This module scans the database for Entity rows that are missing embeddings,
generates embeddings for their text fields using the OpenAI embeddings API,
and writes the binary-encoded vectors back to the database. Embeddings are
generated concurrently using a ThreadPoolExecutor to improve throughput.
"""

from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import logging
import time

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from models import Entity
from utils import parse_crawler_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
CLIENT = OpenAI(timeout=600.0)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text, max_retries):
    """Generate and serialize an embedding for the given text.

    Uses the given OpenAI embedding model to create a vector
    representation of the input text and returns it as a binary
    blob suitable for storage in the database. The request is
    retried with exponential backoff if the OpenAI API call fails.

    Args:
        text: The input text to embed.
        max_retries: Maximum number of retry attempts for the API call.

    Returns:
        bytes: The embedding encoded as a float32 array in little-endian
        binary format.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = CLIENT.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            vec = resp.data[0].embedding
            return array("f", vec).tobytes()
        except Exception as exc:
            logging.error(
                "OpenAI embedding call failed, attempt %s/%s: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def embed_entities(db_path, config_path):
    """Backfill embeddings for Entity rows missing an embedding.

    Finds all Entity records without an embedding, generates embeddings
    for their text fields using the OpenAI client, and writes the
    binary-encoded vectors back to the database. Embedding generation
    and updates are performed concurrently using a thread pool.

    Args:
        db_path: Path to the SQLite database file.
        config_path: Path to the crawler configuration file used to
            determine worker concurrency and retry settings.
    """
    config = parse_crawler_config(config_path)
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    entities = (
        session.query(Entity.id, Entity.text)
        .filter(Entity.embedding.is_(None))
        .all()
    )
    session.close()

    logging.info("Found %d entities without embeddings", len(entities))

    def worker(row):
        entity_id, text = row
        embedding_bytes = embed_text(text, config["max_fetch_retries"])
        session = Session()
        entity = session.get(Entity, entity_id)
        entity.embedding = embedding_bytes
        session.commit()
        session.close()
        return entity_id

    with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
        list(tqdm(executor.map(worker, entities), total=len(entities)))

    logging.info("Completed embedding backfill")


def main():
    """Parse command-line arguments and run the entity embedding backfill.

    This entry point configures an argument parser, reads the database
    and configuration paths from the command line, and invokes
    embed_entities to generate embeddings for Entity rows missing
    an embedding.

    The expected arguments are:
        db_path: Path to the SQLite database file.
        config_path: Path to the YAML configuration file used for
            embedding settings and worker configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Backfill OpenAI embeddings for entities stored in the database."
        ),
    )

    parser.add_argument(
        "db_path", type=Path, help="Path to the SQLite database file"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help='Path to YAML config with an "entities" list',
    )
    args = parser.parse_args()
    embed_entities(args.db_path, args.config_path)


if __name__ == "__main__":
    main()
