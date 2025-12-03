"""Backfill OpenAI text embeddings for Entity rows.

This module scans the database for Entity rows that are missing embeddings,
generates embeddings for their text fields using the OpenAI embeddings API,
and writes the binary-encoded vectors back to the database. Embeddings are
generated concurrently using a ThreadPoolExecutor to improve throughput.
"""

from queue import Queue
from array import array
from threading import Thread
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
    total = session.query(Entity).filter(Entity.embedding.is_(None)).count()
    session.close()

    logging.info("Found %d entities without embeddings", len(str(total)))

    entity_to_process_queue = Queue()
    entity_embedded_queue = Queue()
    num_workers = config["num_workers"]

    def database_reader():
        """Read entities without embeddings from the database into work queue.

        Opens a new SQLAlchemy session, streams all Entity rows that do not yet
        have an embedding, and enqueues each (entity_id, text) pair onto
        entity_to_process_queue. After all entities are enqueued, a None
        is placed into the queue once per worker to signal completion.

        This function is intended to run in a dedicated background thread.
        """
        session = Session()
        entities = session.query(Entity.id, Entity.text).filter(
            Entity.embedding.is_(None)
        )
        for entity_id, text in entities:
            entity_to_process_queue.put((entity_id, text))
        for _ in range(num_workers):
            entity_to_process_queue.put(None)
        session.close()

    def database_writer():
        """Write generated embeddings from the result queue back to database.

        Opens a new SQLAlchemy session and continuously consumes
        (entity_id, embedding_bytes) pairs from entity_embedded_queue, updating
        the corresponding Entity.embedding field and committing each change.
        A None payload is treated as a sentinel from a worker; once a sentinel
        has been received from each worker, the writer exits.

        This function is intended to run in a dedicated background thread.
        """
        session = Session()
        finished = 0
        while True:
            payload = entity_embedded_queue.get()
            if payload is None:
                finished += 1
                entity_embedded_queue.task_done()
                if finished == num_workers:
                    break
                continue
            entity_id, embedding_bytes = payload
            entity = session.get(Entity, entity_id)
            entity.embedding = embedding_bytes
            session.commit()
            entity_embedded_queue.task_done()
        session.close()

    reader_thread = Thread(target=database_reader)
    writer_thread = Thread(target=database_writer)

    reader_thread.start()
    writer_thread.start()

    with tqdm(total=total) as pbar:

        def embedding_worker():
            """Generate embeddings for queued entities in a worker thread.

            This worker repeatedly pulls (entity_id, text) tuples from
            entity_to_process_queue, generates an embedding for each text using
            embed_text, and pushes (entity_id, embedding_bytes) results into
            entity_embedded_queue. A None payload is treated as a sentinel value
            indicating no more work; upon receiving it, the worker forwards a
            None sentinel to the output queue and exits. Progress is tracked by
            incrementing the shared tqdm progress bar.

            This function does not take any arguments and is intended to be run
            in one or more background threads.
            """
            while True:
                payload = entity_to_process_queue.get()
                if payload is None:
                    entity_to_process_queue.task_done()
                    entity_embedded_queue.put(None)
                    break
                entity_id, text = payload
                embedding_bytes = embed_text(
                    text,
                    config["max_fetch_retries"],
                )
                entity_embedded_queue.put((entity_id, embedding_bytes))
                entity_to_process_queue.task_done()
                pbar.update(1)

        threads = []
        for _ in range(num_workers):
            t = Thread(target=embedding_worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    reader_thread.join()
    entity_to_process_queue.join()
    entity_embedded_queue.join()
    writer_thread.join()

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
