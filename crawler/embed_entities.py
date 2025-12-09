from queue import Queue
from threading import Thread
from pathlib import Path
import argparse
import logging

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, selectinload

from tqdm import tqdm

from .models import CombinedEntity
from .utils import parse_crawler_config, embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
CLIENT = OpenAI(timeout=600.0)


def embed_entities(db_path, config_path):
    config = parse_crawler_config(config_path)
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    session = Session()
    total = (
        session.query(CombinedEntity).filter(CombinedEntity.embedding.is_(None)).count()
    )
    session.close()

    logging.info(f"Found {total} combined entities without embeddings")

    entity_to_process_queue = Queue()
    entity_embedded_queue = Queue()
    num_workers = config["num_workers"]

    session = Session()
    try:
        q = (
            session.query(CombinedEntity)
            .options(selectinload(CombinedEntity.raw_entities))
            .filter(CombinedEntity.embedding.is_(None))
        )
        for ce in tqdm(q, total=total, desc="Queuing combined entities"):
            texts = [re.text for re in ce.raw_entities if re.text]
            combined_text = f"** {ce.name} **\n\n" + "\n\n".join(texts)
            entity_to_process_queue.put((ce.id, combined_text))
        for _ in range(num_workers):
            entity_to_process_queue.put(None)
    finally:
        session.close()

    def embedding_worker():
        try:
            while True:
                payload = entity_to_process_queue.get()
                if payload is None:
                    break
                entity_id, text = payload
                embedding_bytes = embed_text(
                    text,
                    CLIENT,
                    config["max_fetch_retries"],
                )
                entity_embedded_queue.put((entity_id, embedding_bytes))
        except Exception:
            logging.exception("error in embedding_worker")
        finally:
            entity_embedded_queue.put(None)

    logging.info("waiting for reader thread to terminate")

    with tqdm(total=total) as pbar:

        def database_writer():
            session = Session()
            WRITE_BATCH_LIMIT = 100
            step = 0
            finished = 0
            try:
                while True:
                    payload = entity_embedded_queue.get()
                    if payload is None:
                        finished += 1
                        if finished == num_workers:
                            break
                        continue
                    entity_id, embedding_bytes = payload
                    entity = session.get(CombinedEntity, entity_id)
                    entity.embedding = embedding_bytes
                    step += 1
                    if step >= WRITE_BATCH_LIMIT:
                        session.commit()
                        pbar.update(step)
                        step = 0
            except Exception:
                logging.exception("error in database_writer")
            finally:
                if step:
                    session.commit()
                    pbar.update(step)
                session.close()

        writer_thread = Thread(target=database_writer)
        writer_thread.start()
        threads = []
        for _ in range(num_workers):
            t = Thread(target=embedding_worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        writer_thread.join()

    logging.info("Completed embedding backfill")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill OpenAI embeddings for combined entities stored in the database.",
    )
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database file")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to YAML config with settings",
    )
    args = parser.parse_args()
    embed_entities(args.db_path, args.config_path)


if __name__ == "__main__":
    main()
