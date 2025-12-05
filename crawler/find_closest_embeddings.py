from array import array
from pathlib import Path
import argparse
import os
import math
import logging
from collections import defaultdict
import json
import time

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from utils import parse_crawler_config, embed_text
from models import Entity
from tqdm import tqdm

load_dotenv()
CLIENT = OpenAI(timeout=600.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_embedding_index(db_url):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("loading entities from the db")
    rows = (
        session.query(Entity.id, Entity.embedding)
        .filter(Entity.embedding.isnot(None))
        .all()
    )
    session.close()

    ids = []
    vectors = []
    for entity_id, emb_bytes in tqdm(rows, total=len(rows)):
        vec = array("f")
        vec.frombytes(emb_bytes)
        v = list(vec)
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]
        ids.append(entity_id)
        vectors.append(v)
    return ids, vectors


def find_nearest(ids, vectors, query_vec, top_k=10):
    norm = math.sqrt(sum(x * x for x in query_vec))
    q = [x / norm for x in query_vec]

    scores = []
    for entity_id, v in zip(ids, vectors):
        s = sum(a * b for a, b in zip(q, v))
        scores.append((s, entity_id))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]


def build_index(vectors):
    x = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def search_index(index, ids, query_bytes, top_k=10):
    q = np.frombuffer(query_bytes, dtype="float32")  # [dim]
    q = q.reshape(1, -1)  # [1, dim]
    scores, idxs = index.search(q, top_k)
    return [(float(s), ids[i]) for s, i in zip(scores[0], idxs[0])]


def build_entity_context_for_query(
    user_query, db_url, embedding_index, entity_ids, config
):
    """Rewrite a user query, embed it, and build an entity context string.

    The user_query is first rewritten into a dense semantic search query
    using a chat model, then embedded with the embedding model and used
    to search the FAISS index of entity embeddings. The top matching
    entities are fetched from the database and concatenated into a
    newline-separated context string of the form "name: text".

    Args:
        user_query: Natural-language query string from the user.
        db_url: SQLAlchemy database URL for the entities database.
        embedding_index: FAISS index containing entity embeddings.
        entity_ids: List of entity IDs aligned with the index vectors.
        config: Configuration mapping with search parameters, including
            'max_fetch_retries' for the embedding call.

    Returns:
        str: Newline-separated "name: text" lines for the most similar
        entities to the rewritten query.
    """
    rewrite_prompt = f"""
Rewrite this user query into a dense search query that would match
academic bios and paper abstracts. Keep it short but include synonyms
and related terms.

User query: {user_query}
"""
    rewrite_response = CLIENT.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": "You turn user questions into semantic search queries.",
            },
            {"role": "user", "content": rewrite_prompt},
        ],
    )

    rewritten_query = rewrite_response.choices[0].message.content.strip()
    query_embedding_bytes = embed_text(
        rewritten_query, CLIENT, config["max_fetch_retries"]
    )
    query_embedding = array("f")
    query_embedding.frombytes(query_embedding_bytes)

    results = search_index(
        embedding_index, entity_ids, query_embedding, top_k=50
    )

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    result_by_name = defaultdict(str)
    try:
        for score, entity_id in results:
            entity = session.query(Entity).filter(Entity.id == entity_id).one()
            result_by_name[(entity.name, entity.type)] += entity.text
        return result_by_name
    finally:
        session.close()


def analyze_results_by_name(
    result_by_name, user_query, model, max_fetch_retries
):
    analyses = {}

    for (name, entity_type), context in result_by_name.items():
        prompt = f"""
You are helping to answer a user query by analyzing a single {entity_type}.

User query:
{user_query}

{entity_type.capitalize()} name:
{name}

Context about this {entity_type}:
\"\"\"{context}\"\"\"

Based only on the context above, return a JSON object with this exact schema:

{{
  "name": "string",
  "is_relevant": true or false,
  "answer": "string",
  "reason": "string",
  "supporting_phrases": ["string", "string", ...]
}}

Definitions:
- "is_relevant": true if this {entity_type} is a good match to help answer the user query, false otherwise.
- "answer": a short explanation of how this {entity_type} relates to the query, or "" if not relevant.
- "reason": a brief justification referencing the context.
- "supporting_phrases": short phrases copied or lightly paraphrased from the context that support your reasoning.

Follow the schema exactly and do not include any extra fields.
"""

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
                    'OpenAI analysis call failed for %s "%s", attempt %s/%s: %s',
                    entity_type,
                    name,
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
            continue

        data = json.loads(completion.choices[0].message.content)
        analyses[name] = data

    return analyses


def main():
    parser = argparse.ArgumentParser(
        description=("fill in"),
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
    config = parse_crawler_config(args.config_path)
    db_url = f"sqlite:///{args.db_path}"
    entity_ids, embedding_vectors = load_embedding_index(db_url)
    index_path = "entity_index.fiass"
    if not os.path.exists(index_path):
        embedding_index = build_index(embedding_vectors)
        faiss.write_index(embedding_index, index_path)
    else:
        embedding_index = faiss.read_index(index_path)
    user_query = "find me researchers that work on ecosystem services in developing countries"
    result_by_name = build_entity_context_for_query(
        user_query, db_url, embedding_index, entity_ids, config
    )
    model = "gpt-5.1"
    analyses = analyze_results_by_name(
        result_by_name, user_query, model, config["max_fetch_retries"]
    )
    logging.info("all done")


if __name__ == "__main__":
    main()
