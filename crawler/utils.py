"""Shared utility functions for the crawler package.

Currently this module only exposes `parse_crawler_config`, which loads and
normalizes crawler YAML configuration files. As the crawler grows, additional
shared helpers should be added here to keep cross-cutting logic in one place.
"""

from array import array
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import logging
import math
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, selectinload
from tqdm import tqdm
import faiss
import numpy as np
import tiktoken
import yaml

from .models import CombinedEntity

load_dotenv()
CLIENT = OpenAI(timeout=600.0)
EMBEDDING_MODEL = "text-embedding-3-small"


def parse_crawler_config(yaml_config_path):
    """Parse a crawler YAML configuration file.

    The YAML file is expected to have a single top-level key whose name matches
    the stem of the YAML filename. For example, for ``example_crawler.yaml``,
    the top-level key must be ``example_crawler``. Within that section, this
    function reads crawl settings including start URLs, allowed scopes, page
    limits, and output database path.

    Args:
        yaml_config_path (str): Filesystem path to the YAML configuration file.

    Returns:
        dict: A dictionary containing:
            - name (str): The configuration section name (filename stem).
            - start_urls (list[str]): Seed URLs for the crawl.
            - allowed_scopes (list[str]): Allowed domains/hosts/URL prefixes.
            - max_pages (int | None): Maximum number of pages to crawl, or
              ``None`` if the configuration uses ``'no_limit'``.
            - sqlite_path (str): Path to the SQLite database file.

    Raises:
        ValueError: If the YAML file does not contain a top-level key matching
            the filename stem.
        KeyError: If required keys (e.g., ``start_urls``, ``allowed_scopes``,
            ``output.sqlite_path``) are missing from the configuration.
    """
    with open(yaml_config_path, "r") as f:
        data = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(yaml_config_path))[0]
    if stem not in data:
        raise ValueError(
            f"Expected a section named the filename stem '{stem}' but none "
            f"was found in {yaml_config_path}"
        )
    section = data[stem]
    max_pages_raw = section.get("max_pages")
    if isinstance(max_pages_raw, str) and max_pages_raw.lower() == "no_limit":
        max_pages = None
    else:
        max_pages = max_pages_raw

    raw_sqlite_path = section["sqlite_path"]
    if os.path.isabs(raw_sqlite_path):
        sqlite_path = raw_sqlite_path
    else:
        sqlite_path = str(Path(yaml_config_path).parent / raw_sqlite_path)

    return {
        "name": stem,
        "start_urls": section["start_urls"],
        "allowed_scopes": section["allowed_scopes"],
        "max_pages": max_pages,
        "sqlite_path": sqlite_path,
        "content_sections": section["content_sections"],
        "num_workers": section["num_workers"],
        "requests_per_second": section["requests_per_second"],
        "drop_elements": section["drop_elements"],
        "max_fetch_retries": section["max_fetch_retries"],
        "entities": section["entities"],
    }


MAX_EMBED_TOKENS = 8000


def embed_text(text, client, max_retries):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = enc.encode(text)
    if len(tokens) > MAX_EMBED_TOKENS:
        text = enc.decode(tokens[:MAX_EMBED_TOKENS])

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
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


def load_embedding_index(db_url):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("loading entities from the db")
    rows = (
        session.query(CombinedEntity.id, CombinedEntity.embedding)
        .filter(CombinedEntity.embedding.isnot(None))
        .order_by(CombinedEntity.id)
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


def load_entity_ids(db_url):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("loading entities from the db")
    rows = (
        session.query(CombinedEntity.id)
        .filter(CombinedEntity.embedding.isnot(None))
        .order_by(CombinedEntity.id)
        .all()
    )
    session.close()

    ids = [row[0] for row in rows]
    return ids


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

    results = search_index(embedding_index, entity_ids, query_embedding, top_k=50)

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    result_by_name = defaultdict(str)
    try:
        ids = [entity_id for score, entity_id in results]
        entities = (
            session.query(CombinedEntity)
            .options(selectinload(CombinedEntity.raw_entities))
            .filter(CombinedEntity.id.in_(ids))
            .all()
        )
        by_id = {e.id: e for e in entities}

        for score, entity_id in results:
            entity = by_id[entity_id]
            texts = [re.text for re in entity.raw_entities if re.text]
            result_by_name[(entity.name, entity.type)] += "\n\n".join(texts)
        return result_by_name
    finally:
        session.close()


def analyze_results_by_name(
    result_by_name, user_query, model, max_fetch_retries, max_workers=8
):
    analyses = {}

    def worker(item):
        (name, entity_type), context = item
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
            return None

        data = json.loads(completion.choices[0].message.content)
        return name, data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(worker, result_by_name.items()):
            if result is None:
                continue
            name, data = result
            analyses[name] = data

    return analyses
