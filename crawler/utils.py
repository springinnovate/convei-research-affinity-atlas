"""Shared utility functions for the crawler package.

Currently this module only exposes `parse_crawler_config`, which loads and
normalizes crawler YAML configuration files. As the crawler grows, additional
shared helpers should be added here to keep cross-cutting logic in one place.
"""

from pathlib import Path
import os
import time
from array import array
import logging

from dotenv import load_dotenv
import yaml

load_dotenv()

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


def embed_text(text, client, max_retries):
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
