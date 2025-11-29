"""
This script crawls a set of urls storing the information in an sqlite database
based off of the input YAML configuration provided. The YAML configuration
example can be seen at `example_crawler.yaml` in this directory.

To run this inside the Docker environment from the current directory:

  docker build -t crawl_env .
  docker run --rm -it -v %CD%:/app crawl_env

Then, inside the container:

  python crawl_runner.py path/to/configuration.yaml
"""

import argparse
import logging
import os

import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def parse_crawler_config(path):
    """
    Parse a crawler YAML configuration file.

    The YAML file is expected to have a single top-level key whose name matches
    the stem of the YAML filename. For example, for ``example_crawler.yaml``,
    the top-level key must be ``example_crawler``. Within that section, this
    function reads crawl settings including start URLs, allowed scopes, page
    limits, and output database path.

    Args:
        path (str): Filesystem path to the YAML configuration file.

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
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem not in data:
        raise ValueError(
            f"Expected a section named the filename stem '{stem}' but none was found in {path}"
        )
    section = data[stem]
    max_pages_raw = section.get("max_pages")
    if isinstance(max_pages_raw, str) and max_pages_raw.lower() == "no_limit":
        max_pages = None
    else:
        max_pages = max_pages_raw
    return {
        "name": stem,
        "start_urls": section["start_urls"],
        "allowed_scopes": section["allowed_scopes"],
        "max_pages": max_pages,
        "sqlite_path": section["output"]["sqlite_path"],
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse a crawler YAML configuration and print the "
            "resolved settings."
        )
    )
    parser.add_argument(
        "config_path",
        help=(
            "Path to the YAML configuration file defining the "
            "crawler settings."
        ),
    )
    args = parser.parse_args()
    config = parse_crawler_config(args.config_path)
    logging.debug(config)


if __name__ == "__main__":
    main()
