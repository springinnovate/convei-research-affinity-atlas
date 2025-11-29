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
import os

import yaml


def parse_crawler_config(path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    config = parse_crawler_config(args.config_path)
    print(config)


if __name__ == "__main__":
    main()
