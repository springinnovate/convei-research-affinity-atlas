from dataclasses import dataclass
from typing import List
from pathlib import Path
import argparse
import yaml


@dataclass
class RequestConfig:
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    delay_between_requests_seconds: float
    concurrent_requests: int
    proxies: List[str]


@dataclass
class StorageConfig:
    type: str
    base_path: str
    overwrite: bool


@dataclass
class CrawlerConfig:
    name: str
    start_urls: List[str]
    allowed_domains: List[str]
    max_depth: int
    max_pages: int
    request: RequestConfig
    storage: StorageConfig


def load_crawler_config(path: Path) -> CrawlerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)["crawler"]

    request = RequestConfig(
        timeout_seconds=raw["request"]["timeout_seconds"],
        max_retries=raw["request"]["max_retries"],
        retry_backoff_seconds=raw["request"]["retry_backoff_seconds"],
        delay_between_requests_seconds=raw["request"][
            "delay_between_requests_seconds"
        ],
        concurrent_requests=raw["request"]["concurrent_requests"],
        proxies=raw["request"]["proxies"],
    )

    storage = StorageConfig(
        type=raw["storage"]["type"],
        base_path=raw["storage"]["base_path"],
        overwrite=raw["storage"]["overwrite"],
    )

    config_result = CrawlerConfig(
        name=raw["name"],
        start_urls=raw["start_urls"],
        allowed_domains=raw["allowed_domains"],
        max_depth=raw["max_depth"],
        max_pages=raw["max_pages"],
        request=request,
        storage=storage,
    )
    if config_result.name not in path.stem:
        raise ValueError(
            f"Expected the name defined in the yaml file "
            f'("{config_result.name}") to be at least a substring '
            f'of the config file path but that is only "{path.name}"'
        )

    return config_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    config = load_crawler_config(Path(args.config_file))

    # placeholder crawl logic
    print("starting crawl:", config.name)
    print("start urls:", config.start_urls)
    print("allowed domains:", config.allowed_domains)


if __name__ == "__main__":
    main()
