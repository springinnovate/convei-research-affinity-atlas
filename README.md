# CONVEI Academic Affinity Atlas

**CONVEI Academic Affinity Atlas** is an open-source tool designed to automatically map academic conferences, extract researcher profiles, and discover scholarly affinities among participants. By leveraging advanced web scraping techniques and state-of-the-art language models, CONVEI creates comprehensive research dossiers, enabling academics to effortlessly discover collaborators, related research interests, and unexplored scholarly networks.

## Core Features

- **Web Mapping**: Automatically crawl and parse academic conference websites.
- **Researcher Dossiers**: Extract and organize detailed profiles including research interests, papers, and group affiliations.
- **Affinity Discovery**: Identify and suggest similar researchers using advanced semantic embeddings.
- **Interactive Search**: Query via command-line interface or intuitive web application.


## Project Structure:

research_network_mapper/
├── src/
│   ├── webmapper/
│   │   ├── __init__.py
│   │   ├── scraper.py
│   │   └── fetcher.py
│   │
│   ├── parser/
│   │   ├── __init__.py
│   │   └── llm_parser.py
│   │
│   ├── vectorizer/
│   │   ├── __init__.py
│   │   └── llm_vectorizer.py
│   │
│   ├── datastore/
│   │   ├── __init__.py
│   │   └── storage.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   └── engine.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli.py
│   │
│   ├── app/
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── main.py
│   │
│   └── common/
│       ├── __init__.py
│       ├── config.py
│       └── models.py
│
├── data/
│   └── ...
│
├── tests/
│   ├── test_webmapper.py
│   ├── test_parser.py
│   ├── test_vectorizer.py
│   ├── test_datastore.py
│   └── test_search.py
│
├── pyproject.toml
├── README.md
└── .env