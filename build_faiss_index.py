import argparse
from pathlib import Path

import faiss
import numpy as np

from crawler.utils import load_embedding_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    db_url = f"sqlite:///{db_path}"

    base_dir = db_path.parent
    index_path = base_dir / "entity_index.faiss"

    if index_path.exists():
        print(f"{index_path} exists, so skipping")
        return

    entity_ids, embedding_vectors = load_embedding_index(db_url)

    x = np.asarray(embedding_vectors, dtype="float32")
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)

    faiss.write_index(index, str(index_path))


if __name__ == "__main__":
    main()
