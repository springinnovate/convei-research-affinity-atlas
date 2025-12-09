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

    print(f"Loading embeddings from {db_url}...")
    entity_ids, embedding_vectors = load_embedding_index(db_url)
    print(f"Loaded {len(entity_ids)} embeddings")

    print("Converting embeddings to float32 array...")
    x = np.asarray(embedding_vectors, dtype="float32")
    print(f"Embedding matrix shape: {x.shape}")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    print("FAISS index built")

    print(f"Writing index to {index_path}...")
    faiss.write_index(index, str(index_path))
    print("Done")


if __name__ == "__main__":
    main()
