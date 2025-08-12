from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    emb = model.encode(text)
    return emb


def search_similar(target_emb, entities, top_k=5):
    entity_embs = np.array([e.get_embedding() for e in entities])
    scores = util.cos_sim(target_emb, entity_embs)[0]
    top_indices = scores.topk(top_k).indices
    return [entities[i] for i in top_indices]
