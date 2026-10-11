"""Dense retrieval: compare texts by meaning using embedding vectors.

An embedding model turns a text into a list of numbers such that texts with
similar meaning point in similar directions. Every vector is scaled to length
1, so the dot product of two vectors is their cosine similarity.
"""
import numpy as np
import requests

from claimcheck.download import DATA_DIR

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBEDDINGS_PATH = DATA_DIR.parent / "embeddings.npy"
# nomic-embed-text was trained with these prefixes and expects them
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def embed(texts, prefix):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [prefix + text for text in texts]},
        timeout=600,
    )
    response.raise_for_status()
    return normalize(np.array(response.json()["embeddings"], dtype=np.float32))


def top_k(query, docs, k):
    """Return the k best (row index, cosine similarity) pairs, best first."""
    scores = docs @ query
    best = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in best]
