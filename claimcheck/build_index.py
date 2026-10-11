"""Embed every abstract once and save the vectors to data/embeddings.npy.

Run once after downloading:  python -m claimcheck.build_index
"""
import time

import numpy as np

from claimcheck.data import load_corpus
from claimcheck.dense import DOCUMENT_PREFIX, EMBEDDINGS_PATH, embed
from claimcheck.retrieve import searchable_text

BATCH = 32


def main():
    corpus = load_corpus()
    start = time.time()
    batches = []
    for offset in range(0, len(corpus), BATCH):
        texts = [searchable_text(doc) for doc in corpus[offset : offset + BATCH]]
        batches.append(embed(texts, DOCUMENT_PREFIX))
        print(f"\rembedded {offset + len(texts)}/{len(corpus)}", end="", flush=True)
    embeddings = np.vstack(batches)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nsaved {embeddings.shape} in {time.time() - start:.0f}s to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
