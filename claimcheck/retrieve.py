"""The three retrievers compared in this project, over one corpus.

Every method returns corpus row numbers, best first.
"""
import numpy as np

from claimcheck.bm25 import BM25, tokenize
from claimcheck.data import load_corpus
from claimcheck.dense import EMBEDDINGS_PATH, top_k
from claimcheck.fusion import reciprocal_rank_fusion

# how many results each retriever contributes to the hybrid ranking
POOL = 100


def searchable_text(doc):
    return doc["title"] + " " + doc["text"]


class Retriever:
    def __init__(self, corpus, embeddings):
        self.corpus = corpus
        self.bm25 = BM25([tokenize(searchable_text(doc)) for doc in corpus])
        self.embeddings = embeddings

    def keyword(self, claim, k):
        return [row for row, _ in self.bm25.search(tokenize(claim), k)]

    def dense(self, claim_vector, k):
        return [row for row, _ in top_k(claim_vector, self.embeddings, k)]

    def hybrid(self, claim, claim_vector, k):
        rankings = [self.keyword(claim, POOL), self.dense(claim_vector, POOL)]
        return reciprocal_rank_fusion(rankings)[:k]


def load_retriever():
    return Retriever(load_corpus(), np.load(EMBEDDINGS_PATH))
