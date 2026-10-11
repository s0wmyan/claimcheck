"""Reciprocal rank fusion: merge several rankings into one.

BM25 scores and cosine similarities are on different scales, so they cannot be
added directly. Fusion ignores the scores and uses only positions: a document
at rank r in a list earns 1 / (k + r), and its total decides the final order.
k = 60 is the value from the original paper; it stops rank 1 from dominating.
"""
from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    """rankings is a list of ranked lists of document ids, best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1 / (k + rank)
    return sorted(scores, key=lambda doc: -scores[doc])
