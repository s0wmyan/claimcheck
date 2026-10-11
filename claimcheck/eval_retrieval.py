"""Score the three retrievers on the 300 labelled claims.

A retrieved abstract counts as relevant if the claim cites it.

Run:  python -m claimcheck.eval_retrieval
"""
import json
from pathlib import Path

from claimcheck.data import load_claims
from claimcheck.dense import QUERY_PREFIX, embed
from claimcheck.metrics import ndcg_at_k, recall_at_k
from claimcheck.retrieve import load_retriever

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "retrieval.json"


def mean(values):
    return sum(values) / len(values)


def main():
    retriever = load_retriever()
    claims = load_claims()
    vectors = embed([claim["claim"] for claim in claims], QUERY_PREFIX)

    retrievers = {
        "BM25 (keyword)": lambda claim, vector: retriever.keyword(claim, 10),
        "Dense (embeddings)": lambda claim, vector: retriever.dense(vector, 10),
        "Hybrid (rank fusion)": lambda claim, vector: retriever.hybrid(claim, vector, 10),
    }
    results = {}
    for name, search in retrievers.items():
        ndcg, recall3, recall10 = [], [], []
        for claim, vector in zip(claims, vectors):
            rows = search(claim["claim"], vector)
            ranked = [retriever.corpus[row]["doc_id"] for row in rows]
            relevant = set(claim["cited_doc_ids"])
            ndcg.append(ndcg_at_k(ranked, relevant, 10))
            recall3.append(recall_at_k(ranked, relevant, 3))
            recall10.append(recall_at_k(ranked, relevant, 10))
        results[name] = {
            "ndcg@10": round(mean(ndcg), 4),
            "recall@3": round(mean(recall3), 4),
            "recall@10": round(mean(recall10), 4),
        }

    RESULTS_PATH.write_text(json.dumps({"claims": len(claims), "retrievers": results}, indent=2))
    print(f"{'retriever':<22} nDCG@10  recall@3  recall@10")
    for name, scores in results.items():
        print(f"{name:<22} {scores['ndcg@10']:.3f}    {scores['recall@3']:.3f}     {scores['recall@10']:.3f}")


if __name__ == "__main__":
    main()
