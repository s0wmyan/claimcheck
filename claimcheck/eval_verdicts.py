"""Score the judge's verdicts under three conditions.

- closed_book: the model sees only the claim.
- retrieved:   the model also sees the top 3 abstracts from hybrid retrieval.
- oracle:      the model sees the abstracts the claim cites (perfect retrieval).

Each prediction is appended to results/predictions.jsonl as it is made, and
claims already in that file are skipped, so an interrupted run can be resumed.

Run:  python -m claimcheck.eval_verdicts            (all 300 claims)
      python -m claimcheck.eval_verdicts --sample 150
"""
import argparse
import json
import random
import time
from pathlib import Path

from claimcheck.data import LABELS, load_claims
from claimcheck.dense import QUERY_PREFIX, embed
from claimcheck.judge import PARSE_FAILED, judge
from claimcheck.metrics import accuracy, macro_f1
from claimcheck.retrieve import load_retriever

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
CONDITIONS = ["closed_book", "retrieved", "oracle"]
TOP_K = 3


def read_predictions():
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH) as file:
        return [json.loads(line) for line in file]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="use the first N claims of a fixed shuffle")
    args = parser.parse_args()

    retriever = load_retriever()
    claims = load_claims()
    # shuffling with a fixed seed means a smaller sample is always a subset of a larger one
    random.Random(0).shuffle(claims)
    claims = claims[: args.sample]
    vectors = embed([claim["claim"] for claim in claims], QUERY_PREFIX)
    by_doc_id = {doc["doc_id"]: doc for doc in retriever.corpus}

    def abstracts_for(condition, claim, vector):
        if condition == "closed_book":
            return []
        if condition == "oracle":
            return [by_doc_id[doc_id] for doc_id in claim["cited_doc_ids"]]
        return [retriever.corpus[row] for row in retriever.hybrid(claim["claim"], vector, TOP_K)]

    done = {(p["condition"], p["id"]) for p in read_predictions()}
    with open(PREDICTIONS_PATH, "a") as file:
        for condition in CONDITIONS:
            for number, (claim, vector) in enumerate(zip(claims, vectors), start=1):
                if (condition, claim["id"]) in done:
                    continue
                start = time.time()
                verdict, reason = judge(claim["claim"], abstracts_for(condition, claim, vector))
                prediction = {
                    "condition": condition,
                    "id": claim["id"],
                    "gold": claim["label"],
                    "predicted": verdict,
                    "reason": reason,
                    "seconds": round(time.time() - start, 2),
                }
                file.write(json.dumps(prediction) + "\n")
                file.flush()
                print(f"{condition} {number}/{len(claims)} {prediction['seconds']}s", flush=True)

    wanted = {claim["id"] for claim in claims}
    predictions = [p for p in read_predictions() if p["id"] in wanted]
    gold_all = [claim["label"] for claim in claims]
    majority = max(LABELS, key=gold_all.count)
    summary = {
        "claims": len(claims),
        "majority_baseline": {
            "label": majority,
            "accuracy": round(gold_all.count(majority) / len(claims), 4),
        },
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = [p for p in predictions if p["condition"] == condition]
        gold = [p["gold"] for p in rows]
        predicted = [p["predicted"] for p in rows]
        summary["conditions"][condition] = {
            "accuracy": round(accuracy(gold, predicted), 4),
            "macro_f1": round(macro_f1(gold, predicted, LABELS), 4),
            "unreadable_replies": sum(p["reason"] == PARSE_FAILED for p in rows),
            "mean_seconds": round(sum(p["seconds"] for p in rows) / len(rows), 2),
        }
    (RESULTS_DIR / "verdicts.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
