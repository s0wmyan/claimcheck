"""Load the SciFact corpus and labelled claims."""
import json

from claimcheck.download import DATA_DIR

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def label_from_evidence(evidence):
    """SciFact attaches evidence to a claim only when an abstract settles it."""
    labels = {a["label"] for annotations in evidence.values() for a in annotations}
    if "SUPPORT" in labels:
        return "SUPPORTED"
    if "CONTRADICT" in labels:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def read_jsonl(name):
    with open(DATA_DIR / "data" / name) as file:
        return [json.loads(line) for line in file]


def load_corpus():
    return [
        {"doc_id": doc["doc_id"], "title": doc["title"], "text": " ".join(doc["abstract"])}
        for doc in read_jsonl("corpus.jsonl")
    ]


def load_claims():
    return [
        {
            "id": claim["id"],
            "claim": claim["claim"],
            "label": label_from_evidence(claim["evidence"]),
            "cited_doc_ids": claim["cited_doc_ids"],
        }
        for claim in read_jsonl("claims_dev.jsonl")
    ]
