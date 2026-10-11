"""Evaluation metrics for retrieval and for verdicts."""
import math


def ndcg_at_k(ranked, relevant, k):
    """How well the top k puts relevant documents first (1.0 is perfect).

    A relevant document at rank r earns 1 / log2(r + 1). The total is divided
    by the best total possible, so scores are comparable across queries.
    """
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def recall_at_k(ranked, relevant, k):
    """Fraction of the relevant documents that appear in the top k."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold, pred, labels):
    """Average of each label's F1, so a rare label counts as much as a common one."""
    scores = []
    for label in labels:
        true_positives = sum(g == label and p == label for g, p in zip(gold, pred))
        predicted = sum(p == label for p in pred)
        actual = sum(g == label for g in gold)
        if true_positives == 0:
            scores.append(0.0)
            continue
        precision = true_positives / predicted
        recall = true_positives / actual
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(labels)
