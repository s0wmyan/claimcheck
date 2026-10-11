import math

import pytest

from claimcheck.bm25 import BM25, tokenize

DOCS = [["cat", "sat"], ["cat", "cat", "dog"], ["bird"]]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Vitamin-D reduces risk (by 20%).") == [
        "vitamin", "d", "reduces", "risk", "by", "20",
    ]


def test_score_matches_hand_worked_example():
    # 3 docs, average length 2. "cat" is in 2 docs: idf = ln(1 + 1.5/2.5) = ln(1.6)
    # doc 0: tf=1, length 2 -> 1*2.2 / (1 + 1.2*1.0)   = 1.0
    # doc 1: tf=2, length 3 -> 2*2.2 / (2 + 1.2*1.375) = 4.4 / 3.65
    scores = dict(BM25(DOCS).search(["cat"], k=3))
    assert scores[0] == pytest.approx(math.log(1.6) * 1.0)
    assert scores[1] == pytest.approx(math.log(1.6) * 4.4 / 3.65)


def test_documents_without_any_query_term_are_not_returned():
    assert [doc for doc, _ in BM25(DOCS).search(["cat"], k=3)] == [1, 0]


def test_rare_term_counts_for_more_than_common_term():
    # "dog" is in one doc, "cat" in two; all docs are the same length
    index = BM25([["cat", "x"], ["dog", "x"], ["cat", "y"]])
    assert index.search(["cat", "dog"], k=3)[0][0] == 1


def test_search_returns_at_most_k_results_best_first():
    results = BM25(DOCS).search(["cat", "bird"], k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_unknown_query_term_matches_nothing():
    assert BM25(DOCS).search(["zebra"], k=3) == []
