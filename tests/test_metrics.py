import math

import pytest

from claimcheck.metrics import accuracy, macro_f1, ndcg_at_k, recall_at_k


def test_ndcg_is_one_when_the_only_relevant_doc_is_first():
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == pytest.approx(1.0)


def test_ndcg_discounts_a_relevant_doc_at_rank_two():
    assert ndcg_at_k(["b", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_with_two_relevant_docs_at_ranks_one_and_three():
    # DCG = 1 + 1/log2(4) = 1.5;  ideal DCG = 1 + 1/log2(3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_ignores_results_beyond_k():
    assert ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_recall_is_fraction_of_relevant_docs_in_top_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5


def test_accuracy_is_fraction_of_matching_labels():
    assert accuracy(["A", "B", "A", "A"], ["A", "A", "A", "B"]) == 0.5


def test_macro_f1_averages_per_class_f1_equally():
    # S: precision 1, recall 2/3 -> F1 0.8      R: precision 1/2, recall 1 -> F1 2/3
    gold = ["S", "S", "S", "R"]
    pred = ["S", "S", "R", "R"]
    assert macro_f1(gold, pred, ["S", "R"]) == pytest.approx((0.8 + 2 / 3) / 2)


def test_macro_f1_scores_a_never_predicted_class_as_zero():
    # S: precision 1/2, recall 1 -> F1 2/3      R: never predicted -> 0
    assert macro_f1(["S", "R"], ["S", "S"], ["S", "R"]) == pytest.approx(1 / 3)
