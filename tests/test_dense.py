import math

import numpy as np
import pytest

from claimcheck.dense import normalize, top_k


def test_normalize_gives_every_row_unit_length():
    rows = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.linalg.norm(rows, axis=1) == pytest.approx([1.0, 1.0])


def test_top_k_ranks_by_direction_not_length():
    docs = normalize(np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 3.0]]))
    query = normalize(np.array([[0.1, 0.0]]))[0]
    assert [doc for doc, _ in top_k(query, docs, 2)] == [0, 1]


def test_top_k_scores_are_cosine_similarities():
    docs = normalize(np.array([[10.0, 0.0], [1.0, 1.0]]))
    query = normalize(np.array([[1.0, 0.0]]))[0]
    scores = dict(top_k(query, docs, 2))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1 / math.sqrt(2))
