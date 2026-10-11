import numpy as np

from claimcheck.retrieve import Retriever

CORPUS = [
    {"doc_id": 10, "title": "Aspirin trial", "text": "aspirin reduces stroke risk"},
    {"doc_id": 20, "title": "Statin cohort", "text": "statins lower cholesterol"},
    {"doc_id": 30, "title": "Sleep study", "text": "rest improves memory"},
]
# one made-up direction per document, so vector search is easy to predict
EMBEDDINGS = np.eye(3, dtype=np.float32)


def test_keyword_search_finds_the_document_sharing_words_with_the_claim():
    retriever = Retriever(CORPUS, EMBEDDINGS)
    assert retriever.keyword("Does aspirin prevent stroke?", 1) == [0]


def test_keyword_search_also_matches_words_in_the_title():
    retriever = Retriever(CORPUS, EMBEDDINGS)
    assert retriever.keyword("sleep", 1) == [2]


def test_dense_search_ranks_by_vector_similarity():
    retriever = Retriever(CORPUS, EMBEDDINGS)
    assert retriever.dense(np.array([0.0, 0.6, 0.8], dtype=np.float32), 2) == [2, 1]


def test_hybrid_puts_a_document_found_by_both_searches_first():
    # keyword search finds only doc 0; vector search ranks 1, 0, 2
    retriever = Retriever(CORPUS, EMBEDDINGS)
    vector = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    assert retriever.hybrid("aspirin stroke", vector, 3) == [0, 1, 2]
