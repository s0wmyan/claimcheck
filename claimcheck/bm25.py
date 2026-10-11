"""BM25 keyword ranking, written from scratch.

For each query word that a document contains, BM25 adds

    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length))

to that document's score, where

- tf is how often the word appears in the document. The fraction grows with tf
  but flattens out, so the tenth mention counts far less than the first.
- idf is large for rare words and small for common ones, so "osteoporosis"
  matters more than "the".
- length / average_length penalises long documents, which contain more words
  by chance. b sets how strong that penalty is; k1 sets how fast tf flattens.
"""
import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        """docs is a list of token lists, one per document."""
        self.k1 = k1
        self.b = b
        self.num_docs = len(docs)
        self.lengths = [len(doc) for doc in docs]
        self.average_length = sum(self.lengths) / len(docs)
        # inverted index: word -> [(document index, times the word appears)]
        self.postings = defaultdict(list)
        for index, doc in enumerate(docs):
            for word, count in Counter(doc).items():
                self.postings[word].append((index, count))

    def search(self, query_tokens, k):
        """Return the k best (document index, score) pairs, best first."""
        scores = defaultdict(float)
        for word in query_tokens:
            postings = self.postings.get(word, [])
            containing = len(postings)
            idf = math.log(1 + (self.num_docs - containing + 0.5) / (containing + 0.5))
            for index, tf in postings:
                length_ratio = self.lengths[index] / self.average_length
                penalty = self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += idf * tf * (self.k1 + 1) / (tf + penalty)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]
