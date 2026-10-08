import collections, math, re, unittest
import numpy as np
from matching.benchmark.embeddings import chunks, plain, rrf
from matching.benchmark.lexical import BM25


def scalar_bm25(corpus, queries):
    docs = [re.findall("[a-z0-9]+", x.lower()) for x in corpus]
    n = len(docs)
    avg = sum(map(len, docs)) / n
    freq = [collections.Counter(d) for d in docs]
    df = collections.Counter(t for d in docs for t in set(d))
    out = []
    for query in queries:
        qt = collections.Counter(re.findall("[a-z0-9]+", query.lower()))
        row = []
        for tokens, tf in zip(docs, freq):
            row.append(
                sum(
                    count
                    * math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
                    * tf[term]
                    * 2.2
                    / (tf[term] + 1.2 * (0.25 + 0.75 * len(tokens) / avg))
                    for term, count in qt.items()
                )
            )
        out.append(row)
    return np.asarray(out)


class BaselineTests(unittest.TestCase):
    def test_chunks_retain_all_words(self):
        import re

        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                spans = [m.span() for m in re.finditer(r"\S+", text)]

                class Encoding:
                    pass

                e = Encoding()
                e.ids = list(range(len(spans) + (2 if add_special_tokens else 0)))
                e.offsets = spans
                return e

        text = "one two three four five six seven"
        parts = chunks(text, Tokenizer(), budget=3)
        self.assertEqual(" ".join(parts), text)
        self.assertEqual(len(parts), 3)

    def test_rrf_and_plain_negation(self):
        self.assertEqual(rrf([["a", "b"], ["b", "a"]]), ["a", "b"])
        text = plain({"preferences": ["hiking"], "explicit_rejections": ["skiing"]})
        self.assertIn("explicit rejections: skiing", text)

    def test_bm25_matches_scalar_reference(self):
        docs = ["hiking hiking quiet", "cooking quiet"]
        queries = ["hiking hiking", "cooking", "absent"]
        np.testing.assert_allclose(
            scalar_bm25(docs, queries),
            np.asarray([BM25(docs).score(q) for q in queries]),
            rtol=1e-12,
        )
