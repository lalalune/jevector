"""Shared tables for direct retrieval and the complete matching pipeline."""

import json
from pathlib import Path
from matching.providers import digest
from matching.engine import POLICY_HASH
from matching.ann import RETRIEVAL_POLICY


def sections():
    result = json.loads(Path("runs/matching-v2/comparison.json").read_text())
    data = json.loads(Path("matching/qualification-data.json").read_text())
    if (
        result["data_hash"] != digest(data)
        or result["policy_hash"] != POLICY_HASH
        or result["retrieval_policy"] != RETRIEVAL_POLICY
    ):
        raise ValueError("Stale comparison; run matching.comparison")
    headers = [
        "Method",
        "Top-1",
        "Top-5",
        "Top-10",
        "MRR@10",
        "nDCG@10",
        "Tie-adjusted top-1",
    ]
    out = []
    for mode, title in [
        ("retrieval", "Retrieval comparison"),
        ("pipeline", "Complete pipeline comparison"),
    ]:
        rows = []
        for name, splits in result[mode].items():
            v = splits["validation"]
            rows.append(
                [name]
                + [f"{v[k]:.1%}" for k in ("top1", "top5", "top10")]
                + [f"{v[k]:.3f}" for k in ("mrr10", "ndcg10")]
                + [f"{v['tie_top1']:.1%}"]
            )
        notes = [
            "Same 52-profile gallery and eight validation queries from four held-out pair families. Each query has one intended partner. Self-matches are excluded. Four no-match queries are excluded from these ranking metrics. Development and full-set metrics and ranked IDs are saved in runs/matching-v2/comparison.json.",
            "Top-k is the fraction of queries with the intended partner in the first k results. With one relevant partner, this is also Recall@k. MRR@10 measures reciprocal partner rank; nDCG@10 discounts partner rank. Missing partners score zero. Higher is better for every column.",
            "Tie-adjusted top-1 is expected credit under random ordering of equal scores (absolute tolerance 0.0000001). Other columns use descending score, then stable profile ID. Small numeric differences can affect rank.",
        ]
        if mode == "retrieval":
            notes += [
                "No mandatory-condition filtering or preference-coverage reranking is used in this table. Clef and Jev use the existing decision-vector projection and HNSW with 32 candidates. BGE-small uses 384-dimensional cosine scores over the full gallery. BM25 uses lexical scores over the full gallery. Scores account for both profile-to-request directions.",
                "BM25 + BGE-small selects ten BM25 candidates, then sorts them by BGE-small cosine score. It is not a cross-encoder. BGE uses unprefixed requests, token-bounded chunks, normalized mean pooling across chunk embeddings, and cached Cloudflare outputs from qualification.",
                "Clef and Jev receive a task-specific attribute catalog. BGE and BM25 receive the profile and request text, including declared facts and rules. This compares configured retrieval systems, not equal training data or equal task supervision. Clef and Jev are trained models too; the distinction is decision vectors versus general-purpose embeddings.",
                "On this synthetic validation set, decision vectors retrieve the intended partners more reliably than these BGE and BM25 configurations. This does not establish an advantage on general search or real profiles. The data was built around the catalog, includes near-duplicate alternatives, and has only four independent validation families. The 256-value advantage tests one added interest: cycling.",
            ]
        else:
            notes += [
                "Clef and Jev now apply reciprocal mandatory checks and preference-coverage ranking after HNSW. The three reference methods keep their retrieval rankings. This table measures the complete configured systems; gains here cannot be attributed to vector encoding alone.",
                "The detailed condition and no-match checks remain below. Use the retrieval comparison above to assess decision vectors without these rule advantages.",
            ]
        out.append((title, headers, rows, notes))
    return out
