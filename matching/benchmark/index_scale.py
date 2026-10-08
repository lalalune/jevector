"""Profile-rule index mechanics on structured synthetic data, not live model quality."""

import random, time
from matching.providers import atomic_json
from matching.schema import attributes
from matching.engine import from_structured, rank_decoded
from matching.ann import ProfileIndex


def main():
    results = []
    for size in (64, 256):
        rng = random.Random(521 + size)
        keys = list(attributes(size))
        vectors = {}
        for i in range(1024):
            pid = f"profile-{i:04}"
            declared = {
                k: (None if rng.random() < 0.05 else rng.random() < 0.4) for k in keys
            }
            prefer = rng.sample(keys, 5)
            required = rng.sample(keys, 2)
            excluded = rng.sample([k for k in keys if k not in required], 1)
            state = {
                "profile_id": pid,
                "about_me": "Structured control",
                "looking_for": {
                    "prefer": prefer,
                    "require": required,
                    "exclude": excluded,
                },
            }
            vectors[pid] = from_structured(state, size, declared)
        start = time.perf_counter()
        index = ProfileIndex(vectors)
        build = time.perf_counter() - start
        queries = list(vectors)[:16]
        start = time.perf_counter()
        exact = {q: rank_decoded(index.decoded, q) for q in queries}
        exact_ms = 1000 * (time.perf_counter() - start) / len(queries)
        for count in (16, 64, 128):
            found = total = equivalent = 0
            start = time.perf_counter()
            for q in queries:
                result = index.search(q, count, 5)
                expected = exact[q][:5]
                expected_ids = {r["id"] for r in expected}
                found += len(expected_ids & set(result["candidate_ids"]))
                total += len(expected_ids)
                equivalent += [r["score"] for r in result["results"]] == [
                    r["score"] for r in expected
                ]
            results.append(
                {
                    "dimensions": size,
                    "profiles": len(vectors),
                    "queries": len(queries),
                    "candidates": count,
                    "exact_top5_id_candidate_recall": found / total,
                    "top5_score_equivalence": equivalent / len(queries),
                    "ann_ms_per_query": 1000
                    * (time.perf_counter() - start)
                    / len(queries),
                    "exact_ms_per_query": exact_ms,
                    "index_build_seconds": build,
                }
            )
            print(results[-1], flush=True)
    atomic_json(
        "benchmarks/index-scale.json",
        {
            "source": "Structured synthetic attributes, not provider inference. Timings exclude shared validation and decoding. Tied IDs may differ despite equal scores.",
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
