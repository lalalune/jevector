"""Live qualification of production matching and HNSW against an independent oracle."""

import argparse, concurrent.futures, getpass, json, time
from pathlib import Path
from matching.providers import Client, atomic_json, digest
from matching.engine import extract, rank_pool, POLICY_HASH
from matching.ann import ProfileIndex, RETRIEVAL_POLICY
from matching.benchmark.reference import match


def measure(data, orders):
    people = {p["id"]: p for p in data["people"]}
    out = {}
    for split in ("development", "validation", "all"):
        queries = [
            q
            for q in data["query_ids"]
            if split == "all" or people[q]["split"] == split
        ]
        paired = [q for q in queries if people[q]["kind"] == "paired"]
        nomatch = [q for q in queries if people[q]["kind"] == "no_match"]
        correct = credit = violations = returned = abstentions = eligible_found = (
            eligible_total
        ) = 0
        for q in queries:
            rows = orders[q]
            returned += len(rows)
            for row in rows:
                violations += not match(people[q]["truth"], people[row["id"]]["truth"])[
                    "eligible"
                ]
            truthids = {x["id"] for x in data["oracle"][q]}
            eligible_found += len(truthids & {r["id"] for r in rows})
            eligible_total += len(truthids)
            if q in paired:
                correct += bool(rows and rows[0]["id"] == people[q]["partner"])
                abstentions += not rows
                tied = (
                    [
                        r["id"]
                        for r in rows
                        if abs(r["score"] - rows[0]["score"]) < 1e-12
                    ]
                    if rows
                    else []
                )
                if people[q]["partner"] in tied:
                    credit += 1 / len(tied)
        out[split] = {
            "paired_queries": len(paired),
            "partner_top1": correct,
            "tie_adjusted_partner_top1": credit,
            "paired_abstentions": abstentions,
            "no_match_queries": len(nomatch),
            "no_match_rejected": sum(not orders[q] for q in nomatch),
            "false_accepts_all_results": violations,
            "returned_candidates": returned,
            "source_eligible_found": eligible_found,
            "source_eligible_total": eligible_total,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True)
    p.add_argument("--prompt-key", action="store_true")
    p.add_argument("--providers", nargs="+", default=["clef", "jev"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--baselines-only", action="store_true")
    args = p.parse_args()
    key = getpass.getpass("Jev API key: ") if args.prompt_key else None
    data = json.loads(Path("benchmarks/profiles.json").read_text())
    destination = Path("benchmarks/qualification.json")
    result = {"data_hash": digest(data), "policy_hash": POLICY_HASH, "methods": {}}
    if destination.exists():
        old = json.loads(destination.read_text())
        if (
            old["data_hash"] == result["data_hash"]
            and old["policy_hash"] == POLICY_HASH
        ):
            result = old
    result["retrieval_policy"] = RETRIEVAL_POLICY
    for provider in [] if args.baselines_only else args.providers:
        client = Client(provider, args.account, key=key, cache="runs/matching/cache")
        for size in (64, 256):
            bundle = Path(f"benchmarks/vectors/{provider}-{size}.json")
            saved = json.loads(bundle.read_text()) if bundle.exists() else {}
            vectors = {}

            def work(person):
                if person["id"] in saved:
                    v = saved[person["id"]]
                    if (
                        v.get("input_hash") == digest(person["state"])
                        and v.get("policy_hash") == POLICY_HASH
                    ):
                        return person["id"], v
                v = extract(client, person["state"], size)
                return person["id"], v

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as pool:
                for i, (pid, v) in enumerate(pool.map(work, data["people"])):
                    vectors[pid] = v
                    saved[pid] = v
                    atomic_json(bundle, saved)
                    if (i + 1) % 10 == 0:
                        print(provider, size, i + 1, flush=True)
            start = time.perf_counter()
            exact = rank_pool(vectors, data["query_ids"])
            exact_ms = 1000 * (time.perf_counter() - start) / len(data["query_ids"])
            name = f"{provider.capitalize()} {size}"
            entry = {
                "exact": measure(data, exact),
                "exact_ms_per_query": exact_ms,
                "hnsw": {},
                "unrepresented_optional_conditions": sum(
                    len(v["unresolved_preferences"]) for v in vectors.values()
                ),
            }
            start = time.perf_counter()
            index = ProfileIndex(vectors)
            entry["index_build_seconds"] = time.perf_counter() - start
            for limit in (8, 16, 32, 51):
                orders = {}
                found = total = agreement = 0
                start = time.perf_counter()
                for q in data["query_ids"]:
                    search = index.search(q, limit, len(vectors))
                    orders[q] = search["results"]
                    wanted = {r["id"] for r in exact[q]}
                    found += len(wanted & set(search["candidate_ids"]))
                    total += len(wanted)
                    agreement += [r["id"] for r in exact[q][:5]] == [
                        r["id"] for r in orders[q][:5]
                    ]
                entry["hnsw"][str(limit)] = {
                    "candidate_recall_vs_exact": found / total if total else None,
                    "top5_list_agreement_vs_exact": agreement / len(data["query_ids"]),
                    "milliseconds_per_query": 1000
                    * (time.perf_counter() - start)
                    / len(data["query_ids"]),
                    "metrics": measure(data, orders),
                }
            result["methods"][name] = entry
            atomic_json(destination, result)
            print(
                name,
                entry["exact"]["all"],
                "HNSW32 recall",
                entry["hnsw"]["32"]["candidate_recall_vs_exact"],
                flush=True,
            )
    from matching.engine import from_structured
    from matching.schema import attributes

    form_vectors = {
        p["id"]: from_structured(p["state"], 256, p["truth"]["self"])
        for p in data["people"]
    }
    result["structured_control"] = measure(
        data, rank_pool(form_vectors, data["query_ids"])
    )
    from matching.benchmark.embeddings import embed, plain, PREFIX, rrf
    from matching.benchmark.lexical import BM25
    import numpy as np

    people = data["people"]
    ids = [p["id"] for p in people]
    positions = {k: i for i, k in enumerate(ids)}
    own = [
        p["state"]["about_me"]
        + "\nDeclared facts: "
        + plain(p["state"].get("facts", {}))
        for p in people
    ]
    wants = [
        plain(p["state"]["looking_for"])
        + "\nMandatory rules: "
        + json.dumps(p["state"].get("rules", []))
        for p in people
    ]

    def ordered(matrix):
        return {
            q: sorted(
                (k for k in ids if k != q),
                key=lambda k: (-float(matrix[positions[q], positions[k]]), k),
            )
            for q in data["query_ids"]
        }

    def baseline(name, orders):
        # Similarity baselines return ranked candidates without enforcing conditions.
        rows = {
            q: [{"id": k, "score": 1 / (i + 1)} for i, k in enumerate(order)]
            for q, order in orders.items()
        }
        result.setdefault("baselines", {})[name] = measure(data, rows)

    lexical = BM25(own)
    s = np.asarray([lexical.score(q) for q in wants])
    s /= np.maximum(s.max(axis=1, keepdims=True), 1e-9)
    s = (s + s.T) / 2
    bm = ordered(s)
    baseline("BM25", bm)
    a, metadata = embed(args.account, own)
    for prefix, label in [("", "BGE-small 384"), (PREFIX, "BGE-small 384 with prefix")]:
        b, _ = embed(args.account, wants, prefix=prefix)
        s = b @ a.T
        s = (s + s.T) / 2
        ranks = ordered(s)
        baseline(label, ranks)
        if not prefix:
            baseline(
                "BM25 + BGE-small cosine rerank",
                {
                    q: sorted(
                        bm[q][:10],
                        key=lambda k: (-float(s[positions[q], positions[k]]), k),
                    )
                    for q in bm
                },
            )
            baseline("BM25 + BGE RRF", {q: rrf([bm[q], ranks[q]]) for q in bm})
    result["baseline_protocol"] = (
        "All self text and declared facts; all requested conditions and mandatory rules. Unfiltered scores. Cosine rerank is not a cross-encoder."
    )
    atomic_json(destination, result)
    print("Saved", destination)


if __name__ == "__main__":
    main()
