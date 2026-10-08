"""Reports for the active model-vector/HNSW pipeline."""

import json
from pathlib import Path
from matching.providers import digest
from matching.engine import POLICY_HASH


def sections():
    path = Path("runs/matching-v2/qualification.json")
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    data = json.loads(Path("matching/qualification-data.json").read_text())
    if result["data_hash"] != digest(data) or result["policy_hash"] != POLICY_HASH:
        raise ValueError("Stale qualification results")
    rows = []
    for name, m in result["methods"].items():
        h = m["hnsw"]["32"]
        v = h["metrics"]["validation"]
        rows.append(
            [
                name,
                f"{v['partner_top1']}/{v['paired_queries']}",
                f"{v['tie_adjusted_partner_top1']:.2f}/{v['paired_queries']}",
                f"{h['candidate_recall_vs_exact']:.1%}",
                f"{v['false_accepts_all_results']}/{v['returned_candidates']}",
                f"{v['no_match_rejected']}/{v['no_match_queries']}",
            ]
        )
    out = [
        (
            "Decision vectors with HNSW",
            [
                "Method",
                "Validation top-1",
                "Tie-adjusted top-1",
                "Candidate recall vs exact",
                "All returned violations",
                "No-match rejected",
            ],
            rows,
            [
                "Decision models encode the self, prefer, require, and exclude channels. HNSW compares cross-role projections of these values. Reciprocal rules then filter and rerank candidates.",
                "52 newly worded synthetic profiles in eight pair families. Four families are development data; four are validation data. The validation queries include eight paired searches and four different no-match cases.",
                "The displayed HNSW run retrieves up to 32 of 51 other profiles, with ef=64. Candidate recall is measured across all 20 queries against exact reciprocal matching.",
                "The 64-value schema cannot represent cycling. The unsupported optional preference remains in the score denominator and is reported. The 256-value schema can distinguish these alternatives.",
                "Tie-adjusted credit divides one correct result by the number of equal best scores. This prevents stable IDs from creating apparent ranking accuracy.",
                "Mandatory conditions must use supported catalog IDs or explicit rules. Numeric facts and location facts are declared input; models do not infer them.",
                "These are synthetic qualification results, not evidence of real relationship outcomes. An independent reference evaluator supplies expected eligibility and ranking.",
            ],
        )
    ]
    baseline = []
    for name, m in result.get("baselines", {}).items():
        v = m["validation"]
        baseline.append(
            [
                name,
                f"{v['partner_top1']}/{v['paired_queries']}",
                f"{v['no_match_rejected']}/{v['no_match_queries']}",
            ]
        )
    if "structured_control" in result:
        v = result["structured_control"]["validation"]
        baseline.append(
            [
                "Declared attributes + rules (no model)",
                f"{v['partner_top1']}/{v['paired_queries']}",
                f"{v['no_match_rejected']}/{v['no_match_queries']}",
            ]
        )
    out.append(
        (
            "Qualification reference methods",
            ["Method", "Validation partner top-1", "No-match rejected"],
            baseline,
            [
                "BM25 and BGE receive the descriptions, declared facts, requested conditions, and mandatory rules. They rank by reciprocal similarity without enforcing conditions or abstaining.",
                "The cosine rerank method sorts ten BM25 candidates using BGE-small. It is not a BGE cross-encoder.",
                "The declared-attribute control receives source attribute values instead of text. It isolates rule matching from model extraction; it is not a blind retrieval baseline.",
            ],
        )
    )
    rows = []
    for name, m in result["methods"].items():
        for count, h in m["hnsw"].items():
            rows.append(
                [
                    name,
                    count,
                    f"{h['candidate_recall_vs_exact']:.1%}",
                    f"{h['top5_list_agreement_vs_exact']:.1%}",
                ]
            )
    out.append(
        (
            "HNSW candidate checks",
            [
                "Method",
                "Candidates",
                "Eligible candidate recall",
                "Exact top-5 list agreement",
            ],
            rows,
            [
                "No exact fallback was used for these measurements. Returning no HNSW result does not prove there is no eligible profile outside the candidate pool."
            ],
        )
    )
    scale = Path("runs/matching-v2/index-scale.json")
    if scale.exists():
        source = json.loads(scale.read_text())
        rows = []
        for r in source["results"]:
            rows.append(
                [
                    str(r["dimensions"]),
                    str(r["candidates"]),
                    f"{r['exact_top5_id_candidate_recall']:.1%}",
                    f"{r['top5_score_equivalence']:.1%}",
                    f"{r['ann_ms_per_query']:.2f} ms",
                    f"{r['exact_ms_per_query']:.2f} ms",
                ]
            )
        out.append(
            (
                "HNSW on 1,024 structured profiles",
                [
                    "Values",
                    "Candidates",
                    "Exact top-5 ID recall",
                    "Top-5 score agreement",
                    "HNSW + checks",
                    "Exact checks",
                ],
                rows,
                [
                    "Sixteen queries per size. This is a structured synthetic index test, not a live model benchmark. Both paths use the actual reciprocal matcher.",
                    "Shared validation and decoding are excluded from query timings. HNSW index construction is excluded and recorded in the JSON artifact. These single-run local timings are not a service latency guarantee.",
                    "Larger candidate pools recover more exact results. Returned candidates pass the same checks, but approximate retrieval can omit better candidates. Equal-scoring candidate IDs may differ.",
                ],
            )
        )
    from matching.comparison_reporting import sections as comparison_sections

    return comparison_sections() + out
