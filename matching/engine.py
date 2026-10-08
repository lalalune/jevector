"""Versioned, identity-safe matching with explicit condition coverage."""

import math
from matching.providers import digest, validate_answers
from matching.schema import SCHEMA_HASH, ATTRIBUTES, attributes, questions, dimensions
from matching import rules

POLICY = {
    "format": "matching-2",
    "self_decode": "raw-probability-0.7",
    "conditions": "model-vectors-explicit-coverage-1",
    "ranking": "specified-preference-coverage-2",
    "rules": "three-valued-1",
}
POLICY_HASH = digest(POLICY)


def contract(state, size):
    if (
        not isinstance(state, dict)
        or set(state) - {"profile_id", "about_me", "looking_for", "facts", "rules"}
        or not {"profile_id", "about_me", "looking_for"} <= set(state)
    ):
        raise ValueError(
            "Expected profile_id, about_me, looking_for; optional facts and rules"
        )
    if (
        not isinstance(state["profile_id"], str)
        or not state["profile_id"].strip()
        or not isinstance(state["about_me"], str)
    ):
        raise ValueError("Nonempty profile_id and about_me text required")
    preferences = state["looking_for"]
    if not isinstance(preferences, dict) or set(preferences) != {
        "prefer",
        "require",
        "exclude",
    }:
        raise ValueError("Expected prefer, require, exclude lists")
    catalog = attributes(size)
    aliases = {v: k for k, v in catalog.items()}
    mapped = {}
    unresolved = []
    for channel, items in preferences.items():
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            raise ValueError(
                "Conditions must be lists of catalog IDs or exact descriptions"
            )
        keys = []
        for item in items:
            key = item if item in catalog else aliases.get(item)
            if key is None:
                if channel == "prefer":
                    if item in unresolved:
                        raise ValueError("Duplicate unsupported preference")
                    unresolved.append(item)
                    continue
                raise ValueError(
                    f"Unsupported {channel} condition for {size} values: {item}. Select a supported catalog ID or an explicit rule."
                )
            if key in keys:
                raise ValueError("Duplicate condition: " + key)
            keys.append(key)
        mapped[channel] = keys
    if set(mapped["require"]) & set(mapped["exclude"]):
        raise ValueError("Conflicting mandatory conditions")
    facts = rules.validate_facts(state.get("facts", {}))
    extra = state.get("rules", [])
    if not isinstance(extra, list) or len(extra) > 64:
        raise ValueError("rules must be a list of at most 64 rules")
    for rule in extra:
        rules.validate(rule, size)
    return mapped, facts, extra, unresolved


def seal(record):
    record["record_hash"] = digest(
        {k: v for k, v in record.items() if k != "record_hash"}
    )
    return record


def from_answers(state, size, answers, provider, models, diagnostics=None):
    mapped, facts, extra, unresolved = contract(state, size)
    qs = questions(size)
    validate_answers(answers, qs)
    values = []
    known = []
    for key in qs:
        if not key.startswith("self."):
            continue
        answer = answers[key]
        p = answer["probabilities"]
        mass = sum(p.values())
        value = p["yes"] / mass
        # Do not remove unknown probability and inflate a weak answer into a strong one.
        yes = answer["choice"] == "yes" and value >= 0.7
        no = answer["choice"] == "no" and p["no"] / mass >= 0.7
        values.append(value if yes else 0.0 if no else 0.5)
        known.append(yes or no)
    for channel in ("prefer", "require", "exclude"):
        for k in attributes(size):
            a = answers[channel + "." + k]
            values.append(a["probabilities"]["yes"] / sum(a["probabilities"].values()))
            known.append(a["choice"] in ("yes", "no"))
    record = {
        "profile_id": state["profile_id"],
        "input_hash": digest(state),
        "schema_hash": SCHEMA_HASH,
        "policy": dict(POLICY),
        "policy_hash": POLICY_HASH,
        "dimensions": size,
        "dimension_ids": dimensions(size),
        "provider": provider,
        "models": models,
        "values": values,
        "known": known,
        "conditions": mapped,
        "unresolved_preferences": unresolved,
        "facts": facts,
        "rules": extra,
        "answers": answers,
        "diagnostics": diagnostics or {},
    }
    seal(record)
    unpack(record)
    return record


def extract(client, state, size):
    mapped, _, _, unresolved = contract(
        state, size
    )  # Fail before any billable request for unsupported conditions.
    answers = {}
    records = []
    for channel in ("self", "prefer", "require", "exclude"):
        items = [
            (k, v) for k, v in questions(size).items() if k.startswith(channel + ".")
        ]
        text = (
            state["about_me"]
            if channel == "self"
            else [ATTRIBUTES[k] for k in mapped[channel]]
            + (unresolved if channel == "prefer" else [])
        )
        for i in range(0, len(items), client.batch_size):
            r = client.call({"input": text}, dict(items[i : i + client.batch_size]))
            answers.update(r["response"]["answers"])
            records.append(r)
    return from_answers(
        state,
        size,
        answers,
        client.provider,
        sorted({r["response"]["model"] for r in records}),
        {
            "request_hashes": [r["request_hash"] for r in records],
            "input_tokens": sum(
                r["response"].get("usage", {}).get("input_tokens", 0) for r in records
            ),
        },
    )


def unpack(v):
    if not isinstance(v, dict):
        raise ValueError("Vector must be an object")
    for k in (
        "profile_id",
        "provider",
        "models",
        "input_hash",
        "policy_hash",
        "record_hash",
    ):
        if not v.get(k):
            raise ValueError("Missing vector provenance: " + k)
    if (
        not isinstance(v["profile_id"], str)
        or not isinstance(v["provider"], str)
        or not isinstance(v["models"], list)
        or not all(isinstance(m, str) and m for m in v["models"])
    ):
        raise ValueError("Invalid provenance")
    if v.get("policy") != POLICY or v["policy_hash"] != POLICY_HASH:
        raise ValueError("Incompatible decoder or matching policy")
    if v["record_hash"] != digest({k: x for k, x in v.items() if k != "record_hash"}):
        raise ValueError("Vector record changed; re-extract it")
    size = v["dimensions"]
    keys = list(attributes(size))
    n = len(keys)
    if v["schema_hash"] != SCHEMA_HASH or v["dimension_ids"] != dimensions(size):
        raise ValueError("Incompatible matching schema")
    if (
        len(v["values"]) != size
        or len(v["known"]) != size
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
            for x in v["values"]
        )
        or any(type(x) is not bool for x in v["known"])
    ):
        raise ValueError("Invalid vector values")
    conditions = {k: list(x) for k, x in v["conditions"].items()}
    conditions["prefer"] += v["unresolved_preferences"]
    mapped, _, _, unresolved = contract(
        {
            "profile_id": v["profile_id"],
            "about_me": "",
            "looking_for": conditions,
            "facts": v["facts"],
            "rules": v["rules"],
        },
        size,
    )
    result = {
        "self": {},
        "prefer": {},
        "require": {},
        "exclude": {},
        "facts": v["facts"],
        "rules": v["rules"],
        "unresolved_preference_count": len(unresolved),
    }
    for i, k in enumerate(keys):
        value = v["values"][i]
        result["self"][k] = (
            (True if value >= 0.7 else False if value <= 0.3 else None)
            if v["known"][i]
            else None
        )
    for j, channel in enumerate(("prefer", "require", "exclude"), 1):
        for i, k in enumerate(keys):
            expected = k in mapped[channel]
            result[channel][k] = expected
    return result


def direction(a, b):
    violations = []
    missing = []
    for k in a["self"]:
        req = a["require"].get(k)
        exc = a["exclude"].get(k)
        actual = b["self"].get(k)
        if req is None or exc is None:
            missing.append("uncertain condition: " + k)
        elif req and exc:
            violations.append("contradictory condition: " + k)
        elif req or exc:
            if actual is None:
                missing.append(k)
            elif actual != req:
                violations.append(k)
    for i, rule in enumerate(a.get("rules", [])):
        result = rules.evaluate(
            rule, a["self"], b["self"], a.get("facts", {}), b.get("facts", {})
        )
        if result is None:
            missing.append("rule " + str(i))
        elif not result:
            violations.append("rule " + str(i))
    # Uncertain preference membership remains in the denominator, never earns credit.
    wanted = [k for k, v in a["prefer"].items() if v is not False]
    hits = sum(a["prefer"][k] is True and b["self"].get(k) is True for k in wanted)
    evidence = sum(
        a["prefer"][k] is not None and b["self"].get(k) is not None for k in wanted
    )
    count = len(wanted) + a.get("unresolved_preference_count", 0)
    return {
        "eligible": not violations and not missing,
        "score": hits / count if count else None,
        "preference_count": count,
        "evidence_coverage": evidence / count if count else None,
        "violations": violations,
        "missing": missing,
    }


def reciprocal(a, b):
    ab = direction(a, b)
    ba = direction(b, a)
    scores = [x["score"] for x in (ab, ba) if x["score"] is not None]
    signal = bool(
        scores
        or any(a["require"].values())
        or any(a["exclude"].values())
        or any(b["require"].values())
        or any(b["exclude"].values())
        or a.get("rules")
        or b.get("rules")
    )
    own_evidence = any(v is not None for v in a["self"].values()) and any(
        v is not None for v in b["self"].values()
    )
    return {
        "eligible": ab["eligible"] and ba["eligible"] and signal and own_evidence,
        "score": sum(scores) / len(scores) if scores else 0.0,
        "insufficient_evidence": not signal or not own_evidence,
        "forward": ab,
        "reverse": ba,
    }


def prepare(vectors):
    decoded = {}
    reference = None
    for key, v in vectors.items():
        if key != v.get("profile_id"):
            raise ValueError("Dictionary key must equal stable profile_id")
        decoded[key] = unpack(v)
        identity = tuple(
            str(v[k])
            for k in ("schema_hash", "policy_hash", "dimensions", "provider", "models")
        )
        if reference is not None and identity != reference:
            raise ValueError("Mixed provider, model, schema, or policy")
        reference = identity
    return decoded


def rank_decoded(decoded, query_id, candidates=None):
    if query_id not in decoded:
        raise ValueError("Query profile is missing")
    out = []
    for key in decoded if candidates is None else candidates:
        if key == query_id:
            continue
        row = reciprocal(decoded[query_id], decoded[key])
        if row["eligible"]:
            out.append({"id": key, **row})
    return sorted(out, key=lambda row: (-row["score"], row["id"]))


def rank_pool(vectors, query_ids):
    decoded = prepare(vectors)
    return {q: rank_decoded(decoded, q) for q in query_ids}


def rank_candidates(query_id, vectors):
    return rank_pool(vectors, [query_id])[query_id]


def from_structured(state, size, declared):
    """Reference path for explicit form answers; no model inference is claimed."""
    mapped, _, _, _ = contract(state, size)
    if (
        not isinstance(declared, dict)
        or set(declared) - set(attributes(size))
        or any(v is not None and type(v) is not bool for v in declared.values())
    ):
        raise ValueError("Invalid declared attributes")
    answers = {}
    for key, q in questions(size).items():
        channel, attribute = key.split(".", 1)
        value = (
            declared.get(attribute)
            if channel == "self"
            else attribute in mapped[channel]
        )
        choice = "unknown" if value is None else "yes" if value else "no"
        answers[key] = {
            "type": "choice",
            "choice": choice,
            "confidence": 1.0,
            "probabilities": {c: float(c == choice) for c in q["criteria"]},
        }
    return from_answers(state, size, answers, "structured", ["declared-attributes-1"])
