"""Separate family splits and independently worded validation cases; no matcher import."""

import copy, random
from matching.schema import ATTRIBUTES, CORE, EXTRA
from matching.benchmark.reference import match
from matching.providers import atomic_json, digest


def build():
    rng = random.Random(920173)
    people = []
    pairs = []

    def state(person):
        sentences = []
        for key, value in person["truth"]["self"].items():
            if value is None:
                continue
            words = ATTRIBUTES[key].split()
            words[0] = {
                "seeks": "seek",
                "enjoys": "enjoy",
                "prefers": "prefer",
                "requires": "require",
                "follows": "follow",
                "handles": "handle",
                "actively": "actively",
                "stays": "stay",
                "communicates": "communicate",
                "keeps": "keep",
                "reflects": "reflect",
                "shares": "share",
                "wants": "want",
                "is": "am",
                "travels": "travel",
            }.get(words[0], words[0])
            phrase = " ".join(words)
            sentences.append(
                ("In daily life, I " + phrase + ".")
                if value
                else ("It would be incorrect to say that I " + phrase + ".")
            )
        rng.shuffle(sentences)
        return {
            "profile_id": person["id"],
            "about_me": " ".join(sentences),
            "looking_for": {
                c: [ATTRIBUTES[k] for k, v in person["truth"][c].items() if v]
                for c in ("prefer", "require", "exclude")
            },
            "facts": person["truth"]["facts"],
            "rules": person["truth"]["rules"],
        }

    for family in range(8):
        ids = [f"family{family}-person{side}" for side in ("a", "b")]
        members = []
        for side, pid in enumerate(ids):
            own = {k: None for k in ATTRIBUTES}
            for k in CORE:
                own[k] = rng.random() < 0.55
            for k in list(CORE)[:4]:
                own[k] = k == list(CORE)[family % 4]
            own["style.pet_friendly"] = False
            own["activity.cycling"] = True
            for k in rng.sample([k for k in EXTRA if k != "activity.cycling"], 8):
                own[k] = bool(rng.getrandbits(1))
            members.append(
                {
                    "id": pid,
                    "family": family,
                    "split": "development" if family < 4 else "validation",
                    "kind": "paired",
                    "partner": ids[1 - side],
                    "truth": {
                        "self": own,
                        "facts": {
                            "budget": 50 + family,
                            "location": [30 + family * 0.1, 10 + side * 0.1],
                        },
                        "rules": [],
                    },
                }
            )
        for i, p in enumerate(members):
            target = members[1 - i]["truth"]["self"]
            positive = [k for k in CORE if target[k] and not k.startswith("intent.")]
            negative = [
                k for k in CORE if target[k] is False and not k.startswith("intent.")
            ]
            required = [list(CORE)[family % 4]] + rng.sample(
                positive, min(2, len(positive))
            )
            excluded = rng.sample(negative, 2)
            preferred = rng.sample(positive, min(3, len(positive))) + [
                "activity.cycling"
            ]
            for c, selected in [
                ("prefer", preferred),
                ("require", required),
                ("exclude", excluded),
            ]:
                p["truth"][c] = {k: k in selected for k in ATTRIBUTES}
        people += members
        pairs.append(ids)
        for requester, target in zip(members, reversed(members)):
            alternative = copy.deepcopy(target)
            alternative.update(
                id=target["id"] + "-alternative", kind="extra_attribute_alternative"
            )
            alternative.pop("partner")
            alternative["truth"]["self"]["activity.cycling"] = False
            people.append(alternative)
            missing = copy.deepcopy(target)
            missing.update(id=target["id"] + "-missing", kind="missing_evidence")
            missing.pop("partner")
            key = next(
                k
                for k, v in requester["truth"]["require"].items()
                if v and not k.startswith("intent.")
            )
            missing["truth"]["self"][key] = None
            people.append(missing)
    for i, rule in enumerate(
        [
            {"field": "budget", "min": 1000},
            {"distance_km": 0},
            {"field": "response_hours", "max": 2},
            {"attribute": "style.pet_friendly", "value": True},
        ]
    ):
        p = copy.deepcopy([x for x in people if x["split"] == "validation"][i])
        p.update(id=f"no-match-{i}", kind="no_match", split="validation")
        p.pop("partner", None)
        p["truth"]["rules"] = [rule]
        p["truth"]["facts"]["location"] = [-30 - i, 100]
        people.append(p)
    for p in people:
        p["state"] = state(p)
    queries = [p["id"] for p in people if p["kind"] in ("paired", "no_match")]
    oracle = {}
    byid = {p["id"]: p for p in people}
    for q in queries:
        candidates = []
        for p in people:
            if p["id"] == q:
                continue
            result = match(byid[q]["truth"], p["truth"])
            if result["eligible"]:
                candidates.append({"id": p["id"], "score": result["score"]})
        candidates.sort(key=lambda x: (-x["score"], x["id"]))
        oracle[q] = candidates
        if byid[q]["kind"] == "paired":
            assert byid[q]["partner"] in [
                r["id"] for r in candidates if r["score"] == candidates[0]["score"]
            ]
        else:
            assert not candidates
    return {
        "seed": 920173,
        "people": people,
        "query_ids": queries,
        "pairs": pairs,
        "oracle": oracle,
        "protocol": "No threshold selection on validation families. Independent reference evaluator. Additional-attribute alternatives and four no-match causes.",
    }


def main():
    data = build()
    atomic_json("benchmarks/profiles.json", data)
    print(
        len(data["people"]), "profiles", len(data["query_ids"]), "queries", digest(data)
    )


if __name__ == "__main__":
    main()
