"""Validated three-valued rules over explicit attributes and declared facts."""

import math
from matching.schema import attributes


def validate_facts(facts):
    if not isinstance(facts, dict):
        raise ValueError("facts must be an object")
    for key, value in facts.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Invalid fact name")
        if key == "location":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in value
                )
                or not -90 <= value[0] <= 90
                or not -180 <= value[1] <= 180
            ):
                raise ValueError("location must be [latitude, longitude]")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Facts must be finite numbers")
    return facts


def validate(rule, size, depth=0):
    if depth > 12 or not isinstance(rule, dict):
        raise ValueError("Invalid or excessively nested rule")
    keys = set(rule)
    if keys in ({"all"}, {"any"}):
        group = next(iter(rule))
        items = rule[group]
        if not isinstance(items, list) or not items or len(items) > 64:
            raise ValueError("Rule groups need 1–64 children")
        for child in items:
            validate(child, size, depth + 1)
    elif keys == {"attribute", "value"}:
        if rule["attribute"] not in attributes(size) or type(rule["value"]) is not bool:
            raise ValueError("Unsupported rule attribute or value")
    elif keys == {"if", "then"}:
        validate(rule["if"], size, depth + 1)
        validate(rule["then"], size, depth + 1)
    elif "field" in keys and keys <= {"field", "min", "max"} and len(keys) >= 2:
        if (
            not isinstance(rule["field"], str)
            or not rule["field"]
            or rule["field"] == "location"
        ):
            raise ValueError("Invalid numeric field")
        for k in keys - {"field"}:
            if type(rule[k]) not in (int, float) or not math.isfinite(rule[k]):
                raise ValueError("Invalid bound")
        if rule.get("min", -math.inf) > rule.get("max", math.inf):
            raise ValueError("Reversed numeric range")
    elif keys == {"distance_km"}:
        if (
            type(rule["distance_km"]) not in (int, float)
            or not math.isfinite(rule["distance_km"])
            or rule["distance_km"] < 0
        ):
            raise ValueError("Invalid distance")
    else:
        raise ValueError("Unsupported rule syntax")
    return rule


def evaluate(rule, own, candidate, own_facts, candidate_facts):
    if "all" in rule or "any" in rule:
        group = "all" if "all" in rule else "any"
        values = [
            evaluate(r, own, candidate, own_facts, candidate_facts) for r in rule[group]
        ]
        if group == "all":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    if "attribute" in rule:
        value = candidate.get(rule["attribute"])
        return None if value is None else value == rule["value"]
    if "if" in rule:
        condition = evaluate(rule["if"], own, candidate, own_facts, candidate_facts)
        return (
            True
            if condition is False
            else (
                None
                if condition is None
                else evaluate(rule["then"], own, candidate, own_facts, candidate_facts)
            )
        )
    if "field" in rule:
        value = candidate_facts.get(rule["field"])
        return (
            None
            if value is None
            else rule.get("min", -math.inf) <= value <= rule.get("max", math.inf)
        )
    a = own_facts.get("location")
    b = candidate_facts.get("location")
    if a is None or b is None:
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, [*a, *b])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    distance = 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, h))))
    return distance <= rule["distance_km"]
