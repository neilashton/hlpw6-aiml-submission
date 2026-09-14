from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .contracts import SURFACE_ONLY_UNAVAILABLE, ContractError, scope_domains, scoring


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite number")
    return float(value)


def component_scores(metrics: dict[str, Any], scope: str) -> dict[str, float]:
    scope_domains(scope)
    result = {}
    for component in scoring()["overall_score_composite"]["components"]:
        key = component["metric_id"]
        if scope == "surface_only" and key in SURFACE_ONLY_UNAVAILABLE:
            if metrics.get(key) is not None:
                raise ContractError(f"{key} must be null for surface-only results")
            result[key] = 0.0
            continue
        value = finite(metrics.get(key), key)
        if component["transform"] == "bounded_error":
            if value < 0:
                raise ContractError(f"{key} cannot be negative")
            value = 1.0 - value / component["cap"]
        result[key] = 100.0 * max(0.0, min(1.0, value))
    return result


def calculate_scores(metrics: dict[str, Any], scope: str) -> dict[str, Any]:
    components = scoring()["overall_score_composite"]["components"]
    points = component_scores(metrics, scope)
    weights = {item["metric_id"]: item["weight"] for item in components}
    scores = {"overall_score": math.fsum(points[key] * weights[key] for key in weights)}
    for group in scoring()["component_score_groups"]["groups"]:
        keys = group["component_metric_ids"]
        scores[group["metric_id"]] = math.fsum(
            points[key] * weights[key] for key in keys
        ) / math.fsum(weights[key] for key in keys)
    return {
        "metric_values": {**metrics, **scores},
        "component_scores": points,
        "component_availability": {
            key: not (scope == "surface_only" and key in SURFACE_ONLY_UNAVAILABLE)
            for key in weights
        },
        "scoring": {
            "maximum_score": 60 if scope == "surface_only" else 100,
            "weights_renormalized": False,
            "weights": weights,
        },
    }


def displayed_score(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def competition_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Rank only rows already selected from one exact split and evaluator contract."""
    identities = {
        (
            row["split_id"],
            row.get("case_set_sha256"),
            row.get("training_case_set_sha256"),
            row.get("validation_case_set_sha256"),
            row.get("evaluator_version"),
        )
        for row in rows
    }
    if len(identities) > 1:
        raise ContractError("rank inputs contain different split or evaluator identities")
    ordered = sorted(rows, key=lambda row: -displayed_score(row["metric_values"]["overall_score"]))
    ranks, previous, rank = {}, None, 0
    for index, row in enumerate(ordered, 1):
        value = displayed_score(row["metric_values"]["overall_score"])
        if value != previous:
            rank, previous = index, value
        ranks[row["import_id"]] = rank
    return ranks
