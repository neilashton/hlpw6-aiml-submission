"""Streaming native-point field reductions with complete-case aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ContractError
from .scores import finite


@dataclass
class FieldAccumulator:
    """Accumulate additive statistics; never average chunk-local errors."""

    components: int
    count: int = 0
    sums: dict[str, list[float]] = field(
        default_factory=lambda: {
            key: []
            for key in (
                "weight",
                "squared_error",
                "squared_truth",
                "absolute_error",
                "absolute_truth",
            )
        }
    )

    def add(self, prediction: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> None:
        pred, target, weight = (
            np.asarray(x, dtype=np.float64) for x in (prediction, truth, weights)
        )
        expected = (len(weight),) if self.components == 1 else (len(weight), self.components)
        if pred.shape != expected or target.shape != expected or weight.ndim != 1:
            raise ContractError("field prediction, truth, weight, or component shape differs")
        if not all(np.all(np.isfinite(x)) for x in (pred, target, weight)) or np.any(weight < 0):
            raise ContractError("fields and non-negative weights must be finite")
        error = pred - target
        squared_error, squared_truth = error * error, target * target
        absolute_error, absolute_truth = np.abs(error), np.abs(target)
        if self.components > 1:
            squared_error, squared_truth, absolute_error, absolute_truth = (
                np.sum(x, axis=1, dtype=np.float64)
                for x in (squared_error, squared_truth, absolute_error, absolute_truth)
            )
        self.count += len(weight)
        self.sums["weight"].append(float(np.sum(weight, dtype=np.float64)))
        for key, values in zip(
            ("squared_error", "squared_truth", "absolute_error", "absolute_truth"),
            (squared_error, squared_truth, absolute_error, absolute_truth),
            strict=True,
        ):
            self.sums[key].append(float(np.dot(weight, values)))

    def statistics(self) -> dict[str, Any]:
        return {
            "entity_count": self.count,
            "component_count": self.components,
            **{key: math.fsum(values) for key, values in self.sums.items()},
        }

    def values(self, prefix: str, *, dimensional_scale: float) -> dict[str, float]:
        return field_values(self.statistics(), prefix, dimensional_scale=dimensional_scale)


def field_values(
    stats: dict[str, Any], prefix: str, *, dimensional_scale: float
) -> dict[str, float]:
    numerator = finite(stats["squared_error"], "squared_error")
    denominator = finite(stats["squared_truth"], "squared_truth")
    absolute = finite(stats["absolute_error"], "absolute_error")
    absolute_truth = finite(stats["absolute_truth"], "absolute_truth")
    weight = finite(stats["weight"], "weight")
    scale = finite(dimensional_scale, "dimensional_scale")
    components = stats["component_count"]
    if components not in (1, 3) or isinstance(components, bool):
        raise ContractError("field component count must be 1 or 3")
    if min(numerator, absolute) < 0 or min(denominator, absolute_truth, weight, scale) <= 0:
        raise ContractError("field statistic denominator, weight, or scale is invalid")
    return {
        f"{prefix}_rel_l2": 100.0 * math.sqrt(numerator / denominator),
        f"{prefix}_rel_l1": 100.0 * absolute / absolute_truth,
        f"{prefix}_mae": scale * absolute / (weight * components),
        f"{prefix}_rmse": scale * math.sqrt(numerator / (weight * components)),
    }


def r2(truth: list[float], prediction: list[float], label: str) -> float:
    if len(truth) != len(prediction) or len(truth) < 2:
        raise ContractError(f"{label} requires matching arrays with at least two cases")
    targets = [finite(value, label) for value in truth]
    predictions = [finite(value, label) for value in prediction]
    mean = math.fsum(targets) / len(targets)
    sst = math.fsum((value - mean) ** 2 for value in targets)
    if sst <= 0:
        raise ContractError(f"{label} is undefined because reference variance is zero")
    return 1.0 - math.fsum((a - b) ** 2 for a, b in zip(targets, predictions, strict=True)) / sst


def aggregate_cases(cases: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    """Recompute split metrics from complete per-case evidence."""
    if not cases:
        raise ContractError("no cases to aggregate")
    field_keys = [
        f"{prefix}_{suffix}"
        for prefix in ("surface_pressure", "surface_wall_shear")
        for suffix in ("rel_l2", "rel_l1", "mae", "rmse", "equal_entity_rel_l2")
    ]
    volume_keys = [
        f"{prefix}_{suffix}"
        for prefix in ("volume_pressure", "volume_velocity")
        for suffix in ("rel_l2", "rel_l1", "mae", "rmse")
    ]
    mean_keys = [*field_keys, "cp_cut_r2", "c_drag_mae", "c_lift_mae", "c_pitch_mae"]
    if scope == "surface_and_volume":
        mean_keys += [*volume_keys, "velocity_profile_r2"]
    result = {
        key: math.fsum(
            finite(case["metric_values"].get(key), f"{case['case_id']}.{key}") for case in cases
        )
        / len(cases)
        for key in mean_keys
    }
    for coefficient, metric in (("Cd", "cd_r2"), ("Cl", "cl_r2")):
        result[metric] = r2(
            [case["forces"]["truth"][coefficient] for case in cases],
            [case["forces"]["prediction"][coefficient] for case in cases],
            metric,
        )
    if scope == "surface_only":
        result.update({key: None for key in (*volume_keys, "velocity_profile_r2")})
    return result


def validate_case_metrics(case: dict[str, Any], scope: str) -> None:
    metrics = case["metric_values"]
    required_domains = ("surface", "volume") if scope == "surface_and_volume" else ("surface",)
    expected_fields = {"surface_pressure", "surface_wall_shear"}
    if scope == "surface_and_volume":
        expected_fields |= {"volume_pressure", "volume_velocity"}
    if set(case["field_statistics"]) != expected_fields:
        raise ContractError("field evidence differs from the declared prediction scope")
    if set(case["coverage"]) != set(required_domains):
        raise ContractError("case coverage must match its prediction scope")
    for domain in required_domains:
        coverage = case["coverage"][domain]
        count = coverage.get("expected_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError("native coverage count must be a positive integer")
        if (
            coverage.get("predicted_count") != count
            or coverage.get("missing_count") != 0
            or coverage.get("duplicate_count") != 0
        ):
            raise ContractError("complete duplicate-free native-point coverage is required")
    for prefix in ("surface_pressure", "surface_wall_shear", "volume_pressure", "volume_velocity"):
        if prefix.startswith("volume_") and scope == "surface_only":
            if any(value is not None for key, value in metrics.items() if key.startswith(prefix)):
                raise ContractError("surface-only case contains a volume metric")
            continue
        record = case["field_statistics"][prefix]
        components = 1 if prefix.endswith("pressure") else 3
        if record["primary"]["component_count"] != components:
            raise ContractError("field evidence has the wrong number of scalar components")
        domain = "surface" if prefix.startswith("surface_") else "volume"
        if record["primary"]["entity_count"] != case["coverage"][domain]["expected_count"]:
            raise ContractError("field statistics do not cover the complete native support")
        recomputed = field_values(
            record["primary"], prefix, dimensional_scale=record["dimensional_scale"]
        )
        if domain == "surface":
            secondary = record["equal_entity"]
            if (
                secondary["component_count"] != components
                or secondary["weight"] != secondary["entity_count"]
            ):
                raise ContractError("equal-entity evidence has incorrect components or weights")
            if secondary["entity_count"] != record["primary"]["entity_count"]:
                raise ContractError("equal-entity surface evidence has different coverage")
            recomputed[f"{prefix}_equal_entity_rel_l2"] = field_values(
                secondary, prefix, dimensional_scale=record["dimensional_scale"]
            )[f"{prefix}_rel_l2"]
        for key, value in recomputed.items():
            if not math.isclose(finite(metrics.get(key), key), value, rel_tol=1e-9, abs_tol=1e-10):
                raise ContractError(f"case metric {key} differs from its additive statistics")
    for coefficient, metric in (
        ("Cd", "c_drag_mae"),
        ("Cl", "c_lift_mae"),
        ("CmPitch", "c_pitch_mae"),
    ):
        expected = abs(
            finite(case["forces"]["truth"][coefficient], coefficient)
            - finite(case["forces"]["prediction"][coefficient], coefficient)
        )
        if not math.isclose(finite(metrics[metric], metric), expected, rel_tol=1e-9, abs_tol=1e-10):
            raise ContractError(f"case force error {metric} differs from its coefficients")
