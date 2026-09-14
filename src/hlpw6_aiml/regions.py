"""Zero-weight regional diagnostics from the same field evaluation stream."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import ContractError, contract_root, scope_domains
from .jsonio import read_json


def region_ids(
    domain: str, xyz: np.ndarray, *, signed_distance: np.ndarray | None = None
) -> np.ndarray:
    """Coordinates are the frozen checkpoint-normalized body-frame coordinates."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ContractError("regional coordinates must be a finite (N, 3) array")
    x, y, z = points.T
    result = np.full(len(points), 3, dtype=np.uint8)
    if domain == "surface":
        outboard = (-1.75 <= x) & (x < 1.85) & (-0.25 <= y) & (y < 2.50) & (-0.35 <= z) & (z < 0.35)
        inboard = (-1.75 <= x) & (x < 1.85) & (-1.35 <= y) & (y < -0.25) & (-0.35 <= z) & (z < 0.35)
        nacelle = (-2 <= x) & (x < 0.25) & (-0.75 <= y) & (y < 0.05) & (-0.75 <= z) & (z < 0.10)
        result[outboard], result[inboard], result[nacelle] = 2, 1, 0
    elif domain == "volume":
        if (
            signed_distance is None
            or signed_distance.shape != (len(points),)
            or not np.all(np.isfinite(signed_distance))
        ):
            raise ContractError("volume regions require the frozen native signed-distance support")
        near = (-5 <= x) & (x < 5) & (np.abs(y) < 5) & (-4 <= z) & (z < 4)
        wake = (1.50 <= x) & (x < 5) & (np.abs(y) < 2.75) & (-1.50 <= z) & (z < 1.50)
        result[near], result[wake], result[np.abs(signed_distance) < 0.02] = 2, 1, 0
    else:
        raise ContractError("unknown regional domain")
    return result


def aggregate_regions(cases: list[dict[str, Any]], scope: str) -> dict[str, Any] | None:
    if not all("regional_statistics" in case for case in cases):
        if any("regional_statistics" in case for case in cases):
            raise ContractError("regional diagnostics must cover every evaluated case")
        return None
    definition = read_json(contract_root() / "regional-diagnostics.json")
    fields = {}
    for domain in scope_domains(scope):
        for field in definition[domain]["fields"]:
            rows = []
            for code, region in enumerate(definition[domain]["region_order"]):
                stats = [case["regional_statistics"][field][code] for case in cases]
                pooled = {
                    key: math.fsum(s[key] for s in stats)
                    for key in ("squared_error", "squared_truth", "weight")
                }
                count = sum(s["entity_count"] for s in stats)
                global_error = math.fsum(
                    case["field_statistics"][field]["primary"]["squared_error"] for case in cases
                )
                local = [
                    100 * math.sqrt(s["squared_error"] / s["squared_truth"])
                    for s in stats
                    if s["squared_truth"] > 0
                ]
                normalized = []
                for case, s in zip(cases, stats, strict=True):
                    whole = case["field_statistics"][field]["primary"]
                    if s["weight"] > 0 and whole["squared_truth"] > 0:
                        normalized.append(
                            100
                            * math.sqrt(
                                (s["squared_error"] / s["weight"])
                                / (whole["squared_truth"] / whole["weight"])
                            )
                        )
                rows.append(
                    {
                        "region_id": region,
                        "entity_count": count,
                        **pooled,
                        "error_share_percent": 100 * pooled["squared_error"] / global_error
                        if global_error
                        else 0.0,
                        "pooled_relative_l2_percent": 100
                        * math.sqrt(pooled["squared_error"] / pooled["squared_truth"])
                        if pooled["squared_truth"] > 0
                        else None,
                        "case_mean_relative_l2_percent": math.fsum(local) / len(local)
                        if local
                        else None,
                        "case_mean_whole_support_normalized_rmse_percent": math.fsum(normalized)
                        / len(normalized)
                        if normalized
                        else None,
                    }
                )
            fields[field] = rows
    return {
        "schema": "hlpw6-aiml-regional-aggregate-v1",
        "weight": 0.0,
        "prediction_scope": scope,
        "case_ids": [case["case_id"] for case in cases],
        "fields": fields,
    }


def validate_regional_report(
    report: dict[str, Any], cases: list[dict[str, Any]], scope: str
) -> None:
    for case in cases:
        if set(case.get("regional_statistics", {})) != set(case["field_statistics"]):
            raise ContractError("regional diagnostics must cover every submitted field")
        for field, regions in case.get("regional_statistics", {}).items():
            if len(regions) != 4:
                raise ContractError("each field needs all four regional bins")
            whole = case["field_statistics"][field]["primary"]
            for key in ("squared_error", "squared_truth", "weight", "entity_count"):
                values = [r[key] for r in regions]
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                    for value in values
                ):
                    raise ContractError(
                        "regional sufficient statistics must be finite and non-negative"
                    )
                if not math.isclose(math.fsum(values), whole[key], rel_tol=1e-8, abs_tol=1e-8):
                    raise ContractError("regional statistics do not reconstruct global statistics")
    expected = aggregate_regions(cases, scope)
    if expected is None or report != expected:
        raise ContractError("regional report differs from its complete case evidence")
