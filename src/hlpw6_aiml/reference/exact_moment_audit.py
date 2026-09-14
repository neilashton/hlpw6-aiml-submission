# SPDX-License-Identifier: Apache-2.0
"""Exact triangle integration for the HiLiftAeroML pitching-moment audit.

The released HiLift surface fields are nodal and are interpreted as
piecewise-linear on the ordered fan triangulation of each native polygon.
Force is therefore integrated exactly by the familiar barycentric nodal
dual.  Pitching moment needs one additional term: position and traction are
both linear, so their cross product is quadratic.

For triangle vertices ``r_i`` and nodal tractions ``t_j``, the exact moment
about ``R`` is

``sum_ij integral(lambda_i lambda_j) * (r_i - R) cross t_j``.

The triangle mass matrix has diagonal ``A/6`` and off-diagonal ``A/12``.
This module applies that identity directly in Float64.  It also evaluates the
old vertex-lumped moment in the same pass so a campaign receipt can quantify
the quadrature correction without modifying any source field or CSV.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

ALGORITHM_ID = "hiliftaeroml-ordered-fan-exact-linear-traction-moment-v1"
CASE_RECEIPT_SCHEMA = "hiliftaeroml-exact-moment-case-receipt-v1"
AGGREGATE_SCHEMA = "hiliftaeroml-exact-moment-campaign-aggregate-v1"
CASE_ID_PATTERN = re.compile(r"geo_LHC(?P<geometry>[0-9]{3})_AoA_(?P<aoa>[0-9]+)")
EXPECTED_AOA = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
EXPECTED_ALL_CASE_COUNT = 1800
EXPECTED_PUBLIC_CASE_COUNT = 1355
EXPECTED_ALL_CASE_SET_SHA256 = (
    "fb20b620338bd7f1671294e050a28adc673897c87e8cb7601ac57e528fca3056"
)
EXPECTED_PUBLIC_NUMERIC_SHA256 = (
    "00ce591c7c6c6c2e29e02f797c8cfd09dcc0d1d3911674ac1b07631c0125d352"
)
EXPECTED_SUPPORT_LEXICAL_SHA256 = (
    "57dcf0b23cee897ed22e28e32a64ae1d1fb9cf03303fdafcc5e91066a36746d2"
)
FORCE_EXCEPTION_CASES = (
    "geo_LHC012_AoA_16",
    "geo_LHC018_AoA_16",
    "geo_LHC028_AoA_18",
    "geo_LHC028_AoA_22",
    "geo_LHC172_AoA_22",
)


class ExactMomentAuditError(ValueError):
    """Raised when exact-moment inputs or campaign identities differ."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's stable JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def content_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash a JSON object after removing its self-referential fingerprint."""

    unsigned = dict(value)
    unsigned.pop("content_fingerprint", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def case_sort_key(case_id: str) -> tuple[int, int]:
    """Return the canonical numeric geometry/AoA ordering key."""

    match = CASE_ID_PATTERN.fullmatch(case_id)
    if match is None:
        raise ExactMomentAuditError(f"invalid HiLift case ID: {case_id!r}")
    geometry = int(match.group("geometry"))
    aoa = int(match.group("aoa"))
    if geometry < 1 or geometry > 180 or aoa not in EXPECTED_AOA:
        raise ExactMomentAuditError(
            f"case ID is outside the 1800-case scope: {case_id}"
        )
    return geometry, aoa


def newline_case_set_sha256(case_ids: Sequence[str]) -> str:
    """Hash ``case_id + newline`` in the supplied order."""

    digest = hashlib.sha256()
    for case_id in case_ids:
        case_sort_key(case_id)
        digest.update(case_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_all_case_universe(case_ids: Sequence[str]) -> tuple[str, ...]:
    """Require the exact 180 geometries by ten AoAs, in numeric order."""

    observed = tuple(case_ids)
    expected = tuple(
        f"geo_LHC{geometry:03d}_AoA_{aoa}"
        for geometry in range(1, 181)
        for aoa in EXPECTED_AOA
    )
    if observed != expected:
        raise ExactMomentAuditError("all-case membership or numeric order differs")
    if len(set(observed)) != EXPECTED_ALL_CASE_COUNT:
        raise ExactMomentAuditError("all-case universe contains duplicates")
    digest = newline_case_set_sha256(observed)
    if digest != EXPECTED_ALL_CASE_SET_SHA256:
        raise ExactMomentAuditError("all-case universe SHA-256 differs")
    return observed


def _as_float_array(
    value: Any, *, shape_tail: tuple[int, ...], label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise ExactMomentAuditError(
            f"{label} must have shape (N, {', '.join(map(str, shape_tail))})"
        )
    if not np.all(np.isfinite(result)):
        raise ExactMomentAuditError(f"{label} contains non-finite values")
    return result


def _reference_vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ExactMomentAuditError("moment reference must contain three finite values")
    return result


@dataclass(frozen=True)
class TriangleBatchIntegrals:
    """Float64 sufficient statistics for one triangle batch."""

    triangle_count: int
    zero_area_triangle_count: int
    area_sum: float
    oriented_area_sum: np.ndarray
    pressure_force: np.ndarray
    viscous_force: np.ndarray
    pressure_moment_exact: np.ndarray
    viscous_moment_exact: np.ndarray
    pressure_moment_vertex_lumped: np.ndarray
    viscous_moment_vertex_lumped: np.ndarray

    def vectors(self) -> tuple[np.ndarray, ...]:
        return (
            self.oriented_area_sum,
            self.pressure_force,
            self.viscous_force,
            self.pressure_moment_exact,
            self.viscous_moment_exact,
            self.pressure_moment_vertex_lumped,
            self.viscous_moment_vertex_lumped,
        )


def integrate_triangle_batch(
    *,
    triangle_points: np.ndarray,
    pressure_coefficient: np.ndarray,
    wall_shear_coefficient: np.ndarray,
    moment_reference: Sequence[float] | np.ndarray,
) -> TriangleBatchIntegrals:
    """Integrate linear nodal pressure/shear exactly on flat triangles.

    ``triangle_points`` has shape ``(T, 3, 3)``, pressure has shape ``(T, 3)``,
    and wall shear has shape ``(T, 3, 3)``.  Pressure force follows the frozen
    HiLift sign convention ``+Cp * oriented_area_vector``.  Wall shear is
    already nondimensionalized by dynamic pressure.
    """

    points = np.asarray(triangle_points, dtype=np.float64)
    cp = np.asarray(pressure_coefficient, dtype=np.float64)
    shear = np.asarray(wall_shear_coefficient, dtype=np.float64)
    reference = _reference_vector(moment_reference)
    if points.ndim != 3 or points.shape[1:] != (3, 3):
        raise ExactMomentAuditError("triangle_points must have shape (T, 3, 3)")
    count = points.shape[0]
    if cp.shape != (count, 3):
        raise ExactMomentAuditError("pressure_coefficient must have shape (T, 3)")
    if shear.shape != (count, 3, 3):
        raise ExactMomentAuditError("wall_shear_coefficient must have shape (T, 3, 3)")
    if not (
        np.all(np.isfinite(points))
        and np.all(np.isfinite(cp))
        and np.all(np.isfinite(shear))
    ):
        raise ExactMomentAuditError("triangle fields and coordinates must be finite")

    oriented = 0.5 * np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = np.linalg.norm(oriented, axis=1)
    if not np.all(np.isfinite(area)):
        raise ExactMomentAuditError("triangle areas are non-finite")
    relative = points - reference[None, None, :]

    pressure_nodal_force = cp[:, :, None] * oriented[:, None, :] / 3.0
    viscous_nodal_force = shear * area[:, None, None] / 3.0
    pressure_force = np.sum(pressure_nodal_force, axis=(0, 1), dtype=np.float64)
    viscous_force = np.sum(viscous_nodal_force, axis=(0, 1), dtype=np.float64)

    # For traction node j, sum_i int(lambda_i lambda_j) * (r_i-R) is
    # A * (sum_i(r_i-R) + (r_j-R)) / 12.  Pressure carries its oriented
    # area vector directly, so its geometric weight omits the scalar A.
    relative_sum = np.sum(relative, axis=1, dtype=np.float64)
    quadratic_position_weight = (relative_sum[:, None, :] + relative) / 12.0
    pressure_exact = np.sum(
        np.cross(
            quadratic_position_weight,
            cp[:, :, None] * oriented[:, None, :],
        ),
        axis=(0, 1),
        dtype=np.float64,
    )
    viscous_exact = np.sum(
        np.cross(
            area[:, None, None] * quadratic_position_weight,
            shear,
        ),
        axis=(0, 1),
        dtype=np.float64,
    )
    pressure_vertex = np.sum(
        np.cross(relative, pressure_nodal_force),
        axis=(0, 1),
        dtype=np.float64,
    )
    viscous_vertex = np.sum(
        np.cross(relative, viscous_nodal_force),
        axis=(0, 1),
        dtype=np.float64,
    )

    result = TriangleBatchIntegrals(
        triangle_count=count,
        zero_area_triangle_count=int(np.count_nonzero(area == 0.0)),
        area_sum=float(np.sum(area, dtype=np.float64)),
        oriented_area_sum=np.sum(oriented, axis=0, dtype=np.float64),
        pressure_force=pressure_force,
        viscous_force=viscous_force,
        pressure_moment_exact=pressure_exact,
        viscous_moment_exact=viscous_exact,
        pressure_moment_vertex_lumped=pressure_vertex,
        viscous_moment_vertex_lumped=viscous_vertex,
    )
    if not math.isfinite(result.area_sum) or any(
        not np.all(np.isfinite(vector)) for vector in result.vectors()
    ):
        raise ExactMomentAuditError("triangle integration produced non-finite values")
    return result


class CompensatedIntegrals:
    """Neumaier-compensated campaign accumulator for deterministic batches."""

    _VECTOR_NAMES = (
        "oriented_area_sum",
        "pressure_force",
        "viscous_force",
        "pressure_moment_exact",
        "viscous_moment_exact",
        "pressure_moment_vertex_lumped",
        "viscous_moment_vertex_lumped",
    )

    def __init__(self) -> None:
        self.triangle_count = 0
        self.zero_area_triangle_count = 0
        self._area_sum = 0.0
        self._area_compensation = 0.0
        self._vectors = {
            name: np.zeros(3, dtype=np.float64) for name in self._VECTOR_NAMES
        }
        self._compensations = {
            name: np.zeros(3, dtype=np.float64) for name in self._VECTOR_NAMES
        }

    @staticmethod
    def _neumaier(
        total: np.ndarray, compensation: np.ndarray, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate = total + value
        correction = np.where(
            np.abs(total) >= np.abs(value),
            (total - candidate) + value,
            (value - candidate) + total,
        )
        return candidate, compensation + correction

    def add(self, batch: TriangleBatchIntegrals) -> None:
        self.triangle_count += batch.triangle_count
        self.zero_area_triangle_count += batch.zero_area_triangle_count
        candidate = self._area_sum + batch.area_sum
        if abs(self._area_sum) >= abs(batch.area_sum):
            correction = (self._area_sum - candidate) + batch.area_sum
        else:
            correction = (batch.area_sum - candidate) + self._area_sum
        self._area_sum = candidate
        self._area_compensation += correction
        for name in self._VECTOR_NAMES:
            total, compensation = self._neumaier(
                self._vectors[name], self._compensations[name], getattr(batch, name)
            )
            self._vectors[name] = total
            self._compensations[name] = compensation

    def finalize(self) -> TriangleBatchIntegrals:
        return TriangleBatchIntegrals(
            triangle_count=self.triangle_count,
            zero_area_triangle_count=self.zero_area_triangle_count,
            area_sum=self._area_sum + self._area_compensation,
            **{
                name: self._vectors[name] + self._compensations[name]
                for name in self._VECTOR_NAMES
            },
        )


def ordered_fan_triangle_ids(
    connectivity: np.ndarray,
    offsets: np.ndarray,
    *,
    begin_cell: int = 0,
    end_cell: int | None = None,
    point_count: int | None = None,
) -> np.ndarray:
    """Materialize ordered ``(v0, vj, vj+1)`` IDs for a cell range."""

    conn = np.asarray(connectivity)
    ends = np.asarray(offsets)
    if conn.ndim != 1 or conn.dtype.kind not in "iu":
        raise ExactMomentAuditError(
            "connectivity must be a one-dimensional integer array"
        )
    if ends.ndim != 1 or ends.dtype.kind not in "iu" or ends.size < 1:
        raise ExactMomentAuditError("offsets must be a nonempty integer vector")
    cell_count = ends.size - 1
    if end_cell is None:
        end_cell = cell_count
    if not (0 <= begin_cell <= end_cell <= cell_count):
        raise ExactMomentAuditError("cell range is invalid")
    if int(ends[0]) != 0 or int(ends[-1]) != conn.size:
        raise ExactMomentAuditError("offset endpoints differ from connectivity")
    selected_begin = np.asarray(ends[begin_cell:end_cell], dtype=np.int64)
    selected_end = np.asarray(ends[begin_cell + 1 : end_cell + 1], dtype=np.int64)
    arity = selected_end - selected_begin
    if np.any(arity < 3):
        raise ExactMomentAuditError("surface cells must have at least three vertices")
    triangle_counts = arity - 2
    total = int(np.sum(triangle_counts, dtype=np.int64))
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)
    repeated_begin = np.repeat(selected_begin, triangle_counts)
    first_triangle = np.cumsum(triangle_counts, dtype=np.int64) - triangle_counts
    local_j = (
        np.arange(total, dtype=np.int64)
        - np.repeat(first_triangle, triangle_counts)
        + 1
    )
    result = np.column_stack(
        (
            conn[repeated_begin],
            conn[repeated_begin + local_j],
            conn[repeated_begin + local_j + 1],
        )
    ).astype(np.int64, copy=False)
    if point_count is not None:
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < 0
        ):
            raise ExactMomentAuditError("point_count must be a non-negative integer")
        if result.size and (int(result.min()) < 0 or int(result.max()) >= point_count):
            raise ExactMomentAuditError("connectivity references an invalid point ID")
    return result


def integrate_ordered_fan_mesh(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    pressure_coefficient: np.ndarray,
    wall_shear_coefficient: np.ndarray,
    moment_reference: Sequence[float] | np.ndarray,
    cell_chunk: int = 100_000,
) -> TriangleBatchIntegrals:
    """Integrate an ordered polygon mesh without retaining all triangles."""

    xyz = _as_float_array(points, shape_tail=(3,), label="points")
    cp = np.asarray(pressure_coefficient, dtype=np.float64)
    shear = np.asarray(wall_shear_coefficient, dtype=np.float64)
    if cp.shape != (xyz.shape[0],) or not np.all(np.isfinite(cp)):
        raise ExactMomentAuditError(
            "pressure_coefficient must be a finite point vector"
        )
    if shear.shape != (xyz.shape[0], 3) or not np.all(np.isfinite(shear)):
        raise ExactMomentAuditError(
            "wall_shear_coefficient must be finite with shape (N, 3)"
        )
    if (
        isinstance(cell_chunk, bool)
        or not isinstance(cell_chunk, int)
        or cell_chunk <= 0
    ):
        raise ExactMomentAuditError("cell_chunk must be a positive integer")
    cell_count = np.asarray(offsets).size - 1
    accumulator = CompensatedIntegrals()
    for begin in range(0, cell_count, cell_chunk):
        end = min(begin + cell_chunk, cell_count)
        triangles = ordered_fan_triangle_ids(
            connectivity,
            offsets,
            begin_cell=begin,
            end_cell=end,
            point_count=xyz.shape[0],
        )
        accumulator.add(
            integrate_triangle_batch(
                triangle_points=xyz[triangles],
                pressure_coefficient=cp[triangles],
                wall_shear_coefficient=shear[triangles],
                moment_reference=moment_reference,
            )
        )
    return accumulator.finalize()


def body_to_wind(vector: Sequence[float] | np.ndarray, alpha_rad: float) -> np.ndarray:
    """Rotate a body vector to ``drag, side, lift`` components."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ExactMomentAuditError("body vector must contain three finite values")
    if not math.isfinite(alpha_rad):
        raise ExactMomentAuditError("alpha must be finite")
    cosine = math.cos(alpha_rad)
    sine = math.sin(alpha_rad)
    return np.asarray(
        (
            value[0] * cosine + value[2] * sine,
            value[1],
            -value[0] * sine + value[2] * cosine,
        ),
        dtype=np.float64,
    )


def coefficient_document(
    integrals: TriangleBatchIntegrals,
    *,
    reference_area: float,
    reference_length: float,
    alpha_rad: float,
) -> dict[str, Any]:
    """Normalize dimensional area/moment numerators into coefficients."""

    if not math.isfinite(reference_area) or reference_area <= 0.0:
        raise ExactMomentAuditError("reference area must be finite and positive")
    if not math.isfinite(reference_length) or reference_length <= 0.0:
        raise ExactMomentAuditError("reference length must be finite and positive")
    if integrals.triangle_count <= 0:
        raise ExactMomentAuditError("at least one triangle is required")

    pressure_force_body = integrals.pressure_force / reference_area
    viscous_force_body = integrals.viscous_force / reference_area
    total_force_body = pressure_force_body + viscous_force_body
    denominator = reference_area * reference_length
    pressure_exact = integrals.pressure_moment_exact / denominator
    viscous_exact = integrals.viscous_moment_exact / denominator
    pressure_vertex = integrals.pressure_moment_vertex_lumped / denominator
    viscous_vertex = integrals.viscous_moment_vertex_lumped / denominator

    def force_item(body: np.ndarray) -> dict[str, Any]:
        wind = body_to_wind(body, alpha_rad)
        return {
            "body": body.tolist(),
            "wind_drag_side_lift": wind.tolist(),
            "cd": float(wind[0]),
            "cy": float(wind[1]),
            "cl": float(wind[2]),
        }

    def moment_item(body: np.ndarray) -> dict[str, Any]:
        wind = body_to_wind(body, alpha_rad)
        return {
            "body": body.tolist(),
            "wind": wind.tolist(),
            "cm_body_y": float(body[1]),
        }

    return {
        "force": {
            "pressure": force_item(pressure_force_body),
            "viscous": force_item(viscous_force_body),
            "total": force_item(total_force_body),
            "closure_residual_body": (
                total_force_body - pressure_force_body - viscous_force_body
            ).tolist(),
        },
        "moment_exact_degree2": {
            "pressure": moment_item(pressure_exact),
            "viscous": moment_item(viscous_exact),
            "total": moment_item(pressure_exact + viscous_exact),
            "closure_residual_body": (
                (pressure_exact + viscous_exact) - pressure_exact - viscous_exact
            ).tolist(),
        },
        "moment_vertex_lumped": {
            "pressure": moment_item(pressure_vertex),
            "viscous": moment_item(viscous_vertex),
            "total": moment_item(pressure_vertex + viscous_vertex),
            "closure_residual_body": (
                (pressure_vertex + viscous_vertex) - pressure_vertex - viscous_vertex
            ).tolist(),
        },
    }


def decimal_half_quantum(token: str) -> float:
    """Return half of the final decimal unit represented by a CSV token."""

    try:
        value = Decimal(token.strip())
    except (AttributeError, InvalidOperation) as error:
        raise ExactMomentAuditError("published decimal token is invalid") from error
    if not value.is_finite():
        raise ExactMomentAuditError("published decimal token must be finite")
    return float(Decimal("0.5").scaleb(value.as_tuple().exponent))


def published_force_component_closure_audit(
    published: Mapping[str, float], published_tokens: Mapping[str, str]
) -> dict[str, Any]:
    """Apply the canonical independent CSV component-closure guard."""

    axes = {
        "drag": ("cd", "cdp", "cdv"),
        "lift": ("cl", "clp", "clv"),
    }
    result: dict[str, Any] = {}
    for axis, (total, pressure, viscous) in axes.items():
        values: dict[str, float] = {}
        half_quanta: dict[str, float] = {}
        for key in (total, pressure, viscous):
            value = published.get(key)
            token = published_tokens.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not isinstance(token, str)
            ):
                raise ExactMomentAuditError(
                    f"published {axis} component value/token is invalid"
                )
            try:
                token_value = float(token)
            except ValueError as error:
                raise ExactMomentAuditError(
                    f"published {axis} component token is not numeric"
                ) from error
            if token_value != float(value):
                raise ExactMomentAuditError(
                    f"published {axis} component token/value differs"
                )
            values[key] = float(value)
            half_quanta[key] = decimal_half_quantum(token)
        residual = abs(values[total] - values[pressure] - values[viscous])
        rounding_bound = sum(half_quanta.values())
        scale = max(1.0, *(abs(value) for value in values.values()))
        slack = 64.0 * np.finfo(np.float64).eps * scale
        limit = rounding_bound + slack
        status = "pass" if residual <= limit else "fail"
        result[axis] = {
            "total_key": total,
            "pressure_key": pressure,
            "viscous_key": viscous,
            "published_values": values,
            "published_csv_half_quanta": half_quanta,
            "absolute_residual": residual,
            "csv_rounding_bound": rounding_bound,
            "floating_point_scale": scale,
            "floating_point_slack_64eps": slack,
            "acceptance_limit": limit,
            "status": status,
        }
        if status != "pass":
            raise ExactMomentAuditError(
                f"published {axis} total/pressure/viscous component closure "
                f"{residual:.17g} exceeds its CSV rounding bound "
                f"{rounding_bound:.17g} plus 64eps slack {slack:.17g}"
            )
    return {
        "protocol_id": "published-force-component-closure-csv-rounding-v1",
        "protocol": (
            "abs(total-pressure-viscous) <= sum(three CSV half-quanta) + "
            "64*float64_eps*max(1,abs(total),abs(pressure),abs(viscous))"
        ),
        "status": "pass",
        "axes": result,
    }


def comparison_record(
    calculated: float,
    published: float,
    published_ci95: float,
    *,
    published_half_quantum: float = 5.0e-7,
) -> dict[str, Any]:
    """Return strict numerical and published-CI comparison diagnostics."""

    if not all(
        math.isfinite(value)
        for value in (
            calculated,
            published,
            published_ci95,
            published_half_quantum,
        )
    ):
        raise ExactMomentAuditError("comparison values must be finite")
    if published_ci95 < 0.0 or published_half_quantum < 0.0:
        raise ExactMomentAuditError("published CI95/half-quantum must be non-negative")
    numerical_tolerance = (
        1.0e-4 * max(1.0, abs(calculated), abs(published)) + published_half_quantum
    )
    delta = calculated - published
    absolute = abs(delta)
    return {
        "calculated": calculated,
        "published": published,
        "calculated_minus_published": delta,
        "absolute_error": absolute,
        "relative_error_percent": (
            100.0 * delta / abs(published) if published != 0.0 else None
        ),
        "numerical_tolerance": numerical_tolerance,
        "published_csv_half_quantum": published_half_quantum,
        "published_ci95": published_ci95,
        "ci95_plus_numerical_tolerance": published_ci95 + numerical_tolerance,
        "numerical_status": "pass" if absolute <= numerical_tolerance else "fail",
        "ci95_status": (
            "pass" if absolute <= published_ci95 + numerical_tolerance else "fail"
        ),
    }


def validate_case_receipt(
    receipt: Mapping[str, Any], *, expected_case_id: str | None = None
) -> None:
    """Fail closed on a completed per-case audit receipt."""

    expected_keys = {
        "schema_id",
        "status",
        "algorithm_id",
        "case_id",
        "input_identity",
        "reference",
        "quadrature",
        "integral_diagnostics",
        "coefficients",
        "published",
        "comparison",
        "source_fields_modified",
        "published_coefficients_modified",
        "content_fingerprint",
    }
    if set(receipt) != expected_keys or receipt.get("schema_id") != CASE_RECEIPT_SCHEMA:
        raise ExactMomentAuditError("case receipt schema differs")
    if (
        receipt.get("status") != "complete"
        or receipt.get("algorithm_id") != ALGORITHM_ID
    ):
        raise ExactMomentAuditError("case receipt status or algorithm differs")
    case_id = receipt.get("case_id")
    if not isinstance(case_id, str):
        raise ExactMomentAuditError("case receipt lacks a case ID")
    case_sort_key(case_id)
    if expected_case_id is not None and case_id != expected_case_id:
        raise ExactMomentAuditError("case receipt ID differs")
    fingerprint = receipt.get("content_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ExactMomentAuditError("case receipt fingerprint is invalid")
    try:
        observed_fingerprint = content_fingerprint(receipt)
    except (TypeError, ValueError) as error:
        raise ExactMomentAuditError(
            "case receipt contains non-JSON-finite values"
        ) from error
    if observed_fingerprint != fingerprint:
        raise ExactMomentAuditError("case receipt fingerprint differs")

    reference = receipt.get("reference")
    if not isinstance(reference, Mapping) or set(reference) != {
        "p_inf",
        "q_ref",
        "area_ref_in2",
        "length_ref_in",
        "alpha_deg",
        "forcesCoR_in",
    }:
        raise ExactMomentAuditError("case receipt reference differs")
    scalar_reference = tuple(
        reference[name]
        for name in ("p_inf", "q_ref", "area_ref_in2", "length_ref_in", "alpha_deg")
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in scalar_reference
    ):
        raise ExactMomentAuditError("case receipt reference is non-finite")
    if any(
        float(reference[name]) <= 0.0
        for name in ("q_ref", "area_ref_in2", "length_ref_in")
    ):
        raise ExactMomentAuditError("case receipt normalization is non-positive")
    expected_alpha = float(case_sort_key(case_id)[1])
    if not math.isclose(float(reference["alpha_deg"]), expected_alpha, abs_tol=1.0e-10):
        raise ExactMomentAuditError("case receipt alpha differs")
    _reference_vector(reference["forcesCoR_in"])

    quadrature = receipt.get("quadrature")
    if (
        not isinstance(quadrature, Mapping)
        or quadrature.get("surface_triangulation") != "ordered fan (v0, vj, vj+1)"
        or quadrature.get("field_interpolation") != "piecewise-linear nodal"
        or quadrature.get("pressure_sign") != "+Cp times ordered oriented area vector"
        or quadrature.get("cm_convention")
        != "positive body-y moment about forcesCoR divided by areaRef*chordRef"
    ):
        raise ExactMomentAuditError("case receipt quadrature convention differs")

    diagnostics = receipt.get("integral_diagnostics")
    vector_names = (
        "oriented_area_sum_in2",
        "pressure_force_numerator_in2",
        "viscous_force_numerator_in2",
        "pressure_moment_exact_in3",
        "viscous_moment_exact_in3",
        "pressure_moment_vertex_lumped_in3",
        "viscous_moment_vertex_lumped_in3",
    )
    if not isinstance(diagnostics, Mapping):
        raise ExactMomentAuditError("case receipt integral diagnostics are absent")
    triangle_count = diagnostics.get("triangle_count")
    zero_count = diagnostics.get("zero_area_triangle_count")
    area = diagnostics.get("surface_area_in2")
    if (
        isinstance(triangle_count, bool)
        or not isinstance(triangle_count, int)
        or triangle_count <= 0
        or isinstance(zero_count, bool)
        or not isinstance(zero_count, int)
        or zero_count < 0
        or zero_count > triangle_count
        or not isinstance(area, (int, float))
        or isinstance(area, bool)
        or not math.isfinite(float(area))
        or area <= 0.0
    ):
        raise ExactMomentAuditError("case receipt triangle/area diagnostics differ")
    vectors = {name: _reference_vector(diagnostics.get(name)) for name in vector_names}

    input_identity = receipt.get("input_identity")
    native = (
        input_identity.get("native") if isinstance(input_identity, Mapping) else None
    )
    if not isinstance(native, Mapping):
        raise ExactMomentAuditError("case receipt native identity is absent")
    topology = native.get("topology")
    field_hashes = native.get("logical_field_sha256")
    if (
        not isinstance(topology, Mapping)
        or topology.get("n_fan_triangles") != triangle_count
        or topology.get("n_points") is None
        or not isinstance(field_hashes, Mapping)
        or set(field_hashes)
        != {
            "PROJ(AVG(P))",
            "AVG(TAU_WALL(0))",
            "AVG(TAU_WALL(1))",
            "AVG(TAU_WALL(2))",
        }
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in field_hashes.values()
        )
    ):
        raise ExactMomentAuditError("case receipt topology/field identity differs")

    integrals = TriangleBatchIntegrals(
        triangle_count=triangle_count,
        zero_area_triangle_count=zero_count,
        area_sum=float(area),
        oriented_area_sum=vectors["oriented_area_sum_in2"],
        pressure_force=vectors["pressure_force_numerator_in2"],
        viscous_force=vectors["viscous_force_numerator_in2"],
        pressure_moment_exact=vectors["pressure_moment_exact_in3"],
        viscous_moment_exact=vectors["viscous_moment_exact_in3"],
        pressure_moment_vertex_lumped=vectors["pressure_moment_vertex_lumped_in3"],
        viscous_moment_vertex_lumped=vectors["viscous_moment_vertex_lumped_in3"],
    )
    recomputed_coefficients = coefficient_document(
        integrals,
        reference_area=float(reference["area_ref_in2"]),
        reference_length=float(reference["length_ref_in"]),
        alpha_rad=math.radians(float(reference["alpha_deg"])),
    )
    if receipt.get("coefficients") != recomputed_coefficients:
        raise ExactMomentAuditError("case receipt coefficients do not close")

    published = receipt.get("published")
    required_published = {
        "cd",
        "cl",
        "cm",
        "cdp",
        "cdv",
        "clp",
        "clv",
        "cd_stderr",
        "cl_stderr",
        "cm_stderr",
        "cd_ci95",
        "cl_ci95",
        "cm_ci95",
        "cm_csv_token",
        "force_csv_tokens",
    }
    if not isinstance(published, Mapping) or set(published) != required_published:
        raise ExactMomentAuditError("case receipt published values differ")
    for name in required_published.difference({"cm_csv_token", "force_csv_tokens"}):
        value = published[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ExactMomentAuditError("case receipt published value is non-finite")
    for axis in ("cd", "cl", "cm"):
        if float(published[f"{axis}_stderr"]) < 0.0 or float(
            published[f"{axis}_ci95"]
        ) < float(published[f"{axis}_stderr"]):
            raise ExactMomentAuditError("case receipt published uncertainty differs")
    cm_token = published["cm_csv_token"]
    if not isinstance(cm_token, str) or float(cm_token) != float(published["cm"]):
        raise ExactMomentAuditError("case receipt CM token differs")
    half_quantum = decimal_half_quantum(cm_token)
    force_tokens = published["force_csv_tokens"]
    force_keys = {"cd", "cl", "cm", "cdp", "cdv", "clp", "clv"}
    if not isinstance(force_tokens, Mapping) or set(force_tokens) != force_keys:
        raise ExactMomentAuditError("case receipt force tokens differ")
    for name, token in force_tokens.items():
        if not isinstance(token, str) or float(token) != float(published[name]):
            raise ExactMomentAuditError("case receipt force token/value differs")
        decimal_half_quantum(token)

    comparison = receipt.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ExactMomentAuditError("case receipt comparison is absent")
    exact_cm = recomputed_coefficients["moment_exact_degree2"]["total"]["cm_body_y"]
    vertex_cm = recomputed_coefficients["moment_vertex_lumped"]["total"]["cm_body_y"]
    expected_exact = comparison_record(
        exact_cm,
        float(published["cm"]),
        float(published["cm_ci95"]),
        published_half_quantum=half_quantum,
    )
    expected_vertex = comparison_record(
        vertex_cm,
        float(published["cm"]),
        float(published["cm_ci95"]),
        published_half_quantum=half_quantum,
    )
    if (
        comparison.get("exact_vs_published") != expected_exact
        or comparison.get("vertex_lumped_vs_published") != expected_vertex
        or comparison.get("exact_minus_vertex_lumped_cm") != exact_cm - vertex_cm
    ):
        raise ExactMomentAuditError("case receipt moment comparison differs")

    coefficient_force = recomputed_coefficients["force"]
    calculated_force = {
        "cdp": coefficient_force["pressure"]["cd"],
        "clp": coefficient_force["pressure"]["cl"],
        "cdv": coefficient_force["viscous"]["cd"],
        "clv": coefficient_force["viscous"]["cl"],
        "cd": coefficient_force["total"]["cd"],
        "cl": coefficient_force["total"]["cl"],
    }
    expected_force_vs = {
        name: {
            "calculated": float(calculated_force[name]),
            "published": float(published[name]),
            "calculated_minus_published": float(
                calculated_force[name] - published[name]
            ),
            "relative_error_percent": (
                100.0
                * (calculated_force[name] - published[name])
                / abs(published[name])
                if published[name] != 0.0
                else None
            ),
        }
        for name in calculated_force
    }
    if comparison.get("force_vs_published") != expected_force_vs:
        raise ExactMomentAuditError("case receipt force comparison differs")

    expected_checks: dict[str, Any] = {}
    expected_closure_audit = published_force_component_closure_audit(
        published, force_tokens
    )
    for axis, (total, pressure, viscous, ci_key) in {
        "drag": ("cd", "cdp", "cdv", "cd_ci95"),
        "lift": ("cl", "clp", "clv", "cl_ci95"),
    }.items():
        closure = expected_closure_audit["axes"][axis]["absolute_residual"]
        for name, role in (
            (total, "total"),
            (pressure, "pressure"),
            (viscous, "viscous"),
        ):
            numerical = (
                1.0e-4 * max(1.0, abs(calculated_force[name]), abs(published[name]))
                + decimal_half_quantum(force_tokens[name])
                + closure
            )
            allowance = published[ci_key] if role in {"total", "pressure"} else 0.0
            absolute = abs(calculated_force[name] - published[name])
            expected_checks[name] = {
                "axis": axis,
                "role": role,
                "calculated": calculated_force[name],
                "published": published[name],
                "calculated_minus_published": calculated_force[name] - published[name],
                "absolute_error": absolute,
                "numerical_tolerance": numerical,
                "published_axis_ci95_allowance": allowance,
                "acceptance_tolerance": numerical + allowance,
                "numerical_status": "pass" if absolute <= numerical else "fail",
                "status": "pass" if absolute <= numerical + allowance else "fail",
            }
    failed = sorted(
        name for name, check in expected_checks.items() if check["status"] != "pass"
    )
    expected_acceptance = {
        "protocol_id": "hilift-native-truth-vs-monitor-ci95-v1",
        "status": "pass" if not failed else "fail",
        "failed_keys": failed,
        "published_component_closure": expected_closure_audit,
        "checks": expected_checks,
    }
    if comparison.get("force_acceptance") != expected_acceptance:
        raise ExactMomentAuditError("case receipt force acceptance differs")
    if (
        receipt.get("source_fields_modified") is not False
        or receipt.get("published_coefficients_modified") is not False
    ):
        raise ExactMomentAuditError("case receipt mutation declaration differs")


__all__ = [
    "AGGREGATE_SCHEMA",
    "ALGORITHM_ID",
    "CASE_RECEIPT_SCHEMA",
    "CompensatedIntegrals",
    "EXPECTED_ALL_CASE_COUNT",
    "EXPECTED_ALL_CASE_SET_SHA256",
    "EXPECTED_PUBLIC_CASE_COUNT",
    "EXPECTED_PUBLIC_NUMERIC_SHA256",
    "EXPECTED_SUPPORT_LEXICAL_SHA256",
    "ExactMomentAuditError",
    "FORCE_EXCEPTION_CASES",
    "TriangleBatchIntegrals",
    "body_to_wind",
    "canonical_json_bytes",
    "case_sort_key",
    "coefficient_document",
    "comparison_record",
    "content_fingerprint",
    "decimal_half_quantum",
    "integrate_ordered_fan_mesh",
    "integrate_triangle_batch",
    "newline_case_set_sha256",
    "ordered_fan_triangle_ids",
    "published_force_component_closure_audit",
    "validate_all_case_universe",
    "validate_case_receipt",
]
