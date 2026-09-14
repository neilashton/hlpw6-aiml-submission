"""Compact, prediction-only HiLiftAeroML diagnostic profile primitives.

This module is deliberately independent of the native profile-v1 package
contract.  Version 1 remains the lossless audit representation.  The compact
version keeps plotting coordinates, topology, quadrature weights, validity
gaps, and reference values in an evaluator-owned support artifact.  A
participant artifact contains exactly two prediction arrays; the surrounding
case metadata binds their support and prediction order with four SHA-256
digests:

* Cp retains every native point in physical ``(row, graph)`` supports of at
  most 128 points; larger supports are sampled at no more than 128 points.
  Predictions use ``int16(round(Cp * 1024))`` and are delta coded, with a reset
  at every retained branch.
* velocity stores scalar ``speed/|U_inf|`` as float32 for all and only valid
  rows.  Its exact IEEE-754 bits are unsigned-delta coded modulo ``2**32``
  and byte-shuffled before ordinary ZIP Deflate compression.  The transform
  is lossless and browser-decodable; the evaluator-owned mask reconstructs
  all five 801-row lines and their explicit NaN gaps.

The Cp score uses the same exact piecewise-linear squared-error quadrature and
per-physical-graph centering as the native evaluator.  Velocity uses the same
physical line weights and per-station centering.  These are diagnostic line
scores only; this module has no full-surface field-scoring path.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


COMPACT_PROFILE_FORMAT = (
    "fluidsbench-hiliftaeroml-compact-profile-chunks-v2-candidate"
)
CP_POINTS_PER_GRAPH = 128
CP_FIXED_POINT_SCALE = 1024
CP_TINY_BRANCH_RELATIVE_LENGTH = 1.0e-4
VELOCITY_STATIONS = ("B.2", "B.3", "C.1", "C.2", "C.3")
VELOCITY_ROWS_PER_STATION = 801
VELOCITY_ROW_COUNT = len(VELOCITY_STATIONS) * VELOCITY_ROWS_PER_STATION
VELOCITY_STORAGE_ENCODING = (
    "little_endian_float32_bits_unsigned_delta_modulo_2pow32_"
    "byte_shuffle_v1"
)

SUPPORT_ARRAYS = (
    "cp_xyz_in",
    "cp_arc_length_in",
    "cp_truth",
    "cp_branch_point_offsets",
    "cp_branch_row_code",
    "cp_branch_graph_component_code",
    "cp_branch_component_code",
    "cp_branch_plane_piece_code",
    "cp_branch_side_code",
    "cp_branch_topology_patch_code",
    "cp_source_branch_index",
    "velocity_requested_xyz_in",
    "velocity_valid_mask",
    "velocity_station_names",
    "velocity_station_row_offsets",
    "velocity_line_length_weights_in",
    "velocity_truth_speed_over_uinf",
)

PREDICTION_ARRAYS = (
    "cp_q_delta",
    "velocity_speed_over_u_inf",
)

_CP_NATIVE_SUPPORT_ARRAYS = (
    "cut_xyz_in",
    "branch_closed",
    "branch_vertex_offsets",
    "branch_vertex_ids",
    "branch_segment_offsets",
    "segment_lengths_in",
    "branch_row_code",
    "branch_graph_component_code",
    "branch_component_code",
    "branch_plane_piece_code",
    "branch_side_code",
    "branch_topology_patch_code",
)

CP_SUPPORT_ARRAYS = SUPPORT_ARRAYS[:11]
VELOCITY_SUPPORT_ARRAYS = SUPPORT_ARRAYS[11:]
CP_PREDICTION_ORDER_ARRAYS = (
    "cp_branch_point_offsets",
    "cp_branch_row_code",
    "cp_branch_graph_component_code",
    "cp_source_branch_index",
    "cp_xyz_in",
    "cp_arc_length_in",
)


class CompactProfileError(ValueError):
    """Raised when compact profile support or predictions violate the contract."""


def _fail(message: str) -> None:
    raise CompactProfileError(message)


def _require_exact_keys(
    arrays: Mapping[str, np.ndarray], expected: Sequence[str], label: str
) -> None:
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        extra = sorted(set(arrays) - set(expected))
        _fail(f"{label} array inventory differs (missing={missing}, extra={extra})")


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        _fail(f"{name} must be a numpy array")
    if dtype is not None and value.dtype != np.dtype(dtype):
        _fail(f"{name} dtype must be {np.dtype(dtype)}, observed {value.dtype}")
    if shape is not None and (
        value.ndim != len(shape)
        or any(
            expected is not None and observed != expected
            for observed, expected in zip(value.shape, shape, strict=True)
        )
    ):
        _fail(f"{name} shape must match {shape}, observed {value.shape}")
    return value


def _require_offsets(value: np.ndarray, *, end: int, label: str) -> None:
    if (
        value.dtype != np.dtype(np.int64)
        or value.ndim != 1
        or len(value) < 2
        or value[0] != 0
        or value[-1] != end
        or np.any(np.diff(value) <= 0)
    ):
        _fail(
            f"{label} must be a non-empty, strictly increasing int64 vector "
            f"from zero to {end}"
        )


def allocate_branch_samples(
    branch_lengths: Sequence[float] | np.ndarray,
    *,
    point_budget: int = CP_POINTS_PER_GRAPH,
    relative_prune: float = CP_TINY_BRANCH_RELATIVE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Select branches and apportion one physical graph's point budget.

    Branches shorter than ``relative_prune * total_graph_length`` are omitted.
    If more than half the point budget survive, the longest branches are kept
    so every retained branch can receive both endpoints.  Remaining points are
    assigned by physical arc length using deterministic largest remainder.
    Returned branch indices are in their original order.
    """

    lengths = np.asarray(branch_lengths, dtype=np.float64)
    if lengths.ndim != 1 or not len(lengths):
        _fail("branch_lengths must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        _fail("branch_lengths must be finite and positive")
    if (
        not isinstance(point_budget, int)
        or isinstance(point_budget, bool)
        or point_budget < 2
    ):
        _fail("point_budget must be an integer of at least two")
    if not math.isfinite(relative_prune) or not 0.0 <= relative_prune < 1.0:
        _fail("relative_prune must be finite and in [0, 1)")

    total = float(np.sum(lengths, dtype=np.float64))
    keep = np.flatnonzero(lengths >= relative_prune * total)
    # This fallback matters only for unusually fragmented synthetic graphs;
    # the longest physical branch must never disappear entirely.
    if not len(keep):
        keep = np.asarray([int(np.argmax(lengths))], dtype=np.int64)
    maximum_branches = point_budget // 2
    if len(keep) > maximum_branches:
        longest_first = np.argsort(-lengths[keep], kind="stable")
        keep = np.sort(keep[longest_first[:maximum_branches]])

    allocation = np.full(len(keep), 2, dtype=np.int64)
    remaining = point_budget - int(np.sum(allocation, dtype=np.int64))
    if remaining:
        shares = lengths[keep] / float(np.sum(lengths[keep], dtype=np.float64))
        exact_extra = shares * remaining
        base_extra = np.floor(exact_extra).astype(np.int64)
        allocation += base_extra
        leftover = remaining - int(np.sum(base_extra, dtype=np.int64))
        if leftover:
            order = np.argsort(-(exact_extra - base_extra), kind="stable")
            allocation[order[:leftover]] += 1
    if int(np.sum(allocation, dtype=np.int64)) != point_budget:
        _fail("internal Cp point allocation did not exhaust the graph budget")
    return keep.astype(np.int64, copy=False), allocation


def _validate_cp_native_support(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    missing = [name for name in _CP_NATIVE_SUPPORT_ARRAYS if name not in arrays]
    if missing:
        _fail(f"native Cp support is missing arrays {missing}")
    xyz = _require_array(arrays, "cut_xyz_in", dtype=np.float64, shape=(None, 3))
    vertex_offsets = _require_array(
        arrays, "branch_vertex_offsets", dtype=np.int64, shape=(None,)
    )
    branch_count = len(vertex_offsets) - 1
    if branch_count < 1:
        _fail("native Cp support must contain at least one branch")
    vertex_ids = _require_array(
        arrays, "branch_vertex_ids", dtype=np.int64, shape=(None,)
    )
    segment_offsets = _require_array(
        arrays,
        "branch_segment_offsets",
        dtype=np.int64,
        shape=(branch_count + 1,),
    )
    segment_lengths = _require_array(
        arrays, "segment_lengths_in", dtype=np.float64, shape=(None,)
    )
    closed = _require_array(
        arrays, "branch_closed", dtype=np.bool_, shape=(branch_count,)
    )
    row_codes = _require_array(
        arrays, "branch_row_code", dtype=np.uint8, shape=(branch_count,)
    )
    graph_codes = _require_array(
        arrays,
        "branch_graph_component_code",
        dtype=np.int64,
        shape=(branch_count,),
    )
    component_codes = _require_array(
        arrays, "branch_component_code", dtype=np.uint8, shape=(branch_count,)
    )
    plane_piece_codes = _require_array(
        arrays, "branch_plane_piece_code", dtype=np.uint8, shape=(branch_count,)
    )
    side_codes = _require_array(
        arrays, "branch_side_code", dtype=np.uint8, shape=(branch_count,)
    )
    topology_patch_codes = _require_array(
        arrays,
        "branch_topology_patch_code",
        dtype=np.uint8,
        shape=(branch_count,),
    )
    _require_offsets(vertex_offsets, end=len(vertex_ids), label="branch_vertex_offsets")
    _require_offsets(
        segment_offsets,
        end=len(segment_lengths),
        label="branch_segment_offsets",
    )
    if not np.all(np.isfinite(xyz)):
        _fail("native Cp coordinates must be finite")
    if (
        not np.all(np.isfinite(segment_lengths))
        or np.any(segment_lengths <= 0.0)
    ):
        _fail("native Cp segment lengths must be finite and positive")
    if np.any(vertex_ids < 0) or np.any(vertex_ids >= len(xyz)):
        _fail("native Cp branch_vertex_ids contain an out-of-range vertex")
    if np.any(row_codes >= 10):
        _fail("native Cp row codes must remain in the A-J catalog")
    if np.any(graph_codes < 0):
        _fail("native Cp graph component codes must be non-negative")

    for branch_index in range(branch_count):
        vertex_start = int(vertex_offsets[branch_index])
        vertex_end = int(vertex_offsets[branch_index + 1])
        segment_start = int(segment_offsets[branch_index])
        segment_end = int(segment_offsets[branch_index + 1])
        vertex_count = vertex_end - vertex_start
        segment_count = segment_end - segment_start
        expected_segment_count = vertex_count if closed[branch_index] else vertex_count - 1
        if vertex_count < 2 or segment_count != expected_segment_count:
            _fail(
                "every native Cp branch must have one segment per vertex when "
                "closed, or one fewer segment than vertices when open"
            )
        branch_xyz = xyz[vertex_ids[vertex_start:vertex_end]]
        if closed[branch_index]:
            branch_xyz = np.concatenate((branch_xyz, branch_xyz[:1]), axis=0)
        geometric_lengths = np.linalg.norm(np.diff(branch_xyz, axis=0), axis=1)
        declared_lengths = segment_lengths[segment_start:segment_end]
        if not np.allclose(
            geometric_lengths, declared_lengths, rtol=1.0e-11, atol=1.0e-12
        ):
            _fail("native Cp branch geometry and physical segment lengths differ")

    return (
        xyz,
        vertex_offsets,
        vertex_ids,
        segment_offsets,
        segment_lengths,
        closed,
        row_codes,
        graph_codes,
        component_codes,
        plane_piece_codes,
        side_codes,
        topology_patch_codes,
    )


def _velocity_truth_speed(
    truth_velocity: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    truth = np.asarray(truth_velocity)
    if truth.dtype != np.dtype(np.float64):
        _fail("velocity truth must use float64")
    if truth.shape == (VELOCITY_ROW_COUNT, 3):
        speed = np.linalg.norm(truth, axis=1)
    elif truth.shape == (VELOCITY_ROW_COUNT,):
        speed = np.array(truth, copy=True)
    else:
        _fail(
            "velocity truth must have shape "
            f"({VELOCITY_ROW_COUNT}, 3) or ({VELOCITY_ROW_COUNT},)"
        )
    if not np.all(np.isfinite(speed[valid])) or np.any(speed[valid] < 0.0):
        _fail("velocity truth speed must be finite and non-negative on valid rows")
    if not np.all(np.isnan(speed[~valid])):
        _fail("velocity truth must preserve NaN on every invalid row")
    return speed.astype(np.float64, copy=False)


def build_compact_support(
    *,
    cp_native: Mapping[str, np.ndarray],
    truth_cp: np.ndarray,
    velocity_native: Mapping[str, np.ndarray],
    truth_velocity_nd: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build evaluator-owned deterministic compact support from native v1 data."""

    (
        xyz,
        vertex_offsets,
        vertex_ids,
        segment_offsets,
        segment_lengths,
        closed,
        row_codes,
        graph_codes,
        component_codes,
        plane_piece_codes,
        side_codes,
        topology_patch_codes,
    ) = _validate_cp_native_support(cp_native)
    truth = np.asarray(truth_cp)
    if truth.dtype != np.dtype(np.float64) or truth.shape != (len(xyz),):
        _fail(f"truth_cp must be float64 with shape ({len(xyz)},)")
    if not np.all(np.isfinite(truth)):
        _fail("truth_cp must be finite")

    branches_by_graph: dict[tuple[int, int], list[int]] = defaultdict(list)
    for branch_index, (row, graph) in enumerate(
        zip(row_codes.tolist(), graph_codes.tolist(), strict=True)
    ):
        branches_by_graph[(int(row), int(graph))].append(branch_index)

    compact_xyz: list[np.ndarray] = []
    compact_arc: list[np.ndarray] = []
    compact_truth: list[np.ndarray] = []
    compact_rows: list[int] = []
    compact_graphs: list[int] = []
    compact_components: list[int] = []
    compact_plane_pieces: list[int] = []
    compact_sides: list[int] = []
    compact_topology_patches: list[int] = []
    source_branches: list[int] = []
    point_offsets = [0]

    for graph_key in sorted(branches_by_graph):
        branch_indices = branches_by_graph[graph_key]
        branch_lengths = np.asarray(
            [
                np.sum(
                    segment_lengths[
                        int(segment_offsets[index]) : int(segment_offsets[index + 1])
                    ],
                    dtype=np.float64,
                )
                for index in branch_indices
            ],
            dtype=np.float64,
        )
        native_point_counts = np.asarray(
            [
                int(
                    vertex_offsets[index + 1]
                    - vertex_offsets[index]
                    + int(closed[index])
                )
                for index in branch_indices
            ],
            dtype=np.int64,
        )
        retain_native_points = (
            int(np.sum(native_point_counts, dtype=np.int64))
            <= CP_POINTS_PER_GRAPH
        )
        if retain_native_points:
            keep = np.arange(len(branch_indices), dtype=np.int64)
            allocations = native_point_counts
        else:
            keep, allocations = allocate_branch_samples(branch_lengths)
        for local_index, point_count in zip(
            keep.tolist(), allocations.tolist(), strict=True
        ):
            source_index = branch_indices[local_index]
            start = int(vertex_offsets[source_index])
            end = int(vertex_offsets[source_index + 1])
            ids = vertex_ids[start:end]
            branch_xyz = xyz[ids]
            branch_truth = truth[ids]
            if closed[source_index]:
                branch_xyz = np.concatenate((branch_xyz, branch_xyz[:1]), axis=0)
                branch_truth = np.concatenate((branch_truth, branch_truth[:1]))
            distance = np.linalg.norm(np.diff(branch_xyz, axis=0), axis=1)
            native_arc = np.concatenate(
                (np.asarray([0.0], dtype=np.float64), np.cumsum(distance))
            )
            sample_arc = (
                np.array(native_arc, copy=True)
                if retain_native_points
                else np.linspace(
                    0.0,
                    float(native_arc[-1]),
                    int(point_count),
                    dtype=np.float64,
                )
            )
            sample_xyz = np.column_stack(
                [
                    np.interp(sample_arc, native_arc, branch_xyz[:, axis])
                    for axis in range(3)
                ]
            )
            sample_truth = np.interp(sample_arc, native_arc, branch_truth)
            compact_xyz.append(sample_xyz)
            compact_arc.append(sample_arc)
            compact_truth.append(sample_truth)
            compact_rows.append(graph_key[0])
            compact_graphs.append(graph_key[1])
            compact_components.append(int(component_codes[source_index]))
            compact_plane_pieces.append(int(plane_piece_codes[source_index]))
            compact_sides.append(int(side_codes[source_index]))
            compact_topology_patches.append(
                int(topology_patch_codes[source_index])
            )
            source_branches.append(source_index)
            point_offsets.append(point_offsets[-1] + int(point_count))

    required_velocity = {
        "requested_xyz_in",
        "valid_mask",
        "station_names",
        "station_row_offsets",
        "line_length_weights_in",
    }
    missing_velocity = sorted(required_velocity - set(velocity_native))
    if missing_velocity:
        _fail(f"native velocity support is missing arrays {missing_velocity}")
    velocity_xyz = _require_array(
        velocity_native,
        "requested_xyz_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT, 3),
    )
    valid = _require_array(
        velocity_native,
        "valid_mask",
        dtype=np.bool_,
        shape=(VELOCITY_ROW_COUNT,),
    )
    station_names = _require_array(
        velocity_native,
        "station_names",
        shape=(len(VELOCITY_STATIONS),),
    )
    if station_names.dtype != np.dtype("<U3") or station_names.tolist() != list(
        VELOCITY_STATIONS
    ):
        _fail("velocity station names/order must be exact B.2/B.3/C.1/C.2/C.3")
    station_offsets = _require_array(
        velocity_native,
        "station_row_offsets",
        dtype=np.int64,
        shape=(len(VELOCITY_STATIONS) + 1,),
    )
    expected_station_offsets = np.arange(
        len(VELOCITY_STATIONS) + 1, dtype=np.int64
    ) * VELOCITY_ROWS_PER_STATION
    if not np.array_equal(station_offsets, expected_station_offsets):
        _fail("velocity station offsets must preserve five 801-row stations")
    velocity_weights = _require_array(
        velocity_native,
        "line_length_weights_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT,),
    )
    if not np.all(np.isfinite(velocity_xyz)):
        _fail("velocity plotting coordinates must be finite")
    if (
        not np.all(np.isfinite(velocity_weights))
        or np.any(velocity_weights < 0.0)
        or np.any(velocity_weights[~valid] != 0.0)
        or np.any(velocity_weights[valid] <= 0.0)
    ):
        _fail(
            "velocity weights must be positive on valid rows and zero on gaps"
        )
    for start, end in zip(
        station_offsets[:-1], station_offsets[1:], strict=True
    ):
        if not np.any(valid[int(start) : int(end)]):
            _fail("every velocity station must retain valid support")
    truth_speed = _velocity_truth_speed(np.asarray(truth_velocity_nd), valid)

    support = {
        "cp_xyz_in": np.concatenate(compact_xyz).astype(np.float64, copy=False),
        "cp_arc_length_in": np.concatenate(compact_arc).astype(
            np.float64, copy=False
        ),
        "cp_truth": np.concatenate(compact_truth).astype(np.float64, copy=False),
        "cp_branch_point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "cp_branch_row_code": np.asarray(compact_rows, dtype=np.uint8),
        "cp_branch_graph_component_code": np.asarray(
            compact_graphs, dtype=np.int64
        ),
        "cp_branch_component_code": np.asarray(
            compact_components, dtype=np.uint8
        ),
        "cp_branch_plane_piece_code": np.asarray(
            compact_plane_pieces, dtype=np.uint8
        ),
        "cp_branch_side_code": np.asarray(compact_sides, dtype=np.uint8),
        "cp_branch_topology_patch_code": np.asarray(
            compact_topology_patches, dtype=np.uint8
        ),
        "cp_source_branch_index": np.asarray(source_branches, dtype=np.int64),
        "velocity_requested_xyz_in": np.array(velocity_xyz, copy=True),
        "velocity_valid_mask": np.array(valid, copy=True),
        "velocity_station_names": np.array(station_names, copy=True),
        "velocity_station_row_offsets": np.array(station_offsets, copy=True),
        "velocity_line_length_weights_in": np.array(velocity_weights, copy=True),
        "velocity_truth_speed_over_uinf": np.array(truth_speed, copy=True),
    }
    validate_compact_support(support)
    return support


def validate_compact_support(support: Mapping[str, np.ndarray]) -> None:
    """Strictly validate one evaluator-owned compact support object."""

    _require_exact_keys(support, SUPPORT_ARRAYS, "compact support")
    xyz = _require_array(support, "cp_xyz_in", dtype=np.float64, shape=(None, 3))
    point_count = len(xyz)
    arc = _require_array(
        support, "cp_arc_length_in", dtype=np.float64, shape=(point_count,)
    )
    truth = _require_array(
        support, "cp_truth", dtype=np.float64, shape=(point_count,)
    )
    offsets = _require_array(
        support, "cp_branch_point_offsets", dtype=np.int64, shape=(None,)
    )
    branch_count = len(offsets) - 1
    if branch_count < 1:
        _fail("compact Cp support must contain at least one branch")
    _require_offsets(offsets, end=point_count, label="cp_branch_point_offsets")
    rows = _require_array(
        support, "cp_branch_row_code", dtype=np.uint8, shape=(branch_count,)
    )
    graphs = _require_array(
        support,
        "cp_branch_graph_component_code",
        dtype=np.int64,
        shape=(branch_count,),
    )
    for name in (
        "cp_branch_component_code",
        "cp_branch_plane_piece_code",
        "cp_branch_side_code",
        "cp_branch_topology_patch_code",
    ):
        _require_array(support, name, dtype=np.uint8, shape=(branch_count,))
    source = _require_array(
        support, "cp_source_branch_index", dtype=np.int64, shape=(branch_count,)
    )
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(arc)):
        _fail("compact Cp plotting coordinates and arc lengths must be finite")
    if not np.all(np.isfinite(truth)):
        _fail("compact Cp truth must be finite")
    if np.any(rows >= 10) or np.any(graphs < 0):
        _fail("compact Cp physical graph codes are invalid")
    if (
        np.any(source < 0)
        or len(set(source.tolist())) != branch_count
    ):
        _fail("compact Cp source branch indices must be unique and non-negative")

    graph_counts: dict[tuple[int, int], int] = defaultdict(int)
    previous_key: tuple[int, int] | None = None
    closed_keys: set[tuple[int, int]] = set()
    for index, (row, graph) in enumerate(
        zip(rows.tolist(), graphs.tolist(), strict=True)
    ):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        local_arc = arc[start:end]
        if len(local_arc) < 2 or local_arc[0] != 0.0 or np.any(np.diff(local_arc) <= 0.0):
            _fail(
                "each compact Cp branch must have two or more strictly ordered "
                "physical arc-length samples starting at zero"
            )
        key = (int(row), int(graph))
        if previous_key is not None and key != previous_key:
            closed_keys.add(previous_key)
        if key in closed_keys:
            _fail("compact Cp branches for one physical graph must be contiguous")
        if previous_key is not None and key < previous_key:
            _fail("compact Cp physical graphs must use deterministic sorted order")
        if index and key == previous_key and source[index] <= source[index - 1]:
            _fail("compact Cp source branches within a graph must be increasing")
        previous_key = key
        graph_counts[key] += end - start
    if any(count > CP_POINTS_PER_GRAPH for count in graph_counts.values()):
        _fail(
            "a compact Cp physical graph exceeds the "
            f"{CP_POINTS_PER_GRAPH}-point maximum"
        )

    velocity_xyz = _require_array(
        support,
        "velocity_requested_xyz_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT, 3),
    )
    valid = _require_array(
        support,
        "velocity_valid_mask",
        dtype=np.bool_,
        shape=(VELOCITY_ROW_COUNT,),
    )
    names = _require_array(
        support,
        "velocity_station_names",
        dtype=np.dtype("<U3"),
        shape=(len(VELOCITY_STATIONS),),
    )
    if names.tolist() != list(VELOCITY_STATIONS):
        _fail("compact velocity station names/order differ")
    station_offsets = _require_array(
        support,
        "velocity_station_row_offsets",
        dtype=np.int64,
        shape=(len(VELOCITY_STATIONS) + 1,),
    )
    expected_station_offsets = np.arange(
        len(VELOCITY_STATIONS) + 1, dtype=np.int64
    ) * VELOCITY_ROWS_PER_STATION
    if not np.array_equal(station_offsets, expected_station_offsets):
        _fail("compact velocity station offsets differ")
    weights = _require_array(
        support,
        "velocity_line_length_weights_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT,),
    )
    velocity_truth = _require_array(
        support,
        "velocity_truth_speed_over_uinf",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT,),
    )
    if not np.all(np.isfinite(velocity_xyz)):
        _fail("compact velocity plotting coordinates must be finite")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights[valid] <= 0.0)
        or np.any(weights[~valid] != 0.0)
    ):
        _fail("compact velocity weights must be positive on valid rows and zero on gaps")
    if (
        not np.all(np.isfinite(velocity_truth[valid]))
        or np.any(velocity_truth[valid] < 0.0)
        or not np.all(np.isnan(velocity_truth[~valid]))
    ):
        _fail("compact velocity truth must be finite on valid rows and NaN on gaps")
    for start, end in zip(
        station_offsets[:-1], station_offsets[1:], strict=True
    ):
        if not np.any(valid[int(start) : int(end)]):
            _fail("every compact velocity station must retain valid support")


def sample_native_cp_prediction(
    cp_native: Mapping[str, np.ndarray], support: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Interpolate one native Cp prediction onto its evaluator-owned support."""

    validate_compact_support(support)
    (
        xyz,
        vertex_offsets,
        vertex_ids,
        _segment_offsets,
        _segment_lengths,
        closed,
        _rows,
        _graphs,
        _components,
        _plane_pieces,
        _sides,
        _topology_patches,
    ) = _validate_cp_native_support(cp_native)
    if "prediction_cp" not in cp_native:
        _fail("native Cp prediction is missing prediction_cp")
    prediction = _require_array(
        cp_native, "prediction_cp", dtype=np.float64, shape=(len(xyz),)
    )
    if not np.all(np.isfinite(prediction)):
        _fail("native Cp prediction must be finite")

    support_offsets = support["cp_branch_point_offsets"]
    support_arc = support["cp_arc_length_in"]
    support_xyz = support["cp_xyz_in"]
    sampled: list[np.ndarray] = []
    for support_branch, source_index_value in enumerate(
        support["cp_source_branch_index"].tolist()
    ):
        source_index = int(source_index_value)
        if source_index >= len(vertex_offsets) - 1:
            _fail("compact support references an absent native Cp branch")
        native_start = int(vertex_offsets[source_index])
        native_end = int(vertex_offsets[source_index + 1])
        ids = vertex_ids[native_start:native_end]
        branch_xyz = xyz[ids]
        branch_prediction = prediction[ids]
        if closed[source_index]:
            branch_xyz = np.concatenate((branch_xyz, branch_xyz[:1]), axis=0)
            branch_prediction = np.concatenate(
                (branch_prediction, branch_prediction[:1])
            )
        distance = np.linalg.norm(np.diff(branch_xyz, axis=0), axis=1)
        native_arc = np.concatenate(
            (np.asarray([0.0], dtype=np.float64), np.cumsum(distance))
        )
        start = int(support_offsets[support_branch])
        end = int(support_offsets[support_branch + 1])
        target_arc = support_arc[start:end]
        if not np.isclose(
            target_arc[-1], native_arc[-1], rtol=1.0e-12, atol=1.0e-12
        ):
            _fail("compact support and native Cp branch lengths differ")
        expected_xyz = np.column_stack(
            [
                np.interp(target_arc, native_arc, branch_xyz[:, axis])
                for axis in range(3)
            ]
        )
        if not np.allclose(
            expected_xyz, support_xyz[start:end], rtol=1.0e-13, atol=1.0e-13
        ):
            _fail("compact support and native Cp plotting coordinates differ")
        sampled.append(np.interp(target_arc, native_arc, branch_prediction))
    return np.concatenate(sampled).astype(np.float64, copy=False)


def native_velocity_speed_prediction(
    velocity_native: Mapping[str, np.ndarray],
    support: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Extract and validate all 4005 native predicted speed rows and gaps."""

    validate_compact_support(support)
    native_xyz = _require_array(
        velocity_native,
        "requested_xyz_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT, 3),
    )
    native_valid = _require_array(
        velocity_native,
        "valid_mask",
        dtype=np.bool_,
        shape=(VELOCITY_ROW_COUNT,),
    )
    native_station_names = _require_array(
        velocity_native,
        "station_names",
        shape=(len(VELOCITY_STATIONS),),
    )
    native_station_offsets = _require_array(
        velocity_native,
        "station_row_offsets",
        dtype=np.int64,
        shape=(len(VELOCITY_STATIONS) + 1,),
    )
    native_weights = _require_array(
        velocity_native,
        "line_length_weights_in",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT,),
    )
    alignments = (
        (
            native_xyz,
            support["velocity_requested_xyz_in"],
            "plotting coordinates",
        ),
        (native_valid, support["velocity_valid_mask"], "validity mask"),
        (
            native_station_names,
            support["velocity_station_names"],
            "station names/order",
        ),
        (
            native_station_offsets,
            support["velocity_station_row_offsets"],
            "station row offsets",
        ),
        (
            native_weights,
            support["velocity_line_length_weights_in"],
            "line-length weights",
        ),
    )
    for observed, expected, label in alignments:
        if not np.array_equal(observed, expected):
            _fail(f"compact support and native velocity {label} differ")
    valid = native_valid
    if "predicted_velocity_nd" not in velocity_native:
        _fail("native velocity prediction is missing predicted_velocity_nd")
    vectors = _require_array(
        velocity_native,
        "predicted_velocity_nd",
        dtype=np.float64,
        shape=(VELOCITY_ROW_COUNT, 3),
    )
    speed = np.linalg.norm(vectors, axis=1)
    if "speed_over_U_inf_pred" in velocity_native:
        alias = _require_array(
            velocity_native,
            "speed_over_U_inf_pred",
            dtype=np.float64,
            shape=(VELOCITY_ROW_COUNT,),
        )
        if not np.array_equal(alias[valid], speed[valid], equal_nan=True):
            _fail("native velocity speed alias differs from vector magnitude")
        if not np.all(np.isnan(alias[~valid])):
            _fail("native velocity speed alias must retain NaN gaps")
    if not np.all(np.isfinite(speed[valid])) or np.any(speed[valid] < 0.0):
        _fail("native predicted speed must be finite and non-negative on valid rows")
    if not np.all(np.isnan(speed[~valid])):
        _fail("native predicted velocity must retain NaN on every invalid row")
    return speed.astype(np.float64, copy=False)


def _support_npz_bytes(support: Mapping[str, np.ndarray]) -> bytes:
    validate_compact_support(support)
    return deterministic_npz_bytes(support, order=SUPPORT_ARRAYS)


def support_sha256(support: Mapping[str, np.ndarray]) -> str:
    """Return the digest of the complete private support NPZ, including truth."""

    return hashlib.sha256(_support_npz_bytes(support)).hexdigest()


def _identity_sha256(
    support: Mapping[str, np.ndarray], order: Sequence[str]
) -> str:
    """Hash the exact deterministic NPZ serialization named by the contract."""

    payload = deterministic_npz_bytes(
        {name: support[name] for name in order}, order=order
    )
    return hashlib.sha256(payload).hexdigest()


def surface_support_identity_sha256(
    support: Mapping[str, np.ndarray],
) -> str:
    """Bind evaluator-owned surface support without exposing or hashing truth."""

    validate_compact_support(support)
    order = tuple(name for name in CP_SUPPORT_ARRAYS if name != "cp_truth")
    return _identity_sha256(support, order)


def surface_prediction_order_sha256(
    support: Mapping[str, np.ndarray],
) -> str:
    """Bind the exact serialized Cp point order and branch reset boundaries."""

    validate_compact_support(support)
    return _identity_sha256(support, CP_PREDICTION_ORDER_ARRAYS)


def velocity_support_identity_sha256(
    support: Mapping[str, np.ndarray],
) -> str:
    """Bind evaluator-owned velocity placement/masks/weights without truth."""

    validate_compact_support(support)
    order = tuple(
        name
        for name in VELOCITY_SUPPORT_ARRAYS
        if name != "velocity_truth_speed_over_uinf"
    )
    return _identity_sha256(support, order)


def velocity_prediction_order_sha256(
    support: Mapping[str, np.ndarray],
) -> str:
    """Bind the station-major sequence of all and only valid velocity rows."""

    validate_compact_support(support)
    arrays = {
        "velocity_station_names": support["velocity_station_names"],
        "velocity_station_row_offsets": support[
            "velocity_station_row_offsets"
        ],
        "velocity_valid_row_indices": np.flatnonzero(
            support["velocity_valid_mask"]
        ).astype(np.int64, copy=False),
    }
    order = tuple(arrays)
    payload = deterministic_npz_bytes(arrays, order=order)
    return hashlib.sha256(payload).hexdigest()


def compact_case_metadata(
    support: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Return the exact support bindings and counts required by chunk JSON."""

    validate_compact_support(support)
    graph_count = len(
        set(
            zip(
                support["cp_branch_row_code"].tolist(),
                support["cp_branch_graph_component_code"].tolist(),
                strict=True,
            )
        )
    )
    valid_count = int(np.count_nonzero(support["velocity_valid_mask"]))
    return {
        "surface_cp": {
            "support_identity_sha256": surface_support_identity_sha256(support),
            "prediction_order_sha256": surface_prediction_order_sha256(support),
            "physical_graph_count": graph_count,
            "retained_branch_count": len(support["cp_branch_point_offsets"]) - 1,
            "retained_point_count": len(support["cp_truth"]),
            "maximum_points_per_physical_graph": CP_POINTS_PER_GRAPH,
            "quantization_scale": CP_FIXED_POINT_SCALE,
            "quantization_dtype": "int16",
            "delta_dtype": "int16",
            "prediction_array": "cp_q_delta",
        },
        "volume_velocity": {
            "support_identity_sha256": velocity_support_identity_sha256(support),
            "prediction_order_sha256": velocity_prediction_order_sha256(support),
            "station_order": list(VELOCITY_STATIONS),
            "station_count": len(VELOCITY_STATIONS),
            "row_count": VELOCITY_ROW_COUNT,
            "valid_row_count": valid_count,
            "invalid_row_count": VELOCITY_ROW_COUNT - valid_count,
            "prediction_dtype": "float32",
            "prediction_array": "velocity_speed_over_u_inf",
            "storage_dtype": "uint8",
            "storage_encoding": VELOCITY_STORAGE_ENCODING,
            "stored_byte_count": valid_count * np.dtype(np.float32).itemsize,
        },
    }


def validate_compact_case_metadata(
    metadata: Mapping[str, Any], *, support: Mapping[str, np.ndarray]
) -> None:
    """Require chunk-side support/order bindings to match live support exactly."""

    expected = compact_case_metadata(support)
    if metadata != expected:
        _fail("compact case metadata differs from evaluator-owned support")


def encode_compact_predictions(
    *,
    support: Mapping[str, np.ndarray],
    cp_prediction: np.ndarray,
    velocity_speed_over_u_inf: np.ndarray,
) -> dict[str, np.ndarray]:
    """Encode aligned prediction values without copying evaluator-owned support."""

    validate_compact_support(support)
    cp = np.asarray(cp_prediction)
    cp_point_count = len(support["cp_truth"])
    if cp.shape != (cp_point_count,) or cp.dtype != np.dtype(np.float64):
        _fail(f"cp_prediction must be float64 with shape ({cp_point_count},)")
    if not np.all(np.isfinite(cp)):
        _fail("cp_prediction must be finite")
    scaled = np.rint(cp * float(CP_FIXED_POINT_SCALE))
    if (
        not np.all(np.isfinite(scaled))
        or np.any(scaled < np.iinfo(np.int16).min)
        or np.any(scaled > np.iinfo(np.int16).max)
    ):
        _fail("quantized Cp prediction overflows int16")
    quantized = scaled.astype(np.int64)
    deltas = np.empty(cp_point_count, dtype=np.int16)
    offsets = support["cp_branch_point_offsets"]
    for start_value, end_value in zip(offsets[:-1], offsets[1:], strict=True):
        start = int(start_value)
        end = int(end_value)
        wide_delta = np.concatenate(
            (quantized[start : start + 1], np.diff(quantized[start:end]))
        )
        if (
            np.any(wide_delta < np.iinfo(np.int16).min)
            or np.any(wide_delta > np.iinfo(np.int16).max)
        ):
            _fail("delta-coded Cp prediction overflows int16 within a branch")
        deltas[start:end] = wide_delta.astype(np.int16)

    speed = np.asarray(velocity_speed_over_u_inf)
    if speed.dtype != np.dtype(np.float64) or speed.shape != (VELOCITY_ROW_COUNT,):
        _fail(
            "velocity_speed_over_u_inf must be float64 with shape "
            f"({VELOCITY_ROW_COUNT},)"
        )
    valid = support["velocity_valid_mask"]
    if not np.all(np.isfinite(speed[valid])) or np.any(speed[valid] < 0.0):
        _fail("predicted speed must be finite and non-negative on valid rows")
    if not np.all(np.isnan(speed[~valid])):
        _fail("predicted speed must retain NaN on every invalid row")
    if np.any(speed[valid] > np.finfo(np.float32).max):
        _fail("predicted speed overflows float32")
    compact_speed = speed[valid].astype(np.float32)
    if not np.all(np.isfinite(compact_speed)):
        _fail("float32 predicted speed is non-finite")

    artifact = {
        "cp_q_delta": deltas,
        "velocity_speed_over_u_inf": compact_speed,
    }
    validate_compact_predictions(
        artifact, support=support, metadata=compact_case_metadata(support)
    )
    return artifact


def encode_native_predictions(
    *,
    support: Mapping[str, np.ndarray],
    cp_native: Mapping[str, np.ndarray],
    velocity_native: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert validated native-v1 prediction arrays to compact-v2 values."""

    return encode_compact_predictions(
        support=support,
        cp_prediction=sample_native_cp_prediction(cp_native, support),
        velocity_speed_over_u_inf=native_velocity_speed_prediction(
            velocity_native, support
        ),
    )


def _decode_cp_delta(
    deltas: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    quantized = np.empty(len(deltas), dtype=np.int64)
    for start_value, end_value in zip(offsets[:-1], offsets[1:], strict=True):
        start = int(start_value)
        end = int(end_value)
        quantized[start:end] = np.cumsum(
            deltas[start:end].astype(np.int64), dtype=np.int64
        )
    if (
        np.any(quantized < np.iinfo(np.int16).min)
        or np.any(quantized > np.iinfo(np.int16).max)
    ):
        _fail("delta-coded Cp reconstruction overflows int16")
    return quantized


def encode_velocity_storage(speed: np.ndarray) -> np.ndarray:
    """Losslessly encode a one-dimensional float32 vector as shuffled bytes.

    Adjacent IEEE-754 bit patterns are differenced as unsigned 32-bit words
    modulo ``2**32``. The four little-endian byte lanes are then stored
    contiguously, which makes smooth profile values substantially more
    compressible with browser-native ZIP Deflate.
    """

    values = np.asarray(speed)
    if values.dtype != np.dtype(np.float32) or values.ndim != 1 or not len(values):
        _fail("velocity storage input must be a non-empty float32 vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        _fail("velocity storage input must be finite and non-negative")
    little = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    words = little.view(np.dtype("<u4"))
    deltas = np.empty_like(words)
    deltas[0] = words[0]
    deltas[1:] = np.subtract(words[1:], words[:-1], dtype=np.uint32)
    byte_rows = deltas.view(np.uint8).reshape(len(deltas), 4)
    return np.ascontiguousarray(byte_rows.T).reshape(-1)


def decode_velocity_storage(storage: np.ndarray) -> np.ndarray:
    """Invert :func:`encode_velocity_storage` exactly."""

    encoded = np.asarray(storage)
    if (
        encoded.dtype != np.dtype(np.uint8)
        or encoded.ndim != 1
        or not len(encoded)
        or len(encoded) % np.dtype(np.float32).itemsize
    ):
        _fail("velocity storage must be a non-empty uint8 vector divisible by four")
    count = len(encoded) // np.dtype(np.float32).itemsize
    byte_rows = np.ascontiguousarray(encoded.reshape(4, count).T)
    deltas = byte_rows.reshape(-1).view(np.dtype("<u4"))
    words = np.bitwise_and(
        np.cumsum(deltas.astype(np.uint64), dtype=np.uint64),
        np.uint64(np.iinfo(np.uint32).max),
    ).astype(np.dtype("<u4"))
    values = np.array(words.view(np.dtype("<f4")), copy=True)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        _fail("decoded compact predicted speed must be finite and non-negative")
    return values


def validate_compact_predictions(
    artifact: Mapping[str, np.ndarray],
    *,
    support: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    """Strictly validate a participant compact-v2 prediction artifact."""

    validate_compact_support(support)
    validate_compact_case_metadata(metadata, support=support)
    _require_exact_keys(artifact, PREDICTION_ARRAYS, "compact prediction")
    deltas = _require_array(
        artifact,
        "cp_q_delta",
        dtype=np.int16,
        shape=(len(support["cp_truth"]),),
    )
    _decode_cp_delta(deltas, support["cp_branch_point_offsets"])
    speed = _require_array(
        artifact,
        "velocity_speed_over_u_inf",
        dtype=np.float32,
        shape=(int(np.count_nonzero(support["velocity_valid_mask"])),),
    )
    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
        _fail("compact predicted speed must be finite and non-negative")


def decode_compact_predictions(
    artifact: Mapping[str, np.ndarray],
    *,
    support: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode Cp and speed to float64 arrays aligned with evaluator support."""

    validate_compact_predictions(artifact, support=support, metadata=metadata)
    quantized = _decode_cp_delta(
        artifact["cp_q_delta"], support["cp_branch_point_offsets"]
    )
    cp = quantized.astype(np.float64) / float(CP_FIXED_POINT_SCALE)
    valid = support["velocity_valid_mask"]
    speed = np.full(VELOCITY_ROW_COUNT, np.nan, dtype=np.float64)
    speed[valid] = artifact["velocity_speed_over_u_inf"].astype(np.float64)
    return cp, speed


def _roundoff_safe_centered(
    squared_truth: float,
    truth_integral: float,
    support_weight: float,
    *,
    unit_floor: bool,
) -> float:
    correction = truth_integral * truth_integral / support_weight
    centered = squared_truth - correction
    floor = 1.0 if unit_floor else np.finfo(np.float64).tiny
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        abs(squared_truth), abs(correction), floor
    )
    if centered < -tolerance:
        _fail("profile truth centered sum of squares is materially negative")
    return max(centered, 0.0)


def _score_cp(
    prediction: np.ndarray, support: Mapping[str, np.ndarray]
) -> float:
    truth = support["cp_truth"]
    arc = support["cp_arc_length_in"]
    offsets = support["cp_branch_point_offsets"]
    rows = support["cp_branch_row_code"]
    graphs = support["cp_branch_graph_component_code"]
    states: dict[tuple[int, int], list[float]] = {}
    for branch_index, (row, graph) in enumerate(
        zip(rows.tolist(), graphs.tolist(), strict=True)
    ):
        start = int(offsets[branch_index])
        end = int(offsets[branch_index + 1])
        lengths = np.diff(arc[start:end])
        pred = prediction[start:end]
        target = truth[start:end]
        error = pred - target
        key = (int(row), int(graph))
        state = states.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        state[0] += float(np.sum(lengths, dtype=np.float64))
        state[1] += float(
            np.sum(
                lengths
                * (
                    error[:-1] * error[:-1]
                    + error[:-1] * error[1:]
                    + error[1:] * error[1:]
                )
                / 3.0,
                dtype=np.float64,
            )
        )
        state[2] += float(
            np.sum(
                lengths
                * (
                    target[:-1] * target[:-1]
                    + target[:-1] * target[1:]
                    + target[1:] * target[1:]
                )
                / 3.0,
                dtype=np.float64,
            )
        )
        state[3] += float(
            np.sum(
                lengths * (target[:-1] + target[1:]) / 2.0,
                dtype=np.float64,
            )
        )
    sse = 0.0
    sst = 0.0
    for key in sorted(states):
        length, graph_sse, squared_truth, truth_integral = states[key]
        if length <= 0.0:
            _fail("compact Cp graph has no positive physical support")
        sse += graph_sse
        sst += _roundoff_safe_centered(
            squared_truth, truth_integral, length, unit_floor=False
        )
    if not math.isfinite(sse) or not math.isfinite(sst) or sst <= 0.0:
        _fail("compact Cp profile R2 target variance is zero or invalid")
    return 1.0 - sse / sst


def _score_velocity(
    prediction: np.ndarray, support: Mapping[str, np.ndarray]
) -> float:
    truth = support["velocity_truth_speed_over_uinf"]
    valid = support["velocity_valid_mask"]
    weights = support["velocity_line_length_weights_in"]
    offsets = support["velocity_station_row_offsets"]
    sse = 0.0
    sst = 0.0
    for start_value, end_value in zip(offsets[:-1], offsets[1:], strict=True):
        start = int(start_value)
        end = int(end_value)
        active = valid[start:end]
        local_weights = weights[start:end][active]
        pred = prediction[start:end][active]
        target = truth[start:end][active]
        error = pred - target
        weight_sum = float(np.sum(local_weights, dtype=np.float64))
        sse += float(np.sum(local_weights * error * error, dtype=np.float64))
        squared_truth = float(
            np.sum(local_weights * target * target, dtype=np.float64)
        )
        truth_integral = float(
            np.sum(local_weights * target, dtype=np.float64)
        )
        sst += _roundoff_safe_centered(
            squared_truth, truth_integral, weight_sum, unit_floor=True
        )
    if not math.isfinite(sse) or not math.isfinite(sst) or sst <= 0.0:
        _fail("compact velocity profile R2 target variance is zero or invalid")
    return 1.0 - sse / sst


def score_compact_profiles(
    artifact: Mapping[str, np.ndarray],
    *,
    support: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, float]:
    """Score only the compact Cp-cut and five-line velocity diagnostics."""

    cp, speed = decode_compact_predictions(
        artifact, support=support, metadata=metadata
    )
    return {
        "cp_cut_r2": _score_cp(cp, support),
        "velocity_profile_r2": _score_velocity(speed, support),
    }


def deterministic_npz_bytes(
    arrays: Mapping[str, np.ndarray], *, order: Sequence[str]
) -> bytes:
    """Return canonical, timestamp-fixed compressed NPZ bytes."""

    _require_exact_keys(arrays, order, "deterministic NPZ")
    output = io.BytesIO()
    try:
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in order:
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        _fail(f"cannot construct deterministic NPZ: {error}")
    return output.getvalue()


def deterministic_prediction_npz_bytes(
    arrays: Mapping[str, np.ndarray], *, order: Sequence[str]
) -> bytes:
    """Return canonical browser-decodable prediction NPZ bytes.

    The logical velocity values remain exact float32. Only their on-disk
    representation is transformed to lossless shuffled delta bytes before
    deterministic level-9 ZIP Deflate.
    """

    _require_exact_keys(arrays, order, "deterministic prediction NPZ")
    stored = dict(arrays)
    stored["velocity_speed_over_u_inf"] = encode_velocity_storage(
        np.asarray(arrays["velocity_speed_over_u_inf"])
    )
    return deterministic_npz_bytes(stored, order=order)


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CompactProfileError(f"refusing to overwrite {label}: {path}") from error
    except OSError as error:
        raise CompactProfileError(f"cannot write {label} {path}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def write_compact_support_npz(
    path: Path, support: Mapping[str, np.ndarray]
) -> str:
    """Write canonical evaluator-owned support and return its SHA-256."""

    payload = _support_npz_bytes(support)
    return _write_exclusive(path, payload, label="compact support")


def write_compact_prediction_npz(
    path: Path,
    artifact: Mapping[str, np.ndarray],
    *,
    support: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    """Write a canonical prediction-only participant artifact."""

    validate_compact_predictions(artifact, support=support, metadata=metadata)
    payload = deterministic_prediction_npz_bytes(
        artifact, order=PREDICTION_ARRAYS
    )
    return _write_exclusive(path, payload, label="compact prediction artifact")


def _load_npz_exact(
    path: Path, *, order: Sequence[str], label: str
) -> tuple[dict[str, np.ndarray], str]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    before = path.stat()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != list(order):
                _fail(f"{label} array inventory or order differs: {archive.files}")
            arrays = {name: np.array(archive[name], copy=True) for name in order}
    except CompactProfileError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise CompactProfileError(f"cannot read {label} {path}: {error}") from error
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        _fail(f"{label} changed while it was read: {path}")
    return arrays, digest


def load_compact_support_npz(path: Path) -> tuple[dict[str, np.ndarray], str]:
    """Load and strictly validate canonical evaluator-owned support."""

    support, digest = _load_npz_exact(path, order=SUPPORT_ARRAYS, label="compact support")
    validate_compact_support(support)
    if digest != support_sha256(support):
        _fail("compact support bytes are not in canonical deterministic form")
    return support, digest


def load_compact_prediction_npz(
    path: Path,
    *,
    support: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    """Load and strictly validate a prediction-only participant artifact."""

    stored, digest = _load_npz_exact(
        path, order=PREDICTION_ARRAYS, label="compact prediction artifact"
    )
    artifact = dict(stored)
    artifact["velocity_speed_over_u_inf"] = decode_velocity_storage(
        stored["velocity_speed_over_u_inf"]
    )
    validate_compact_predictions(artifact, support=support, metadata=metadata)
    canonical = deterministic_prediction_npz_bytes(
        artifact, order=PREDICTION_ARRAYS
    )
    if hashlib.sha256(canonical).hexdigest() != digest:
        _fail("compact prediction bytes are not in canonical deterministic form")
    return artifact, digest


__all__ = [
    "COMPACT_PROFILE_FORMAT",
    "CP_FIXED_POINT_SCALE",
    "CP_POINTS_PER_GRAPH",
    "CP_TINY_BRANCH_RELATIVE_LENGTH",
    "PREDICTION_ARRAYS",
    "SUPPORT_ARRAYS",
    "VELOCITY_ROW_COUNT",
    "VELOCITY_ROWS_PER_STATION",
    "VELOCITY_STORAGE_ENCODING",
    "VELOCITY_STATIONS",
    "CompactProfileError",
    "allocate_branch_samples",
    "build_compact_support",
    "compact_case_metadata",
    "decode_compact_predictions",
    "decode_velocity_storage",
    "deterministic_npz_bytes",
    "deterministic_prediction_npz_bytes",
    "encode_compact_predictions",
    "encode_native_predictions",
    "encode_velocity_storage",
    "load_compact_prediction_npz",
    "load_compact_support_npz",
    "native_velocity_speed_prediction",
    "sample_native_cp_prediction",
    "score_compact_profiles",
    "surface_prediction_order_sha256",
    "surface_support_identity_sha256",
    "support_sha256",
    "validate_compact_case_metadata",
    "validate_compact_predictions",
    "validate_compact_support",
    "velocity_prediction_order_sha256",
    "velocity_support_identity_sha256",
    "write_compact_prediction_npz",
    "write_compact_support_npz",
]
