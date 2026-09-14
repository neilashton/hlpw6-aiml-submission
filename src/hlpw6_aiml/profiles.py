"""HLPW6 scope-aware envelope around the retained compact HiLift profile mathematics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .archive import npz_arrays
from .contracts import ContractError, scope_domains
from .reference import compact_profiles as reference


def encode_cp(cp: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(cp, dtype=np.float64)
    scaled = np.rint(values * 1024.0)
    if not np.all(np.isfinite(scaled)) or np.any(scaled < -32768) or np.any(scaled > 32767):
        raise ContractError("Cp prediction overflows the fixed compact representation")
    result = np.empty(len(values), dtype=np.int16)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        q = scaled[int(start) : int(end)].astype(np.int64)
        delta = np.concatenate((q[:1], np.diff(q)))
        if np.any(delta < -32768) or np.any(delta > 32767):
            raise ContractError("Cp delta overflows the fixed compact representation")
        result[int(start) : int(end)] = delta.astype(np.int16)
    return result


def decode_cp(delta: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if delta.dtype != np.dtype("int16") or delta.ndim != 1:
        raise ContractError("compact Cp must be an int16 vector")
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(delta)
        or np.any(np.diff(offsets) < 2)
    ):
        raise ContractError("Cp branch offsets do not match the prediction")
    result = np.empty(len(delta), dtype=np.float64)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        q = np.cumsum(delta[int(start) : int(end)], dtype=np.int64)
        if np.any(q < -32768) or np.any(q > 32767):
            raise ContractError("decoded Cp overflows the fixed compact representation")
        result[int(start) : int(end)] = q / 1024.0
    return result


def encode_profiles(
    cp: np.ndarray, velocity: np.ndarray | None, support: dict[str, np.ndarray], scope: str
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    scope_domains(scope)
    reference.validate_compact_support(support)
    metadata = reference.compact_case_metadata(support)
    arrays = {"cp_q_delta": encode_cp(cp, support["cp_branch_point_offsets"])}
    decoded_cp = decode_cp(arrays["cp_q_delta"], support["cp_branch_point_offsets"])
    values = {"cp_cut_r2": reference._score_cp(decoded_cp, support), "velocity_profile_r2": None}
    if scope == "surface_and_volume":
        if velocity is None:
            raise ContractError("surface-and-volume profiles require predicted velocity")
        artifact = reference.encode_compact_predictions(
            support=support,
            cp_prediction=np.asarray(cp, dtype=np.float64),
            velocity_speed_over_u_inf=np.asarray(velocity, dtype=np.float64),
        )
        arrays["velocity_speed_over_u_inf"] = reference.encode_velocity_storage(
            artifact["velocity_speed_over_u_inf"]
        )
        speed = np.full(4005, np.nan, dtype=np.float64)
        speed[support["velocity_valid_mask"]] = artifact["velocity_speed_over_u_inf"].astype(
            np.float64
        )
        values["velocity_profile_r2"] = reference._score_velocity(speed, support)
    else:
        if velocity is not None:
            raise ContractError("surface-only profiles must omit volume predictions")
        metadata["volume_velocity"] = {"submitted": False}
    payload = reference.deterministic_npz_bytes(arrays, order=tuple(arrays))
    return payload, metadata, values


def validate_profile_payload(
    payload: bytes, metadata: dict[str, Any], scope: str
) -> dict[str, np.ndarray]:
    scope_domains(scope)
    expected = {"cp_q_delta"}
    if scope == "surface_and_volume":
        expected.add("velocity_speed_over_u_inf")
    arrays = npz_arrays(payload, expected)
    cp = arrays["cp_q_delta"]
    surface = metadata["surface_cp"]
    if (
        cp.dtype != np.dtype("int16")
        or cp.shape != (surface["retained_point_count"],)
        or surface["quantization_scale"] != 1024
        or surface["maximum_points_per_physical_graph"] != 128
    ):
        raise ContractError("compact Cp representation differs from the HLPW6 contract")
    for key in ("support_identity_sha256", "prediction_order_sha256"):
        if not isinstance(surface.get(key), str) or len(surface[key]) != 64:
            raise ContractError("compact Cp support identity is missing")
    volume = metadata["volume_velocity"]
    if scope == "surface_only":
        if volume != {"submitted": False}:
            raise ContractError("surface-only velocity profiles must be marked not submitted")
    else:
        if volume.get("station_order") != ["B.2", "B.3", "C.1", "C.2", "C.3"]:
            raise ContractError("velocity station order differs")
        speed = reference.decode_velocity_storage(arrays["velocity_speed_over_u_inf"])
        if (
            speed.shape != (volume["valid_row_count"],)
            or not np.all(np.isfinite(speed))
            or np.any(speed < 0)
        ):
            raise ContractError("invalid compact velocity predictions")
        if volume["valid_row_count"] + volume["invalid_row_count"] != 4005:
            raise ContractError("velocity profile coverage differs")
    return arrays
