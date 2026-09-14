"""Bounded-memory evaluation of native-point predictions on organiser-owned support."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .archive import safe_relative_path
from .contracts import ContractError, case_identities, require_native_release, scope_domains
from .entry import load_entry
from .jsonio import canonical_json_bytes, read_json, sha256_bytes, sha256_file, write_json
from .metrics import FieldAccumulator
from .packaging import result_document
from .profiles import encode_profiles
from .reference.compact_profiles import load_compact_support_npz
from .reference.exact_moment_audit import (
    CompensatedIntegrals,
    coefficient_document,
    integrate_triangle_batch,
    ordered_fan_triangle_ids,
)
from .regions import aggregate_regions

CHUNK_SIZE = 250_000


def array_file(root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    path = root / safe_relative_path(descriptor["path"])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ContractError("array must be a regular file inside its input root")
    if sha256_file(path) != descriptor["sha256"]:
        raise ContractError(f"array checksum differs: {path.name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype.hasobject or array.dtype.kind not in "fiub" or array.ndim not in (1, 2):
        raise ContractError("native arrays must be numeric vectors or matrices")
    if list(array.shape) != descriptor["shape"] or str(array.dtype) != descriptor["dtype"]:
        raise ContractError("native array shape or dtype differs from its manifest")
    return array


def open_case_support(
    root: Path, case_id: str, *, demonstration: bool = False
) -> tuple[dict[str, Any], Path]:
    release = None if demonstration else require_native_release()
    case_root = root / "cases" / case_id
    document = read_json(case_root / "case.json")
    if (
        document.get("schema") != "hlpw6-native-case-support-v1"
        or document.get("case_id") != case_id
    ):
        raise ContractError("native case support identity differs")
    index = read_json(root / "index.json")
    if index.get("schema") != "hlpw6-native-support-index-v1":
        raise ContractError("native support release has an incompatible index")
    record = index["cases"][case_id]
    if record["sha256"] != sha256_file(case_root / "case.json"):
        raise ContractError("case support differs from its release index")
    if index.get("demonstration", False) != demonstration:
        raise ContractError("demonstration and workshop support cannot be mixed")
    if not demonstration:
        if release.get("index_sha256") != sha256_file(root / "index.json"):
            raise ContractError("native support is not the release pinned by this evaluator")
    return document, case_root


def _domain(
    support: dict[str, Any],
    support_root: Path,
    predictions_root: Path,
    domain: str,
    scratch: Path,
    *,
    expected_count: int | None,
) -> tuple[dict, dict, dict, dict]:
    fields = ("pressure", "wall_shear") if domain == "surface" else ("pressure", "velocity")
    metadata = read_json(predictions_root / "manifest.json")
    if metadata.get("schema") != "hlpw6-native-predictions-v1" or metadata.get("domain") != domain:
        raise ContractError("native prediction manifest has a different schema or domain")
    if metadata.get("basis") != "hiliftaeroml_nondimensional":
        raise ContractError("predictions must use the declared HiLiftAeroML nondimensional basis")
    source = support[domain]
    ids = array_file(support_root, source["raw_point_ids"])
    if ids.dtype != np.dtype("int64") or ids.ndim != 1:
        raise ContractError("native raw point IDs must be an int64 vector")
    count = len(ids)
    if expected_count is not None and count != expected_count:
        raise ContractError("native support point count differs from the pinned case")
    for start in range(0, count, CHUNK_SIZE):
        block = ids[max(0, start - 1) : min(start + CHUNK_SIZE, count)]
        if np.any(block < 0) or np.any(np.diff(block) <= 0):
            raise ContractError("native support IDs must be non-negative and strictly increasing")
        if domain == "surface" and not np.array_equal(
            ids[start : start + CHUNK_SIZE],
            np.arange(start, min(start + CHUNK_SIZE, count), dtype=np.int64),
        ):
            raise ContractError(
                "surface support must preserve all native IDs in original point order"
            )
    truths = {field: array_file(support_root, source[field]) for field in fields}
    weights = array_file(support_root, source["weights"]) if domain == "surface" else None
    if weights is not None and weights.shape != (count,):
        raise ContractError("surface quadrature must have one weight per native point")
    if weights is not None and weights.dtype != np.dtype("float32"):
        raise ContractError("surface quadrature must preserve the published float32 sidecar")
    regions = array_file(support_root, source["region_ids"]) if "region_ids" in source else None
    mapped, accumulators, equal, region_accumulators = {}, {}, {}, {}
    for field in fields:
        components = 1 if field == "pressure" else 3
        shape = (count,) if components == 1 else (count, 3)
        if truths[field].shape != shape:
            raise ContractError("native truth field has an incorrect shape")
        mapped[field] = np.lib.format.open_memmap(
            scratch / f"{domain}-{field}.npy", mode="w+", dtype=np.float64, shape=shape
        )
        accumulators[field], equal[field] = (
            FieldAccumulator(components),
            FieldAccumulator(components),
        )
        region_accumulators[field] = [FieldAccumulator(components) for _ in range(4)]
    seen = np.memmap(scratch / f"{domain}-seen.bin", dtype=np.uint8, mode="w+", shape=(count,))
    seen[:] = 0
    if regions is not None and (regions.shape != (count,) or regions.dtype != np.dtype("uint8")):
        raise ContractError("regional native support must be a uint8 vector")
    chunks = metadata.get("chunks", [])
    if not chunks:
        raise ContractError("native prediction manifest has no chunks")
    for chunk in chunks:
        if set(chunk) != {"raw_point_ids", *fields}:
            raise ContractError(
                "prediction chunks must contain exactly raw IDs and the required fields"
            )
        raw = array_file(predictions_root, chunk["raw_point_ids"])
        if raw.dtype != np.dtype("int64") or raw.ndim != 1 or len(raw) == 0:
            raise ContractError("prediction raw IDs must be a non-empty int64 vector")
        values = {field: array_file(predictions_root, chunk[field]) for field in fields}
        for field in fields:
            shape = (len(raw),) if field == "pressure" else (len(raw), 3)
            if values[field].shape != shape:
                raise ContractError("prediction chunk field shape differs from its raw IDs")
        for start in range(0, len(raw), CHUNK_SIZE):
            current = raw[start : start + CHUNK_SIZE]
            locations = np.searchsorted(ids, current)
            if (
                np.any(locations >= count)
                or not np.array_equal(ids[locations], current)
                or len(np.unique(locations)) != len(locations)
                or np.any(seen[locations])
            ):
                raise ContractError(
                    "predictions contain an unknown, excluded, or duplicate raw point ID"
                )
            seen[locations] = 1
            for field in fields:
                value = np.asarray(values[field][start : start + CHUNK_SIZE], dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ContractError("native prediction contains non-finite values")
                mapped[field][locations] = value
    predicted_count = sum(
        int(np.count_nonzero(seen[start : start + CHUNK_SIZE]))
        for start in range(0, count, CHUNK_SIZE)
    )
    if predicted_count != count:
        raise ContractError(
            f"native {domain} prediction is incomplete: {predicted_count}/{count} points"
        )
    # Reduce in canonical point order, independently of the participant's chunk order.
    for start in range(0, count, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, count)
        weight = (
            np.asarray(weights[start:end], dtype=np.float64)
            if weights is not None
            else np.ones(end - start)
        )
        for field in fields:
            pred, truth = mapped[field][start:end], truths[field][start:end]
            accumulators[field].add(pred, truth, weight)
            if domain == "surface":
                equal[field].add(pred, truth, np.ones(end - start))
            if regions is not None:
                zone = regions[start:end]
                if np.any(zone > 3):
                    raise ContractError("regional support includes an unknown region")
                for code in range(4):
                    selected = zone == code
                    region_accumulators[field][code].add(
                        pred[selected], truth[selected], weight[selected]
                    )
    statistics, metrics, regional = {}, {}, {}
    for field in fields:
        prefix = f"{domain}_{field}"
        scale = (
            support["reference"]["u_inf"] if field == "velocity" else support["reference"]["q_inf"]
        )
        metrics.update(accumulators[field].values(prefix, dimensional_scale=scale))
        statistics[prefix] = {
            "primary": accumulators[field].statistics(),
            "dimensional_scale": scale,
        }
        if domain == "surface":
            statistics[prefix]["equal_entity"] = equal[field].statistics()
            metrics[f"{prefix}_equal_entity_rel_l2"] = equal[field].values(
                prefix, dimensional_scale=scale
            )[f"{prefix}_rel_l2"]
        if regions is not None:
            regional[prefix] = [acc.statistics() for acc in region_accumulators[field]]
    coverage = {
        "expected_count": count,
        "predicted_count": count,
        "missing_count": 0,
        "duplicate_count": 0,
    }
    return (
        {"metrics": metrics, "statistics": statistics, "coverage": coverage, "regional": regional},
        mapped,
        truths,
        {"raw_point_ids": ids},
    )


def _stencil(field: np.ndarray, ids: np.ndarray, record: dict, root: Path) -> np.ndarray:
    raw = array_file(root, record["raw_point_ids"])
    weights = array_file(root, record["weights"])
    offsets = array_file(root, record["offsets"])
    if (
        raw.ndim != 1
        or weights.shape != raw.shape
        or offsets.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != len(raw)
        or np.any(np.diff(offsets) < 1)
    ):
        raise ContractError("profile interpolation stencil is malformed")
    locations = np.searchsorted(ids, raw)
    if np.any(locations >= len(ids)) or not np.array_equal(ids[locations], raw):
        raise ContractError("profile stencil references unknown native points")
    output = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        local_weights = weights[int(start) : int(end)]
        if not np.all(np.isfinite(local_weights)) or not math.isclose(
            float(np.sum(local_weights)), 1.0, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ContractError("profile interpolation weights must be finite and sum to one")
        values = field[locations[int(start) : int(end)]]
        output.append(
            np.sum(values * (local_weights[:, None] if values.ndim == 2 else local_weights), axis=0)
        )
    return np.asarray(output, dtype=np.float64)


def _loads(source: dict, root: Path, predictions: dict, truths: dict, reference: dict) -> dict:
    points = array_file(root, source["points"])
    connectivity, offsets = (
        array_file(root, source["connectivity"]),
        array_file(root, source["offsets"]),
    )
    results = {}
    for label, fields in (("prediction", predictions), ("truth", truths)):
        total = CompensatedIntegrals()
        for begin in range(0, len(offsets) - 1, 100_000):
            triangles = ordered_fan_triangle_ids(
                connectivity,
                offsets,
                begin_cell=begin,
                end_cell=min(begin + 100_000, len(offsets) - 1),
                point_count=len(points),
            )
            scale = reference["q_inf"] / reference["q_ref"]
            total.add(
                integrate_triangle_batch(
                    triangle_points=points[triangles],
                    pressure_coefficient=fields["pressure"][triangles] * scale,
                    wall_shear_coefficient=fields["wall_shear"][triangles] * scale,
                    moment_reference=reference["moment_reference"],
                )
            )
        document = coefficient_document(
            total.finalize(),
            reference_area=reference["area_ref"],
            reference_length=reference["chord_ref"],
            alpha_rad=math.radians(reference["aoa_degrees"]),
        )
        results[label] = {
            "Cd": document["force"]["total"]["cd"],
            "Cl": document["force"]["total"]["cl"],
            "CmPitch": document["moment_exact_degree2"]["total"]["cm_body_y"],
        }
    return results


def evaluate_case(
    entry_root: Path,
    support_root: Path,
    case_id: str,
    scope: str,
    output: Path,
    *,
    demonstration: bool = False,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    support, root = open_case_support(support_root, case_id, demonstration=demonstration)
    case = {"case_id": case_id, "metric_values": {}, "field_statistics": {}, "coverage": {}}
    output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="hlpw6-native-", dir=scratch_root) as directory:
            scratch = Path(directory)
            domain_outputs = {}
            regions = {}
            for domain in scope_domains(scope):
                expected = (
                    None
                    if demonstration
                    else case_identities("native")[case_id][domain]["expected_count"]
                )
                summary, mapped, truths, identifiers = _domain(
                    support,
                    root,
                    entry_root / "cases" / case_id / domain,
                    domain,
                    scratch,
                    expected_count=expected,
                )
                case["metric_values"].update(summary["metrics"])
                case["field_statistics"].update(summary["statistics"])
                case["coverage"][domain] = summary["coverage"]
                regions.update(summary["regional"])
                domain_outputs[domain] = (mapped, truths, identifiers)
            mapped, truths, identifiers = domain_outputs["surface"]
            forces = _loads(support["surface"], root, mapped, truths, support["reference"])
            case["forces"] = forces
            for coefficient, key in (
                ("Cd", "c_drag_mae"),
                ("Cl", "c_lift_mae"),
                ("CmPitch", "c_pitch_mae"),
            ):
                case["metric_values"][key] = abs(
                    forces["prediction"][coefficient] - forces["truth"][coefficient]
                )
            profile_path = root / safe_relative_path(support["profiles"]["path"])
            if sha256_file(profile_path) != support["profiles"]["sha256"]:
                raise ContractError("compact profile support checksum differs")
            profile_support, _ = load_compact_support_npz(profile_path)
            cp = _stencil(
                mapped["pressure"], identifiers["raw_point_ids"], support["cp_stencil"], root
            )
            velocity = None
            if scope == "surface_and_volume":
                mapped, _, identifiers = domain_outputs["volume"]
                vector = _stencil(
                    mapped["velocity"],
                    identifiers["raw_point_ids"],
                    support["velocity_stencil"],
                    root,
                )
                velocity = np.full(4005, np.nan, dtype=np.float64)
                velocity[profile_support["velocity_valid_mask"]] = np.linalg.norm(vector, axis=1)
            payload, metadata, metrics = encode_profiles(cp, velocity, profile_support, scope)
            case["metric_values"].update(metrics)
            if regions:
                case["regional_statistics"] = regions
            case["profile_metadata"] = metadata
            (output / "profiles.npz").write_bytes(payload)
            case["profile_sha256"] = sha256_bytes(payload)
            write_json(output / "case.json", case, exclusive=True)
            return case
    except BaseException:
        shutil.rmtree(output)
        raise


def evaluate_entry(
    entry_root: Path,
    support_root: Path,
    output: Path,
    *,
    resume: bool = False,
    demonstration: bool = False,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    entry = load_entry(entry_root)
    if not demonstration:
        require_native_release()
    if output.exists() and not resume:
        raise ContractError("output already exists; use --resume for the same entry and support")
    output.mkdir(parents=True, exist_ok=True)
    # Rehash the complete declared prediction set before allowing cached cases.
    prediction_hashes = {}
    for case_id in entry["test_case_ids"]:
        for domain in scope_domains(entry["prediction_scope"]):
            root = entry_root / "cases" / case_id / domain
            manifest = read_json(root / "manifest.json")
            for chunk in manifest["chunks"]:
                for descriptor in chunk.values():
                    array_file(root, descriptor)
            prediction_hashes[f"{case_id}/{domain}"] = sha256_file(root / "manifest.json")
    binding = {
        "entry_sha256": sha256_bytes(canonical_json_bytes(entry)),
        "support_index_sha256": sha256_file(support_root / "index.json"),
        "prediction_manifests": prediction_hashes,
        "demonstration": demonstration,
    }
    if (output / "evaluation.json").exists():
        if read_json(output / "evaluation.json") != binding:
            raise ContractError("resume entry or support differs from the original evaluation")
    else:
        write_json(output / "evaluation.json", binding, exclusive=True)
    cases, profiles = [], []
    for case_id in entry["test_case_ids"]:
        directory = output / "working" / case_id
        if resume and (directory / "case.json").is_file():
            case = read_json(directory / "case.json")
            if sha256_file(directory / "profiles.npz") != case["profile_sha256"]:
                raise ContractError("resumed profile artifact checksum differs")
        else:
            case = evaluate_case(
                entry_root,
                support_root,
                case_id,
                entry["prediction_scope"],
                directory,
                demonstration=demonstration,
                scratch_root=scratch_root,
            )
        destination = output / "profiles/cases" / (case_id + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "profiles.npz", destination)
        profiles.append(
            {
                "case_id": case_id,
                "path": destination.relative_to(output).as_posix(),
                "sha256": case["profile_sha256"],
                **case["profile_metadata"],
            }
        )
        cases.append(case)
    result = result_document(
        entry, cases, evaluation_kind="demonstration" if demonstration else "native_predictions"
    )
    regional = aggregate_regions(cases, entry["prediction_scope"])
    result["has_regional_diagnostics"] = regional is not None
    write_json(output / "entry.json", entry)
    write_json(
        output / "metrics/cases.json", {"schema": "hlpw6-aiml-case-metrics-v1", "cases": cases}
    )
    write_json(
        output / "profiles/index.json",
        {
            "schema": "hlpw6-aiml-profile-index-v1",
            "prediction_scope": entry["prediction_scope"],
            "cases": profiles,
        },
    )
    if regional is not None:
        write_json(output / "regional-diagnostics.json", regional)
    write_json(output / "result.json", result)
    return result
