"""Small, deliberately synthetic native meshes for offline workflow checks.

These fixtures reuse two valid case identifiers solely to exercise split handling.
They are not HiLiftAeroML predictions, scoring support, or benchmark results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import ContractError, official_splits, scope_domains
from .jsonio import sha256_file, write_json
from .reference.compact_profiles import write_compact_support_npz


def save_array(root: Path, name: str, value: np.ndarray) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {
        "path": name,
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def create_demo(root: Path, scope: str = "surface_and_volume", *, error: float = 0.0) -> dict:
    scope_domains(scope)
    if root.exists():
        raise ContractError("demo destination already exists")
    case_ids = official_splits()["full"]["case_ids"][:2]
    entry = {
        "schema": "hlpw6-aiml-entry-v1",
        "schema_version": 1,
        "submission_id": "demo",
        "method_name": "Synthetic workflow demonstration",
        "contact_email": "demo@example.invalid",
        "split_id": "demo-two-cases",
        "test_case_ids": case_ids,
        "train_case_ids": [],
        "validation_case_ids": [],
        "prediction_scope": scope,
        "training_regime": "from_scratch",
        "target_data_used": "official_train",
        "external_pretraining": False,
        "pretraining_data": [],
    }
    write_json(root / "entry/entry.json", entry, exclusive=True)
    index = {"schema": "hlpw6-native-support-index-v1", "demonstration": True, "cases": {}}
    for number, case_id in enumerate(case_ids):
        support_root = root / "support/cases" / case_id
        reference = {
            "q_inf": 2.0,
            "q_ref": 4.0,
            "u_inf": 10.0,
            "area_ref": 1.0,
            "chord_ref": 1.0,
            "aoa_degrees": 4.0 + 4 * number,
            "moment_reference": [0.0, 0.0, 0.0],
        }
        document = {
            "schema": "hlpw6-native-case-support-v1",
            "case_id": case_id,
            "reference": reference,
        }
        field_arrays = {}
        for domain in scope_domains(scope):
            count = 6 if domain == "surface" else 12
            ids = np.arange(count, dtype=np.int64) * (1 if domain == "surface" else 2)
            pressure = np.arange(1, count + 1, dtype=np.float64) / 4 + number
            vector = np.column_stack((pressure, pressure / 2, pressure / 4))
            field_arrays[domain] = {
                "pressure": pressure,
                "wall_shear" if domain == "surface" else "velocity": vector,
            }
            arrays = {
                "raw_point_ids": ids,
                **field_arrays[domain],
                "region_ids": (np.arange(count) % 4).astype(np.uint8),
            }
            if domain == "surface":
                arrays.update(
                    weights=np.array([1, 2, 3, 1, 2, 3], dtype=np.float32) / 12,
                    points=np.array(
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
                        dtype=np.float64,
                    ),
                    connectivity=np.arange(6, dtype=np.int64),
                    offsets=np.array([0, 3, 6], dtype=np.int64),
                )
            document[domain] = {
                key: save_array(support_root, f"{domain}/{key}.npy", value)
                for key, value in arrays.items()
            }
            prediction_root = root / "entry/cases" / case_id / domain
            # Deliberately non-canonical ordering, split across chunks.
            manifest = {
                "schema": "hlpw6-native-predictions-v1",
                "domain": domain,
                "basis": "hiliftaeroml_nondimensional",
                "chunks": [],
            }
            for part, order in enumerate(np.array_split(np.arange(count)[::-1], 2)):
                values = {
                    "raw_point_ids": ids[order],
                    **{key: value[order] + error for key, value in field_arrays[domain].items()},
                }
                manifest["chunks"].append(
                    {
                        key: save_array(prediction_root, f"chunk-{part}/{key}.npy", value)
                        for key, value in values.items()
                    }
                )
            write_json(prediction_root / "manifest.json", manifest, exclusive=True)
        cp_ids = np.tile(np.arange(3, dtype=np.int64), 10)
        arc = np.tile(np.array([0, 0.5, 1.0], dtype=np.float64), 10)
        velocity_ids = np.arange(4005, dtype=np.int64) % 12
        valid = np.ones(4005, dtype=bool)
        valid[::801] = False
        speed = np.linalg.norm(
            np.column_stack(
                (
                    np.arange(1, 13) / 4 + number,
                    (np.arange(1, 13) / 4 + number) / 2,
                    (np.arange(1, 13) / 4 + number) / 4,
                )
            ),
            axis=1,
        )[velocity_ids]
        speed[~valid] = np.nan
        support = {
            "cp_xyz_in": np.column_stack((arc, np.repeat(np.arange(10), 3), np.zeros(30))).astype(
                np.float64
            ),
            "cp_arc_length_in": arc,
            "cp_truth": field_arrays["surface"]["pressure"][cp_ids],
            "cp_branch_point_offsets": np.arange(11, dtype=np.int64) * 3,
            "cp_branch_row_code": np.arange(10, dtype=np.uint8),
            "cp_branch_graph_component_code": np.zeros(10, dtype=np.int64),
            "cp_source_branch_index": np.arange(10, dtype=np.int64),
            "velocity_requested_xyz_in": np.column_stack(
                (np.zeros(4005), np.repeat(np.arange(5), 801), np.tile(np.linspace(0, 1, 801), 5))
            ),
            "velocity_valid_mask": valid,
            "velocity_station_names": np.array(["B.2", "B.3", "C.1", "C.2", "C.3"], dtype="<U3"),
            "velocity_station_row_offsets": np.arange(6, dtype=np.int64) * 801,
            "velocity_line_length_weights_in": valid.astype(np.float64) / 800,
            "velocity_truth_speed_over_uinf": speed,
        }
        for key in ("component", "plane_piece", "side", "topology_patch"):
            support[f"cp_branch_{key}_code"] = np.zeros(10, dtype=np.uint8)
        path = support_root / "profiles.npz"
        write_compact_support_npz(path, support)
        document["profiles"] = {"path": path.name, "sha256": sha256_file(path)}
        for name, raw in (("cp_stencil", cp_ids), ("velocity_stencil", velocity_ids[valid] * 2)):
            if name == "velocity_stencil" and scope == "surface_only":
                continue
            document[name] = {
                key: save_array(support_root, f"{name}/{key}.npy", value)
                for key, value in {
                    "raw_point_ids": raw,
                    "weights": np.ones(len(raw), dtype=np.float64),
                    "offsets": np.arange(len(raw) + 1, dtype=np.int64),
                }.items()
            }
        write_json(support_root / "case.json", document, exclusive=True)
        index["cases"][case_id] = {"sha256": sha256_file(support_root / "case.json")}
    write_json(root / "support/index.json", index, exclusive=True)
    return {"entry": str(root / "entry"), "support": str(root / "support"), "demonstration": True}
