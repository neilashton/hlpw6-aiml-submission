from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import deterministic_zip, verify_manifest
from .contracts import (
    EVALUATOR_VERSION,
    ContractError,
    case_identities,
    contract_identity,
    dataset_identity,
    require_native_release,
)
from .entry import split_record, validate_entry
from .jsonio import read_json, read_json_bytes, sha256_file
from .metrics import aggregate_cases, validate_case_metrics
from .profiles import validate_profile_payload
from .scores import calculate_scores, finite

PackageError = ContractError


@dataclass(frozen=True)
class VerifiedPackage:
    path: Path
    package_sha256: str
    size_bytes: int
    entry_count: int
    result: dict[str, Any]
    profile_index: dict[str, Any]


def _same_numbers(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    if left.keys() != right.keys():
        raise ContractError(f"{label} metric keys differ")
    for key, expected in right.items():
        if expected is None:
            if left[key] is not None:
                raise ContractError(f"{label}.{key} must be unavailable")
        elif not math.isclose(finite(left[key], key), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ContractError(f"{label}.{key} differs from the retained case evidence")


def verify_package(path: Path | str, *, allow_demonstration: bool = False) -> VerifiedPackage:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractError("package must be a regular ZIP file")
    try:
        with zipfile.ZipFile(source) as archive:
            members = verify_manifest(archive)
            required = {
                "entry.json",
                "result.json",
                "metrics/cases.json",
                "profiles/index.json",
                "package-manifest.json",
            }
            if not required <= members.keys():
                raise ContractError("package is missing required documents")
            entry = validate_entry(read_json_bytes(archive.read("entry.json")))
            result = read_json_bytes(archive.read("result.json"))
            if result.get("schema") != "hlpw6-aiml-result-v1" or result.get("schema_version") != 1:
                raise ContractError("package is not an HLPW6 evaluator result")
            if result.get("status") != "complete":
                raise ContractError("only complete split results may be submitted")
            if result.get("evaluation_kind") not in {"native_predictions", "demonstration"}:
                raise ContractError("unknown evaluation kind")
            if result["evaluation_kind"] == "demonstration" and not allow_demonstration:
                raise ContractError("demonstration packages are not workshop submissions")
            if result["evaluation_kind"] == "native_predictions":
                require_native_release()
            if result.get("dataset") != dataset_identity():
                raise ContractError("dataset identity differs from the pinned HiLiftAeroML release")
            expected_evaluator = {
                "version": EVALUATOR_VERSION,
                "contract_sha256": contract_identity(),
            }
            if result.get("evaluator") != expected_evaluator:
                raise ContractError("evaluator or scientific contract identity differs")
            if result.get("submission") != entry or result.get("split") != split_record(entry):
                raise ContractError("result and entry split or participant declarations differ")
            scope = entry["prediction_scope"]
            if result.get("prediction_scope") != scope:
                raise ContractError("result and entry prediction scopes differ")
            evidence = read_json_bytes(archive.read("metrics/cases.json"))
            cases = evidence["cases"]
            if (
                evidence.get("schema") != "hlpw6-aiml-case-metrics-v1"
                or [case["case_id"] for case in cases] != entry["test_case_ids"]
            ):
                raise ContractError("case evidence must cover the exact ordered test split")
            for case in cases:
                validate_case_metrics(case, scope)
                if result["evaluation_kind"] == "native_predictions":
                    expected = case_identities("native")[case["case_id"]]
                    for domain, coverage in case["coverage"].items():
                        if coverage["expected_count"] != expected[domain]["expected_count"]:
                            raise ContractError(
                                "case point count differs from the pinned native support"
                            )
            recomputed = calculate_scores(aggregate_cases(cases, scope), scope)
            _same_numbers(result["metric_values"], recomputed["metric_values"], "aggregate")
            _same_numbers(result["component_scores"], recomputed["component_scores"], "components")
            if (
                result["component_availability"] != recomputed["component_availability"]
                or result["scoring"] != recomputed["scoring"]
            ):
                raise ContractError("score availability or fixed weights differ")
            index = read_json_bytes(archive.read("profiles/index.json"))
            if (
                index.get("schema") != "hlpw6-aiml-profile-index-v1"
                or index.get("prediction_scope") != scope
            ):
                raise ContractError("profile index identity differs")
            rows = index.get("cases", [])
            if [row["case_id"] for row in rows] != entry["test_case_ids"]:
                raise ContractError("profile coverage differs from the exact ordered test split")
            allowed = set(required)
            hashes = {
                row["path"]: row["sha256"]
                for row in read_json_bytes(archive.read("package-manifest.json"))["files"]
            }
            for row in rows:
                expected_path = f"profiles/cases/{row['case_id']}.npz"
                if row["path"] != expected_path or expected_path not in members:
                    raise ContractError("profile artifact has an unexpected package location")
                if row["sha256"] != hashes[expected_path]:
                    raise ContractError("profile index and package manifest hashes differ")
                arrays = validate_profile_payload(archive.read(expected_path), row, scope)
                del arrays
                if result["evaluation_kind"] == "native_predictions":
                    expected = case_identities("profiles")[row["case_id"]]
                    if row["surface_cp"] != expected["surface_cp"]:
                        raise ContractError("surface profile identity differs from pinned support")
                    if (
                        scope == "surface_and_volume"
                        and row["volume_velocity"] != expected["volume_velocity"]
                    ):
                        raise ContractError("velocity profile identity differs from pinned support")
                allowed.add(expected_path)
            if result.get("has_regional_diagnostics") is not (
                "regional-diagnostics.json" in members
            ):
                raise ContractError("regional availability differs from package contents")
            if "regional-diagnostics.json" in members:
                from .regions import validate_regional_report

                regional = read_json_bytes(archive.read("regional-diagnostics.json"))
                validate_regional_report(regional, cases, scope)
                allowed.add("regional-diagnostics.json")
            if set(members) != allowed:
                raise ContractError(
                    "package contains undeclared files or raw prediction/support data"
                )
            return VerifiedPackage(
                source, sha256_file(source), source.stat().st_size, len(members), result, index
            )
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        zipfile.BadZipFile,
        RuntimeError,
        OSError,
    ) as error:
        raise ContractError(f"invalid HLPW6 package: {error}") from error


def package_result(
    root: Path | str, output: Path | str, *, allow_demonstration: bool = False
) -> dict[str, Any]:
    root, output = Path(root), Path(output)
    index = read_json(root / "profiles/index.json")
    paths = [
        "entry.json",
        "result.json",
        "metrics/cases.json",
        "profiles/index.json",
        *[row["path"] for row in index["cases"]],
    ]
    if (root / "regional-diagnostics.json").is_file():
        paths.append("regional-diagnostics.json")
    receipt = deterministic_zip(root, output, paths)
    try:
        verify_package(output, allow_demonstration=allow_demonstration)
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix + ".sha256").unlink(missing_ok=True)
        raise
    return receipt


def result_document(
    entry: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    evaluation_kind: str = "native_predictions",
) -> dict[str, Any]:
    validate_entry(entry)
    if [case["case_id"] for case in cases] != entry["test_case_ids"]:
        raise ContractError("cannot aggregate a partial or reordered split")
    scope = entry["prediction_scope"]
    for case in cases:
        validate_case_metrics(case, scope)
    return {
        "schema": "hlpw6-aiml-result-v1",
        "schema_version": 1,
        "status": "complete",
        "evaluation_kind": evaluation_kind,
        "dataset": dataset_identity(),
        "evaluator": {"version": EVALUATOR_VERSION, "contract_sha256": contract_identity()},
        "submission": entry,
        "split": split_record(entry),
        "prediction_scope": scope,
        **calculate_scores(aggregate_cases(cases, scope), scope),
    }
