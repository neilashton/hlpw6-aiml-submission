from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hlpw6_aiml.archive import deterministic_zip, npz_arrays
from hlpw6_aiml.contracts import ContractError, official_splits
from hlpw6_aiml.demo import create_demo
from hlpw6_aiml.entry import init_entry, validate_entry
from hlpw6_aiml.evaluation import evaluate_entry
from hlpw6_aiml.fetch import data_plan
from hlpw6_aiml.jsonio import read_json, read_json_bytes, sha256_file, write_json
from hlpw6_aiml.metrics import FieldAccumulator
from hlpw6_aiml.packaging import package_result, verify_package
from hlpw6_aiml.predictions import PredictionWriter
from hlpw6_aiml.reference.exact_moment_audit import integrate_triangle_batch
from hlpw6_aiml.report import write_report
from hlpw6_aiml.scores import competition_ranks


@pytest.fixture(params=["surface_only", "surface_and_volume"])
def workflow(tmp_path, request):
    root = tmp_path / "demo"
    create_demo(root, request.param)
    output = root / "output"
    result = evaluate_entry(root / "entry", root / "support", output, demonstration=True)
    return root, output, result


def test_complete_scope_workflow_and_determinism(workflow):
    root, output, result = workflow
    scope = result["prediction_scope"]
    assert result["metric_values"]["overall_score"] == pytest.approx(
        60 if scope == "surface_only" else 100
    )
    if scope == "surface_only":
        assert not list((root / "entry").rglob("volume"))
        assert result["metric_values"]["volume_pressure_rel_l2"] is None
        assert result["component_scores"]["volume_pressure_rel_l2"] == 0
    first = package_result(output, root / "first.zip", allow_demonstration=True)
    second = package_result(output, root / "second.zip", allow_demonstration=True)
    assert first["sha256"] == second["sha256"]
    verified = verify_package(root / "first.zip", allow_demonstration=True)
    assert verified.result == result
    assert (
        evaluate_entry(root / "entry", root / "support", output, resume=True, demonstration=True)
        == result
    )
    with pytest.raises(ContractError, match="not workshop submissions"):
        verify_package(root / "first.zip")
    write_report(output, root / "report.html")
    assert "Synthetic workflow demonstration" in (root / "report.html").read_text()
    assert "NaN" not in (root / "report.html").read_text()


def test_native_activation_is_explicit(workflow):
    root, _, _ = workflow
    with pytest.raises(ContractError, match="not activated"):
        evaluate_entry(root / "entry", root / "support", root / "production")


def test_changed_predictions_invalidate_resume(workflow):
    root, output, _ = workflow
    path = next((root / "entry").rglob("pressure.npy"))
    values = np.load(path)
    values[0] += 1
    np.save(path, values)
    with pytest.raises(ContractError, match="checksum differs"):
        evaluate_entry(root / "entry", root / "support", output, resume=True, demonstration=True)


def test_writer_and_changed_chunk_layout_preserve_scores(workflow):
    root, output, result = workflow
    for path in list((root / "entry").rglob("manifest.json")):
        source = read_json(path)
        data = {
            key: np.concatenate(
                [np.load(path.parent / chunk[key]["path"]) for chunk in source["chunks"]]
            )
            for key in source["chunks"][0]
        }
        shutil.rmtree(path.parent)
        writer = PredictionWriter(path.parent, source["domain"])
        raw = data.pop("raw_point_ids")
        writer.write_chunk(raw[::-1], **{key: values[::-1] for key, values in data.items()})
        writer.finish()
    changed = evaluate_entry(
        root / "entry", root / "support", root / "new-output", demonstration=True
    )
    assert changed == result
    with pytest.raises(ContractError, match="resume entry or support differs"):
        evaluate_entry(root / "entry", root / "support", output, demonstration=True, resume=True)


def test_retained_numerical_sources_match_upstream():
    root = Path(__file__).resolve().parents[1]
    for record in read_json(root / "UPSTREAM.json")["numerical_reference_files"]:
        assert sha256_file(root / "src/hlpw6_aiml" / record["path"]) == record["sha256"]


def test_recomputed_score_and_manifest_tampering_are_rejected(workflow):
    root, output, result = workflow
    result["metric_values"]["overall_score"] -= 1
    write_json(output / "result.json", result)
    paths = [
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and "working" not in p.parts and p.name != "evaluation.json"
    ]
    package = root / "forged.zip"
    deterministic_zip(output, package, paths)
    with pytest.raises(ContractError, match="differs from the retained case evidence"):
        verify_package(package, allow_demonstration=True)
    with zipfile.ZipFile(package) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    contents["entry.json"] += b" "
    with zipfile.ZipFile(root / "bad-checksum.zip", "w") as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
    with pytest.raises(ContractError, match="checksum or size"):
        verify_package(root / "bad-checksum.zip", allow_demonstration=True)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "nonfinite"])
def test_invalid_native_predictions_fail_before_packaging(tmp_path, mutation):
    create_demo(tmp_path / "input", "surface_only")
    root = tmp_path / "input"
    manifest_path = next((root / "entry").rglob("manifest.json"))
    manifest = read_json(manifest_path)
    if mutation == "missing":
        manifest["chunks"].pop()
    else:
        descriptor = manifest["chunks"][0][
            "pressure" if mutation == "nonfinite" else "raw_point_ids"
        ]
        path = manifest_path.parent / descriptor["path"]
        values = np.load(path)
        values[0] = (
            np.nan if mutation == "nonfinite" else (values[1] if mutation == "duplicate" else 999)
        )
        np.save(path, values)
        descriptor["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ContractError, match="incomplete|duplicate raw point|non-finite"):
        evaluate_entry(root / "entry", root / "support", root / "output", demonstration=True)
    assert not (root / "output/result.json").exists()


def test_statistics_are_additive_with_one_vector_weight():
    truth = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 4.0, 6.0]])
    pred = truth + np.array([1.0, -1.0, 2.0])
    weights = np.array([1.0, 2.0, 4.0])
    whole, chunks = FieldAccumulator(3), FieldAccumulator(3)
    whole.add(pred, truth, weights)
    for i in range(3):
        chunks.add(pred[i : i + 1], truth[i : i + 1], weights[i : i + 1])
    assert whole.statistics() == chunks.statistics()
    assert whole.statistics()["squared_error"] == 42
    assert whole.values("v", dimensional_scale=2)["v_rmse"] == pytest.approx(2 * np.sqrt(2))


def test_exact_pressure_force_and_linear_pressure_pitch():
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = integrate_triangle_batch(
        triangle_points=triangle,
        pressure_coefficient=np.array([[0.0, 1.0, 0.0]]),
        wall_shear_coefficient=np.zeros((1, 3, 3)),
        moment_reference=[0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(result.pressure_force, [0, 0, 1 / 6])
    # Integral -x * Cp over this triangle, Cp=x: -integral(x^2) = -1/12.
    np.testing.assert_allclose(result.pressure_moment_exact, [1 / 24, -1 / 12, 0])


def test_official_training_regimes_and_surface_downloads(tmp_path):
    assert len(official_splits()) == 14
    assert len({case for split in official_splits().values() for case in split["case_ids"]}) == 1355
    entry = init_entry(
        tmp_path / "entry",
        split_id="full",
        scope="surface_only",
        submission_id="team-1",
        method_name="Model",
        contact_email="a@example.invalid",
    )
    plan = data_plan(entry["test_case_ids"][:1], "surface_only")
    assert len(plan) == 1 and plan[0]["domain"] == "surface"
    entry["train_case_ids"] = []
    with pytest.raises(ContractError, match="published training regime"):
        validate_entry(entry)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "x\\bad", "x/../bad"])
def test_unsafe_zip_paths_are_rejected(tmp_path, path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(path, "bad")
    with pytest.raises(ContractError, match="unsafe path"):
        verify_package(archive)


def test_json_and_numpy_reject_unsafe_values():
    with pytest.raises(ValueError):
        read_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):
        read_json_bytes(b'{"a":NaN}')
    stream = io.BytesIO()
    np.savez(stream, unsafe=np.array([{}], dtype=object))
    with pytest.raises(ContractError, match="unsafe prediction array"):
        npz_arrays(stream.getvalue(), {"unsafe"})


def test_displayed_ties_and_split_isolation():
    rows = [
        {
            "import_id": str(i),
            "split_id": "full",
            "case_set_sha256": "same",
            "evaluator_version": "same",
            "metric_values": {"overall_score": score},
        }
        for i, score in enumerate([95.04, 95.01, 94.95, 90])
    ]
    assert competition_ranks(rows) == {"0": 1, "1": 1, "2": 1, "3": 4}
    rows[-1]["split_id"] = "medium"
    with pytest.raises(ContractError, match="different split"):
        competition_ranks(rows)
