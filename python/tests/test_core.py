from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from cross_runtime_inference_tester.cli import app
from cross_runtime_inference_tester.core import (
    benchmark,
    compare,
    generate_dataset,
    run_verification,
    train_export,
    validate_features,
    write_reports,
)
from typer.testing import CliRunner


def test_dataset_is_deterministic_and_reasonably_balanced() -> None:
    first = generate_dataset(1200, 19)
    second = generate_dataset(1200, 19)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert 0.17 < float(first[1].mean()) < 0.31


def test_training_and_onnx_export_are_reproducible(tmp_path: Path) -> None:
    first = train_export(tmp_path / "first", seed=21, samples=400)
    second = train_export(tmp_path / "second", seed=21, samples=400)
    assert first["manifest"]["model_sha256"] == second["manifest"]["model_sha256"]


def test_comparison_math_and_strict_negative_case() -> None:
    ok = compare(np.asarray([0.2, 0.8]), np.asarray([0.20001, 0.79999]), 1e-4, 1e-3, 0.5)
    assert ok.status == "PASS"
    assert ok.agreement is True
    drift = compare(np.asarray([0.499, 0.501]), np.asarray([0.501, 0.499]), 1e-4, 1e-3, 0.5)
    assert drift.status == "FAIL"
    assert drift.agreement is False
    assert drift.violations > 0


def test_comparison_rejects_shape_and_nonfinite_outputs() -> None:
    assert compare(np.ones((1, 2)), np.ones((2, 2))).status == "FAIL"
    assert compare(np.asarray([np.nan]), np.asarray([1.0])).status == "FAIL"


def test_input_validation() -> None:
    assert validate_features([1, 2, 0, 0, 20, 30, 50, 2]).dtype == np.float32
    with pytest.raises(ValueError, match="expected rows"):
        validate_features([1, 2])
    with pytest.raises(ValueError, match="finite"):
        validate_features([1, 2, 0, 0, 20, 30, 50, float("inf")])
    with pytest.raises(ValueError, match="finite"):
        validate_features([1, 2, 0, 0, 20, 30, 50, float("nan")])
    with pytest.raises(ValueError, match="tenure_months"):
        validate_features([121, 2, 0, 0, 20, 30, 50, 2])
    with pytest.raises(ValueError, match="numeric"):
        validate_features(["1", 2, 0, 0, 20, 30, 50, 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="numeric"):
        validate_features([True, 2, 0, 0, 20, 30, 50, 2])  # type: ignore[list-item]


def test_full_export_fixture_hash_and_verification(tmp_path: Path) -> None:
    generated = train_export(tmp_path, seed=13, samples=1200)
    manifest = tmp_path / "manifest.json"
    assert generated["manifest"]["model_sha256"]
    vectors = json.loads((tmp_path / "golden-vectors.json").read_text())
    assert sum(1 for case in vectors["fixtures"] if case["category"] == "held-out-test") >= 200
    assert {"typical", "boundary", "numerical", "batch"}.issubset(
        {case["category"] for case in vectors["fixtures"]}
    )
    result = run_verification(manifest)
    assert result["summary"]["status"] == "PASS"
    assert result["summary"]["failures"] == 0
    write_reports(result, tmp_path / "reports")
    assert all((tmp_path / "reports" / f"report.{ext}").is_file() for ext in ("json", "md", "html"))


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    train_export(tmp_path, seed=14, samples=500)
    model = tmp_path / "model.onnx"
    model.write_bytes(model.read_bytes() + b"corruption")
    from cross_runtime_inference_tester.core import OnnxRuntime

    with pytest.raises(ValueError, match="hash mismatch"):
        OnnxRuntime(tmp_path / "manifest.json")


def test_benchmark_records_independent_sessions_and_batch_sizes(tmp_path: Path) -> None:
    train_export(tmp_path, seed=33, samples=500)
    result = benchmark(tmp_path / "manifest.json", warmup=0, iterations=2, independent_runs=2)
    assert result["independent_run_count"] == 2
    assert [run["run_index"] for run in result["independent_runs"]] == [1, 2]
    assert all(len(run["batch_workloads"]) == 4 for run in result["independent_runs"])


def test_cli_exits_nonzero_on_injected_drift(tmp_path: Path) -> None:
    train_export(tmp_path, seed=42, samples=1200)
    fixtures_path = tmp_path / "golden-vectors.json"
    fixtures = json.loads(fixtures_path.read_text())
    fixtures["fixtures"][0]["expected_probabilities"] = [0.0, 1.0]
    fixtures_path.write_text(json.dumps(fixtures))
    output = tmp_path / "failure-report"
    result = CliRunner().invoke(
        app,
        ["verify", "--manifest", str(tmp_path / "manifest.json"), "--output", str(output)],
    )
    assert result.exit_code == 1
    report = json.loads((output / "report.json").read_text())
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["failures"] > 0
