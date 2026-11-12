"""Dataset, training, export, inference, fixture, and comparison primitives."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
from jsonschema import validate as validate_json_schema
from onnx import TensorProto
from pydantic import BaseModel, ConfigDict, Field
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class FeatureDefinition(TypedDict):
    name: str
    label: str
    description: str
    type: str
    unit: str
    min: float
    max: float
    default: float


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "models" / "subscription-risk-v1"
FEATURES: list[FeatureDefinition] = [
    {
        "name": "tenure_months",
        "label": "Tenure",
        "description": "Time since account started",
        "type": "number",
        "unit": "months",
        "min": 0,
        "max": 120,
        "default": 18,
    },
    {
        "name": "active_days_30d",
        "label": "Active days",
        "description": "Days with product activity in the past 30 days",
        "type": "number",
        "unit": "days",
        "min": 0,
        "max": 30,
        "default": 15,
    },
    {
        "name": "support_tickets_90d",
        "label": "Support tickets",
        "description": "Support requests in the past 90 days",
        "type": "number",
        "unit": "tickets",
        "min": 0,
        "max": 40,
        "default": 1,
    },
    {
        "name": "failed_payments_12m",
        "label": "Failed payments",
        "description": "Failed payment attempts in the past year",
        "type": "number",
        "unit": "payments",
        "min": 0,
        "max": 12,
        "default": 0,
    },
    {
        "name": "avg_session_minutes",
        "label": "Average session",
        "description": "Mean product session duration",
        "type": "number",
        "unit": "minutes",
        "min": 0,
        "max": 300,
        "default": 24,
    },
    {
        "name": "monthly_plan_usd",
        "label": "Monthly plan",
        "description": "Fictional monthly subscription price",
        "type": "number",
        "unit": "USD",
        "min": 0,
        "max": 500,
        "default": 49,
    },
    {
        "name": "feature_adoption_pct",
        "label": "Feature adoption",
        "description": "Share of available features used",
        "type": "number",
        "unit": "percent",
        "min": 0,
        "max": 100,
        "default": 45,
    },
    {
        "name": "days_since_last_login",
        "label": "Last login",
        "description": "Days since the account was last active",
        "type": "number",
        "unit": "days",
        "min": 0,
        "max": 365,
        "default": 4,
    },
]
FEATURE_ORDER = [feature["name"] for feature in FEATURES]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    model_id: str
    model_version: str
    source_model_version: str
    model_type: str
    task_type: str
    model_filename: str
    model_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    feature_order: list[str]
    input_name: str
    input_dtype: str
    input_shape: list[int | None]
    output_names: list[str]
    output_types: list[str]
    probability_output_name: str
    class_output_name: str
    class_labels: list[int]
    decision_threshold: float = Field(ge=0, le=1)
    opset_imports: dict[str, int]
    training_seed: int
    training_config: dict[str, Any]
    training_library_versions: dict[str, str]
    export_library_versions: dict[str, str]
    expected_runtime_compatibility: list[str]
    artifact_provenance: dict[str, str]


@dataclass
class Comparison:
    status: str
    max_absolute_difference: float
    max_relative_difference: float
    mean_absolute_difference: float
    rmse: float
    violations: int
    outputs: int
    agreement: bool | None
    reference_class: int | None
    target_class: int | None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_schema(value: Any, schema_name: str) -> None:
    schema_path = ROOT / "packages" / "schemas" / f"{schema_name}.schema.json"
    validate_json_schema(value, json.loads(schema_path.read_text()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_schema() -> dict[str, Any]:
    return {"schema_version": "1.0.0", "features": FEATURES}


def generate_dataset(
    samples: int = 7000, seed: int = 2026, class_balance: float = 0.24
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic, fictional renewal dataset with controlled signal and noise."""
    if samples < 100 or not 0.05 <= class_balance <= 0.6:
        raise ValueError("samples must be >=100 and class_balance must be between 0.05 and 0.6")
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.gamma(2.4, 13, samples), 0, 120)
    active = np.clip(rng.poisson(15, samples) - 0.055 * tenure, 0, 30)
    tickets = np.clip(
        rng.poisson(1.3, samples) + (rng.random(samples) < 0.08) * rng.poisson(4, samples), 0, 40
    )
    failed = np.clip(
        rng.poisson(0.28, samples) + (rng.random(samples) < 0.035) * rng.poisson(2, samples), 0, 12
    )
    session = np.clip(rng.lognormal(3.05, 0.62, samples), 0, 300)
    price = np.clip(
        rng.choice([19, 39, 59, 99, 149], samples, p=[0.2, 0.29, 0.28, 0.17, 0.06])
        + rng.normal(0, 3, samples),
        0,
        500,
    )
    adoption = np.clip(100 * rng.beta(2.1, 2.6, samples), 0, 100)
    last_login = np.clip(rng.exponential(15, samples) + 0.22 * (30 - active), 0, 365)
    X = np.column_stack(
        [tenure, active, tickets, failed, session, price, adoption, last_login]
    ).astype(np.float32)
    z = (
        -0.027 * tenure
        - 0.095 * active
        + 0.31 * tickets
        + 0.86 * failed
        - 0.018 * session
        + 0.007 * (price - 50)
        - 0.027 * adoption
        + 0.035 * last_login
        + 0.013 * tickets * (failed > 0)
        + 0.0008 * price * last_login
        + rng.normal(0, 0.42, samples)
    )
    # Calibrate average event probability to the requested balance without using generated labels.
    low, high = -15.0, 15.0
    for _ in range(60):
        midpoint = (low + high) / 2
        if float(np.mean(1 / (1 + np.exp(-np.clip(z + midpoint, -30, 30))))) < class_balance:
            low = midpoint
        else:
            high = midpoint
    z += (low + high) / 2
    y = (1 / (1 + np.exp(-np.clip(z, -30, 30))) > rng.random(samples)).astype(np.int64)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 2026) -> tuple[np.ndarray, ...]:
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=2 / 3, random_state=seed, stratify=y_hold
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def metrics(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }


def train_export(output: Path = ARTIFACT, seed: int = 2026, samples: int = 7000) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    X, y = generate_dataset(samples, seed)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed, C=0.8)),
        ]
    )
    with threadpool_limits(limits=1):
        model.fit(X_train, y_train)
    import onnxruntime

    model_onnx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        options={"classifier": {"zipmap": False}},
        target_opset=17,
    )
    # skl2onnx includes a process-specific graph scope in its graph name; stabilize it
    # so repeated exports from the same seed produce identical model bytes.
    model_onnx.graph.name = "cross_runtime_inference_tester_subscription_risk_v1"
    onnx.checker.check_model(model_onnx)
    model_path = output / "model.onnx"
    model_path.write_bytes(model_onnx.SerializeToString())
    graph = onnx.load(model_path)
    input_meta = graph.graph.input[0]
    output_meta = graph.graph.output
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_map = dict(
        zip(
            [out.name for out in session.get_outputs()],
            session.run(None, {input_meta.name: X_test}),
            strict=True,
        )
    )
    probability_name = next(
        out.name
        for out in session.get_outputs()
        if out.type == "tensor(float)" and len(out.shape) == 2 and out.shape[-1] == 2
    )
    class_name = next(out.name for out in session.get_outputs() if out.name != probability_name)
    ort_prob = np.asarray(output_map[probability_name], dtype=np.float32)
    native_prob = model.predict_proba(X_test).astype(np.float32)
    parity = compare(native_prob, ort_prob, 0.0001, 0.001, 0.5)
    schema = feature_schema()
    schema_path = output / "feature-schema.json"
    schema_path.write_bytes(canonical_json(schema))
    digest = sha256(model_path)
    manifest = Manifest(
        schema_version="1.0.0",
        model_id="subscription-risk",
        model_version="1.0.0",
        source_model_version="sklearn-pipeline-v1",
        model_type="StandardScaler+LogisticRegression",
        task_type="binary_classification",
        model_filename="model.onnx",
        model_sha256=digest,
        feature_schema_version="1.0.0",
        feature_schema_sha256=hashlib.sha256(canonical_json(schema)).hexdigest(),
        feature_order=FEATURE_ORDER,
        input_name=input_meta.name,
        input_dtype="float32",
        input_shape=[None, len(FEATURES)],
        output_names=[x.name for x in output_meta],
        output_types=[
            "float32" if x.type.tensor_type.elem_type == TensorProto.FLOAT else "int64"
            for x in output_meta
        ],
        probability_output_name=probability_name,
        class_output_name=class_name,
        class_labels=[int(v) for v in model.classes_],
        decision_threshold=0.5,
        opset_imports={x.domain or "ai.onnx": x.version for x in graph.opset_import},
        training_seed=seed,
        training_config={
            "samples": samples,
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
            "class_balance_target": 0.24,
        },
        training_library_versions={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        export_library_versions={
            "onnx": onnx.__version__,
            "skl2onnx": __import__("skl2onnx").__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        expected_runtime_compatibility=["python-onnxruntime-cpu", "browser-onnxruntime-wasm"],
        artifact_provenance={
            "generator": "Cross-Runtime Inference Tester synthetic renewal risk data",
            "source": "synthetic; not validated business estimates",
        },
    )
    validate_schema(manifest.model_dump(), "manifest")
    (output / "manifest.json").write_bytes(manifest.model_dump_json(indent=2).encode())
    train_metrics = {
        "schema_version": "1.0.0",
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "seed": seed,
        "class_distribution": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "training": metrics(model, X_train, y_train),
        "validation": metrics(model, X_val, y_val),
        "test": metrics(model, X_test, y_test),
        "native_vs_onnx": asdict(parity),
    }
    (output / "training-metrics.json").write_bytes(canonical_json(train_metrics))
    write_fixtures(output, X_test, model, digest, seed, session, manifest)
    return {
        "manifest": manifest.model_dump(),
        "metrics": train_metrics,
        "artifact_dir": str(output),
    }


def compare(
    reference: np.ndarray,
    target: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    threshold: float | None = None,
) -> Comparison:
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if a.shape != b.shape:
        return Comparison(
            "FAIL",
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            max(a.size, b.size),
            max(a.size, b.size),
            False,
            None,
            None,
        )
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all() or a.size == 0:
        return Comparison(
            "FAIL",
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            int((~finite).sum()),
            int(a.size),
            False,
            None,
            None,
        )
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(np.abs(a), np.finfo(float).tiny)
    within = absolute <= (atol + rtol * np.abs(a))
    ref_class = target_class = None
    agreement: bool | None = None
    if threshold is not None and a.ndim >= 1 and a.shape[-1] == 2:
        ref_classes = (a.reshape(-1, 2)[:, 1] >= threshold).astype(np.int64)
        target_classes = (b.reshape(-1, 2)[:, 1] >= threshold).astype(np.int64)
        ref_class = int(ref_classes[0]) if len(ref_classes) == 1 else None
        target_class = int(target_classes[0]) if len(target_classes) == 1 else None
        agreement = bool(np.array_equal(ref_classes, target_classes))
    status = "PASS" if bool(within.all()) and agreement is not False else "FAIL"
    return Comparison(
        status,
        float(absolute.max()),
        float(relative.max()),
        float(absolute.mean()),
        float(np.sqrt(np.mean(np.square(a - b)))),
        int((~within).sum()),
        int(a.size),
        agreement,
        ref_class,
        target_class,
    )


def write_fixtures(
    output: Path,
    X: np.ndarray,
    model: Pipeline,
    digest: str,
    seed: int,
    session: ort.InferenceSession,
    manifest: Manifest,
    count: int = 256,
) -> None:
    rng = np.random.default_rng(seed + 7)
    indices = rng.choice(len(X), size=min(count, len(X)), replace=False)
    rows = np.ascontiguousarray(X[indices], dtype=np.float32)
    outputs = session.run(None, {manifest.input_name: rows})
    probs = np.asarray(
        dict(zip([item.name for item in session.get_outputs()], outputs, strict=True))[
            manifest.probability_output_name
        ],
        dtype=np.float32,
    )
    fixtures = []
    for i, (row, p) in enumerate(zip(rows, probs, strict=True)):
        fixtures.append(
            {
                "fixture_id": f"golden-{i:04d}",
                "category": "held-out-test",
                "description": "Deterministic held-out synthetic account",
                "features": row.tolist(),
                "shape": [len(FEATURES)],
                "dtype": "float32",
                "expected_probabilities": p.tolist(),
                "expected_class": int(np.argmax(p)),
                "source_runtime": "python-onnxruntime-cpu",
                "model_sha256": digest,
                "generation": {"seed": seed, "held_out_index": int(indices[i])},
            }
        )

    probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32)
    curated = [
        (
            "typical-low",
            "typical",
            "Held-out case with low class-1 probability",
            int(np.argmin(probabilities)),
        ),
        (
            "typical-high",
            "typical",
            "Held-out case with high class-1 probability",
            int(np.argmax(probabilities)),
        ),
        (
            "threshold-near",
            "boundary",
            "Held-out case closest to the 0.5 decision threshold",
            int(np.argmin(np.abs(probabilities - manifest.decision_threshold))),
        ),
    ]
    defaults = np.asarray([feature["default"] for feature in FEATURES], dtype=np.float32)
    minima = np.asarray([feature["min"] for feature in FEATURES], dtype=np.float32)
    maxima = np.asarray([feature["max"] for feature in FEATURES], dtype=np.float32)
    curated_inputs = [
        ("boundary-min", "boundary", "Valid input at each declared feature minimum", minima),
        ("boundary-max", "boundary", "Valid input at each declared feature maximum", maxima),
        (
            "numeric-zero",
            "numerical",
            "All-zero feature vector within the declared ranges",
            np.zeros(len(FEATURES), dtype=np.float32),
        ),
        (
            "numeric-small",
            "numerical",
            "Small finite float32 values",
            np.full(len(FEATURES), np.float32(1e-6)),
        ),
        (
            "numeric-rounded",
            "numerical",
            "Float32 rounding-sensitive values near defaults",
            np.nextafter(defaults, np.float32(np.inf)),
        ),
    ]
    rows_to_add: list[tuple[str, str, str, np.ndarray]] = [
        (case_id, category, description, X[index])
        for case_id, category, description, index in curated
    ] + curated_inputs
    for case_id, category, description, row in rows_to_add:
        row = np.asarray(row, dtype=np.float32)
        probability_outputs = session.run(None, {manifest.input_name: row.reshape(1, -1)})
        output_map = dict(
            zip([item.name for item in session.get_outputs()], probability_outputs, strict=True)
        )
        probability = np.asarray(output_map[manifest.probability_output_name], dtype=np.float32)[0]
        fixtures.append(
            {
                "fixture_id": case_id,
                "category": category,
                "description": description,
                "features": row.tolist(),
                "shape": [len(FEATURES)],
                "dtype": "float32",
                "expected_probabilities": probability.tolist(),
                "expected_class": int(np.argmax(probability)),
                "source_runtime": "python-onnxruntime-cpu",
                "model_sha256": digest,
                "generation": {
                    "seed": seed,
                    "method": "held-out selection or declared feature schema",
                },
            }
        )
    for batch_size in (1, 8, 32, 128):
        batch = rows[:batch_size]
        if len(batch) < batch_size:
            batch = np.concatenate([batch, rows[: batch_size - len(batch)]])
        batch_outputs = session.run(None, {manifest.input_name: batch})
        batch_prob = np.asarray(
            dict(zip([item.name for item in session.get_outputs()], batch_outputs, strict=True))[
                manifest.probability_output_name
            ],
            dtype=np.float32,
        )
        fixtures.append(
            {
                "fixture_id": f"batch-{batch_size}",
                "category": "batch",
                "description": f"Batch inference with {batch_size} rows",
                "features": batch.tolist(),
                "shape": list(batch.shape),
                "dtype": "float32",
                "expected_probabilities": batch_prob.tolist(),
                "expected_class": np.argmax(batch_prob, axis=1).tolist(),
                "source_runtime": "python-onnxruntime-cpu",
                "model_sha256": digest,
                "generation": {"seed": seed, "batch_size": batch_size},
            }
        )
    for fixture in fixtures:
        validate_schema(fixture, "fixture")
    (output / "golden-vectors.json").write_bytes(
        canonical_json(
            {
                "schema_version": "1.0.0",
                "model_sha256": digest,
                "feature_order": FEATURE_ORDER,
                "fixtures": fixtures,
            }
        )
    )
    (output / "reference-predictions.json").write_bytes(
        canonical_json(
            {
                "schema_version": "1.0.0",
                "model_sha256": digest,
                "source_runtime": "python-onnxruntime-cpu",
                "fixtures": [
                    {
                        "fixture_id": f["fixture_id"],
                        "probabilities": f["expected_probabilities"],
                        "class": f["expected_class"],
                    }
                    for f in fixtures
                ],
            }
        )
    )


def validate_features(
    values: list[float] | list[list[float]], order: list[str] = FEATURE_ORDER
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in raw.flat
    ):
        raise ValueError("input values must be numeric integers or floats, not booleans or strings")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(order):
        raise ValueError(f"expected rows with shape [batch, {len(order)}] as a float32 tensor")
    if not np.isfinite(array).all():
        raise ValueError("input values must all be finite")
    if order == FEATURE_ORDER:
        for index, feature in enumerate(FEATURES):
            if np.any(array[:, index] < feature["min"]) or np.any(array[:, index] > feature["max"]):
                raise ValueError(
                    f"{feature['name']} must be between {feature['min']} and {feature['max']}"
                )
    return np.ascontiguousarray(array, dtype=np.float32)


class OnnxRuntime:
    """Verified CPU session; construction and inference timings remain distinct."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest = Manifest.model_validate_json(manifest_path.read_text())
        validate_schema(self.manifest.model_dump(), "manifest")
        self.model_path = manifest_path.parent / self.manifest.model_filename
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model missing: {self.model_path}")
        actual = sha256(self.model_path)
        if actual != self.manifest.model_sha256:
            raise ValueError(
                f"model hash mismatch: manifest={self.manifest.model_sha256}, actual={actual}"
            )
        graph = onnx.load(self.model_path)
        onnx.checker.check_model(graph)
        started = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.initialization_ms = (time.perf_counter() - started) * 1000
        actual_input = self.session.get_inputs()[0]
        if actual_input.name != self.manifest.input_name or actual_input.type != "tensor(float)":
            raise ValueError("runtime input does not match manifest")
        self.output_names = [out.name for out in self.session.get_outputs()]
        if self.output_names != self.manifest.output_names:
            raise ValueError("runtime outputs do not match manifest")

    def predict(self, values: list[float] | list[list[float]]) -> dict[str, Any]:
        x = validate_features(values, self.manifest.feature_order)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.manifest.input_name: x})
        elapsed = (time.perf_counter() - started) * 1000
        output_map = dict(zip(self.output_names, outputs, strict=True))
        probabilities = np.asarray(
            output_map[self.manifest.probability_output_name], dtype=np.float32
        )
        return {
            "runtime_id": "python-onnxruntime-cpu",
            "probabilities": probabilities.tolist(),
            "classes": np.argmax(probabilities, axis=1).tolist(),
            "latency_ms": elapsed,
            "initialization_ms": self.initialization_ms,
            "model_sha256": self.manifest.model_sha256,
            "input_shape": list(x.shape),
            "input_dtype": "float32",
        }


def benchmark(
    manifest_path: Path, warmup: int = 10, iterations: int = 100, independent_runs: int = 3
) -> dict[str, Any]:
    if independent_runs < 1 or independent_runs > 10:
        raise ValueError("independent_runs must be between 1 and 10")
    runtime = OnnxRuntime(manifest_path)
    sample = np.asarray([f["default"] for f in FEATURES], dtype=np.float32)
    first_inference_ms = runtime.predict(sample.tolist())["latency_ms"]
    for _ in range(warmup):
        runtime.predict(sample.tolist())
    measurements = []
    for _ in range(iterations):
        measurements.append(runtime.predict(sample.tolist())["latency_ms"])
    arr = np.asarray(measurements)
    def measure_batches(session: OnnxRuntime) -> list[dict[str, Any]]:
        workloads = []
        for batch_size in (1, 8, 32, 128):
            batch = np.repeat(sample.reshape(1, -1), batch_size, axis=0).tolist()
            for _ in range(warmup):
                session.predict(batch)
            values = np.asarray(
                [session.predict(batch)["latency_ms"] for _ in range(iterations)]
            )
            median = float(np.median(values))
            workloads.append(
                {
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "warmup_iterations": warmup,
                    "latency_ms": {
                        "median": median,
                        "mean": float(values.mean()),
                        "p95": float(np.percentile(values, 95)),
                        "p99": float(np.percentile(values, 99)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "stdev": float(values.std()),
                    },
                    "throughput_rows_per_second": float(batch_size * 1000 / median)
                    if median
                    else None,
                }
            )
        return workloads

    batch_workloads = measure_batches(runtime)
    run_summaries = [
        {
            "run_index": 1,
            "session_initialization_ms": runtime.initialization_ms,
            "first_inference_ms": first_inference_ms,
            "batch_workloads": batch_workloads,
        }
    ]
    for run_index in range(2, independent_runs + 1):
        fresh_runtime = OnnxRuntime(manifest_path)
        fresh_first_inference_ms = fresh_runtime.predict(sample.tolist())["latency_ms"]
        run_summaries.append(
            {
                "run_index": run_index,
                "session_initialization_ms": fresh_runtime.initialization_ms,
                "first_inference_ms": fresh_first_inference_ms,
                "batch_workloads": measure_batches(fresh_runtime),
            }
        )
    model = runtime.model_path
    result = {
        "schema_version": "1.0.0",
        "runtime_id": "python-onnxruntime-cpu",
        "model_sha256": runtime.manifest.model_sha256,
        "model_size_bytes": model.stat().st_size,
        "environment": {
            "python": sys.version.split()[0],
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "os": platform.platform(),
            "architecture": platform.machine(),
            "provider": "CPUExecutionProvider",
            "intra_op_threads": 1,
        },
        "initialization_ms": runtime.initialization_ms,
        "first_inference_ms": first_inference_ms,
        "warmup_iterations": warmup,
        "iterations": iterations,
        "batch_size": 1,
        "batch_workloads": batch_workloads,
        "independent_run_count": len(run_summaries),
        "independent_runs": run_summaries,
        "latency_ms": {
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "stdev": float(arr.std()),
        },
    }
    validate_schema(result, "benchmark-result")
    return result


def write_reports(report: dict[str, Any], output: Path) -> None:
    validate_schema(report, "report")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(canonical_json(report))
    status = report["summary"]["status"]
    summary = report["summary"]
    model_hash = report["model"]["sha256"]
    (output / "report.md").write_text(
        f"# Cross-Runtime Inference Tester report\n\nStatus: **{status}**\n\n"
        f"Model SHA-256: `{model_hash}`\n\n"
        f"Cases: {summary['cases']}\n\nFailures: {summary['failures']}\n\n"
        f"Maximum absolute difference: {summary['max_absolute_difference']:.8g}\n\n"
        "This report describes synthetic demonstration data, not validated business risk estimates."
        "\n"
    )
    rows = "".join(
        f"<tr><td>{c['case_id']}</td><td>{c['status']}</td><td>{c['max_absolute_difference']:.8g}</td></tr>"
        for c in report.get("cases", [])
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Cross-Runtime Inference Tester report</title>"
        f"<h1>Cross-Runtime Inference Tester report</h1><p>Status: <strong>{status}</strong></p>"
        f"<p>Model: <code>{model_hash}</code></p>"
        "<table><thead><tr><th>Case</th><th>Status</th><th>Max abs diff</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def run_verification(manifest_path: Path, atol: float = 1e-4, rtol: float = 1e-3) -> dict[str, Any]:
    runtime = OnnxRuntime(manifest_path)
    data = json.loads((manifest_path.parent / "golden-vectors.json").read_text())
    if data["model_sha256"] != runtime.manifest.model_sha256:
        raise ValueError("fixture and active model hashes differ")
    cases = []
    for fixture in data["fixtures"]:
        if fixture["model_sha256"] != runtime.manifest.model_sha256:
            raise ValueError(f"model hash mismatch in {fixture['fixture_id']}")
        x = validate_features(fixture["features"], runtime.manifest.feature_order)
        actual_outputs = runtime.session.run(None, {runtime.manifest.input_name: x})
        output = np.asarray(
            dict(zip(runtime.output_names, actual_outputs, strict=True))[
                runtime.manifest.probability_output_name
            ]
        )
        expected = np.asarray(fixture["expected_probabilities"])
        if expected.ndim == 1 and output.ndim == 2 and output.shape[0] == 1:
            output = output[0]
        result = compare(
            expected,
            output,
            atol,
            rtol,
            runtime.manifest.decision_threshold,
        )
        cases.append(
            {
                "case_id": fixture["fixture_id"],
                "reference_runtime": fixture["source_runtime"],
                "target_runtime": "python-onnxruntime-cpu",
                "reference_output": expected.tolist(),
                "target_output": output.tolist(),
                "tolerance_limit": float(np.max(atol + rtol * np.abs(expected))),
                "within_tolerance": result.status == "PASS",
                "class_agreement": result.agreement,
                **asdict(result),
            }
        )
    failures = sum(c["status"] != "PASS" for c in cases)
    report = {
        "schema_version": "1.0.0",
        "model": {
            "id": runtime.manifest.model_id,
            "version": runtime.manifest.model_version,
            "sha256": runtime.manifest.model_sha256,
        },
        "runtimes": [
            {
                "id": "python-onnxruntime-cpu",
                "provider": "CPUExecutionProvider",
                "version": ort.__version__,
                "status": "executed",
            }
        ],
        "policy": {
            "atol": atol,
            "rtol": rtol,
            "formula": "abs(a-b) <= atol + rtol * abs(reference)",
        },
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "failures": failures,
            "passes": len(cases) - failures,
            "max_absolute_difference": max(
                (c["max_absolute_difference"] for c in cases), default=0
            ),
            "status": "PASS" if failures == 0 else "FAIL",
        },
        "benchmark": None,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    validate_schema(report, "report")
    for case in cases:
        validate_schema(case, "numerical-comparison")
    return report
