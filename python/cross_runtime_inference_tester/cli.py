"""First-class command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from cross_runtime_inference_tester.core import (
    ARTIFACT,
    OnnxRuntime,
    benchmark,
    run_verification,
    train_export,
    validate_schema,
    write_reports,
)

app = typer.Typer(
    help="Measure ML inference correctness and performance across runtimes.", no_args_is_help=True
)


@app.command()
def train(
    seed: int = typer.Option(2026, help="Deterministic data and training seed."),
    samples: int = typer.Option(7000, min=100),
    output: Path = ARTIFACT,
) -> None:
    """Train, export, validate, and generate default model artifacts."""
    result = train_export(output, seed, samples)
    print(
        f"[green]Artifacts ready[/green]: {result['artifact_dir']}\n"
        f"Model SHA-256: {result['manifest']['model_sha256']}"
    )


@app.command()
def demo(seed: int = 2026, samples: int = 7000) -> None:
    """Prepare the bundled synthetic model and golden fixtures."""
    train(seed=seed, samples=samples, output=ARTIFACT)


@app.command()
def validate(manifest: Path = ARTIFACT / "manifest.json") -> None:
    """Validate manifest, hash, ONNX graph, and Python CPU runtime compatibility."""
    runtime = OnnxRuntime(manifest)
    print(
        f"[green]Valid[/green] {runtime.manifest.model_id} {runtime.manifest.model_version} "
        f"({runtime.manifest.model_sha256})"
    )


@app.command()
def predict(
    features: str = typer.Argument(..., help="JSON array of ordered numeric features."),
    manifest: Path = ARTIFACT / "manifest.json",
) -> None:
    """Run actual local ONNX Runtime CPU inference."""
    try:
        result = OnnxRuntime(manifest).predict(json.loads(features))
    except (ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    print(json.dumps(result, indent=2))


@app.command()
def verify(
    manifest: Path = ARTIFACT / "manifest.json",
    atol: float = 1e-4,
    rtol: float = 1e-3,
    output: Path = Path("artifacts/reports"),
) -> None:
    """Verify reference fixtures against the actual Python ONNX Runtime execution."""
    report = run_verification(manifest, atol, rtol)
    write_reports(report, output)
    color = "green" if report["summary"]["status"] == "PASS" else "red"
    print(
        f"[{color}]{report['summary']['status']}[/] "
        f"{report['summary']['passes']}/{report['summary']['cases']} cases; reports: {output}"
    )
    if report["summary"]["failures"]:
        raise typer.Exit(1)


@app.command()
def compare(
    manifest: Path = ARTIFACT / "manifest.json",
    atol: float = 1e-4,
    rtol: float = 1e-3,
    output: Path = Path("artifacts/reports"),
) -> None:
    """Run strict local fixture comparison and write JSON, HTML, and Markdown reports."""
    verify(manifest, atol, rtol, output)


@app.command()
def report(
    manifest: Path = ARTIFACT / "manifest.json", output: Path = Path("artifacts/reports")
) -> None:
    """Generate a reproducible compatibility report."""
    result = run_verification(manifest)
    write_reports(result, output)
    print(f"Reports written to {output}")


@app.command("benchmark")
def benchmark_command(
    manifest: Path = ARTIFACT / "manifest.json",
    warmup: int = 10,
    iterations: int = typer.Option(100, min=1),
    independent_runs: int = typer.Option(3, min=1, max=10),
    output: Path = Path("artifacts/benchmarks/python.json"),
) -> None:
    """Measure Python runtime initialization and warm inference latency."""
    result = benchmark(manifest, warmup, iterations, independent_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


@app.command()
def fixtures(manifest: Path = ARTIFACT / "manifest.json") -> None:
    """Show location and count of deterministic model fixtures."""
    path = manifest.parent / "golden-vectors.json"
    if not path.exists():
        raise typer.BadParameter("fixtures missing; run `cross_runtime_inference_tester demo` first")
    data = json.loads(path.read_text())
    print(f"{len(data['fixtures'])} fixtures at {path}")


@app.command()
def models() -> None:
    """List the bundled model if its manifest exists."""
    path = ARTIFACT / "manifest.json"
    if path.exists():
        model = json.loads(path.read_text())
        print(f"{model['model_id']} {model['model_version']} {model['model_sha256']}")
    else:
        print("No model artifacts yet. Run `cross_runtime_inference_tester demo`.")


@app.command()
def doctor() -> None:
    """Inspect prerequisites and generated artifacts."""
    import importlib.util
    import platform
    import sys

    checks = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": importlib.util.find_spec("numpy") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "model_artifact": (ARTIFACT / "model.onnx").exists(),
        "browser_app": (Path.cwd() / "apps/web/package.json").exists(),
        "playwright_package": (Path.cwd() / "apps/web/node_modules/@playwright/test").exists(),
        "onnxruntime_web_wasm": (
            Path.cwd() / "apps/web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"
        ).exists(),
    }
    print(json.dumps(checks, indent=2))


@app.command()
def register(
    model: Path, manifest: Path, output: Path = Path("artifacts/models/registered")
) -> None:
    """Validate a trusted numerical float32 ONNX model and register its files locally."""
    import shutil

    from cross_runtime_inference_tester.core import Manifest, sha256

    if model.stat().st_size > 100 * 1024 * 1024:
        raise typer.BadParameter("model exceeds 100 MiB registration limit")
    try:
        metadata = Manifest.model_validate_json(manifest.read_text())
        validate_schema(metadata.model_dump(), "manifest")
        graph = __import__("onnx").load(model)
        __import__("onnx").checker.check_model(graph)
    except Exception as error:
        raise typer.BadParameter(f"unsupported model or manifest: {error}") from error
    if metadata.model_sha256 != sha256(model):
        raise typer.BadParameter("manifest model_sha256 does not match the supplied model")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, output / metadata.model_filename)
    shutil.copy2(manifest, output / "manifest.json")
    OnnxRuntime(output / "manifest.json")
    print(f"Registered validated model in {output}")


if __name__ == "__main__":
    app()
