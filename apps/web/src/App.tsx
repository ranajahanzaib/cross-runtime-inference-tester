import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { compareOutputs, validateManifest } from "./comparison";
import type { Comparison, Feature, Fixture, Manifest } from "./contracts";
import { InferenceClient } from "./inference/client";

const featurePath = "/models/subscription-risk-v1/feature-schema.json";
const fixturePath = "/models/subscription-risk-v1/golden-vectors.json";
const manifestPath = "/models/subscription-risk-v1/manifest.json";

function fmt(n: number, digits = 7) {
  return Number.isFinite(n) ? n.toPrecision(digits) : "—";
}
function median(values: number[]) {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
}
function browserIdentity() {
  const ua = navigator.userAgent;
  const match =
    ua.match(/Edg\/([\d.]+)/) ??
    ua.match(/Firefox\/([\d.]+)/) ??
    ua.match(/Chrome\/([\d.]+)/) ??
    ua.match(/Version\/([\d.]+).*Safari/);
  const name = /Edg\//.test(ua)
    ? "Edge"
    : /Firefox\//.test(ua)
      ? "Firefox"
      : /Chrome\//.test(ua)
        ? "Chrome"
        : /Safari\//.test(ua)
          ? "Safari"
          : "Unknown";
  return { name, version: match?.[1] ?? "unknown" };
}

export function App() {
  const [manifest, setManifest] = useState<Manifest>();
  const [features, setFeatures] = useState<Feature[]>([]);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [runtime, setRuntime] = useState<InferenceClient>();
  const [ready, setReady] = useState(false);
  const [runtimeInfo, setRuntimeInfo] = useState<Record<string, unknown>>({});
  const [error, setError] = useState("");
  const [active, setActive] = useState("Overview");
  const [pred, setPred] = useState<Record<string, unknown>>();
  const [inputs, setInputs] = useState<number[]>([]);
  const [atol, setAtol] = useState(0.0001);
  const [rtol, setRtol] = useState(0.001);
  const [selected, setSelected] = useState("golden-0000");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"absolute" | "relative">("absolute");
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [bench, setBench] = useState<Record<string, unknown>>();
  const [pythonBench, setPythonBench] = useState<Record<string, unknown>>();
  const [busy, setBusy] = useState(false);
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchmarkCancelled, setBenchmarkCancelled] = useState(false);
  const [failure, setFailure] = useState("feature-order");
  const [failureResult, setFailureResult] = useState(
    "No fault has been run yet.",
  );

  useEffect(() => {
    const client = new InferenceClient();
    setRuntime(client);
    Promise.all([
      fetch(manifestPath).then((r) => {
        if (!r.ok)
          throw new Error(
            "Model artifacts are unavailable. Run `cross-runtime-inference-tester demo` first.",
          );
        return r.json();
      }),
      fetch(featurePath).then((r) => r.json()),
      fetch(fixturePath).then((r) => r.json()),
    ])
      .then(async ([m, schema, vectors]) => {
        if (!validateManifest(m))
          throw new Error(
            "Model manifest failed the browser contract checks. Recreate the artifacts with `cross-runtime-inference-tester demo`.",
          );
        if (
          !Array.isArray(schema.features) ||
          schema.features.length !== m.feature_order.length ||
          schema.features.some(
            (feature: Feature, index: number) =>
              feature.name !== m.feature_order[index] ||
              !Number.isFinite(feature.min) ||
              !Number.isFinite(feature.max) ||
              feature.min > feature.max,
          )
        )
          throw new Error(
            "Feature schema does not match the model manifest or contains invalid ranges.",
          );
        if (
          !Array.isArray(vectors.fixtures) ||
          !vectors.fixtures.length ||
          vectors.fixtures.some(
            (fixture: Fixture) =>
              fixture.model_sha256 !== m.model_sha256 ||
              !Array.isArray(fixture.features) ||
              !Array.isArray(fixture.expected_probabilities),
          )
        )
          throw new Error(
            "Golden fixture contract is invalid or references a different model hash.",
          );
        const loadedManifest = m as Manifest;
        setManifest(loadedManifest);
        setFeatures(schema.features as Feature[]);
        setFixtures(vectors.fixtures as Fixture[]);
        setInputs((schema.features as Feature[]).map((f) => f.default));
        const result = await client.call<Record<string, unknown>>({
          type: "initialize",
          baseUrl: "/",
        });
        setRuntimeInfo(result);
        setReady(true);
        const curated = vectors.fixtures as Fixture[];
        const actual = await client.call<
          Array<{ id: string; probabilities: number[] | number[][] }>
        >({
          type: "batch",
          cases: curated.map((f) => ({
            id: f.fixture_id,
            features: f.features,
          })),
        });
        const byId = new Map(actual.map((a) => [a.id, a.probabilities]));
        setComparisons(
          curated.map((f) => {
            const expected = (
              Array.isArray(f.expected_probabilities[0])
                ? (f.expected_probabilities as number[][]).flat()
                : f.expected_probabilities
            ) as number[];
            const result = byId.get(f.fixture_id) ?? [];
            const got = (
              Array.isArray(result[0]) ? (result as number[][]).flat() : result
            ) as number[];
            const diffs = expected.map((a, i) =>
              Math.abs(a - (got[i] ?? Number.NaN)),
            );
            const limits = expected.map((a) => atol + rtol * Math.abs(a));
            const max = Math.max(...diffs);
            const limit = Math.min(...limits);
            const expectedRows = Array.isArray(f.expected_probabilities[0])
              ? (f.expected_probabilities as number[][])
              : [f.expected_probabilities as number[]];
            const actualRows = Array.isArray(result[0])
              ? (result as number[][])
              : [result as number[]];
            const expectedClasses = Array.isArray(f.expected_class)
              ? f.expected_class
              : [f.expected_class];
            const targetClasses = expectedRows.map((row, i) => {
              const target = actualRows[i] ?? [];
              const targetClass =
                row.length === 2
                  ? target[1] >= loadedManifest.decision_threshold
                    ? loadedManifest.class_labels[1]
                    : loadedManifest.class_labels[0]
                  : loadedManifest.class_labels[
                      target.indexOf(Math.max(...target))
                    ];
              return Number(targetClass);
            });
            const referenceClasses = expectedClasses.map(Number);
            const classAgreement = referenceClasses.every(
              (classId, i) => classId === targetClasses[i],
            );
            return {
              id: f.fixture_id,
              category: f.category,
              expected,
              actual: got,
              abs: max,
              rel: Math.max(
                ...diffs.map(
                  (d, i) =>
                    d / Math.max(Math.abs(expected[i]), Number.MIN_VALUE),
                ),
              ),
              limit: limits[diffs.indexOf(max)] ?? limit,
              pass: diffs.every((d, i) => d <= limits[i]) && classAgreement,
              classAgreement,
              referenceClasses,
              targetClasses,
            };
          }),
        );
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    return () => client.terminate();
  }, []);

  useEffect(() => {
    fetch("/benchmarks/python.json")
      .then((response) => (response.ok ? response.json() : undefined))
      .then((value) => {
        if (value) setPythonBench(value as Record<string, unknown>);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    setComparisons((items) =>
      items.map((item) => ({
        ...item,
        limit: Math.min(...item.expected.map((v) => atol + rtol * Math.abs(v))),
        pass:
          item.classAgreement &&
          item.expected.every(
            (v, i) => Math.abs(v - item.actual[i]) <= atol + rtol * Math.abs(v),
          ),
      })),
    );
  }, [atol, rtol]);
  const selectedFixture = fixtures.find((f) => f.fixture_id === selected);
  const filtered = useMemo(
    () =>
      comparisons
        .filter(
          (c) =>
            c.id.toLowerCase().includes(search.toLowerCase()) &&
            (filter === "all" ||
              c.category === filter ||
              (filter === "pass"
                ? c.pass
                : filter === "fail"
                  ? !c.pass
                  : filter === "threshold-sensitive"
                    ? c.id === "threshold-near"
                    : false)),
        )
        .sort((a, b) =>
          sortBy === "absolute" ? b.abs - a.abs : b.rel - a.rel,
        ),
    [comparisons, search, filter, sortBy],
  );
  const failed = comparisons.filter((c) => !c.pass).length;
  const maxError = Math.max(0, ...comparisons.map((c) => c.abs));
  const classMatrix = [
    [0, 0],
    [0, 0],
  ];
  comparisons.forEach((comparison) =>
    comparison.referenceClasses.forEach((referenceClass, index) => {
      const targetClass = comparison.targetClasses[index];
      if (
        (referenceClass === 0 || referenceClass === 1) &&
        (targetClass === 0 || targetClass === 1)
      )
        classMatrix[referenceClass][targetClass]++;
    }),
  );
  const histogram = useMemo(
    () =>
      Array.from({ length: 9 }, (_, i) => ({
        band: i === 8 ? "≥8e-7" : `${i}e-7`,
        count: comparisons.filter(
          (c) => c.abs >= i * 1e-7 && c.abs < (i + 1) * 1e-7,
        ).length,
      })),
    [comparisons],
  );
  const browserWorkloads =
    (bench?.workloads as
      | Array<{
          batch_size: number;
          samples: number[];
          latency_ms: { median: number; p95: number };
          throughput_rows_per_second: number;
        }>
      | undefined) ?? [];
  const pythonRunWorkloads =
    (pythonBench?.independent_runs as
      | Array<{
          batch_workloads: Array<{
            batch_size: number;
            latency_ms: { median: number; p95: number };
            throughput_rows_per_second: number | null;
          }>;
        }>
      | undefined) ?? [];
  const pythonWorkloads = pythonRunWorkloads.length
    ? [1, 8, 32, 128].map((size) => {
        const items = pythonRunWorkloads.flatMap((run) =>
          run.batch_workloads.filter((item) => item.batch_size === size),
        );
        return {
          batch_size: size,
          latency_ms: {
            median: median(items.map((item) => item.latency_ms.median)),
            p95: median(items.map((item) => item.latency_ms.p95)),
          },
          throughput_rows_per_second: items.length
            ? items.reduce(
                (sum, item) => sum + (item.throughput_rows_per_second ?? 0),
                0,
              ) / items.length
            : 0,
        };
      })
    : ((pythonBench?.batch_workloads as
        | Array<{
            batch_size: number;
            latency_ms: { median: number; p95: number };
            throughput_rows_per_second: number | null;
          }>
        | undefined) ?? []);
  const batchChartData = [1, 8, 32, 128].map((size) => ({
    batch_size: size,
    browser_median: browserWorkloads.find((w) => w.batch_size === size)
      ?.latency_ms.median,
    browser_p95: browserWorkloads.find((w) => w.batch_size === size)?.latency_ms
      .p95,
    python_median: pythonWorkloads.find((w) => w.batch_size === size)
      ?.latency_ms.median,
  }));
  const pythonLatencyRuns = pythonRunWorkloads
    .map((run) =>
      run.batch_workloads.find((workload) => workload.batch_size === 1),
    )
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const runtimePercentiles = [
    pythonBench && {
      runtime: "Python",
      median: pythonLatencyRuns.length
        ? median(pythonLatencyRuns.map((run) => run.latency_ms.median))
        : Number((pythonBench.latency_ms as Record<string, number>).median),
      p95: pythonLatencyRuns.length
        ? median(pythonLatencyRuns.map((run) => run.latency_ms.p95))
        : Number((pythonBench.latency_ms as Record<string, number>).p95),
    },
    bench &&
      browserWorkloads.find((w) => w.batch_size === 1) && {
        runtime: "Browser",
        median: browserWorkloads.find((w) => w.batch_size === 1)!.latency_ms
          .median,
        p95: browserWorkloads.find((w) => w.batch_size === 1)!.latency_ms.p95,
      },
  ].filter((entry): entry is { runtime: string; median: number; p95: number } =>
    Boolean(entry),
  );
  const browserStages = runtimeInfo.startup_stages_ms as
    | Record<string, number>
    | undefined;
  const startupSegments = [
    pythonBench && {
      runtime: "Python",
      python_session_setup: Number(pythonBench.initialization_ms),
      python_first_inference: Number(pythonBench.first_inference_ms),
    },
    browserStages && {
      runtime: "Browser",
      manifest: browserStages.manifest_load,
      feature_schema: browserStages.feature_schema_load,
      model_fetch_hash: browserStages.model_fetch_and_hash,
      wasm_session_setup: browserStages.session_creation_and_wasm_setup,
      browser_first_inference: Number(runtimeInfo.first_inference_ms),
    },
  ].filter(Boolean);

  const runPredict = async () => {
    if (!runtime) return;
    setBusy(true);
    setError("");
    try {
      setPred(await runtime.call({ type: "predict", features: inputs }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const runBenchmark = async () => {
    if (!runtime) return;
    setBusy(true);
    setBenchmarking(true);
    setBenchmarkCancelled(false);
    try {
      const result = await runtime.call<Record<string, unknown>>({
        type: "benchmark",
        features: features.map((feature) => feature.default),
        warmup: 10,
        iterations: 100,
        batch_sizes: [1, 8, 32, 128],
      });
      if (result.cancelled) setBenchmarkCancelled(true);
      else setBench(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setBenchmarking(false);
    }
  };
  const cancelBenchmark = () => {
    if (runtime && benchmarking)
      void runtime.call({ type: "cancel-benchmark" });
  };
  const downloadReport = () => {
    const appAssets = performance
      .getEntriesByType("resource")
      .map((entry) => {
        const timing = entry as PerformanceResourceTiming;
        return {
          path: new URL(entry.name).pathname,
          bytes: timing.decodedBodySize || timing.encodedBodySize,
          encoded_bytes: timing.encodedBodySize,
        };
      })
      .filter((asset) => asset.path.startsWith("/assets/") && asset.bytes > 0);
    const appAssetBytes = appAssets.length
      ? appAssets.reduce((total, asset) => total + asset.bytes, 0)
      : null;
    const appTransferBytes = appAssets.some((asset) => asset.encoded_bytes > 0)
      ? appAssets.reduce((total, asset) => total + asset.encoded_bytes, 0)
      : null;
    const measuredSizes = [
      runtimeInfo.model_size_bytes,
      runtimeInfo.wasm_size_bytes,
      runtimeInfo.wasm_js_size_bytes,
    ];
    const totalDeployedAssetBytes =
      appAssetBytes === null ||
      measuredSizes.some((size) => typeof size !== "number")
        ? null
        : appAssetBytes +
          measuredSizes.reduce<number>(
            (total, size) => total + Number(size),
            0,
          );
    const report = {
      schema_version: "1.0.0",
      model: manifest
        ? {
            id: manifest.model_id,
            version: manifest.model_version,
            sha256: manifest.model_sha256,
          }
        : null,
      runtimes: [
        {
          id: "python-onnxruntime-cpu",
          source: "stored reference predictions",
        },
        {
          id: "browser-onnxruntime-wasm",
          version: runtimeInfo.version,
          provider: runtimeInfo.provider,
          status: ready ? "executed" : "unavailable",
        },
      ],
      policy: {
        atol,
        rtol,
        formula: "abs(a-b) <= atol + rtol * abs(reference)",
      },
      cases: comparisons.map((c) => {
        const fixture = fixtures.find((f) => f.fixture_id === c.id);
        return {
          case_id: c.id,
          category: c.category,
          features: fixture?.features,
          reference_runtime: "python-onnxruntime-cpu",
          target_runtime: "browser-onnxruntime-wasm",
          reference_output: c.expected,
          target_output: c.actual,
          max_absolute_difference: c.abs,
          max_relative_difference: c.rel,
          tolerance_limit: c.limit,
          within_tolerance: c.pass,
          class_agreement: c.classAgreement,
          status: c.pass && c.classAgreement ? "PASS" : "FAIL",
        };
      }),
      summary: {
        cases: comparisons.length,
        failures: comparisons.filter((c) => !c.pass || !c.classAgreement)
          .length,
        passes: comparisons.filter((c) => c.pass && c.classAgreement).length,
        max_absolute_difference: maxError,
        status:
          comparisons.length &&
          comparisons.every((c) => c.pass && c.classAgreement)
            ? "PASS"
            : "FAIL",
      },
      benchmark: bench ?? null,
      startup: {
        initialization_ms: runtimeInfo.initialization_ms,
        stages_ms: runtimeInfo.startup_stages_ms,
        first_inference_ms: runtimeInfo.first_inference_ms,
        time_to_first_usable_prediction_ms:
          runtimeInfo.time_to_first_usable_prediction_ms,
      },
      environment: {
        browser: browserIdentity(),
        provider: runtimeInfo.provider,
        crossOriginIsolated:
          runtimeInfo.capabilities &&
          (runtimeInfo.capabilities as Record<string, unknown>)
            .crossOriginIsolated,
        hardwareConcurrency:
          runtimeInfo.capabilities &&
          (runtimeInfo.capabilities as Record<string, unknown>)
            .hardwareConcurrency,
        wasm_size_bytes: runtimeInfo.wasm_size_bytes,
        wasm_js_size_bytes: runtimeInfo.wasm_js_size_bytes,
        model_size_bytes: runtimeInfo.model_size_bytes,
        application_bundle_asset_bytes: appAssetBytes,
        application_bundle_encoded_bytes: appTransferBytes,
        total_deployed_asset_bytes: totalDeployedAssetBytes,
        app_asset_size_availability:
          appAssetBytes === null
            ? "unavailable from this browser context"
            : "measured from decoded resource sizes",
        total_asset_size_availability:
          totalDeployedAssetBytes === null
            ? "unavailable because at least one asset size was not exposed"
            : "measured uncompressed deployed asset sizes combined",
        threading: false,
        cache_configuration:
          "fresh application page; browser HTTP cache behavior is not controlled",
        navigation_timing: performance
          .getEntriesByType("navigation")
          .map((entry) => ({
            dom_content_loaded_ms: (entry as PerformanceNavigationTiming)
              .domContentLoadedEventEnd,
            load_ms: (entry as PerformanceNavigationTiming).loadEventEnd,
            response_ms: (entry as PerformanceNavigationTiming).responseEnd,
          })),
        app_assets: appAssets,
      },
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "cross-runtime-inference-tester-browser-report.json";
    link.click();
    URL.revokeObjectURL(url);
  };
  const runFailure = async () => {
    const fixture =
      selectedFixture ?? fixtures.find((f) => f.category === "held-out-test");
    if (!manifest || !runtime || !fixture) {
      setFailureResult(
        "Cannot run the fault until the browser runtime and fixtures are ready.",
      );
      return;
    }
    if (failure === "model-hash") {
      setFailureResult(
        `FAIL · rejected before comparison: fixture model hash ${"0".repeat(64)} does not match active artifact ${manifest.model_sha256}.`,
      );
      return;
    }
    const values = (fixture.features as number[]).slice();
    const expected = fixture.expected_probabilities as number[];
    if (failure === "threshold") {
      const boundaryCase =
        fixtures.find((f) => f.fixture_id === "threshold-near") ?? fixture;
      const boundaryProbabilities =
        boundaryCase.expected_probabilities as number[];
      const positiveProbability = boundaryProbabilities[1];
      const baseline =
        positiveProbability >= manifest.decision_threshold ? 1 : 0;
      const injectedThreshold =
        baseline === 1
          ? (positiveProbability + 1) / 2
          : positiveProbability / 2;
      const injectedClass = positiveProbability >= injectedThreshold ? 1 : 0;
      setFailureResult(
        `FAIL · injected threshold ${injectedThreshold.toPrecision(5)} predicts class ${injectedClass}; manifest threshold ${manifest.decision_threshold} predicts class ${baseline}. Raw probabilities remain unchanged.`,
      );
      return;
    }
    [values[0], values[1]] = [values[1], values[0]];
    try {
      const actual = await runtime.call<{ probabilities: number[] }>({
        type: "predict",
        features: values,
      });
      const result = compareOutputs(expected, actual.probabilities, atol, rtol);
      setFailureResult(
        `${result.status} · feature positions 0 and 1 swapped. Max absolute difference ${result.maxAbsolute.toExponential(4)} across ${fixture.fixture_id}; ${result.violations} values outside tolerance.`,
      );
    } catch (e) {
      setFailureResult(
        `ERROR · injected feature-order fault failed execution: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">EP</div>
          <div>
            <b>Cross-Runtime Inference Tester</b>
            <small>INFERENCE LAB</small>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {[
            "Overview",
            "Live inference",
            "Parity explorer",
            "Runtime comparison",
            "Performance lab",
            "Failure lab",
            "Artifacts",
            "Methodology",
          ].map((name, i) => (
            <button
              key={name}
              className={`nav-item ${active === name ? "active" : ""}`}
              onClick={() => {
                setActive(name);
                document
                  .getElementById(
                    (
                      {
                        Overview: "overview",
                        "Live inference": "inference",
                        "Parity explorer": "parity",
                        "Runtime comparison": "runtimes",
                        "Performance lab": "performance",
                        "Failure lab": "failure",
                        Artifacts: "artifacts",
                        Methodology: "methodology",
                      } as Record<string, string>
                    )[name],
                  )
                  ?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
            >
              <span className="nav-symbol">
                {["◫", "⌁", "⊞", "⇄", "◷", "⚑", "▤", "ⓘ"][i]}
              </span>
              {name}
              {name === "Overview" && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-mark">
            <span />
            Local execution
          </div>
          <p>No prediction data leaves this browser.</p>
          <div className="version">
            EDGEPARITY&nbsp;&nbsp;·&nbsp;&nbsp;0.1.0
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="crumb">
            Workspace <span>/</span> {active}
          </div>
          <div className="top-right">
            <span className={`runtime-pill ${ready ? "online" : ""}`}>
              <i />
              {ready ? "WASM runtime ready" : "Runtime initializing"}
            </span>
            <button
              className="quiet-button"
              onClick={() => setActive("Methodology")}
            >
              Methodology ↗
            </button>
          </div>
        </header>
        <div className="content">
          <div id="overview" className="eyebrow">
            MODEL COMPATIBILITY / BINARY CLASSIFICATION
          </div>
          <div className="title-row">
            <div>
              <h1>
                {active === "Overview" ? "Runtime parity overview" : active}
              </h1>
              <p className="subtitle">
                One model. Multiple runtimes. Verified predictions.
              </p>
            </div>
            <div
              className={`status-chip ${ready ? "status-pass" : "status-warn"}`}
            >
              <span className="status-icon">{ready ? "✓" : "…"}</span>
              {ready ? "RUNTIME ACTIVE" : "INITIALIZING"}
            </div>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <b>Runtime issue</b>
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          <section className="model-strip">
            <div className="model-symbol">Σ</div>
            <div className="model-name">
              <small>ACTIVE MODEL</small>
              <b>
                {manifest?.model_id ?? "subscription-risk"}{" "}
                <span className="tag">v{manifest?.model_version ?? "—"}</span>
              </b>
              <span className="hash">
                SHA-256 · {manifest?.model_sha256?.slice(0, 16) ?? "loading"}…
              </span>
            </div>
            <div className="strip-divider" />
            <div className="model-detail">
              <small>MODEL TYPE</small>
              <b>Logistic regression</b>
            </div>
            <div className="model-detail">
              <small>INPUT CONTRACT</small>
              <b>8 × float32</b>
            </div>
            <div className="model-detail">
              <small>ARTIFACT SIZE</small>
              <b>
                {runtimeInfo.model_size_bytes
                  ? `${runtimeInfo.model_size_bytes} B · ONNX`
                  : "—"}
              </b>
            </div>
            <a className="strip-link" href={manifestPath} download>
              Manifest ↗
            </a>
          </section>
          <section className="metrics-grid">
            <Metric
              label="FIXTURES COMPARED"
              value={comparisons.length ? String(comparisons.length) : "—"}
              foot="Python ORT ↔ browser WASM"
              mark="#"
            />
            <Metric
              label="WITHIN TOLERANCE"
              value={
                comparisons.length
                  ? `${comparisons.length - failed}/${comparisons.length}`
                  : "—"
              }
              foot={`atol ${atol} · rtol ${rtol}`}
              mark="✓"
              good
            />
            <Metric
              label="MAX ABS. DIFFERENCE"
              value={comparisons.length ? maxError.toExponential(2) : "—"}
              foot="Across compared probability outputs"
              mark="Δ"
            />
            <Metric
              label="CLASS AGREEMENT"
              value={
                comparisons.length
                  ? `${Math.round((comparisons.filter((c) => c.classAgreement).length / comparisons.length) * 100)}%`
                  : "—"
              }
              foot="Same predicted class at threshold 0.5"
              mark="↔"
              good
            />
          </section>
          <section id="runtimes" className="runtime-flow panel">
            <div className="section-head">
              <div>
                <span className="section-kicker">VERIFICATION PATH</span>
                <h2>Same artifact. Independent execution.</h2>
              </div>
              <span className="live-indicator">
                <i /> LIVE
              </span>
            </div>
            <div className="flow">
              <FlowNode
                icon="sk"
                title="scikit-learn"
                sub="Reference pipeline"
              />
              <span className="flow-line" />
              <FlowNode
                icon="ON"
                title="ONNX export"
                sub="One shared artifact"
                active
              />
              <span className="flow-split">
                <i />
                <i />
              </span>
              <div className="flow-targets">
                <FlowNode
                  icon="Py"
                  title="Python runtime"
                  sub="ONNX Runtime · CPU"
                />
                <FlowNode
                  icon="W"
                  title="Browser runtime"
                  sub="ONNX Runtime · WASM"
                  active
                />
              </div>
              <span className="flow-line" />
              <FlowNode
                icon="Δ"
                title="Parity check"
                sub={
                  comparisons.length
                    ? `${comparisons.length - failed} pass · ${failed} fail`
                    : "Awaiting results"
                }
                active
              />
            </div>
            <div className="flow-note">
              <span>●</span> Input tensors use canonical float32 feature order
              from the model manifest. No server-side prediction endpoint.
            </div>
          </section>
          <div className="work-grid">
            <section id="inference" className="panel inference-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">LIVE INFERENCE LAB</span>
                  <h2>Run in this browser</h2>
                </div>
                <span className="small-tag">LOCAL · WASM</span>
              </div>
              <p className="panel-copy">
                Adjust the synthetic account features and execute the actual
                ONNX model in your browser.
              </p>
              <div className="preset-row">
                <span>PRESETS</span>
                <button
                  onClick={() => setInputs([42, 25, 0, 0, 42, 39, 82, 1])}
                >
                  Low risk
                </button>
                <button onClick={() => setInputs([2, 2, 8, 3, 6, 149, 8, 88])}>
                  Elevated risk
                </button>
                <button
                  onClick={() => setInputs([16, 12, 2, 1, 22, 59, 40, 14])}
                >
                  Borderline
                </button>
              </div>
              <div className="feature-list">
                {features.map((feature, i) => (
                  <label className="feature" key={feature.name}>
                    <span className="feature-title" title={feature.description}>
                      {feature.label}
                      <small>{feature.unit}</small>
                    </span>
                    <input
                      aria-label={feature.label}
                      type="range"
                      min={feature.min}
                      max={feature.max}
                      step={feature.max > 100 ? 1 : 0.1}
                      value={inputs[i] ?? feature.default}
                      onChange={(e) =>
                        setInputs((v) =>
                          v.map((n, j) =>
                            j === i ? Number(e.target.value) : n,
                          ),
                        )
                      }
                    />
                    <input
                      className="number-input"
                      aria-label={`${feature.label} value`}
                      type="number"
                      min={feature.min}
                      max={feature.max}
                      step={feature.max > 100 ? 1 : 0.1}
                      value={inputs[i] ?? feature.default}
                      onChange={(e) =>
                        setInputs((v) =>
                          v.map((n, j) =>
                            j === i ? Number(e.target.value) : n,
                          ),
                        )
                      }
                    />
                  </label>
                ))}
              </div>
              <button
                className="primary-button"
                disabled={!ready || busy}
                onClick={runPredict}
              >
                {busy ? "Running inference…" : "Run browser inference"}{" "}
                <span>→</span>
              </button>
              {pred && (
                <div className="prediction-result" aria-live="polite">
                  <div>
                    <small>BROWSER PREDICTION</small>
                    <b>
                      {(pred.classes as number[])?.[0] === 1
                        ? "At-risk class"
                        : "Lower-risk class"}
                    </b>
                  </div>
                  <div>
                    <small>CLASS 0</small>
                    <b>{fmt((pred.probabilities as number[])?.[0] ?? 0)}</b>
                  </div>
                  <div>
                    <small>CLASS 1</small>
                    <b>{fmt((pred.probabilities as number[])?.[1] ?? 0)}</b>
                  </div>
                  <div>
                    <small>LATENCY</small>
                    <b>{Number(pred.latency_ms).toFixed(3)} ms</b>
                  </div>
                </div>
              )}
              <p className="disclosure">
                Synthetic demonstration only. Outputs are not validated
                real-world cancellation estimates.
              </p>
            </section>
            <section id="parity" className="panel explorer-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">PARITY EXPLORER</span>
                  <h2>Golden fixture results</h2>
                </div>
                <button
                  className="text-button"
                  disabled={!comparisons.length}
                  onClick={downloadReport}
                >
                  Download JSON ↧
                </button>
              </div>
              <p className="panel-copy">
                Actual browser WASM output against committed Python ONNX Runtime
                references.
              </p>
              <div className="explorer-controls">
                <input
                  aria-label="Search fixture"
                  placeholder="Search fixture ID"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select
                  aria-label="Filter result status"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  <option value="all">All cases</option>
                  <option value="pass">Pass</option>
                  <option value="fail">Fail</option>
                  <option value="threshold-sensitive">
                    Threshold-sensitive
                  </option>
                  <option value="held-out-test">Held-out</option>
                  <option value="typical">Typical</option>
                  <option value="boundary">Boundary</option>
                  <option value="numerical">Numerical</option>
                  <option value="batch">Batch</option>
                </select>
                <select
                  aria-label="Sort comparisons"
                  value={sortBy}
                  onChange={(e) =>
                    setSortBy(e.target.value as "absolute" | "relative")
                  }
                >
                  <option value="absolute">Sort by absolute</option>
                  <option value="relative">Sort by relative</option>
                </select>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>CASE</th>
                      <th>REFERENCE · PYTHON ORT</th>
                      <th>TARGET · BROWSER WASM</th>
                      <th>MAX ABS. DIFF</th>
                      <th>TOLERANCE</th>
                      <th>CLASS</th>
                      <th>STATUS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.slice(0, 8).map((row) => (
                      <tr
                        key={row.id}
                        onClick={() => setSelected(row.id)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelected(row.id);
                          }
                        }}
                        tabIndex={0}
                        aria-label={`Inspect ${row.id}`}
                        className={selected === row.id ? "selected-row" : ""}
                      >
                        <td className="mono">
                          {row.id} <small>{row.category}</small>
                        </td>
                        <td className="mono">
                          {fmt(row.expected[1] ?? row.expected[0], 5)}
                        </td>
                        <td className="mono">
                          {fmt(row.actual[1] ?? row.actual[0], 5)}
                        </td>
                        <td className="mono">{row.abs.toExponential(2)}</td>
                        <td className="mono">{row.limit.toExponential(1)}</td>
                        <td>{row.classAgreement ? "Match" : "Differ"}</td>
                        <td>
                          <Status pass={row.pass} />
                        </td>
                      </tr>
                    ))}
                    {!filtered.length && (
                      <tr>
                        <td colSpan={7}>No cases match these filters.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="tolerance-controls">
                <label>
                  ABSOLUTE TOLERANCE{" "}
                  <input
                    type="number"
                    min="0"
                    step="0.00001"
                    value={atol}
                    onChange={(e) => setAtol(Number(e.target.value))}
                  />
                </label>
                <label>
                  RELATIVE TOLERANCE{" "}
                  <input
                    type="number"
                    min="0"
                    step="0.0001"
                    value={rtol}
                    onChange={(e) => setRtol(Number(e.target.value))}
                  />
                </label>
                <span className="recalc-note">
                  Verdicts recalculate locally
                </span>
              </div>
            </section>
          </div>
          <div className="work-grid lower-grid">
            <section id="parity-chart" className="panel chart-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">NUMERICAL ERROR</span>
                  <h2>Absolute difference distribution</h2>
                </div>
                <span className="unit-label">Probability units</span>
              </div>
              <div className="chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    accessibilityLayer
                    data={histogram}
                    margin={{ top: 10, right: 10, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#e9edf1"
                    />
                    <XAxis
                      dataKey="band"
                      tickLine={false}
                      axisLine={false}
                      fontSize={10}
                    />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      fontSize={10}
                      allowDecimals={false}
                      width={38}
                    />
                    <Tooltip />
                    <Bar
                      dataKey="count"
                      name="Fixtures"
                      fill="#39836e"
                      radius={[3, 3, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-caption">
                <span>
                  <i className="legend-dot" />
                  Browser ↔ Python · {comparisons.length || "—"} fixtures
                </span>
                <span>Worst: {maxError.toExponential(2)}</span>
              </div>
            </section>
            <section className="panel selected-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">CASE INSPECTOR</span>
                  <h2>{selectedFixture?.fixture_id ?? "Choose a fixture"}</h2>
                </div>
                {selectedFixture && (
                  <Status
                    pass={
                      comparisons.find((c) => c.id === selected)?.pass ?? false
                    }
                  />
                )}
              </div>
              <div className="case-meta">
                {selectedFixture?.category ?? "—"} <span>·</span>{" "}
                {selectedFixture?.description ?? "Fixture details appear here"}
              </div>
              <div className="comparison-pairs">
                <div>
                  <small>INPUT · MANIFEST ORDER</small>
                  <code>
                    {JSON.stringify(selectedFixture?.features ?? "—")}
                  </code>
                </div>
                <div>
                  <small>PYTHON ORT REFERENCE</small>
                  <code>
                    {JSON.stringify(
                      selectedFixture?.expected_probabilities ?? "—",
                    )}
                  </code>
                </div>
                <div>
                  <small>BROWSER WASM ACTUAL</small>
                  <code>
                    {JSON.stringify(
                      comparisons.find((c) => c.id === selected)?.actual ??
                        "Awaiting execution",
                    )}
                  </code>
                </div>
              </div>
              <div className="selected-foot">
                <span>
                  Model hash{" "}
                  {selectedFixture?.model_sha256 === manifest?.model_sha256
                    ? "verified"
                    : "mismatch"}{" "}
                  · max Δ{" "}
                  {comparisons
                    .find((c) => c.id === selected)
                    ?.abs.toExponential(3) ?? "—"}
                </span>
                <span>
                  Float32 · threshold {manifest?.decision_threshold} · atol{" "}
                  {atol} / rtol {rtol}
                </span>
              </div>
            </section>
          </div>
          <div className="work-grid lower-grid">
            <section className="panel chart-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">PARITY MATRIX</span>
                  <h2>Reference vs target outcomes</h2>
                </div>
                <span className="unit-label">Python ORT → Browser WASM</span>
              </div>
              <div className="matrix-summary">
                <div>
                  <small>NUMERIC PASS</small>
                  <b>{comparisons.filter((c) => c.pass).length}</b>
                </div>
                <div>
                  <small>CLASS MATCH</small>
                  <b>{comparisons.filter((c) => c.classAgreement).length}</b>
                </div>
                <div>
                  <small>CASES CHECKED</small>
                  <b>{comparisons.length}</b>
                </div>
                <div>
                  <small>FAILURES</small>
                  <b>{failed}</b>
                </div>
              </div>
              <table
                className="parity-matrix"
                aria-label="Reference and target class agreement"
              >
                <thead>
                  <tr>
                    <th rowSpan={2}>REFERENCE CLASS</th>
                    <th colSpan={2}>TARGET CLASS</th>
                  </tr>
                  <tr>
                    <th>0</th>
                    <th>1</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>0</th>
                    <td>{classMatrix[0][0]}</td>
                    <td>{classMatrix[0][1]}</td>
                  </tr>
                  <tr>
                    <th>1</th>
                    <td>{classMatrix[1][0]}</td>
                    <td>{classMatrix[1][1]}</td>
                  </tr>
                </tbody>
              </table>
              <p className="chart-caption">
                Counts are recalculated from all actual output rows in the
                fixtures. No browser threaded runtime is enabled.
              </p>
            </section>
            <section className="panel chart-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">BATCH PERFORMANCE</span>
                  <h2>Measured latency by batch size</h2>
                </div>
                <span className="unit-label">Milliseconds per inference</span>
              </div>
              {bench ? (
                <>
                  <div className="chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        accessibilityLayer
                        data={batchChartData}
                        margin={{ top: 10, right: 12, bottom: 0, left: 8 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          vertical={false}
                          stroke="#e9edf1"
                        />
                        <XAxis
                          dataKey="batch_size"
                          tickLine={false}
                          axisLine={false}
                          fontSize={10}
                        />
                        <YAxis
                          tickLine={false}
                          axisLine={false}
                          fontSize={10}
                          width={42}
                        />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="python_median"
                          name="Python median"
                          stroke="#738294"
                          strokeWidth={2}
                          dot
                          connectNulls={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="browser_median"
                          name="Browser median"
                          stroke="#39836e"
                          strokeWidth={2}
                          dot
                          connectNulls={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="browser_p95"
                          name="Browser p95"
                          stroke="#b27c37"
                          strokeWidth={2}
                          dot
                          connectNulls={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-caption">
                    Lines: Python median · Browser median · Browser p95. Same
                    default row; 50 Python and 100 browser measurements per
                    session at each size (3 sessions).
                  </div>
                  <div className="batch-throughput">
                    <b>Throughput · rows/s</b>
                    {[1, 8, 32, 128].map((size) => (
                      <span key={size}>
                        Batch {size}
                        <small>
                          Python{" "}
                          {pythonWorkloads
                            .find((w) => w.batch_size === size)
                            ?.throughput_rows_per_second?.toLocaleString(
                              undefined,
                              { maximumFractionDigits: 0 },
                            ) ?? "—"}{" "}
                          · Browser{" "}
                          {browserWorkloads
                            .find((w) => w.batch_size === size)
                            ?.throughput_rows_per_second?.toLocaleString(
                              undefined,
                              { maximumFractionDigits: 0 },
                            ) ?? "—"}
                        </small>
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <p className="chart-caption">
                  Run the browser benchmark to measure batches of 1, 8, 32, and
                  128 rows. Python measurements are available from the bundled
                  local benchmark report.
                </p>
              )}
            </section>
          </div>
          <section className="panel chart-panel">
            <div className="section-head">
              <div>
                <span className="section-kicker">RUNTIME COMPARISON</span>
                <h2>Measured runtime and startup characteristics</h2>
              </div>
              <span className="unit-label">No cross-device speed ranking</span>
            </div>
            <div className="runtime-matrix">
              <div>
                <b>Python ONNX Runtime</b>
                <span>
                  CPUExecutionProvider ·{" "}
                  {pythonBench?.environment
                    ? String(
                        (pythonBench.environment as Record<string, unknown>)
                          .onnxruntime,
                      )
                    : "Not measured"}
                </span>
                <span>
                  Initialization{" "}
                  {pythonBench
                    ? `${Number(pythonBench.initialization_ms).toFixed(2)} ms`
                    : "Not measured"}{" "}
                  · first inference{" "}
                  {pythonBench
                    ? `${Number(pythonBench.first_inference_ms).toFixed(3)} ms`
                    : "Not measured"}
                </span>
                <span>
                  Warm median{" "}
                  {pythonBench
                    ? `${Number((pythonBench.latency_ms as Record<string, number>).median).toFixed(4)} ms`
                    : "Not measured"}{" "}
                  · batch sizes {pythonBench ? "1, 8, 32, 128" : "Not measured"}
                </span>
              </div>
              <div>
                <b>Browser ONNX Runtime Web</b>
                <span>
                  WASM · {String(runtimeInfo.version ?? "Not measured")}
                </span>
                <span>
                  Initialization{" "}
                  {runtimeInfo.initialization_ms
                    ? `${Number(runtimeInfo.initialization_ms).toFixed(2)} ms`
                    : "Not measured"}{" "}
                  · first inference{" "}
                  {runtimeInfo.first_inference_ms
                    ? `${Number(runtimeInfo.first_inference_ms).toFixed(3)} ms`
                    : "Not measured"}
                </span>
                <span>
                  Warm median{" "}
                  {bench
                    ? `${Number(browserWorkloads.find((w) => w.batch_size === 1)?.latency_ms.median ?? 0).toFixed(4)} ms`
                    : "Not measured"}{" "}
                  · single thread ·{" "}
                  {bench ? "1, 8, 32, 128" : "batch sizes not measured"}
                </span>
              </div>
              <div>
                <b>Browser WASM multithreaded</b>
                <span>Unsupported in this configuration</span>
                <span>
                  Not run: cross-origin isolated threading adapter is not
                  enabled.
                </span>
              </div>
            </div>
            <div className="runtime-charts">
              <div>
                <h3>Warm latency percentiles (ms)</h3>
                {runtimePercentiles.length ? (
                  <div className="chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        accessibilityLayer
                        data={runtimePercentiles}
                        margin={{ left: 50, right: 8 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="runtime" />
                        <YAxis unit=" ms" width={55} />
                        <Tooltip />
                        <Bar dataKey="median" name="Median" fill="#39836e" />
                        <Bar dataKey="p95" name="p95" fill="#b27c37" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p>Measurements unavailable.</p>
                )}
                <div className="chart-caption">
                  Green · median &nbsp; Amber · p95
                </div>
              </div>
              <div>
                <h3>Startup breakdown (ms)</h3>
                {startupSegments.length ? (
                  <div className="chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        accessibilityLayer
                        data={startupSegments}
                        margin={{ left: 50, right: 8 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="runtime" />
                        <YAxis unit=" ms" width={55} />
                        <Tooltip />
                        <Bar
                          dataKey="python_session_setup"
                          stackId="a"
                          name="Python session setup"
                          fill="#738294"
                        />
                        <Bar
                          dataKey="manifest"
                          stackId="a"
                          name="Browser manifest fetch"
                          fill="#b0c5b8"
                        />
                        <Bar
                          dataKey="feature_schema"
                          stackId="a"
                          name="Browser feature schema fetch"
                          fill="#91b09f"
                        />
                        <Bar
                          dataKey="model_fetch_hash"
                          stackId="a"
                          name="Browser model fetch and hash"
                          fill="#67947e"
                        />
                        <Bar
                          dataKey="wasm_session_setup"
                          stackId="a"
                          name="Browser ONNX session and WASM setup"
                          fill="#39836e"
                        />
                        <Bar
                          dataKey="python_first_inference"
                          stackId="a"
                          name="Python first inference"
                          fill="#d0a266"
                        />
                        <Bar
                          dataKey="browser_first_inference"
                          stackId="a"
                          name="Browser first inference"
                          fill="#b27c37"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p>Measurements unavailable.</p>
                )}
              </div>
            </div>
            <h3>Largest observed fixture differences</h3>
            <ol className="worst-cases">
              {comparisons
                .slice()
                .sort((a, b) => b.abs - a.abs)
                .slice(0, 5)
                .map((c) => (
                  <li key={c.id}>
                    <button
                      onClick={() => {
                        setSelected(c.id);
                        document
                          .getElementById("parity")
                          ?.scrollIntoView({ behavior: "smooth" });
                      }}
                    >
                      {c.id} · {c.category}
                    </button>
                    <code>{c.abs.toExponential(4)}</code>
                  </li>
                ))}
            </ol>
          </section>
          <div className="work-grid lower-grid">
            <section id="performance" className="panel benchmark-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">PERFORMANCE LAB</span>
                  <h2>Warm inference benchmark</h2>
                </div>
                <span className="small-tag">MEASURED HERE</span>
              </div>
              <p className="panel-copy">
                Each batch size uses 100 measured runs after 10 warmups in each
                of 3 fresh ONNX sessions in this browser context. Performance is
                device-specific and informational.
              </p>
              <button
                className="secondary-button"
                disabled={!ready || busy}
                onClick={runBenchmark}
              >
                {bench ? "Run again" : "Run browser benchmark"} →
              </button>
              {benchmarking && (
                <button className="cancel-button" onClick={cancelBenchmark}>
                  Cancel benchmark
                </button>
              )}
              {benchmarkCancelled && (
                <p className="chart-caption" role="status">
                  Benchmark cancelled; completed batch measurements remain
                  available.
                </p>
              )}
              {bench && (
                <div className="benchmark-result">
                  <div>
                    <small>BATCH 1 MEDIAN</small>
                    <b>
                      {Number(
                        (
                          bench.workloads as Array<{
                            batch_size: number;
                            latency_ms: { median: number };
                          }>
                        ).find((w) => w.batch_size === 1)?.latency_ms.median ??
                          0,
                      ).toFixed(3)}{" "}
                      ms
                    </b>
                  </div>
                  <div>
                    <small>BATCH 128 MEDIAN</small>
                    <b>
                      {Number(
                        (
                          bench.workloads as Array<{
                            batch_size: number;
                            latency_ms: { median: number };
                          }>
                        ).find((w) => w.batch_size === 128)?.latency_ms
                          .median ?? 0,
                      ).toFixed(3)}{" "}
                      ms
                    </b>
                  </div>
                  <div>
                    <small>SESSIONS · ITERATIONS EACH</small>
                    <b>
                      {String(bench.independent_run_count)} ·{" "}
                      {String(bench.iterations)}
                    </b>
                  </div>
                </div>
              )}
            </section>
            <section id="failure" className="panel failure-panel">
              <div className="section-head">
                <div>
                  <span className="section-kicker">FAILURE LAB</span>
                  <h2>Prove drift is detected</h2>
                </div>
                <span className="small-tag warn-tag">INJECTED SCENARIO</span>
              </div>
              <p className="panel-copy">
                Run an isolated fault against a real fixture. The canonical
                model and results are not changed.
              </p>
              <div className="failure-controls">
                <select
                  value={failure}
                  onChange={(e) => setFailure(e.target.value)}
                >
                  <option value="feature-order">Feature order swapped</option>
                  <option value="threshold">Decision threshold changed</option>
                  <option value="model-hash">Model hash mismatch</option>
                </select>
                <button className="secondary-button" onClick={runFailure}>
                  Run injected fault →
                </button>
              </div>
              <div className="failure-note" aria-live="polite">
                {failureResult}
              </div>
            </section>
          </div>
          <section id="artifacts" className="panel artifact-panel">
            <div className="section-head">
              <div>
                <span className="section-kicker">ARTIFACT INSPECTOR</span>
                <h2>Model contract and downloads</h2>
              </div>
              <span className="small-tag">HASH VERIFIED</span>
            </div>
            <p className="panel-copy">
              The same versioned ONNX artifact and golden inputs are used by the
              local Python runtime and this browser.
            </p>
            <div className="artifact-detail">
              <span>ONNX input</span>
              <code>
                {manifest?.input_name ?? "—"} · float32 · [batch,{" "}
                {manifest?.feature_order.length ?? "—"}]
              </code>
              <span>Probability output</span>
              <code>
                {manifest?.probability_output_name ?? "—"} · class order{" "}
                {JSON.stringify(manifest?.class_labels ?? [])}
              </code>
              <span>ONNX Runtime Web</span>
              <code>
                {String(runtimeInfo.version ?? "loading")} · WASM · one thread ·{" "}
                {runtimeInfo.wasm_size_bytes
                  ? `${Math.round(Number(runtimeInfo.wasm_size_bytes) / 1024)} KiB WASM`
                  : "WASM size unavailable"}{" "}
                ·{" "}
                {runtimeInfo.wasm_js_size_bytes
                  ? `${Math.round(Number(runtimeInfo.wasm_js_size_bytes) / 1024)} KiB loader`
                  : "loader size unavailable"}
              </code>
              <span>Initialization</span>
              <code>
                {Number(runtimeInfo.initialization_ms ?? 0).toFixed(2)} ms ·
                current browser context
              </code>
              <span>SHA-256</span>
              <code>{manifest?.model_sha256 ?? "—"}</code>
            </div>
            <div className="artifact-links">
              <a href="/models/subscription-risk-v1/model.onnx" download>
                ONNX model ↓
              </a>
              <a href={manifestPath} download>
                Manifest ↓
              </a>
              <a href={featurePath} download>
                Feature schema ↓
              </a>
              <a href={fixturePath} download>
                Golden vectors ↓
              </a>
              <a
                href="/models/subscription-risk-v1/training-metrics.json"
                download
              >
                Training metrics ↓
              </a>
              <a href="/reports/report.json" download>
                Python comparison JSON ↓
              </a>
              <a href="/benchmarks/python.json" download>
                Python benchmark JSON ↓
              </a>
            </div>
          </section>
          <section id="methodology" className="panel methodology-panel">
            <div className="section-head">
              <div>
                <span className="section-kicker">METHODOLOGY</span>
                <h2>What these results mean</h2>
              </div>
              <span className="unit-label">REPRODUCIBLE LOCALLY</span>
            </div>
            <div className="method-grid">
              <p>
                <b>Training</b>
                <br />
                Seeded synthetic rows are split by class into 70% train, 10%
                validation and 20% test. A scaler and logistic classifier are
                exported together.
              </p>
              <p>
                <b>Parity</b>
                <br />
                Stored float32 references from Python ONNX Runtime are checked
                in this browser with absolute and relative tolerance. The
                fixture file preserves the exact inputs and model hash.
              </p>
              <p>
                <b>Measurement</b>
                <br />
                Cold initialization and warm calls have different costs. Browser
                timings describe this device and context; they are not a
                universal runtime ranking.
              </p>
              <p>
                <b>Interpretation</b>
                <br />
                The fictional subscription-risk task is a technical example.
                Predictions are not validated real-world risk estimates.
              </p>
            </div>
          </section>
          <footer className="footer-note">
            <span>EDGE PARITY / SYNTHETIC SUBSCRIPTION RISK DEMO</span>
            <span>Python ORT · Browser WASM · No telemetry</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  foot,
  mark,
  good = false,
}: {
  label: string;
  value: string;
  foot: string;
  mark: string;
  good?: boolean;
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${good ? "green-text" : ""}`}>{value}</div>
      <div className="metric-foot">{foot}</div>
      <div className="metric-mark">{mark}</div>
    </div>
  );
}
function FlowNode({
  icon,
  title,
  sub,
  active = false,
}: {
  icon: string;
  title: string;
  sub: string;
  active?: boolean;
}) {
  return (
    <div className={`flow-node ${active ? "node-active" : ""}`}>
      <span className="flow-icon">{icon}</span>
      <b>{title}</b>
      <small>{sub}</small>
    </div>
  );
}
function Status({ pass }: { pass: boolean }) {
  return (
    <span className={`status-label ${pass ? "pass" : "fail"}`}>
      <i />
      {pass ? "PASS" : "FAIL"}
    </span>
  );
}
