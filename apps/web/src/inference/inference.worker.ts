/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/wasm";
import type { WorkerRequest, WorkerResponse } from "./messages";

declare const self: DedicatedWorkerGlobalScope;
type Manifest = {
  schema_version: string;
  feature_schema_version: string;
  model_filename: string;
  model_sha256: string;
  input_name: string;
  output_names: string[];
  probability_output_name: string;
  feature_order: string[];
  decision_threshold: number;
  class_labels: number[];
};
type BrowserRunWorkload = {
  batch_size: number;
  repetitions_per_sample: number;
  samples: number[];
  latency_ms: {
    median: number;
    mean: number;
    p95: number;
    min: number;
    max: number;
  };
  throughput_rows_per_second: number;
};
type BrowserIndependentRun = {
  run_index: number;
  session_initialization_ms: number;
  first_inference_ms: number;
  workloads: BrowserRunWorkload[];
};
let session: ort.InferenceSession | undefined;
let modelUrl: URL | undefined;
let manifest: Manifest | undefined;
let featureLimits: Array<{ min: number; max: number }> = [];
let initMs = 0;
let benchmarkCancelRequested = false;

function orderedTensor(input: number[] | number[][]): ort.Tensor {
  const rows: number[][] = Array.isArray(input[0])
    ? (input as number[][])
    : [input as number[]];
  if (
    !rows.length ||
    rows.some((row) => row.length !== (manifest?.feature_order.length ?? 0))
  )
    throw new Error(
      `Expected rows with ${manifest?.feature_order.length ?? 0} features in manifest order.`,
    );
  if (rows.flat().some((x) => !Number.isFinite(x)))
    throw new Error("Feature values must be finite numbers.");
  for (const row of rows)
    for (const [index, value] of row.entries())
      if (
        featureLimits[index] &&
        (value < featureLimits[index].min || value > featureLimits[index].max)
      )
        throw new Error(
          `${manifest?.feature_order[index]} must be between ${featureLimits[index].min} and ${featureLimits[index].max}.`,
        );
  const data = new Float32Array(rows.flat());
  return new ort.Tensor("float32", data, [rows.length, rows[0].length]);
}

async function digest(
  url: string,
): Promise<{ hash: string; sizeBytes: number }> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Asset download failed (${response.status}): ${url}`);
  const bytes = await response.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return {
    hash: [...new Uint8Array(hash)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join(""),
    sizeBytes: bytes.byteLength,
  };
}

async function initialize(baseUrl: string) {
  const started = performance.now();
  const root = new URL(baseUrl, self.location.href);
  const manifestStarted = performance.now();
  const manifestUrl = new URL(
    "models/subscription-risk-v1/manifest.json",
    root,
  );
  const response = await fetch(manifestUrl);
  if (!response.ok)
    throw new Error(
      `Could not load manifest (${response.status}). Run cross-runtime-inference-tester demo to create local artifacts.`,
    );
  manifest = (await response.json()) as Manifest;
  const manifestLoadMs = performance.now() - manifestStarted;
  if (
    manifest.schema_version !== "1.0.0" ||
    !manifest.input_name ||
    !manifest.probability_output_name ||
    !manifest.output_names?.length ||
    !manifest.model_sha256 ||
    !Array.isArray(manifest.feature_order)
  )
    throw new Error(
      "Model manifest is missing required graph metadata or has an unsupported schema version.",
    );
  if (
    !/^[a-f\d]{64}$/.test(manifest.model_sha256) ||
    !manifest.output_names.includes(manifest.probability_output_name) ||
    !manifest.model_filename.endsWith(".onnx")
  )
    throw new Error(
      "Model manifest contains invalid integrity or output mapping metadata.",
    );
  const schemaStarted = performance.now();
  const featureResponse = await fetch(
    new URL("models/subscription-risk-v1/feature-schema.json", root),
  );
  if (!featureResponse.ok)
    throw new Error(
      "Could not load the feature schema needed to validate browser inputs.",
    );
  const featureSchema = (await featureResponse.json()) as {
    schema_version: string;
    features: Array<{
      name: string;
      min: number;
      max: number;
      default: number;
    }>;
  };
  if (
    featureSchema.schema_version !== manifest.feature_schema_version ||
    featureSchema.features.map((f) => f.name).join("|") !==
      manifest.feature_order.join("|")
  )
    throw new Error(
      "Feature schema order or version does not match the model manifest.",
    );
  if (
    featureSchema.features.some(
      (f) =>
        !Number.isFinite(f.min) ||
        !Number.isFinite(f.max) ||
        !Number.isFinite(f.default) ||
        f.min > f.max ||
        f.default < f.min ||
        f.default > f.max,
    )
  )
    throw new Error("Feature schema has an invalid range or default value.");
  const featureSchemaLoadMs = performance.now() - schemaStarted;
  featureLimits = featureSchema.features.map((f) => ({
    min: f.min,
    max: f.max,
  }));
  const artifactUrl = new URL(
    `models/subscription-risk-v1/${manifest.model_filename}`,
    root,
  );
  modelUrl = artifactUrl;
  const modelStarted = performance.now();
  const modelArtifact = await digest(artifactUrl.href);
  const modelFetchVerifyMs = performance.now() - modelStarted;
  if (modelArtifact.hash !== manifest.model_sha256)
    throw new Error(
      `Model integrity check failed: expected ${manifest.model_sha256}, received ${modelArtifact.hash}.`,
    );
  ort.env.wasm.wasmPaths = new URL("ort/", root).href;
  ort.env.wasm.numThreads = 1;
  const sessionStarted = performance.now();
  session = await ort.InferenceSession.create(artifactUrl.href, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const sessionCreationMs = performance.now() - sessionStarted;
  if (
    session.inputNames[0] !== manifest.input_name ||
    session.outputNames.join("|") !== manifest.output_names.join("|")
  )
    throw new Error("ONNX graph names do not match the published manifest.");
  initMs = performance.now() - started;
  const firstInference = await predict(
    featureSchema.features.map((f) => f.default),
  );
  const timeToFirstUsablePredictionMs = performance.now() - started;
  let wasmSizeBytes: number | null = null;
  let wasmJsSizeBytes: number | null = null;
  try {
    const [wasmHead, wasmJsHead] = await Promise.all([
      fetch(new URL("ort/ort-wasm-simd-threaded.wasm", root), {
        method: "HEAD",
      }),
      fetch(new URL("ort/ort-wasm-simd-threaded.mjs", root)),
    ]);
    const wasmSize = wasmHead.headers.get("content-length");
    if (wasmHead.ok && wasmSize) wasmSizeBytes = Number(wasmSize);
    if (wasmJsHead.ok)
      wasmJsSizeBytes = (await wasmJsHead.arrayBuffer()).byteLength;
  } catch {
    /* Asset size metadata is optional on static hosts. */
  }
  return {
    runtime_id: "browser-onnxruntime-wasm",
    version: __ORT_WEB_VERSION__,
    provider: "wasm",
    initialization_ms: initMs,
    startup_stages_ms: {
      manifest_load: manifestLoadMs,
      feature_schema_load: featureSchemaLoadMs,
      model_fetch_and_hash: modelFetchVerifyMs,
      session_creation_and_wasm_setup: sessionCreationMs,
    },
    first_inference_ms: firstInference.latency_ms,
    time_to_first_usable_prediction_ms: timeToFirstUsablePredictionMs,
    model_sha256: modelArtifact.hash,
    model_size_bytes: modelArtifact.sizeBytes,
    wasm_size_bytes: wasmSizeBytes,
    wasm_js_size_bytes: wasmJsSizeBytes,
    input_name: session.inputNames[0],
    output_names: session.outputNames,
    capabilities: {
      crossOriginIsolated: self.crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      threading: false,
      threading_reason: self.crossOriginIsolated
        ? "Single-thread baseline selected."
        : "Multithreaded WASM requires cross-origin isolation and SharedArrayBuffer.",
    },
  };
}

async function predict(input: number[] | number[][]) {
  if (!session || !manifest)
    throw new Error("Browser model is not initialized.");
  const tensor = orderedTensor(input);
  const started = performance.now();
  const outputs = await session.run({ [manifest.input_name]: tensor });
  const latency = performance.now() - started;
  const output = outputs[manifest.probability_output_name];
  if (
    !output ||
    !(output.data instanceof Float32Array) ||
    output.dims.length !== 2 ||
    output.dims[1] !== 2
  )
    throw new Error(
      "Unsupported model output: expected [batch, 2] float32 probabilities.",
    );
  const probabilities = Array.from(output.data);
  const rows = Array.from({ length: output.dims[0] }, (_, i) =>
    probabilities.slice(i * 2, i * 2 + 2),
  );
  return {
    runtime_id: "browser-onnxruntime-wasm",
    probabilities: rows.length === 1 ? rows[0] : rows,
    classes: rows.map((row) =>
      row[1] >= (manifest?.decision_threshold ?? 0.5) ? 1 : 0,
    ),
    latency_ms: latency,
    initialization_ms: initMs,
    model_sha256: manifest.model_sha256,
    input_shape: tensor.dims,
    input_dtype: "float32",
  };
}

async function measureBlock(
  input: number[] | number[][],
  repetitions: number,
): Promise<number> {
  if (!session || !manifest)
    throw new Error("Browser model is not initialized.");
  const tensor = orderedTensor(input);
  const feed = { [manifest.input_name]: tensor };
  const started = performance.now();
  for (let i = 0; i < repetitions; i++) await session.run(feed);
  return (performance.now() - started) / repetitions;
}

function summarizeLatency(samples: number[], batchSize: number) {
  const values = samples.slice().sort((a, b) => a - b);
  const median = values[Math.floor(values.length * 0.5)] ?? 0;
  return {
    samples: values,
    latency_ms: {
      median,
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      p95:
        values[Math.min(values.length - 1, Math.floor(values.length * 0.95))] ??
        0,
      min: values[0] ?? 0,
      max: values.at(-1) ?? 0,
    },
    throughput_rows_per_second:
      (batchSize * 1000) / (median || Number.MIN_VALUE),
  };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  const respond = (message: WorkerResponse) => self.postMessage(message);
  try {
    if (request.type === "cancel-benchmark") {
      benchmarkCancelRequested = true;
      respond({ id: request.id, ok: true, result: { cancelled: true } });
      return;
    }
    let result: unknown;
    if (request.type === "initialize")
      result = await initialize(request.baseUrl);
    else if (request.type === "predict")
      result = await predict(request.features);
    else if (request.type === "batch") {
      const predictions = [];
      for (const item of request.cases)
        predictions.push({ id: item.id, ...(await predict(item.features)) });
      result = predictions;
    } else if (request.type === "benchmark") {
      benchmarkCancelRequested = false;
      const sizes = request.batch_sizes ?? [1];
      const measurementRepetitions = 20;
      const liveSession = session;
      if (!liveSession || !manifest || !modelUrl)
        throw new Error("Browser model is not initialized.");
      const independentRuns: BrowserIndependentRun[] = [];
      for (
        let runIndex = 0;
        runIndex < 3 && !benchmarkCancelRequested;
        runIndex++
      ) {
        const sessionStarted = performance.now();
        const freshSession = await ort.InferenceSession.create(modelUrl.href, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        const sessionInitializationMs = performance.now() - sessionStarted;
        session = freshSession;
        try {
          const firstInferenceMs = (await predict(request.features)).latency_ms;
          const runWorkloads: BrowserRunWorkload[] = [];
          for (const batchSize of sizes) {
            const rows = Array.from(
              { length: batchSize },
              () => request.features,
            );
            const workload = batchSize === 1 ? request.features : rows;
            for (
              let i = 0;
              i < request.warmup && !benchmarkCancelRequested;
              i++
            )
              await predict(workload);
            if (benchmarkCancelRequested) break;
            const values: number[] = [];
            for (
              let i = 0;
              i < request.iterations && !benchmarkCancelRequested;
              i++
            ) {
              await new Promise<void>((resolve) => setTimeout(resolve, 0));
              if (!benchmarkCancelRequested)
                values.push(
                  await measureBlock(workload, measurementRepetitions),
                );
            }
            if (benchmarkCancelRequested) break;
            runWorkloads.push({
              batch_size: batchSize,
              repetitions_per_sample: measurementRepetitions,
              ...summarizeLatency(values, batchSize),
            });
          }
          independentRuns.push({
            run_index: runIndex + 1,
            session_initialization_ms: sessionInitializationMs,
            first_inference_ms: firstInferenceMs,
            workloads: runWorkloads,
          });
        } finally {
          freshSession.release();
          session = liveSession;
        }
      }
      session = liveSession;
      const workloads = sizes.flatMap((batchSize) => {
        const runs = independentRuns.flatMap((run) =>
          run.workloads.filter((item) => item.batch_size === batchSize),
        );
        const samples = runs.flatMap((item) => item.samples);
        if (!samples.length) return [];
        return [
          {
            batch_size: batchSize,
            repetitions_per_sample: measurementRepetitions,
            independent_runs: runs.length,
            ...summarizeLatency(samples, batchSize),
          },
        ];
      });
      result = {
        runtime_id: "browser-onnxruntime-wasm",
        iterations: request.iterations,
        warmup: request.warmup,
        independent_run_count: independentRuns.length,
        independent_runs: independentRuns,
        model_sha256: manifest.model_sha256,
        workloads,
        cancelled: benchmarkCancelRequested,
      };
    } else if (request.type === "capabilities")
      result = {
        initialized: Boolean(session),
        crossOriginIsolated: self.crossOriginIsolated,
        sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        threading: false,
      };
    else {
      session?.release();
      session = undefined;
      manifest = undefined;
      modelUrl = undefined;
      result = { released: true };
    }
    respond({ id: request.id, ok: true, result });
  } catch (error) {
    respond({
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

export {};
