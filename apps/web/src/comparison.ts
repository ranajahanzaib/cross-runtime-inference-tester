export function compareOutputs(
  reference: number[],
  target: number[],
  atol: number,
  rtol: number,
) {
  if (reference.length !== target.length || !reference.length)
    return {
      status: "FAIL" as const,
      maxAbsolute: Infinity,
      violations: Math.max(reference.length, target.length),
    };
  if ([...reference, ...target].some((x) => !Number.isFinite(x)))
    return { status: "FAIL" as const, maxAbsolute: Infinity, violations: 1 };
  const differences = reference.map((a, i) => Math.abs(a - target[i]));
  const violations = differences.filter(
    (d, i) => d > atol + rtol * Math.abs(reference[i]),
  ).length;
  return {
    status: violations ? ("FAIL" as const) : ("PASS" as const),
    maxAbsolute: Math.max(...differences),
    violations,
  };
}

export function validateManifest(value: unknown): value is Record<
  string,
  unknown
> & {
  schema_version: string;
  model_id: string;
  model_version: string;
  model_sha256: string;
  model_filename: string;
  input_dtype: string;
  input_name: string;
  output_names: string[];
  probability_output_name: string;
  feature_order: string[];
  class_labels: Array<string | number>;
  decision_threshold: number;
} {
  if (!value || typeof value !== "object") return false;
  const manifest = value as Record<string, unknown>;
  return (
    manifest.schema_version === "1.0.0" &&
    typeof manifest.model_sha256 === "string" &&
    /^[a-f\d]{64}$/.test(manifest.model_sha256) &&
    manifest.input_dtype === "float32" &&
    typeof manifest.model_id === "string" &&
    typeof manifest.model_version === "string" &&
    typeof manifest.model_filename === "string" &&
    manifest.model_filename.endsWith(".onnx") &&
    typeof manifest.input_name === "string" &&
    typeof manifest.probability_output_name === "string" &&
    Array.isArray(manifest.output_names) &&
    manifest.output_names.includes(manifest.probability_output_name) &&
    Array.isArray(manifest.class_labels) &&
    manifest.class_labels.length >= 2 &&
    typeof manifest.decision_threshold === "number" &&
    manifest.decision_threshold >= 0 &&
    manifest.decision_threshold <= 1 &&
    Array.isArray(manifest.feature_order) &&
    manifest.feature_order.length > 0 &&
    manifest.feature_order.every((item) => typeof item === "string")
  );
}
