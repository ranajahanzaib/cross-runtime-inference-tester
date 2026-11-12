import { describe, expect, it } from "vitest";
import { compareOutputs, validateManifest } from "../src/comparison";

describe("shared data contract helpers", () => {
  it("accepts a valid manifest and rejects malformed hashes", () => {
    const manifest = {
      schema_version: "1.0.0",
      model_id: "demo",
      model_version: "1.0.0",
      model_sha256: "a".repeat(64),
      model_filename: "model.onnx",
      input_dtype: "float32",
      input_name: "features",
      output_names: ["probabilities"],
      probability_output_name: "probabilities",
      feature_order: ["x"],
      class_labels: [0, 1],
      decision_threshold: 0.5,
    };
    expect(validateManifest(manifest)).toBe(true);
    expect(validateManifest({ ...manifest, model_sha256: "abc" })).toBe(false);
    expect(
      validateManifest({ ...manifest, probability_output_name: "missing" }),
    ).toBe(false);
  });
  it("applies configurable absolute and relative tolerance", () => {
    expect(compareOutputs([0.8], [0.80001], 1e-4, 1e-3).status).toBe("PASS");
    expect(compareOutputs([0.8], [0.81], 1e-4, 1e-3).status).toBe("FAIL");
  });
  it("fails on shape mismatch and non-finite output", () => {
    expect(compareOutputs([1, 2], [1], 1, 1).status).toBe("FAIL");
    expect(compareOutputs([Number.NaN], [1], 1, 1).status).toBe("FAIL");
  });
});
