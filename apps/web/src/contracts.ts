export type Feature = {
  name: string;
  label: string;
  description: string;
  unit: string;
  min: number;
  max: number;
  default: number;
};
export type Fixture = {
  fixture_id: string;
  category: string;
  description: string;
  features: number[] | number[][];
  expected_probabilities: number[] | number[][];
  expected_class: number | number[];
  model_sha256: string;
};
export type Manifest = {
  model_id: string;
  model_version: string;
  model_sha256: string;
  model_filename: string;
  feature_order: string[];
  input_name: string;
  output_names: string[];
  probability_output_name: string;
  decision_threshold: number;
  class_labels: Array<string | number>;
};
export type Comparison = {
  id: string;
  category: string;
  expected: number[];
  actual: number[];
  referenceClasses: number[];
  targetClasses: number[];
  abs: number;
  rel: number;
  limit: number;
  pass: boolean;
  classAgreement: boolean;
};
