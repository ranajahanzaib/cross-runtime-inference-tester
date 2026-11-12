# Model contract

The manifest schema is in `packages/schemas/manifest.schema.json`; the default model's feature schema is generated at `artifacts/models/subscription-risk-v1/feature-schema.json`.

Feature order: tenure months, active days in 30 days, support tickets in 90 days, failed payments in 12 months, mean session minutes, monthly plan USD, feature adoption percent, days since last login. The manifest describes the ONNX graph's input name, float32 dtype, dynamic batch shape, output names/types, class order, threshold, opsets, hashes, library versions, and provenance.

Golden vector data stores canonical float32 features, shape, model hash, expected outputs, and source runtime. Invalid inputs are rejected independently from inference. The UI currently understands the default binary `[batch, 2]` probabilities contract; Python can validate generic float32 input and compare numeric arrays.
