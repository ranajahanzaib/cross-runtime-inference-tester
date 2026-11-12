# Security policy

Report vulnerabilities privately to the repository maintainers. Do not submit sensitive details in a public issue.

Cross-Runtime Inference Tester validates manifests, tensor dimensions, finite values, and model hashes. The demo performs browser inference locally after downloading static app/model/WASM assets. It has no telemetry or remote prediction endpoint. User entered values are not persisted.

ONNX models and converter toolchains are still untrusted input. Register only trusted models; model graph validation is not a sandbox against malicious artifacts. Registration is limited to 100 MiB and supported float32 numerical tensor contracts.
