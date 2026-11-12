# Architecture

## Data and artifact flow

The seeded generator creates synthetic account features and probabilistic labels. A stratified split separates 70% training, 10% validation, and 20% held-out test rows. The pipeline fits `StandardScaler` and `LogisticRegression` on the training partition only. Metrics are recorded independently for all three partitions.

`skl2onnx` serializes the complete pipeline with float32 dynamic-batch input and ZipMap disabled. The exporter validates the ONNX graph, inspects actual graph metadata, executes the model through Python ONNX Runtime CPU, hashes the artifact, and records the manifest. Held-out vectors are canonicalized to float32 before inference and stored with actual reference probabilities.

The Python adapter verifies the hash before session creation, pins `CPUExecutionProvider`, inspects names and types, validates inputs, and returns output and timing metadata. The browser worker fetches the same static model, hashes its bytes using Web Crypto, loads ONNX Runtime Web's WASM provider, verifies graph names, creates a float32 tensor, and returns probabilities. On startup it records manifest/schema/model fetch and verification, session creation, a real first inference, and time to first usable result. It then runs all held-out, curated, and batch fixtures sequentially through that same worker and ONNX session. It has no prediction server dependency.

The comparison module applies absolute and relative tolerance per output, verifies output structure and finiteness, and checks class agreement. Reports preserve model identity, runtime versions, policy, per-case results, and environment. Browser measurements are computed from actual browser executions and are not substituted with Python timing.

## Trust boundaries

Manifests and inputs are validated. Model hash checks establish artifact identity, not model safety. ONNX models and converter packages are still untrusted. Registration avoids arbitrary Python model imports and is restricted to local trusted float32 tensor artifacts.
