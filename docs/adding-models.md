# Adding models

`cross-runtime-inference-tester register MODEL MANIFEST` accepts a local trusted ONNX file, validates ONNX structure and the strict versioned manifest, verifies SHA-256, enforces a 100 MiB limit, and creates a local artifact directory. It does not execute Python supplied by a model package. Inputs must be numerical float32 tensors. The browser dashboard currently requires the default feature contract and binary `[batch,2]` probability output; generic arbitrary model UI inference is not yet supported.

Use your own fixture-generation process to store exact float32 rows and reference outputs. Preserve model hashes across runtimes and reject references for a different hash. Regression and additional multiclass output adapters are a roadmap item.
