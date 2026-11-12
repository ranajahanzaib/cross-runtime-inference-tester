# Numerical parity

Floating-point values can differ because converters rewrite operations, runtimes fuse graph nodes, and processors use different instruction paths. Bitwise equality is not a general correctness requirement.

For each output, Cross-Runtime Inference Tester uses `abs(a-b) <= atol + rtol * abs(reference)`. Absolute tolerance provides a floor near zero; relative tolerance scales with larger reference values. Defaults (`atol=1e-4`, `rtol=1e-3`) are starting points for this demo, not universal guarantees. Change them based on model sensitivity and downstream risk.

Shape mismatch, missing outputs, non-finite values, tolerance violations, and a class disagreement produce a failure. Threshold-sensitive probability changes deserve separate attention because small numeric drift can change decisions. Batch size is retained in each fixture and should be compared with matching semantics. Never average a severe individual mismatch into an aggregate pass.
