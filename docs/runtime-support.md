# Runtime support

| Adapter                  | Status                      | Scope                                           |
| ------------------------ | --------------------------- | ----------------------------------------------- |
| scikit-learn native      | Export/reference validation | Default training/evaluation pipeline            |
| Python ONNX Runtime      | Implemented                 | Explicit CPU provider, binary probability model |
| Browser ONNX Runtime Web | Implemented                 | WASM CPU, same ONNX file and float32 vectors    |
| Browser threaded WASM    | Not enabled                 | Requires cross-origin isolation; not measured   |
| WebGPU / WebNN           | Unsupported                 | Not part of current verified runtime matrix     |

Browser capability metadata reports cross-origin isolation, SharedArrayBuffer presence, and hardware concurrency. The runtime remains single-threaded and does not infer support from those flags alone.

Chromium is the automated browser baseline. Firefox and WebKit are not currently run in CI and are not claimed as verified. Browser timing combines the installed engine, OS, and WASM build; exported reports include browser name/version and available runtime capability metadata.
