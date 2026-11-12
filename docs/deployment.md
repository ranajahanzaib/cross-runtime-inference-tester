# Static deployment

Run `uv run --project . cross-runtime-inference-tester demo`, `uv run --project . cross-runtime-inference-tester verify`, and `uv run --project . cross-runtime-inference-tester benchmark` before `pnpm build` to include current model, fixtures, comparison reports, and Python benchmark artifacts. Vite copies those generated assets and the ONNX Runtime WASM binaries from the installed package into `apps/web/public/` before bundling. Serve `apps/web/dist/` over HTTPS or localhost.

The `public/_headers` file sets basic security headers and immutable caching for the versioned model path. Vite bundles the matching WASM binary as a hashed `.wasm` asset. Single-threaded WASM needs no COOP/COEP isolation. Threading is not enabled. Cloudflare Pages can host the `dist` output as a static site; no Worker or paid Cloudflare service is required. Verify deployed asset headers and wasm MIME handling on the chosen host. The app has no service worker or offline cache.
