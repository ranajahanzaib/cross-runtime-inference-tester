import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { cpSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const projectRoot = resolve(import.meta.dirname, '../..');
const modelDir = resolve(projectRoot, 'artifacts/models/subscription-risk-v1');
const reportsDir = resolve(projectRoot, 'artifacts/reports');
const benchmarksDir = resolve(projectRoot, 'artifacts/benchmarks');
const ortDir = resolve(projectRoot, 'apps/web/node_modules/onnxruntime-web/dist');
const ortVersion = JSON.parse(readFileSync(resolve(projectRoot, 'apps/web/node_modules/onnxruntime-web/package.json'), 'utf8')).version as string;
const copyAssets = () => {
  const publicDir = resolve(projectRoot, 'apps/web/public');
  if (existsSync(modelDir)) { mkdirSync(resolve(publicDir, 'models/subscription-risk-v1'), { recursive: true }); cpSync(modelDir, resolve(publicDir, 'models/subscription-risk-v1'), { recursive: true }); }
  if (existsSync(reportsDir)) { mkdirSync(resolve(publicDir, 'reports'), { recursive: true }); for (const file of ['report.json', 'report.html', 'report.md']) if (existsSync(resolve(reportsDir, file))) cpSync(resolve(reportsDir, file), resolve(publicDir, 'reports', file)); }
  if (existsSync(benchmarksDir)) { mkdirSync(resolve(publicDir, 'benchmarks'), { recursive: true }); for (const file of ['python.json']) if (existsSync(resolve(benchmarksDir, file))) cpSync(resolve(benchmarksDir, file), resolve(publicDir, 'benchmarks', file)); }
  if (existsSync(ortDir)) { mkdirSync(resolve(publicDir, 'ort'), { recursive: true }); for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) if (existsSync(resolve(ortDir, f))) cpSync(resolve(ortDir, f), resolve(publicDir, 'ort', f)); }
};

export default defineConfig({ root: import.meta.dirname, define: { __ORT_WEB_VERSION__: JSON.stringify(ortVersion) }, plugins: [react(), { name: 'cross-runtime-inference-tester-local-assets'', configResolved() { copyAssets(); }, configureServer(server) { copyAssets(); server.watcher.on('change', (path) => { if (path.startsWith(modelDir)) copyAssets(); }); }, generateBundle(_options, bundle) { for (const fileName of Object.keys(bundle)) if (fileName.endsWith('.wasm')) delete bundle[fileName]; }, closeBundle() { copyAssets(); } }], publicDir: 'public', worker: { format: 'es' }, test: { environment: 'jsdom', include: ['tests/**/*.test.ts'] } });
