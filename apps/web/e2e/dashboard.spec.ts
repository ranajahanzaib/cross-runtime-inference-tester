import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

test("loads actual browser runtime, compares fixtures, and predicts locally", async ({
  page,
}) => {
  const remoteRequests: string[] = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:4173"))
      remoteRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByText("Runtime parity overview")).toBeVisible();
  await expect(page.getByText("WASM runtime ready")).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText("268/268", { exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await page
    .getByRole("textbox", { name: "Search fixture" })
    .fill("golden-0000");
  await expect(
    page.getByRole("cell", { name: "golden-0000 held-out-test" }),
  ).toBeVisible({ timeout: 30_000 });
  await page.getByRole("textbox", { name: "Search fixture" }).fill("");
  await page
    .getByRole("combobox", { name: "Filter result status" })
    .selectOption("batch");
  await expect(
    page.getByRole("cell", { name: "batch-128 batch" }),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Filter result status" })
    .selectOption("all");
  await page.getByRole("button", { name: /Run browser inference/ }).click();
  await expect(page.getByText("BROWSER PREDICTION")).toBeVisible();
  await page.getByRole("button", { name: /Run browser benchmark/ }).click();
  await expect(page.getByText("BATCH 1 MEDIAN", { exact: true })).toBeVisible({
    timeout: 30_000,
  });
  const downloadReady = page.waitForEvent("download");
  await page.getByRole("button", { name: /Download JSON/ }).click();
  const download = await downloadReady;
  const reportDir = resolve(import.meta.dirname, "../../../artifacts/reports");
  mkdirSync(reportDir, { recursive: true });
  const reportPath = resolve(reportDir, "browser-report.json");
  await download.saveAs(reportPath);
  const report = JSON.parse(readFileSync(reportPath, "utf8")) as {
    summary: { cases: number; failures: number; status: string };
    benchmark: {
      independent_run_count: number;
      independent_runs: Array<{
        run_index: number;
        session_initialization_ms: number;
        first_inference_ms: number;
        workloads: unknown[];
      }>;
      workloads: Array<{
        batch_size: number;
        repetitions_per_sample: number;
        latency_ms: { median: number };
      }>;
    };
    startup: { first_inference_ms: number };
    environment: {
      application_bundle_asset_bytes: number;
      total_deployed_asset_bytes: number;
    };
  };
  expect(report.summary).toEqual({
    cases: 268,
    failures: 0,
    passes: 268,
    max_absolute_difference: expect.any(Number),
    status: "PASS",
  });
  expect(
    report.benchmark.workloads.map((workload) => workload.batch_size),
  ).toEqual([1, 8, 32, 128]);
  expect(report.benchmark.workloads[0].repetitions_per_sample).toBeGreaterThan(
    1,
  );
  expect(report.benchmark.independent_run_count).toBe(3);
  expect(report.benchmark.independent_runs.map((run) => run.run_index)).toEqual(
    [1, 2, 3],
  );
  expect(
    report.benchmark.independent_runs.every(
      (run) =>
        run.session_initialization_ms >= 0 &&
        run.first_inference_ms >= 0 &&
        run.workloads.length === 4,
    ),
  ).toBe(true);
  expect(report.benchmark.workloads[0].latency_ms.median).toBeGreaterThan(0);
  expect(report.startup.first_inference_ms).toBeGreaterThan(0);
  expect(report.environment.application_bundle_asset_bytes).toBeGreaterThan(0);
  expect(report.environment.total_deployed_asset_bytes).toBeGreaterThan(
    report.environment.application_bundle_asset_bytes,
  );
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(100);
  const screenshotDir = resolve(import.meta.dirname, "../../../docs/images");
  mkdirSync(screenshotDir, { recursive: true });
  await page.screenshot({
    path: resolve(screenshotDir, "dashboard.png"),
    fullPage: true,
  });
  const failureLab = page.locator("#failure");
  await failureLab.getByRole("combobox").selectOption("threshold");
  await failureLab.getByRole("button", { name: /Run injected fault/ }).click();
  await expect(failureLab.getByText(/FAIL · injected threshold/)).toBeVisible();
  expect(remoteRequests).toEqual([]);
});

test("reports actionable model loading errors", async ({ page }) => {
  await page.route("**/models/subscription-risk-v1/manifest.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText(
    "Model artifacts are unavailable",
  );
});

test("keeps the explorer keyboard and mobile usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("268/268", { exact: true })).toBeVisible({
    timeout: 30_000,
  });
  const firstCase = page.locator('tr[aria-label^="Inspect "]').first();
  await firstCase.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#artifacts")).toBeVisible();
  const dimensions = await page.evaluate(() => ({
    width: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.width + 1);
  const absoluteTolerance = page.getByLabel("ABSOLUTE TOLERANCE");
  await absoluteTolerance.fill("0");
  await page.getByLabel("RELATIVE TOLERANCE").fill("0");
  const toleranceText = await page
    .locator(".metric-card")
    .filter({ hasText: "WITHIN TOLERANCE" })
    .innerText();
  const passingCount = Number(toleranceText.match(/(\d+)\/268/)?.[1]);
  expect(passingCount).toBeLessThan(268);
});

test("cancels the browser benchmark without reporting a failure", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("WASM runtime ready")).toBeVisible({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: /Run browser benchmark/ }).click();
  const cancel = page.getByRole("button", { name: "Cancel benchmark" });
  await expect(cancel).toBeVisible();
  await cancel.click();
  await expect(page.getByRole("status")).toContainText("Benchmark cancelled");
});
