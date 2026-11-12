export type WorkerRequest =
  | { id: number; type: "initialize"; baseUrl: string }
  | { id: number; type: "predict"; features: number[] | number[][] }
  | {
      id: number;
      type: "batch";
      cases: Array<{ id: string; features: number[] | number[][] }>;
    }
  | {
      id: number;
      type: "benchmark";
      features: number[];
      warmup: number;
      iterations: number;
      batch_sizes?: number[];
    }
  | { id: number; type: "cancel-benchmark" }
  | { id: number; type: "capabilities" }
  | { id: number; type: "release" };

export type WorkerResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };
