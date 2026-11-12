import type { WorkerRequest, WorkerResponse } from "./messages";

type WorkerCommand = WorkerRequest extends infer T
  ? T extends { id: number }
    ? Omit<T, "id">
    : never
  : never;

export class InferenceClient {
  private worker = new Worker(
    new URL("./inference.worker.ts", import.meta.url),
    { type: "module" },
  );
  private sequence = 0;
  async call<T>(request: WorkerCommand): Promise<T> {
    const id = ++this.sequence;
    return new Promise<T>((resolve, reject) => {
      const handler = (event: MessageEvent<WorkerResponse>) => {
        if (event.data.id !== id) return;
        this.worker.removeEventListener("message", handler);
        if (event.data.ok) resolve(event.data.result as T);
        else reject(new Error(event.data.error));
      };
      this.worker.addEventListener("message", handler);
      this.worker.postMessage({ ...request, id } as WorkerRequest);
    });
  }
  terminate() {
    this.worker.terminate();
  }
}
