import type { GenerationPoolBackend } from "../contracts/autobench.js";

export interface ScheduledGenerationResult<T, R> {
  item: T;
  result: R;
  backend: GenerationPoolBackend;
}

/**
 * Runs generation work through a FIFO queue consumed by per-backend workers.
 */
export async function runGenerationQueue<T, R>(
  items: T[],
  backends: GenerationPoolBackend[],
  worker: (item: T, backend: GenerationPoolBackend, index: number) => Promise<R>
): Promise<Array<ScheduledGenerationResult<T, R>>> {
  if (items.length === 0) {
    return [];
  }
  const activeBackends = backends.filter((backend) => backend.workers > 0);
  if (activeBackends.length === 0) {
    throw new Error("Generation pool must contain at least one worker");
  }

  const results: Array<ScheduledGenerationResult<T, R>> = new Array(items.length);
  let cursor = 0;
  const runners: Array<Promise<void>> = [];

  for (const backend of activeBackends) {
    for (let workerIndex = 0; workerIndex < backend.workers; workerIndex += 1) {
      runners.push((async () => {
        while (cursor < items.length) {
          const index = cursor;
          cursor += 1;
          const item = items[index]!;
          results[index] = {
            item,
            backend,
            result: await worker(item, backend, index)
          };
        }
      })());
    }
  }

  await Promise.all(runners);
  return results;
}

export function totalGenerationWorkers(backends: GenerationPoolBackend[]): number {
  return backends.reduce((total, backend) => total + Math.max(0, backend.workers), 0);
}
