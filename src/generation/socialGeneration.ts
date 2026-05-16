import { datasetPath } from "../config/loadConfig.js";
import {
  MAX_ATTEMPTS_PER_ROW,
  MAX_ROW_CANDIDATES,
  TEXT_PARSE,
  type CsvRecord,
  type DatasetConfig,
  type GenerationPoolBackend,
  type ProfileConfig,
  type RunContext
} from "../contracts/autobench.js";
import { jsonSafe, readCsvFile, stableSeedOffset } from "../io/filesystem.js";
import { logEvent, quarantine } from "../pipeline/runContext.js";
import type { TerminalObserver } from "../runtime/terminalObserver.js";
import { shuffleIndices } from "../sampling/samplePlanner.js";
import { generateOne } from "./generateOne.js";
import { makeResponseRecord, writeDatasetOutputs } from "./responseRecords.js";
import { BenchmarkAbort } from "../pipeline/benchmarkAbort.js";
import { interactiveMenuWithObserver } from "../runtime/terminalObserver.js";
import { runGenerationQueue, totalGenerationWorkers } from "./generationScheduler.js";

/**
 * Builds the canonical social sample for the reference profile.
 */
export async function canonicalSocialGeneration(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  dataset: DatasetConfig;
  targetN: number;
  generationPool: GenerationPoolBackend[];
  observer: TerminalObserver;
}): Promise<{ acceptedIndices: number[]; records: CsvRecord[] }> {
  const { ctx, profile, dataset, targetN, generationPool, observer } = args;
  const rows = readCsvFile(datasetPath(ctx.repoRoot, profile, dataset));
  const shuffled = shuffleIndices(rows, profile.seed + stableSeedOffset(dataset.name));
  let cursor = 0;
  const acceptedIndices: number[] = [];
  const records: CsvRecord[] = [];
  const isFullBench = targetN >= rows.length;
  const stage = observer.startStage(`${profile.name}:${dataset.name}:build-sample`, targetN);

  logEvent(ctx, "stage_started", observer, {
    profile: profile.name,
    dataset: dataset.name,
    mode: "responses",
    stage: "build-sample",
    target_n: targetN
  });

  try {
    while (acceptedIndices.length < targetN) {
      const batch = nextSocialCandidateBatch(shuffled, cursor, targetN - acceptedIndices.length, generationPool, isFullBench);
      cursor += batch.length;
      if (batch.length === 0) {
        await handleCanonicalSocialFailure(dataset.name, acceptedIndices.length, observer);
        continue;
      }
      const generated = await runGenerationQueue(batch, generationPool, async (rowIndex, backend) => {
        const row = rows[rowIndex]!;
        return generateOne({
          ctx,
          profile,
          dataset,
          row,
          rowIndex,
          mode: "responses",
          parser: TEXT_PARSE,
          backend,
          observer,
          progress: isFullBench ? stage : undefined
        });
      });
      for (const item of generated) {
        const rowIndex = item.item;
        const row = rows[rowIndex]!;
        if (!item.result.ok) {
          const reason = isFullBench ? "canonical_social_fullbench_candidate_failed" : "canonical_social_candidate_failed";
          quarantine(ctx, reason, observer, {
            profile: profile.name,
            dataset: dataset.name,
            row_index: rowIndex,
            error: item.result.error,
            result: jsonSafe(item.result),
            backend_id: item.backend.backendId
          });
          if (isFullBench && acceptedIndices.length < targetN) {
            acceptedIndices.push(rowIndex);
            records.push(makeResponseRecord(row, profile, rowIndex, item.result, "responses", "refused"));
          }
          continue;
        }
        if (acceptedIndices.length < targetN) {
          acceptedIndices.push(rowIndex);
          records.push(makeResponseRecord(row, profile, rowIndex, item.result, "responses", "ok"));
          if (!isFullBench) {
            stage.advance();
            observer.advanceOverall();
          }
        } else {
          quarantine(ctx, "canonical_social_extra_candidate_discarded", observer, {
            profile: profile.name,
            dataset: dataset.name,
            row_index: rowIndex,
            backend_id: item.backend.backendId
          });
        }
      }
    }
  } finally {
    stage.close();
    logEvent(ctx, "stage_finished", observer, {
      profile: profile.name,
      dataset: dataset.name,
      mode: "responses",
      stage: "build-sample",
      accepted: acceptedIndices.length
    });
  }

  writeDatasetOutputs(ctx, profile, dataset, "responses", records);
  return { acceptedIndices, records };
}

/**
 * Replays a canonical social sample for a non-reference profile.
 */
export async function fixedSocialGeneration(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  dataset: DatasetConfig;
  acceptedIndices: number[];
  generationPool: GenerationPoolBackend[];
  observer: TerminalObserver;
}): Promise<CsvRecord[]> {
  const { ctx, profile, dataset, acceptedIndices, generationPool, observer } = args;
  const rows = readCsvFile(datasetPath(ctx.repoRoot, profile, dataset));
  const records: CsvRecord[] = [];
  const stage = observer.startStage(`${profile.name}:${dataset.name}:fixed-sample`, acceptedIndices.length);
  logEvent(ctx, "stage_started", observer, {
    profile: profile.name,
    dataset: dataset.name,
    mode: "responses",
    stage: "fixed-sample",
    target_n: acceptedIndices.length
  });

  try {
    const generatedRecords = await runGenerationQueue(acceptedIndices, generationPool, async (rowIndex, backend) => {
      const row = rows[rowIndex]!;
      const result = await generateOne({
        ctx,
        profile,
        dataset,
        row,
        rowIndex,
        mode: "responses",
        parser: TEXT_PARSE,
        backend,
        observer,
        progress: stage
      });
      const status = await fixedStatusOrAbort(result.ok, `${profile.name}/${dataset.name}/${rowIndex}`, observer);
      return makeResponseRecord(row, profile, rowIndex, result, "responses", status);
    });
    records.push(...generatedRecords.map((item) => item.result));
  } finally {
    stage.close();
    logEvent(ctx, "stage_finished", observer, {
      profile: profile.name,
      dataset: dataset.name,
      mode: "responses",
      stage: "fixed-sample",
      rows: records.length
    });
  }

  writeDatasetOutputs(ctx, profile, dataset, "responses", records);
  return records;
}

export function nextSocialCandidateBatch(
  shuffled: number[],
  cursor: number,
  remainingTarget: number,
  generationPool: GenerationPoolBackend[],
  isFullBench = false
): number[] {
  if (cursor >= shuffled.length) {
    return [];
  }
  if (isFullBench) {
    return shuffled.slice(cursor);
  }
  const workerCount = Math.max(1, totalGenerationWorkers(generationPool));
  const batchSize = Math.max(workerCount, Math.min(workerCount * MAX_ROW_CANDIDATES, remainingTarget));
  return shuffled.slice(cursor, Math.min(shuffled.length, cursor + batchSize));
}

async function handleCanonicalSocialFailure(datasetName: string, slot: number, observer: TerminalObserver): Promise<void> {
  const choice = await interactiveMenuWithObserver(
    observer,
    `${datasetName} slot ${slot} failed after ${MAX_ROW_CANDIDATES} reserve rows x ${MAX_ATTEMPTS_PER_ROW} attempts.`,
    {
      r: "try four more reserve rows",
      a: "fail hard and save current artifacts"
    }
  );
  if (choice === "a") {
    throw new BenchmarkAbort(`User aborted canonical sampling for ${datasetName}`);
  }
}

async function fixedStatusOrAbort(isOk: boolean, label: string, observer: TerminalObserver): Promise<string> {
  if (isOk) {
    return "ok";
  }
  const choice = await interactiveMenuWithObserver(
    observer,
    `${label} failed after ${MAX_ATTEMPTS_PER_ROW} attempts.`,
    {
      f: "mark refused/fail for this profile and continue",
      a: "fail hard and save current artifacts"
    }
  );
  if (choice === "a") {
    throw new BenchmarkAbort(`User aborted fixed generation for ${label}`);
  }
  return "refused";
}
