import {
  type CsvRecord,
  type GenerationResult,
  type RunContext
} from "../contracts/autobench.js";
import { fileExists, readJsonFile, writeJson } from "../io/filesystem.js";

type GenerationCheckpoint = {
  version: 1;
  results: Record<string, GenerationResult>;
};

type JudgeCheckpoint = {
  version: 1;
  labels: Record<string, string | null>;
};

export function generationUnitKey(args: {
  profileName: string;
  datasetName: string;
  mode: string;
  rowIndex: number | string;
}): string {
  return [args.profileName, args.datasetName, args.mode, String(args.rowIndex)].join("::");
}

export function judgeUnitKey(args: {
  profileName: string;
  datasetName: string;
  mode: string;
  rowId: unknown;
  metric: string;
}): string {
  return [args.profileName, args.datasetName, args.mode, String(args.rowId), args.metric].join("::");
}

export function readGenerationResult(ctx: RunContext, key: string): GenerationResult | null {
  const checkpoint = readGenerationCheckpoint(ctx);
  return checkpoint.results[key] ?? null;
}

export function writeGenerationResult(ctx: RunContext, key: string, result: GenerationResult): void {
  const checkpoint = readGenerationCheckpoint(ctx);
  checkpoint.results[key] = result;
  writeJson(ctx.generationCheckpointPath, checkpoint);
}

export function readJudgeLabel(ctx: RunContext, key: string): string | null | undefined {
  const checkpoint = readJudgeCheckpoint(ctx);
  return Object.prototype.hasOwnProperty.call(checkpoint.labels, key) ? checkpoint.labels[key] : undefined;
}

export function writeJudgeLabel(ctx: RunContext, key: string, label: string | null): void {
  const checkpoint = readJudgeCheckpoint(ctx);
  checkpoint.labels[key] = label;
  writeJson(ctx.judgeCheckpointPath, checkpoint);
}

export function checkpointSummary(ctx: RunContext): CsvRecord {
  return {
    generation_results: Object.keys(readGenerationCheckpoint(ctx).results).length,
    judge_labels: Object.keys(readJudgeCheckpoint(ctx).labels).length
  };
}

function readGenerationCheckpoint(ctx: RunContext): GenerationCheckpoint {
  if (!fileExists(ctx.generationCheckpointPath)) {
    return { version: 1, results: {} };
  }
  const payload = readJsonFile<GenerationCheckpoint>(ctx.generationCheckpointPath);
  return {
    version: 1,
    results: payload.results ?? {}
  };
}

function readJudgeCheckpoint(ctx: RunContext): JudgeCheckpoint {
  if (!fileExists(ctx.judgeCheckpointPath)) {
    return { version: 1, labels: {} };
  }
  const payload = readJsonFile<JudgeCheckpoint>(ctx.judgeCheckpointPath);
  return {
    version: 1,
    labels: payload.labels ?? {}
  };
}
