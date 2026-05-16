import fs from "node:fs";
import path from "node:path";

import type { JudgeAfferitionSamplingSummary } from "../contracts/autobench.js";
import { readJsonFile, utcNowIso, writeJson } from "../io/filesystem.js";

export const JUDGE_AFFERITION_RUN_STATE_FILE = "run_state.json";
export const JUDGE_AFFERITION_MAX_ATTEMPTS = 10;
export const JUDGE_AFFERITION_BACKOFF = "exponential_limited";

export type JudgeAfferitionRunStatus = "running" | "failed" | "completed";

export interface JudgeAfferitionLastError {
  rowId: string;
  metric: string;
  attempt: number;
  message: string;
}

export interface JudgeAfferitionRunState {
  kind: "judge_afferition";
  status: JudgeAfferitionRunStatus;
  modelId: string;
  model: string;
  judgeConfigPath: string;
  outputPath: string;
  startedAt: string;
  updatedAt: string;
  completedAt: string | null;
  total: number;
  completed: number;
  remaining: number;
  lastProcessedRowId: string | null;
  lastError: JudgeAfferitionLastError | null;
  afferitionSampling: JudgeAfferitionSamplingSummary | null;
  retryPolicy: {
    maxAttempts: number;
    backoff: typeof JUDGE_AFFERITION_BACKOFF;
  };
}

export interface IncompleteJudgeAfferitionRun {
  outputPath: string;
  state: JudgeAfferitionRunState;
}

export function buildInitialJudgeAfferitionRunState(args: {
  modelId: string;
  model: string;
  judgeConfigPath: string;
  outputPath: string;
  total: number;
  completed: number;
  lastProcessedRowId: string | null;
  afferitionSampling?: JudgeAfferitionSamplingSummary | null;
}): JudgeAfferitionRunState {
  const existing = readJudgeAfferitionRunState(args.outputPath);
  const startedAt = existing?.startedAt ?? utcNowIso();
  return {
    kind: "judge_afferition",
    status: "running",
    modelId: args.modelId,
    model: args.model,
    judgeConfigPath: args.judgeConfigPath,
    outputPath: args.outputPath,
    startedAt,
    updatedAt: utcNowIso(),
    completedAt: null,
    total: args.total,
    completed: args.completed,
    remaining: Math.max(0, args.total - args.completed),
    lastProcessedRowId: args.lastProcessedRowId,
    lastError: null,
    afferitionSampling: existing?.afferitionSampling ?? args.afferitionSampling ?? null,
    retryPolicy: {
      maxAttempts: JUDGE_AFFERITION_MAX_ATTEMPTS,
      backoff: JUDGE_AFFERITION_BACKOFF
    }
  };
}

export function readJudgeAfferitionRunState(outputPath: string): JudgeAfferitionRunState | null {
  const statePath = judgeAfferitionRunStatePath(outputPath);
  if (!fs.existsSync(statePath)) {
    return null;
  }
  const state = readJsonFile<JudgeAfferitionRunState>(statePath);
  return state.kind === "judge_afferition" ? state : null;
}

export function writeJudgeAfferitionRunState(outputPath: string, state: JudgeAfferitionRunState): void {
  writeJson(judgeAfferitionRunStatePath(outputPath), {
    ...state,
    updatedAt: utcNowIso()
  });
}

export function listIncompleteJudgeAfferitionRuns(outputBase: string): IncompleteJudgeAfferitionRun[] {
  if (!fs.existsSync(outputBase)) {
    return [];
  }
  return fs.readdirSync(outputBase)
    .map((entry) => path.join(outputBase, entry))
    .filter((entryPath) => fs.statSync(entryPath).isDirectory())
    .map((entryPath) => ({ outputPath: entryPath, state: readJudgeAfferitionRunState(entryPath) }))
    .filter((entry): entry is IncompleteJudgeAfferitionRun => entry.state !== null && entry.state.status !== "completed")
    .sort((left, right) => right.state.updatedAt.localeCompare(left.state.updatedAt));
}

export function judgeAfferitionRunStatePath(outputPath: string): string {
  return path.join(outputPath, JUDGE_AFFERITION_RUN_STATE_FILE);
}
