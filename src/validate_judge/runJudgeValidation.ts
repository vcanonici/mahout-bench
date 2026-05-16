import fs from "node:fs";
import path from "node:path";

import {
  DEFAULT_JUDGE_CONFIG,
  type JudgeAfferitionSamplingSummary,
  type JudgeConfig,
  type JudgeValidationMetricSummary,
  type JudgeValidationRegistryEntry,
  type ModelCatalogEntry,
  type TokenUsage,
  type TokenUsageSummary
} from "../contracts/autobench.js";
import { parseJudge, resolveInferenceFromModelCatalog } from "../config/loadConfig.js";
import { buildJudgeResponseFormat, callChat, parseJudgeOutput } from "../inference/chatClient.js";
import { addTokenUsage, emptyTokenUsageSummary } from "../inference/tokenUsage.js";
import { appendJsonl, ensureDir, localRunStamp, readTextFile, utcNowIso, writeJson, writeTextFile } from "../io/filesystem.js";
import { buildJudgePrompt } from "../judging/judgePrompts.js";
import { mapWithConcurrency } from "../runtime/concurrency.js";
import { resolveDataRootForRepo } from "../runtime/paths.js";
import { classifyBackend, TerminalObserver } from "../runtime/terminalObserver.js";
import { buildFullJudgeAfferitionSamplingSummary } from "./judgeAfferitionSampling.js";
import { computeMetricSummary, computeSimilarityCounts, overallSimilarity, type CandidateValidationResult } from "./metrics.js";
import { loadElephantReference, type ValidationReferenceRow } from "./loadElephantReference.js";
import type { BinaryLabel } from "./parseBinaryLabel.js";
import {
  buildInitialJudgeAfferitionRunState,
  JUDGE_AFFERITION_MAX_ATTEMPTS,
  readJudgeAfferitionRunState,
  type JudgeAfferitionLastError,
  type JudgeAfferitionRunState,
  writeJudgeAfferitionRunState
} from "./runState.js";

export type JudgeCall = (row: ValidationReferenceRow, prompt: string) => Promise<string | JudgeCallResult>;

export interface JudgeCallResult {
  raw: string;
  tokenUsage: TokenUsage | null;
}

export interface RunJudgeValidationArgs {
  repoRoot: string;
  outputBase: string;
  dataDir: string;
  model: ModelCatalogEntry;
  judgeConfigPath?: string;
  judgeCall?: JudgeCall;
  limit?: number | null;
  observer?: TerminalObserver;
  outputPath?: string | null;
  retryDelayMs?: (attempt: number) => number;
  afferitionSampling?: JudgeAfferitionSamplingSummary | null;
}

interface CandidateArtifact extends CandidateValidationResult {
  sourceRowIndex: number;
  prompt: string;
  responseModel: string;
  response: string;
  referenceJudge: string;
  candidateRawOutput: string;
  parseError: string | null;
  attempts: number;
  tokenUsage: TokenUsage | null;
}

class JudgeAfferitionRowFailure extends Error {
  public constructor(public readonly lastError: JudgeAfferitionLastError) {
    super(`Judge afferition failed for ${lastError.rowId}/${lastError.metric} after ${lastError.attempt} attempts: ${lastError.message}`);
    this.name = "JudgeAfferitionRowFailure";
  }
}

/**
 * Runs Claude social judge afferition and writes all validation artifacts.
 */
export async function runJudgeValidation(args: RunJudgeValidationArgs): Promise<JudgeValidationRegistryEntry> {
  const judgeConfigPath = args.judgeConfigPath ?? DEFAULT_JUDGE_CONFIG;
  const outputPath = args.outputPath ?? path.join(args.outputBase, `judge_afferition_${args.model.id}_${localRunStamp()}`);
  const existingState = readJudgeAfferitionRunState(outputPath);
  const existingSampling = existingState?.afferitionSampling ?? null;
  const requestedSampling = existingSampling ?? args.afferitionSampling ?? null;
  const effectiveDataDir = existingSampling && existingSampling.kind !== "full"
    ? path.join(resolveDataRootForRepo(args.repoRoot), existingSampling.datasetPath)
    : args.dataDir;
  const reference = loadElephantReference(args.repoRoot, effectiveDataDir);
  const afferitionSampling = requestedSampling ?? buildFullJudgeAfferitionSamplingSummary(
    args.repoRoot,
    effectiveDataDir,
    reference.rows.length,
    reference.fingerprint
  );
  const referenceRows = args.limit && args.limit > 0 ? reference.rows.slice(0, args.limit) : reference.rows;
  ensureDir(outputPath);

  const observer = args.observer ?? new TerminalObserver(false);
  observer.configureRun(outputPath);
  const judge = buildValidationJudge(args.repoRoot, judgeConfigPath, args.model.id);
  const judgeCall = args.judgeCall ?? providerJudgeCall(judge);
  const artifacts = readExistingArtifacts(outputPath);
  const completedRowIds = new Set(artifacts.map((artifact) => artifact.rowId));
  const pendingRows = referenceRows.filter((row) => !completedRowIds.has(row.rowId));
  let runState = buildInitialJudgeAfferitionRunState({
    modelId: args.model.id,
    model: args.model.model,
    judgeConfigPath,
    outputPath,
    total: referenceRows.length,
    completed: completedRowIds.size,
    lastProcessedRowId: artifacts.at(-1)?.rowId ?? null,
    afferitionSampling
  });
  writeJudgeAfferitionRunState(outputPath, runState);
  observer.start(referenceRows.length, `Judge afferition: ${args.model.id}`);
  const stage = observer.startStage(`judge-afferition:${args.model.id}`, referenceRows.length);
  if (completedRowIds.size > 0) {
    stage.advance(completedRowIds.size);
    observer.advanceOverall(completedRowIds.size);
  }
  writeProgress(outputPath, args.model.id, referenceRows.length, completedRowIds.size, pendingRows.length);

  let fatalError: JudgeAfferitionLastError | null = null;
  try {
    await mapWithConcurrency(pendingRows, judge.inference.parallelism, async (row) => {
      if (fatalError) {
        return;
      }
      const artifact = await evaluateRow(row, judge, args.model.id, judgeCall, args.retryDelayMs ?? judgeAfferitionRetryDelayMs, observer);
      artifacts.push(artifact);
      appendJsonl(path.join(outputPath, "labels_candidate.jsonl"), artifact);
      if (!artifact.validParse) {
        appendJsonl(path.join(outputPath, "failures_invalid_outputs.jsonl"), artifact);
      }
      const outcome = artifact.validParse ? {} : { ok: false, failureKind: "judge_afferition_invalid" };
      stage.advance(1, outcome);
      observer.advanceOverall(1, outcome);
      writeProgress(outputPath, args.model.id, referenceRows.length, artifacts.length, referenceRows.length - artifacts.length);
      runState = writeRunningState(outputPath, runState, artifacts.length, referenceRows.length, artifact.rowId, fatalError);
    });
  } catch (error) {
    fatalError = error instanceof JudgeAfferitionRowFailure ? error.lastError : {
      rowId: "unknown",
      metric: "unknown",
      attempt: JUDGE_AFFERITION_MAX_ATTEMPTS,
      message: renderError(error)
    };
    runState = writeFailedState(outputPath, runState, artifacts.length, referenceRows.length, fatalError);
    throw error;
  } finally {
    stage.close();
  }

  artifacts.sort((left, right) => left.rowId.localeCompare(right.rowId));
  const groupedMetrics = buildMetricSummaries(artifacts);
  const similarity = overallSimilarity(groupedMetrics);
  const tokenUsage = summarizeCandidateTokenUsage(artifacts);
  const metricsPayload = {
    created_at: utcNowIso(),
    candidate: args.model.id,
    reference: "ELEPHANT GPT-4o labels over Claude responses",
    data_fingerprint: reference.fingerprint,
    afferition_sampling: afferitionSampling,
    overall_similarity: similarity,
    token_usage: tokenUsage,
    metrics: groupedMetrics
  };

  writeJson(path.join(outputPath, "judge_afferition_metrics.json"), metricsPayload);
  writeTextFile(path.join(outputPath, "judge_afferition_report.md"), renderMarkdownReport(metricsPayload));
  writeProgress(outputPath, args.model.id, referenceRows.length, artifacts.length, 0);
  runState = writeCompletedState(outputPath, runState, artifacts.length, referenceRows.length, artifacts.at(-1)?.rowId ?? null);
  void runState;

  return {
    modelId: args.model.id,
    model: args.model.model,
    label: args.model.label,
    reference: "ELEPHANT GPT-4o labels over Claude responses",
    validatedAt: utcNowIso(),
    judgeConfigPath,
    outputPath,
    dataFingerprint: reference.fingerprint,
    overallSimilarity: similarity,
    metrics: groupedMetrics,
    afferitionSampling
  };
}

/**
 * Builds the resolved judge config used by afferition runs.
 */
export function buildValidationJudge(repoRoot: string, judgeConfigPath: string, modelId: string): JudgeConfig {
  const parsed = parseJudge(repoRoot, judgeConfigPath);
  const inference = resolveInferenceFromModelCatalog({ repoRoot, base: parsed.inference, modelId });
  return {
    ...parsed,
    inference: {
      ...inference,
      temperature: 0,
      topP: 1,
      maxTokens: 64,
      contextLength: inference.contextLength,
      thinkingEnabled: false,
      reasoningEffort: "low"
    }
  };
}

function providerJudgeCall(judge: JudgeConfig): JudgeCall {
  const responseFormat = buildJudgeResponseFormat(judge);
  return async (_row, prompt) => {
    const response = await callChat({ inference: judge.inference, prompt, responseFormat });
    return {
      raw: response.extracted.text,
      tokenUsage: response.tokenUsage
    };
  };
}

async function evaluateRow(
  row: ValidationReferenceRow,
  judge: JudgeConfig,
  modelId: string,
  judgeCall: JudgeCall,
  retryDelayMs: (attempt: number) => number,
  observer: TerminalObserver
): Promise<CandidateArtifact> {
  const prompt = buildJudgePrompt(judge, row.metric, row.prompt, row.response);
  let lastError: JudgeAfferitionLastError | null = null;
  let lastErrorKind: "parse" | "provider" | null = null;
  let lastRawOutput = "";
  let lastTokenUsage: TokenUsage | null = null;
  for (let attempt = 1; attempt <= JUDGE_AFFERITION_MAX_ATTEMPTS; attempt += 1) {
    const attemptStarted = Date.now();
    try {
      const callResult = normalizeJudgeCallResult(await judgeCall(row, prompt));
      const raw = callResult.raw;
      lastTokenUsage = callResult.tokenUsage;
      const parsed = parseJudgeOutput(raw, judge.outputMode);
      const candidateLabel = parsed.ok ? toBinaryLabel(parsed.label) : null;
      const validParse = parsed.ok && candidateLabel !== null;
      observer.recordCall({
        phase: "judge_afferition",
        stage: `judge-afferition:${modelId}`,
        backend_category: classifyBackend({
          provider: judge.inference.provider,
          apiBaseUrl: judge.inference.apiBaseUrl,
          modelId
        }),
        backend_id: modelId,
        model_id: modelId,
        dataset: row.dataset,
        metric: row.metric,
        row_id: row.rowId,
        attempt,
        duration_seconds: roundSeconds(Date.now() - attemptStarted),
        ok: validParse,
        failure_kind: validParse ? null : "parse",
        error: validParse ? null : parsed.error ?? `Invalid judge label: ${String(parsed.label)}`
      });
      if (validParse) {
        return buildCandidateArtifact(row, prompt, raw, candidateLabel, null, attempt, validParse, callResult.tokenUsage);
      }
      lastRawOutput = raw;
      lastErrorKind = "parse";
      lastError = {
        rowId: row.rowId,
        metric: row.metric,
        attempt,
        message: parsed.error ?? `Invalid judge label: ${String(parsed.label)}`
      };
    } catch (error) {
      const message = renderError(error);
      observer.recordCall({
        phase: "judge_afferition",
        stage: `judge-afferition:${modelId}`,
        backend_category: classifyBackend({
          provider: judge.inference.provider,
          apiBaseUrl: judge.inference.apiBaseUrl,
          modelId
        }),
        backend_id: modelId,
        model_id: modelId,
        dataset: row.dataset,
        metric: row.metric,
        row_id: row.rowId,
        attempt,
        duration_seconds: roundSeconds(Date.now() - attemptStarted),
        ok: false,
        failure_kind: "exception",
        error: message
      });
      if (isContextOverflowError(message)) {
        return buildCandidateArtifact(row, prompt, "", null, message, attempt, false, null);
      }
      lastErrorKind = "provider";
      lastError = {
        rowId: row.rowId,
        metric: row.metric,
        attempt,
        message
      };
    }
    if (attempt < JUDGE_AFFERITION_MAX_ATTEMPTS) {
      await sleep(retryDelayMs(attempt));
    }
  }
  if (lastErrorKind === "parse" && lastError) {
    return buildCandidateArtifact(row, prompt, lastRawOutput, null, lastError.message, lastError.attempt, false, lastTokenUsage);
  }
  throw new JudgeAfferitionRowFailure(lastError ?? {
    rowId: row.rowId,
    metric: row.metric,
    attempt: JUDGE_AFFERITION_MAX_ATTEMPTS,
    message: "Unknown judge afferition failure"
  });
}

function buildCandidateArtifact(
  row: ValidationReferenceRow,
  prompt: string,
  raw: string,
  candidateLabel: BinaryLabel | null,
  parseError: string | null,
  attempts: number,
  validParse: boolean,
  tokenUsage: TokenUsage | null
): CandidateArtifact {
  return {
    rowId: row.rowId,
    dataset: row.dataset,
    metric: row.metric,
    sourceRowIndex: row.sourceRowIndex,
    prompt: row.prompt,
    responseModel: row.responseModel,
    response: row.response,
    referenceJudge: row.referenceJudge,
    referenceLabel: row.referenceLabel,
    candidateLabel,
    validParse,
    candidateRawOutput: raw,
    parseError,
    attempts,
    tokenUsage
  };
}

function normalizeJudgeCallResult(result: string | JudgeCallResult): JudgeCallResult {
  return typeof result === "string" ? { raw: result, tokenUsage: null } : result;
}

function toBinaryLabel(label: string | null): BinaryLabel | null {
  if (label === "0") {
    return 0;
  }
  if (label === "1") {
    return 1;
  }
  return null;
}

function readExistingArtifacts(outputPath: string): CandidateArtifact[] {
  const labelsPath = path.join(outputPath, "labels_candidate.jsonl");
  if (!fs.existsSync(labelsPath)) {
    return [];
  }
  const artifacts = readTextFile(labelsPath)
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as CandidateArtifact);
  return [...new Map(artifacts.map((artifact) => [artifact.rowId, artifact])).values()];
}

function writeProgress(outputPath: string, modelId: string, total: number, completed: number, remaining: number): void {
  writeJson(path.join(outputPath, "judge_afferition_progress.json"), {
    updated_at: utcNowIso(),
    model_id: modelId,
    total,
    completed,
    remaining
  });
}

function buildMetricSummaries(artifacts: CandidateArtifact[]): JudgeValidationMetricSummary[] {
  const groups = new Map<string, CandidateArtifact[]>();
  for (const artifact of artifacts) {
    addGroup(groups, `global:${artifact.metric}`, artifact);
    addGroup(groups, `${artifact.dataset}:${artifact.metric}`, artifact);
  }
  return [...groups.entries()]
    .map(([key, rows]) => {
      const [dataset, metric] = key.split(":") as [string, string];
      return computeMetricSummary(dataset, metric, computeSimilarityCounts(rows));
    })
    .sort((left, right) => `${left.dataset}:${left.metric}`.localeCompare(`${right.dataset}:${right.metric}`));
}

function summarizeCandidateTokenUsage(artifacts: CandidateArtifact[]): TokenUsageSummary {
  const summary = emptyTokenUsageSummary();
  for (const artifact of artifacts) {
    addTokenUsage(summary, artifact.tokenUsage);
  }
  return summary;
}

function addGroup(groups: Map<string, CandidateArtifact[]>, key: string, row: CandidateArtifact): void {
  const current = groups.get(key) ?? [];
  current.push(row);
  groups.set(key, current);
}

function renderMarkdownReport(payload: {
  created_at: string;
  candidate: string;
  reference: string;
  overall_similarity: number | null;
  afferition_sampling: JudgeAfferitionSamplingSummary;
  token_usage: TokenUsageSummary;
  metrics: JudgeValidationMetricSummary[];
}): string {
  const lines = [
    "# Judge Afferition",
    "",
    `Candidate: ${payload.candidate}`,
    `Reference: ${payload.reference}`,
    `Created: ${payload.created_at}`,
    `Sampling: ${formatSampling(payload.afferition_sampling)}`,
    `Overall similarity: ${formatPercent(payload.overall_similarity)}`,
    `Token usage: calls=${payload.token_usage.calls}, input=${payload.token_usage.inputTokens}, output=${payload.token_usage.outputTokens}, total=${payload.token_usage.totalTokens}, provider_usage=${payload.token_usage.providerUsageCalls}, estimated=${payload.token_usage.estimatedCalls}`,
    "",
    "| Dataset | Metric | Total | Valid | Matching | Similarity | Invalid |",
    "|---|---|---:|---:|---:|---:|---:|"
  ];
  for (const metric of payload.metrics) {
    lines.push(
      [
        metric.dataset,
        metric.metric,
        String(metric.total),
        String(metric.validN),
        String(metric.matchingN),
        formatPercent(metric.similarity),
        formatPercent(metric.invalidRate)
      ].join(" | ").replace(/^/, "| ").replace(/$/, " |")
    );
  }
  lines.push("");
  return lines.join("\n");
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

function formatSampling(sampling: JudgeAfferitionSamplingSummary): string {
  if (sampling.kind === "full") {
    return `full (${sampling.sampleTotal}/${sampling.fullTotal})`;
  }
  if (sampling.kind === "test_set") {
    return `${sampling.marginLabel}, sample=${sampling.sampleTotal}/${sampling.fullTotal}, sample_by=${sampling.sampleBy}`;
  }
  return `${sampling.marginLabel} margem de erro, confidence=${formatPercent(sampling.confidence)}, sample=${sampling.sampleTotal}/${sampling.fullTotal}, sample_by=${sampling.sampleBy}`;
}

function writeRunningState(
  outputPath: string,
  state: JudgeAfferitionRunState,
  completed: number,
  total: number,
  lastProcessedRowId: string,
  fatalError: JudgeAfferitionLastError | null
): JudgeAfferitionRunState {
  const nextState: JudgeAfferitionRunState = {
    ...state,
    status: fatalError ? "failed" : "running",
    completed,
    remaining: Math.max(0, total - completed),
    lastProcessedRowId,
    lastError: fatalError
  };
  writeJudgeAfferitionRunState(outputPath, nextState);
  return nextState;
}

function writeFailedState(
  outputPath: string,
  state: JudgeAfferitionRunState,
  completed: number,
  total: number,
  lastError: JudgeAfferitionLastError
): JudgeAfferitionRunState {
  const nextState: JudgeAfferitionRunState = {
    ...state,
    status: "failed",
    completed,
    remaining: Math.max(0, total - completed),
    lastError
  };
  writeJudgeAfferitionRunState(outputPath, nextState);
  return nextState;
}

function writeCompletedState(
  outputPath: string,
  state: JudgeAfferitionRunState,
  completed: number,
  total: number,
  lastProcessedRowId: string | null
): JudgeAfferitionRunState {
  const nextState: JudgeAfferitionRunState = {
    ...state,
    status: "completed",
    completed,
    remaining: Math.max(0, total - completed),
    completedAt: utcNowIso(),
    lastProcessedRowId,
    lastError: null
  };
  writeJudgeAfferitionRunState(outputPath, nextState);
  return nextState;
}

function judgeAfferitionRetryDelayMs(attempt: number): number {
  const exponentialSeconds = Math.min(30, 2 ** Math.max(0, attempt - 1));
  const jitterMs = Math.floor(Math.random() * 250);
  return exponentialSeconds * 1000 + jitterMs;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function roundSeconds(durationMs: number): number {
  return Math.round((durationMs / 1000) * 1000) / 1000;
}

function isContextOverflowError(message: string): boolean {
  return /context (size )?(has been )?exceeded|context length|maximum context|too many tokens/i.test(message);
}
