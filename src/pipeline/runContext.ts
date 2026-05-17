import path from "node:path";

import {
  DEFAULT_OUTPUT_PREFIX,
  DEFAULT_OUTPUTS_DIR,
  type BenchmarkArgs,
  type RunContext
} from "../contracts/autobench.js";
import { appendJsonl, fileExists, localRunStamp, utcNowIso } from "../io/filesystem.js";
import { defaultPackageRoot, resolveDataRoot } from "../runtime/paths.js";
import type { TerminalObserver } from "../runtime/terminalObserver.js";

/**
 * Builds the benchmark run context and resolves output paths from CLI args.
 * Throws when judge-only mode does not point to an existing run.
 */
export function createRunContext(args: BenchmarkArgs): RunContext {
  const repoRoot = defaultPackageRoot();
  const dataRoot = resolveDataRoot();
  const outputRoot = args.outputRoot
    ? path.resolve(args.outputRoot)
    : path.join(dataRoot, DEFAULT_OUTPUTS_DIR, `${DEFAULT_OUTPUT_PREFIX}_${localRunStamp()}`);

  if (args.judgeOnly && !args.outputRoot) {
    throw new Error("--judge-only requires --output-root pointing to an existing run");
  }
  if (args.judgeOnly && !fileExists(outputRoot)) {
    throw new Error(`--judge-only output root does not exist: ${outputRoot}`);
  }

  return {
    repoRoot,
    dataRoot,
    outputRoot,
    eventsPath: path.join(outputRoot, "run_events.jsonl"),
    rawGenerationPath: path.join(outputRoot, "raw_generation.jsonl"),
    rawJudgePath: path.join(outputRoot, "raw_judge.jsonl"),
    quarantinePath: path.join(outputRoot, "quarantine.jsonl"),
    providerEventsPath: path.join(outputRoot, "provider_events.jsonl"),
    generationCheckpointPath: path.join(outputRoot, "generation_checkpoint.json"),
    judgeCheckpointPath: path.join(outputRoot, "judge_checkpoint.json"),
    profilesRoot: args.profilesRoot,
    judgeConfigPath: args.judgeConfig,
    profileNames: args.profiles,
    benchmarkName: args.benchmarkName,
    generationModelId: args.generationModelId,
    generationPool: args.generationPool,
    judgeModelId: args.judgeModelId,
    judgePool: args.judgePool,
    marginOfError: args.marginOfError,
    resumeMode: args.resumeMode
  };
}

/**
 * Appends a structured event to the run log and mirrors it to the terminal observer.
 */
export function logEvent(
  ctx: RunContext,
  eventType: string,
  observer: TerminalObserver | null,
  payload: Record<string, unknown> = {}
): void {
  appendJsonl(ctx.eventsPath, { timestamp: utcNowIso(), event: eventType, ...payload });
  observer?.event(eventType, payload);
}

/**
 * Records a rejected generation candidate with enough context for later audit.
 */
export function quarantine(
  ctx: RunContext,
  reason: string,
  observer: TerminalObserver | null,
  payload: Record<string, unknown>
): void {
  appendJsonl(ctx.quarantinePath, { timestamp: utcNowIso(), reason, ...payload });
  observer?.event("quarantine", { reason, ...payload });
}

/**
 * Rounds a millisecond duration to seconds with millisecond precision.
 */
export function roundSeconds(durationMs: number): number {
  return Math.round((durationMs / 1000) * 1000) / 1000;
}

/**
 * Converts unknown thrown values to readable error text.
 */
export function renderError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Sleeps for the requested duration; used to bound retry backoff.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
