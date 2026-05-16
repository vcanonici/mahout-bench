#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { resolveOutputBase } from "../runtime/paths.js";

type OutputMode = "pretty" | "json" | "human";
type RunStatus = "active" | "paused" | "completed" | "unknown";
type Freshness = "fresh" | "stale" | "unknown";
type Confidence = "high" | "medium" | "partial";

interface StatusArgs {
  outputRoot: string | null;
  runNumber: number | null;
  outputMode: OutputMode;
  recentHours: number;
  outputsDir: string;
}

interface UiState {
  updated_at: string;
  description?: string;
  stage: {
    name: string | null;
    total: number;
    value: number;
    ok: number;
    failed: number;
  };
  overall: {
    total: number;
    value: number;
    ok: number;
    failed: number;
    eta_seconds: number | null;
  };
  calls?: {
    total: number;
    retries: number;
    failures: number;
    average_duration_seconds: number | null;
  };
  backends?: Record<string, {
    answered_in_stage: number;
    retries: number;
    failures: number;
    calls: number;
    average_duration_seconds: number | null;
  }>;
}

interface RunCandidate {
  number: number;
  name: string;
  outputRoot: string;
  updatedAt: string | null;
  status: RunStatus;
  freshness: Freshness;
  stageName: string | null;
  stageValue: number;
  stageTotal: number;
  overallValue: number;
  overallTotal: number;
  etaSeconds: number | null;
}

interface ThroughputSummary {
  recent_hours: number;
  ok_per_hour: number | null;
  source: string;
}

interface ArtifactSummary {
  ui_state: boolean;
  run_events_lines: number;
  raw_generation_lines: number;
  raw_judge_lines: number;
  ui_calls_lines: number;
}

interface StatusReport {
  schema_version: 1;
  selected_run_number: number | null;
  run: {
    name: string;
    output_root: string;
    updated_at: string | null;
    status: RunStatus;
    freshness: Freshness;
    confidence: Confidence;
  };
  stage: {
    name: string | null;
    value: number;
    total: number;
    remaining: number;
    percent: number | null;
  };
  overall: {
    value: number;
    total: number;
    remaining: number;
    percent: number | null;
    ok: number;
    failed: number;
  };
  eta: {
    seconds: number | null;
    hours: number | null;
    human: string;
    source: string;
  };
  throughput: ThroughputSummary;
  artifacts: ArtifactSummary;
  warnings: string[];
  next_command: string;
}

const JSON_BEGIN = "---BEGIN MAHOUT_STATUS_JSON---";
const JSON_END = "---END MAHOUT_STATUS_JSON---";
const FRESH_MS = 15 * 60 * 1000;

/**
 * Runs the mahout-bench status command.
 */
export async function main(argv = process.argv.slice(2)): Promise<number> {
  const args = parseStatusArgs(argv);
  const candidates = discoverRuns(args.outputsDir);

  if (!args.outputRoot && args.runNumber === null) {
    process.stdout.write(renderRunList(candidates));
    return 0;
  }

  const selected = selectRun(args, candidates);
  const report = buildStatusReport(selected.outputRoot, selected.number, args.recentHours);
  process.stdout.write(renderStatusReport(report, args.outputMode));
  return 0;
}

export function parseStatusArgs(argv: string[]): StatusArgs {
  const args: StatusArgs = {
    outputRoot: null,
    runNumber: null,
    outputMode: "pretty",
    recentHours: 6,
    outputsDir: resolveOutputBase()
  };

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    switch (current) {
      case "--":
        break;
      case "--output-root":
        args.outputRoot = path.resolve(requireValue(argv, ++index, current));
        break;
      case "--run":
        args.runNumber = parseRunNumber(requireValue(argv, ++index, current));
        break;
      case "--json":
        args.outputMode = setOutputMode(args.outputMode, "json");
        break;
      case "--human":
        args.outputMode = setOutputMode(args.outputMode, "human");
        break;
      case "--pretty":
        args.outputMode = setOutputMode(args.outputMode, "pretty");
        break;
      case "--recent-hours":
        args.recentHours = parseRecentHours(requireValue(argv, ++index, current));
        break;
      case "--outputs-dir":
        args.outputsDir = path.resolve(requireValue(argv, ++index, current));
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown status argument: ${current}`);
    }
  }

  if (args.outputRoot && args.runNumber !== null) {
    throw new Error("Use either --output-root or --run, not both.");
  }
  return args;
}

export function discoverRuns(outputsDir = path.resolve("outputs"), now = new Date()): RunCandidate[] {
  if (!fs.existsSync(outputsDir)) {
    return [];
  }

  const candidates = fs.readdirSync(outputsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => buildRunCandidate(path.join(outputsDir, entry.name), entry.name, now))
    .filter((entry): entry is RunCandidate => entry !== null)
    .sort(compareRunCandidates);

  return candidates.map((candidate, index) => ({ ...candidate, number: index + 1 }));
}

export function buildStatusReport(outputRoot: string, selectedRunNumber: number | null, recentHours: number): StatusReport {
  const resolvedRoot = path.resolve(outputRoot);
  if (!fs.existsSync(resolvedRoot)) {
    throw new Error(`Output root not found: ${resolvedRoot}`);
  }

  const warnings: string[] = [];
  const uiState = readUiState(path.join(resolvedRoot, "ui_state.json"), warnings);
  const artifacts = summarizeArtifacts(resolvedRoot, warnings);
  const runEvents = readJsonlRecords(path.join(resolvedRoot, "run_events.jsonl"), warnings);
  const throughput = summarizeThroughput(runEvents, recentHours);
  const candidate = buildRunCandidate(resolvedRoot, path.basename(resolvedRoot), new Date());
  const fallback = fallbackProgress(runEvents);

  const stageTotal = uiState?.stage.total ?? fallback.stageTotal;
  const stageValue = uiState?.stage.value ?? fallback.stageValue;
  const overallTotal = uiState?.overall.total ?? fallback.overallTotal;
  const overallValue = uiState?.overall.value ?? fallback.overallValue;
  const etaSeconds = uiState?.overall.eta_seconds ?? null;
  const confidence = uiState ? (runEvents.length > 0 ? "high" : "medium") : "partial";

  return {
    schema_version: 1,
    selected_run_number: selectedRunNumber,
    run: {
      name: path.basename(resolvedRoot),
      output_root: relativeOrAbsolute(resolvedRoot),
      updated_at: uiState?.updated_at ?? candidate?.updatedAt ?? null,
      status: candidate?.status ?? "unknown",
      freshness: candidate?.freshness ?? "unknown",
      confidence
    },
    stage: {
      name: uiState?.stage.name ?? fallback.stageName,
      value: stageValue,
      total: stageTotal,
      remaining: remaining(stageValue, stageTotal),
      percent: percent(stageValue, stageTotal)
    },
    overall: {
      value: overallValue,
      total: overallTotal,
      remaining: remaining(overallValue, overallTotal),
      percent: percent(overallValue, overallTotal),
      ok: uiState?.overall.ok ?? fallback.ok,
      failed: uiState?.overall.failed ?? fallback.failed
    },
    eta: {
      seconds: etaSeconds,
      hours: etaSeconds === null ? null : round(etaSeconds / 3600, 2),
      human: formatDuration(etaSeconds),
      source: uiState ? "ui_state.json" : "unavailable"
    },
    throughput,
    artifacts,
    warnings,
    next_command: selectedRunNumber === null
      ? `mahout-bench status --output-root "${relativeOrAbsolute(resolvedRoot)}"`
      : `mahout-bench status --run "${selectedRunNumber}"`
  };
}

export function renderRunList(candidates: RunCandidate[]): string {
  const lines = [
    "Mahout Bench status run selector",
    "",
    "AGENTE: se nao sabe qual escolher, mostre esta lista ao user e pergunte qual run ele quer.",
    ""
  ];

  if (candidates.length === 0) {
    lines.push("Nenhuma run ativa/recente com ui_state.json foi encontrada em outputs/.");
    lines.push("Use mahout-bench status --output-root <run-dir> para consultar uma pasta especifica.");
    return lines.join("\n").concat("\n");
  }

  appendGroup(lines, "Active / fresh runs", candidates.filter((run) => run.status === "active"));
  appendGroup(lines, "Paused / stale runs", candidates.filter((run) => run.status === "paused" || run.status === "unknown"));
  appendGroup(lines, "Completed runs", candidates.filter((run) => run.status === "completed"));
  lines.push("Para detalhes:");
  lines.push(`  mahout-bench status --run "${candidates[0]!.number}"`);
  return lines.join("\n").concat("\n");
}

export function renderStatusReport(report: StatusReport, outputMode: OutputMode): string {
  const human = renderHumanReport(report);
  const json = JSON.stringify(report, null, 2);

  if (outputMode === "json") {
    return json.concat("\n");
  }
  if (outputMode === "human") {
    return human;
  }
  return `${human}\n${JSON_BEGIN}\n${json}\n${JSON_END}\n`;
}

function selectRun(args: StatusArgs, candidates: RunCandidate[]): { outputRoot: string; number: number | null } {
  if (args.outputRoot) {
    return { outputRoot: args.outputRoot, number: null };
  }

  const candidate = candidates.find((run) => run.number === args.runNumber);
  if (!candidate) {
    process.stdout.write(renderRunList(candidates));
    throw new Error(`Run number not found: ${args.runNumber}`);
  }
  return { outputRoot: candidate.outputRoot, number: candidate.number };
}

function buildRunCandidate(outputRoot: string, name: string, now: Date): RunCandidate | null {
  const uiState = readUiState(path.join(outputRoot, "ui_state.json"), []);
  if (!uiState) {
    return null;
  }
  const freshness = classifyFreshness(uiState.updated_at, now);
  const isComplete = uiState.overall.total > 0 && uiState.overall.value >= uiState.overall.total;
  const status = isComplete ? "completed" : freshness === "fresh" ? "active" : "paused";

  return {
    number: 0,
    name,
    outputRoot,
    updatedAt: uiState.updated_at,
    status,
    freshness,
    stageName: uiState.stage.name,
    stageValue: uiState.stage.value,
    stageTotal: uiState.stage.total,
    overallValue: uiState.overall.value,
    overallTotal: uiState.overall.total,
    etaSeconds: uiState.overall.eta_seconds
  };
}

function compareRunCandidates(left: RunCandidate, right: RunCandidate): number {
  const groupDiff = groupRank(left.status) - groupRank(right.status);
  if (groupDiff !== 0) {
    return groupDiff;
  }
  return timestampMs(right.updatedAt) - timestampMs(left.updatedAt);
}

function groupRank(status: RunStatus): number {
  if (status === "active") {
    return 0;
  }
  if (status === "paused" || status === "unknown") {
    return 1;
  }
  return 2;
}

function appendGroup(lines: string[], title: string, runs: RunCandidate[]): void {
  lines.push(title);
  if (runs.length === 0) {
    lines.push("  none");
    lines.push("");
    return;
  }
  for (const run of runs) {
    lines.push(`  [${run.number}] ${run.name} updated=${run.updatedAt ?? "unknown"} status=${run.status} stage=${run.stageName ?? "unknown"} progress=${formatCount(run.overallValue, run.overallTotal)} eta=${formatDuration(run.etaSeconds)}`);
  }
  lines.push("");
}

function renderHumanReport(report: StatusReport): string {
  const selected = report.selected_run_number === null ? "explicit output-root" : String(report.selected_run_number);
  const lines = [
    "Mahout Bench status report",
    "",
    `Selected run: ${selected}`,
    `Run name: ${report.run.name}`,
    `Output root: ${report.run.output_root}`,
    `Updated at: ${report.run.updated_at ?? "unknown"}`,
    `Run status: ${report.run.status}`,
    `Data freshness: ${report.run.freshness}`,
    `Confidence: ${report.run.confidence}`,
    "",
    `Current step: ${report.stage.name ?? "unknown"}`,
    `Step progress: ${formatCount(report.stage.value, report.stage.total)} (${formatPercent(report.stage.percent)}), remaining ${report.stage.remaining}`,
    `Overall progress: ${formatCount(report.overall.value, report.overall.total)} (${formatPercent(report.overall.percent)}), remaining ${report.overall.remaining}`,
    `Overall OK: ${report.overall.ok}`,
    `Overall failures: ${report.overall.failed}`,
    "",
    `ETA: ${report.eta.human}`,
    `ETA source: ${report.eta.source}`,
    `Recent throughput: ${formatThroughput(report.throughput.ok_per_hour)} over the last ${report.throughput.recent_hours}h`,
    `Throughput source: ${report.throughput.source}`,
    "",
    "Artifacts read:",
    `  ui_state.json: ${report.artifacts.ui_state ? "yes" : "no"}`,
    `  run_events.jsonl lines: ${report.artifacts.run_events_lines}`,
    `  raw_generation.jsonl lines: ${report.artifacts.raw_generation_lines}`,
    `  raw_judge.jsonl lines: ${report.artifacts.raw_judge_lines}`,
    `  ui_calls.jsonl lines: ${report.artifacts.ui_calls_lines}`,
    "",
    `Meaning: ${meaning(report)}`,
    `Repeat command: ${report.next_command}`
  ];

  if (report.warnings.length > 0) {
    lines.push("");
    lines.push("Warnings:");
    for (const warning of report.warnings) {
      lines.push(`  - ${warning}`);
    }
  }

  return lines.join("\n").concat("\n");
}

function meaning(report: StatusReport): string {
  if (report.run.status === "completed") {
    return "benchmark appears complete because overall progress reached the planned total.";
  }
  if (report.stage.name) {
    return `benchmark is currently at ${report.stage.name}; use the ETA and throughput fields as operational estimates, not as a contractual finish time.`;
  }
  return "benchmark progress is partial because the current stage could not be recovered from ui_state.json.";
}

function summarizeArtifacts(outputRoot: string, warnings: string[]): ArtifactSummary {
  return {
    ui_state: fs.existsSync(path.join(outputRoot, "ui_state.json")),
    run_events_lines: countLines(path.join(outputRoot, "run_events.jsonl"), warnings),
    raw_generation_lines: countLines(path.join(outputRoot, "raw_generation.jsonl"), warnings),
    raw_judge_lines: countLines(path.join(outputRoot, "raw_judge.jsonl"), warnings),
    ui_calls_lines: countLines(path.join(outputRoot, "ui_calls.jsonl"), warnings)
  };
}

function summarizeThroughput(records: Array<Record<string, unknown>>, recentHours: number): ThroughputSummary {
  const attempts = records.filter((record) => isOkAttempt(record));
  if (attempts.length === 0) {
    return { recent_hours: recentHours, ok_per_hour: null, source: "run_events.jsonl" };
  }

  const latest = Math.max(...attempts.map((record) => timestampMs(asString(record.timestamp))));
  const cutoff = latest - recentHours * 3600 * 1000;
  const recent = attempts.filter((record) => timestampMs(asString(record.timestamp)) >= cutoff);
  return {
    recent_hours: recentHours,
    ok_per_hour: round(recent.length / recentHours, 1),
    source: "run_events.jsonl"
  };
}

function fallbackProgress(records: Array<Record<string, unknown>>): {
  stageName: string | null;
  stageValue: number;
  stageTotal: number;
  overallValue: number;
  overallTotal: number;
  ok: number;
  failed: number;
} {
  const started = records.filter((record) => asString(record.event) === "stage_started").at(-1);
  const attempts = records.filter((record) => isAttempt(record));
  const ok = attempts.filter((record) => record.ok === true).length;
  const failed = attempts.filter((record) => record.ok === false).length;
  return {
    stageName: started ? [asString(started.profile), asString(started.dataset), asString(started.stage)].filter(Boolean).join(":") : null,
    stageValue: ok + failed,
    stageTotal: asNumber(started?.target_n) ?? 0,
    overallValue: ok + failed,
    overallTotal: 0,
    ok,
    failed
  };
}

function readUiState(filePath: string, warnings: string[]): UiState | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return normalizeUiState(JSON.parse(fs.readFileSync(filePath, "utf8")));
  } catch (error) {
    warnings.push(`Could not read ui_state.json: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}

function normalizeUiState(value: unknown): UiState {
  const record = requireRecord(value, "ui_state.json");
  const stage = requireRecord(record.stage, "ui_state.stage");
  const overall = requireRecord(record.overall, "ui_state.overall");

  return {
    updated_at: requireString(record.updated_at, "ui_state.updated_at"),
    description: asString(record.description) || undefined,
    stage: {
      name: nullableString(stage.name),
      total: requireNumber(stage.total, "ui_state.stage.total"),
      value: requireNumber(stage.value, "ui_state.stage.value"),
      ok: requireNumber(stage.ok, "ui_state.stage.ok"),
      failed: requireNumber(stage.failed, "ui_state.stage.failed")
    },
    overall: {
      total: requireNumber(overall.total, "ui_state.overall.total"),
      value: requireNumber(overall.value, "ui_state.overall.value"),
      ok: requireNumber(overall.ok, "ui_state.overall.ok"),
      failed: requireNumber(overall.failed, "ui_state.overall.failed"),
      eta_seconds: nullableNumber(overall.eta_seconds)
    }
  };
}

function readJsonlRecords(filePath: string, warnings: string[]): Array<Record<string, unknown>> {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  const text = fs.readFileSync(filePath, "utf8").trim();
  if (!text) {
    return [];
  }
  const records: Array<Record<string, unknown>> = [];
  const lines = text.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    try {
      records.push(requireRecord(JSON.parse(lines[index]!), `${path.basename(filePath)}:${index + 1}`));
    } catch (error) {
      warnings.push(`Could not parse ${path.basename(filePath)} line ${index + 1}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return records;
}

function countLines(filePath: string, warnings: string[]): number {
  if (!fs.existsSync(filePath)) {
    return 0;
  }
  try {
    const text = fs.readFileSync(filePath, "utf8").trim();
    return text ? text.split("\n").length : 0;
  } catch (error) {
    warnings.push(`Could not count ${path.basename(filePath)}: ${error instanceof Error ? error.message : String(error)}`);
    return 0;
  }
}

function isOkAttempt(record: Record<string, unknown>): boolean {
  return isAttempt(record) && record.ok === true && Boolean(asString(record.timestamp));
}

function isAttempt(record: Record<string, unknown>): boolean {
  const event = asString(record.event);
  return event === "generation_attempt" || event === "judge_attempt" || event === "judge_afferition_attempt";
}

function classifyFreshness(value: string, now: Date): Freshness {
  const ms = timestampMs(value);
  if (ms === 0) {
    return "unknown";
  }
  return now.getTime() - ms <= FRESH_MS ? "fresh" : "stale";
}

function timestampMs(value: string | null): number {
  if (!value) {
    return 0;
  }
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : 0;
}

function percent(value: number, total: number): number | null {
  if (total <= 0) {
    return null;
  }
  return round((value / total) * 100, 2);
}

function remaining(value: number, total: number): number {
  return Math.max(total - value, 0);
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatCount(value: number, total: number): string {
  return `${value} / ${total}`;
}

function formatPercent(value: number | null): string {
  return value === null ? "unknown" : `${value.toFixed(2)}%`;
}

function formatThroughput(value: number | null): string {
  return value === null ? "unknown ok/hour" : `${value.toFixed(1)} ok/hour`;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) {
    return "unknown";
  }
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected object for ${label}`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Expected non-empty string for ${label}`);
  }
  return value;
}

function requireNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Expected finite number for ${label}`);
  }
  return value;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : requireString(value, "nullable string");
}

function nullableNumber(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  return requireNumber(value, "nullable number");
}

function relativeOrAbsolute(filePath: string): string {
  const relative = path.relative(process.cwd(), filePath);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative) ? relative : filePath;
}

function parseRunNumber(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`Expected --run to be a positive integer, got: ${value}`);
  }
  return parsed;
}

function parseRecentHours(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Expected --recent-hours to be greater than zero, got: ${value}`);
  }
  return parsed;
}

function setOutputMode(current: OutputMode, next: OutputMode): OutputMode {
  if (current !== "pretty" && current !== next) {
    throw new Error("Use only one of --json, --human, or --pretty.");
  }
  return next;
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function printHelp(): void {
  process.stdout.write(`mahout-bench status\n\n`);
  process.stdout.write(`Lists benchmark runs or reports detailed status for one run.\n\n`);
  process.stdout.write(`Commands:\n`);
  process.stdout.write(`  mahout-bench status\n`);
  process.stdout.write(`  mahout-bench status --run "1"\n`);
  process.stdout.write(`  mahout-bench status --output-root outputs/<run>\n\n`);
  process.stdout.write(`Flags:\n`);
  process.stdout.write(`  --run <n>             Select a numbered run from mahout-bench status.\n`);
  process.stdout.write(`  --output-root <path>  Select a run directory directly.\n`);
  process.stdout.write(`  --json               Print only stable JSON.\n`);
  process.stdout.write(`  --human              Print only the human report.\n`);
  process.stdout.write(`  --pretty             Print human report and marked JSON block. Default.\n`);
  process.stdout.write(`  --recent-hours <n>   Throughput window. Default: 6.\n`);
  process.stdout.write(`  --outputs-dir <path> Override run discovery directory. Default: MAHOUT_BENCH_HOME outputs.\n`);
}

function isDirectCliExecution(): boolean {
  const entryPath = process.argv[1];
  return Boolean(entryPath) && path.resolve(entryPath!) === fileURLToPath(import.meta.url);
}

if (isDirectCliExecution()) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}
