import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { emitKeypressEvents } from "node:readline";
import readline from "node:readline/promises";

import { appendJsonl, readTextFile, utcNowIso, writeJson } from "../io/filesystem.js";

export type BackendCategory = "LMS local" | "LMS remoto" | "Ollama" | "OR" | "MiniMax" | "Outro";
export type CallPhase = "generation" | "judge" | "judge_afferition" | "preflight" | "lms";

export interface UiCallRecord {
  timestamp: string;
  phase: CallPhase;
  stage: string | null;
  backend_category: BackendCategory;
  backend_id: string | null;
  model_id: string | null;
  profile: string | null;
  dataset: string | null;
  mode: string | null;
  metric: string | null;
  row_id: string | number | boolean | null;
  attempt: number | null;
  duration_seconds: number | null;
  ok: boolean;
  failure_kind: string | null;
  error: string | null;
}

export interface UnitOutcome {
  ok?: boolean;
  failureKind?: string | null;
}

interface BackendStats {
  answeredInStage: number;
  retries: number;
  failures: number;
  calls: number;
  durationSeconds: number;
}

interface UiState {
  updated_at: string;
  description: string;
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
  calls: {
    total: number;
    retries: number;
    failures: number;
    average_duration_seconds: number | null;
  };
  backends: Record<string, {
    answered_in_stage: number;
    retries: number;
    failures: number;
    calls: number;
    average_duration_seconds: number | null;
  }>;
}

const ANSI = {
  clear: "\u001b[2J",
  home: "\u001b[H",
  hideCursor: "\u001b[?25l",
  showCursor: "\u001b[?25h",
  reset: "\u001b[0m",
  dim: "\u001b[2m",
  green: "\u001b[32m",
  red: "\u001b[31m",
  yellow: "\u001b[33m",
  cyan: "\u001b[36m",
  inverse: "\u001b[7m"
} as const;

export class StageHandle {
  public constructor(
    private readonly observer: TerminalObserver,
    private readonly stageId: string | null
  ) {}

  public advance(amount = 1, outcome: UnitOutcome = {}): void {
    this.observer.advanceStage(this.stageId, amount, outcome);
  }

  public close(): void {
    this.observer.finishStage(this.stageId);
  }
}

export class TerminalObserver {
  private readonly enabled: boolean;
  private readonly events: string[] = [];
  private readonly calls: UiCallRecord[] = [];
  private readonly backendStats = new Map<BackendCategory, BackendStats>();
  private readonly keypressHandler: (chunk: string, key: KeypressKey) => void;
  private outputRoot: string | null = null;
  private callsPath: string | null = null;
  private statePath: string | null = null;
  private overallTotal = 1;
  private overallValue = 0;
  private overallOk = 0;
  private overallFailed = 0;
  private overallTask = "Overall benchmark";
  private stageTotal = 1;
  private stageValue = 0;
  private stageOk = 0;
  private stageFailed = 0;
  private stageTask: string | null = null;
  private startedAtMs = Date.now();
  private callDurationTotal = 0;
  private callCount = 0;
  private retryCount = 0;
  private failureCount = 0;
  private scrollOffsetFromEnd = 0;
  private isRawModeEnabled = false;
  private lastRenderMs = 0;

  public constructor(enabled = true, private readonly eventLimit = 120) {
    this.enabled = enabled && process.stdout.isTTY;
    this.keypressHandler = (_chunk, key) => this.handleKeypress(key);
  }

  public isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Attaches UI persistence to a run directory and restores the call window when resuming.
   */
  public configureRun(outputRoot: string): void {
    this.outputRoot = outputRoot;
    this.callsPath = path.join(outputRoot, "ui_calls.jsonl");
    this.statePath = path.join(outputRoot, "ui_state.json");
    this.restoreState();
    this.restoreCalls();
    if (this.backfillCallsFromRawArtifacts()) {
      this.persistCalls();
    }
    this.sortCalls();
    this.rebuildCallStats();
    this.restoreProgressFromRunEvents();
    this.writeState();
  }

  public start(total: number, description = "Overall benchmark"): void {
    this.overallTotal = Math.max(1, total);
    this.overallTask = description;
    this.startedAtMs = Date.now();
    this.enableLiveInput();
    this.render(true);
    this.writeState();
  }

  public stop(): void {
    this.disableLiveInput();
    if (this.enabled) {
      process.stdout.write(`${ANSI.showCursor}${ANSI.reset}\n`);
    }
    this.writeState();
  }

  public pause(): boolean {
    if (!this.enabled) {
      return false;
    }
    this.disableLiveInput();
    process.stdout.write(`${ANSI.showCursor}${ANSI.reset}\n`);
    return true;
  }

  public resume(wasLive: boolean): void {
    if (!wasLive || !this.enabled) {
      return;
    }
    this.enableLiveInput();
    this.render(true);
  }

  public event(eventType: string, payload: Record<string, unknown>): void {
    const line = `${localTime()} ${eventType}${formatPayload(payload)}`;
    this.events.push(line);
    while (this.events.length > this.eventLimit) {
      this.events.shift();
    }
    this.recordCallFromEvent(eventType, payload);
    this.render();
  }

  public recordCall(call: Partial<UiCallRecord> & { phase: CallPhase; ok: boolean }): void {
    const record = normalizeCall(call, this.stageTask);
    this.calls.push(record);
    this.updateCallStats(record);
    if (this.callsPath) {
      appendJsonl(this.callsPath, record);
    }
    this.writeState();
    this.render();
  }

  public startStage(description: string, total: number): StageHandle {
    const nextTotal = Math.max(1, total);
    const isSameResumedStage = this.stageTask === description && this.stageTotal === nextTotal;
    this.stageTotal = nextTotal;
    if (!isSameResumedStage) {
      this.stageValue = 0;
      this.stageOk = 0;
      this.stageFailed = 0;
    }
    this.stageTask = description;
    if (!isSameResumedStage) {
      for (const stats of this.backendStats.values()) {
        stats.answeredInStage = 0;
      }
    }
    this.writeState();
    this.render(true);
    return new StageHandle(this, description);
  }

  public advanceOverall(amount = 1, outcome: UnitOutcome = {}): void {
    const normalized = Math.max(0, amount);
    this.overallValue += normalized;
    if (outcome.ok === false) {
      this.overallFailed += normalized;
    } else {
      this.overallOk += normalized;
    }
    this.writeState();
    this.render();
  }

  public advanceStage(stageId: string | null, amount = 1, outcome: UnitOutcome = {}): void {
    if (!stageId) {
      return;
    }
    const normalized = Math.max(0, amount);
    this.stageValue += normalized;
    if (outcome.ok === false) {
      this.stageFailed += normalized;
    } else {
      this.stageOk += normalized;
    }
    this.writeState();
    this.render();
  }

  public completeUnit(outcome: UnitOutcome = {}): void {
    this.advanceStage(this.stageTask, 1, outcome);
    this.advanceOverall(1, outcome);
  }

  public finishStage(stageId: string | null): void {
    if (!stageId) {
      return;
    }
    this.stageTask = null;
    this.stageTotal = 1;
    this.stageValue = 0;
    this.stageOk = 0;
    this.stageFailed = 0;
    this.writeState();
    this.render(true);
  }

  private recordCallFromEvent(eventType: string, payload: Record<string, unknown>): void {
    if (eventType === "generation_attempt" || eventType === "generation_exception") {
      this.recordCall(eventPayloadToCall("generation", eventType, payload));
    } else if (eventType === "judge_attempt" || eventType === "judge_exception") {
      this.recordCall(eventPayloadToCall("judge", eventType, payload));
    }
  }

  private updateCallStats(record: UiCallRecord): void {
    this.callCount += 1;
    this.callDurationTotal += record.duration_seconds ?? 0;
    if ((record.attempt ?? 1) > 1 || !record.ok) {
      this.retryCount += (record.attempt ?? 1) > 1 ? 1 : 0;
    }
    if (!record.ok) {
      this.failureCount += 1;
    }
    const stats = this.statsFor(record.backend_category);
    stats.answeredInStage += 1;
    stats.calls += 1;
    stats.durationSeconds += record.duration_seconds ?? 0;
    if ((record.attempt ?? 1) > 1) {
      stats.retries += 1;
    }
    if (!record.ok) {
      stats.failures += 1;
    }
  }

  private statsFor(category: BackendCategory): BackendStats {
    const current = this.backendStats.get(category);
    if (current) {
      return current;
    }
    const created = { answeredInStage: 0, retries: 0, failures: 0, calls: 0, durationSeconds: 0 };
    this.backendStats.set(category, created);
    return created;
  }

  private restoreCalls(): void {
    if (!this.callsPath || !fs.existsSync(this.callsPath)) {
      return;
    }
    const restored = readTextFile(this.callsPath)
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as UiCallRecord);
    for (const call of restored) {
      this.calls.push(call);
    }
  }

  private backfillCallsFromRawArtifacts(): boolean {
    if (!this.outputRoot) {
      return false;
    }
    const existingKeys = new Set(this.calls.map(callKey));
    const generationCalls = readJsonlRecords(path.join(this.outputRoot, "raw_generation.jsonl"))
      .map(rawGenerationToCall);
    const judgeCalls = readJsonlRecords(path.join(this.outputRoot, "raw_judge.jsonl"))
      .map(rawJudgeToCall);
    let added = false;
    for (const call of [...generationCalls, ...judgeCalls]) {
      const key = callKey(call);
      if (existingKeys.has(key)) {
        continue;
      }
      this.calls.push(call);
      existingKeys.add(key);
      added = true;
    }
    return added;
  }

  private sortCalls(): void {
    this.calls.sort((left, right) => Date.parse(left.timestamp) - Date.parse(right.timestamp));
  }

  private persistCalls(): void {
    if (!this.callsPath) {
      return;
    }
    this.sortCalls();
    fs.writeFileSync(
      this.callsPath,
      this.calls.map((call) => JSON.stringify(call)).join("\n").concat(this.calls.length > 0 ? "\n" : ""),
      "utf8"
    );
  }

  private rebuildCallStats(): void {
    this.backendStats.clear();
    this.callDurationTotal = 0;
    this.callCount = 0;
    this.retryCount = 0;
    this.failureCount = 0;
    for (const call of this.calls) {
      this.updateCallStats(call);
    }
  }

  private restoreState(): void {
    if (!this.statePath || !fs.existsSync(this.statePath)) {
      return;
    }
    try {
      const state = JSON.parse(readTextFile(this.statePath)) as UiState;
      this.overallTotal = Math.max(1, state.overall.total);
      this.overallValue = Math.max(0, state.overall.value);
      this.overallOk = Math.max(0, state.overall.ok);
      this.overallFailed = Math.max(0, state.overall.failed);
      this.overallTask = state.description || this.overallTask;
      this.stageTask = state.stage.name;
      this.stageTotal = Math.max(1, state.stage.total);
      this.stageValue = Math.max(0, state.stage.value);
      this.stageOk = Math.max(0, state.stage.ok);
      this.stageFailed = Math.max(0, state.stage.failed);
    } catch {
      return;
    }
  }

  private restoreProgressFromRunEvents(): void {
    if (!this.outputRoot) {
      return;
    }
    const events = readJsonlRecords(path.join(this.outputRoot, "run_events.jsonl"));
    const latestRun = events.filter((event) => event.event === "run_started").at(-1);
    const latestStage = events.filter((event) => event.event === "stage_started").at(-1);
    if (latestRun) {
      this.overallTotal = Math.max(this.overallTotal, numberOrNull(latestRun.overall_total) ?? 1);
    }
    if (latestStage) {
      this.stageTask = formatStageName(latestStage);
      this.stageTotal = Math.max(1, numberOrNull(latestStage.target_n) ?? this.stageTotal);
    }
    const completed = completedUnitsFromEvents(events);
    this.overallValue = Math.max(this.overallValue, completed.overall);
    this.overallOk = Math.max(this.overallOk, completed.overall);
    if (latestStage) {
      const stageValue = countStageUnits(completed.units, latestStage);
      this.stageValue = Math.max(this.stageValue, stageValue);
      this.stageOk = Math.max(this.stageOk, stageValue);
    }
  }

  private writeState(): void {
    if (!this.statePath) {
      return;
    }
    writeJson(this.statePath, this.buildState());
  }

  private buildState(): UiState {
    return {
      updated_at: utcNowIso(),
      description: this.overallTask,
      stage: {
        name: this.stageTask,
        total: this.stageTotal,
        value: this.stageValue,
        ok: this.stageOk,
        failed: this.stageFailed
      },
      overall: {
        total: this.overallTotal,
        value: this.overallValue,
        ok: this.overallOk,
        failed: this.overallFailed,
        eta_seconds: this.etaSeconds()
      },
      calls: {
        total: this.callCount,
        retries: this.retryCount,
        failures: this.failureCount,
        average_duration_seconds: averageOrNull(this.callDurationTotal, this.callCount)
      },
      backends: Object.fromEntries([...this.backendStats.entries()].map(([category, stats]) => [category, {
        answered_in_stage: stats.answeredInStage,
        retries: stats.retries,
        failures: stats.failures,
        calls: stats.calls,
        average_duration_seconds: averageOrNull(stats.durationSeconds, stats.calls)
      }]))
    };
  }

  private etaSeconds(): number | null {
    if (this.overallValue <= 0 || this.overallValue >= this.overallTotal) {
      return null;
    }
    const elapsedSeconds = Math.max(0, (Date.now() - this.startedAtMs) / 1000);
    const rate = this.overallValue / Math.max(1, elapsedSeconds);
    return Math.round((this.overallTotal - this.overallValue) / Math.max(rate, 0.0001));
  }

  private render(force = false): void {
    if (!this.enabled) {
      return;
    }
    const now = Date.now();
    if (!force && now - this.lastRenderMs < 120) {
      return;
    }
    this.lastRenderMs = now;
    const width = terminalWidth();
    const height = terminalHeight();
    const footerLines = Math.min(11, Math.max(8, height - 4));
    const callLines = Math.max(4, height - footerLines - 2);
    const lines = [
      `${ANSI.inverse}${padRight(` ELEPHANT runtime dashboard | ${this.overallTask}`, width)}${ANSI.reset}`,
      ...this.renderCalls(callLines, width),
      ...this.renderFooter(width)
    ].slice(0, height);
    process.stdout.write(`${ANSI.hideCursor}${ANSI.home}${ANSI.clear}${lines.map((line) => truncateAnsi(line, width)).join("\n")}`);
  }

  private renderCalls(maxLines: number, width: number): string[] {
    const header = `${ANSI.cyan}Chamadas${ANSI.reset} ${this.scrollOffsetFromEnd === 0 ? "(auto-follow)" : `(scroll -${this.scrollOffsetFromEnd})`}`;
    const visible = this.visibleCalls(maxLines - 1);
    const rows = visible.map((call) => renderCallLine(call, width));
    while (rows.length < maxLines - 1) {
      rows.unshift(ANSI.dim + "~" + ANSI.reset);
    }
    return [header, ...rows];
  }

  private visibleCalls(count: number): UiCallRecord[] {
    const end = Math.max(0, this.calls.length - this.scrollOffsetFromEnd);
    const start = Math.max(0, end - count);
    return this.calls.slice(start, end);
  }

  private renderFooter(width: number): string[] {
    const state = this.buildState();
    const overallBar = renderProgressBar(this.overallValue, this.overallTotal, this.overallFailed, Math.max(18, Math.floor(width * 0.34)));
    const stageBar = renderProgressBar(this.stageValue, this.stageTotal, this.stageFailed, Math.max(18, Math.floor(width * 0.34)));
    return [
      separator(width),
      `Total ${overallBar} ${formatCount(this.overallValue, this.overallTotal)} ETA ${formatDuration(state.overall.eta_seconds)} FAILS: ${ANSI.red}${this.overallFailed}${ANSI.reset}`,
      `Etapa ${stageBar} ${formatCount(this.stageValue, this.stageTotal)} ${truncate(this.stageTask ?? "idle", Math.max(10, width - 58))}`,
      `Calls ${this.callCount} | retries ${this.retryCount} | call fails ${this.failureCount} | avg ${formatSeconds(state.calls.average_duration_seconds)} | scroll PgUp/PgDn/Home/End`,
      ...this.renderBackendLines(width)
    ];
  }

  private renderBackendLines(width: number): string[] {
    const categories: BackendCategory[] = ["LMS local", "LMS remoto", "OR", "MiniMax", "Outro"];
    const rendered = categories
      .map((category) => [category, this.backendStats.get(category)] as const)
      .filter(([, stats]) => stats && stats.calls > 0)
      .map(([category, stats]) => `${category}: stage=${stats!.answeredInStage} retries=${stats!.retries} fails=${stats!.failures} avg=${formatSeconds(averageOrNull(stats!.durationSeconds, stats!.calls))}`);
    if (rendered.length === 0) {
      return [`Backends: ${ANSI.dim}sem chamadas ainda${ANSI.reset}`];
    }
    const joined = rendered.join(" | ");
    return wrapLine(`Backends: ${joined}`, width, 3);
  }

  private enableLiveInput(): void {
    if (!this.enabled || this.isRawModeEnabled || !process.stdin.isTTY) {
      return;
    }
    emitKeypressEvents(process.stdin);
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on("keypress", this.keypressHandler);
    this.isRawModeEnabled = true;
  }

  private disableLiveInput(): void {
    if (!this.isRawModeEnabled) {
      return;
    }
    process.stdin.off("keypress", this.keypressHandler);
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
    }
    this.isRawModeEnabled = false;
  }

  private handleKeypress(key: KeypressKey): void {
    if (key.ctrl && key.name === "c") {
      this.stop();
      process.kill(process.pid, "SIGINT");
      return;
    }
    const page = Math.max(5, Math.floor(terminalHeight() / 2));
    if (key.name === "pageup") {
      this.scrollOffsetFromEnd = Math.min(this.calls.length, this.scrollOffsetFromEnd + page);
      this.render(true);
    } else if (key.name === "pagedown") {
      this.scrollOffsetFromEnd = Math.max(0, this.scrollOffsetFromEnd - page);
      this.render(true);
    } else if (key.name === "home") {
      this.scrollOffsetFromEnd = this.calls.length;
      this.render(true);
    } else if (key.name === "end") {
      this.scrollOffsetFromEnd = 0;
      this.render(true);
    }
  }
}

export async function interactiveMenu(prompt: string, choices: Record<string, string>): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  try {
    while (true) {
      process.stdout.write(`\n${prompt}\n`);
      for (const [key, description] of Object.entries(choices)) {
        process.stdout.write(`[${key}] ${description}\n`);
      }
      process.stdout.write("\u0007");
      const answer = (await rl.question("> ")).trim().toLowerCase();
      if (choices[answer]) {
        return answer;
      }
      process.stdout.write("Opcao invalida.\n");
    }
  } finally {
    rl.close();
  }
}

export async function interactiveMenuWithObserver(
  observer: TerminalObserver | null,
  prompt: string,
  choices: Record<string, string>
): Promise<string> {
  const wasLive = observer?.pause() ?? false;
  try {
    return await interactiveMenu(prompt, choices);
  } finally {
    observer?.resume(wasLive);
  }
}

export function requireInteractiveTty(): void {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error("mahout-bench requires an interactive TTY for real benchmark runs.");
  }
}

export function classifyBackend(args: {
  provider?: unknown;
  apiBaseUrl?: unknown;
  backendId?: unknown;
  modelId?: unknown;
}): BackendCategory {
  const provider = String(args.provider ?? "").toLowerCase();
  const apiBaseUrl = String(args.apiBaseUrl ?? "").toLowerCase();
  const backendId = String(args.backendId ?? "").toLowerCase();
  const modelId = String(args.modelId ?? "").toLowerCase();
  const joined = `${provider} ${apiBaseUrl} ${backendId} ${modelId}`;
  if (joined.includes("openrouter") || provider === "or") {
    return "OR";
  }
  if (joined.includes("minimax")) {
    return "MiniMax";
  }
  if (joined.includes("ollama")) {
    return "Ollama";
  }
  if (joined.includes("lmstudio") || joined.includes("lm-studio") || joined.includes("127.0.0.1") || joined.includes("localhost")) {
    return joined.includes("remote") || (!joined.includes("127.0.0.1") && !joined.includes("localhost") && /https?:\/\//.test(apiBaseUrl))
      ? "LMS remoto"
      : "LMS local";
  }
  return "Outro";
}

export function renderProgressBar(value: number, total: number, failed: number, width: number): string {
  const safeTotal = Math.max(1, total);
  const safeWidth = Math.max(6, width);
  const okValue = Math.max(0, value - Math.max(0, failed));
  const okCells = Math.min(safeWidth, Math.round((okValue / safeTotal) * safeWidth));
  const failCells = Math.min(safeWidth, Math.round((Math.max(0, failed) / safeTotal) * safeWidth));
  const emptyCells = Math.max(0, safeWidth - okCells - failCells);
  return `[${ANSI.green}${"=".repeat(okCells)}${ANSI.reset}${" ".repeat(emptyCells)}${ANSI.red}${"!".repeat(failCells)}${ANSI.reset}]`;
}

function eventPayloadToCall(phase: CallPhase, eventType: string, payload: Record<string, unknown>): Partial<UiCallRecord> & { phase: CallPhase; ok: boolean } {
  const ok = eventType.endsWith("_exception") ? false : payload.ok === true;
  return {
    phase,
    stage: stringOrNull(payload.stage) ?? stringOrNull(payload.mode),
    backend_category: classifyBackend({
      provider: payload.provider,
      apiBaseUrl: payload.api_base_url,
      backendId: payload.backend_id,
      modelId: payload.model_id
    }),
    backend_id: stringOrNull(payload.backend_id),
    model_id: stringOrNull(payload.model_id),
    profile: stringOrNull(payload.profile),
    dataset: stringOrNull(payload.dataset),
    mode: stringOrNull(payload.mode),
    metric: stringOrNull(payload.metric),
    row_id: primitiveOrNull(payload.row_id),
    attempt: numberOrNull(payload.attempt),
    duration_seconds: numberOrNull(payload.duration_seconds),
    ok,
    failure_kind: ok ? null : (eventType.endsWith("_exception") ? "exception" : "parse"),
    error: stringOrNull(payload.error)
  };
}

function rawGenerationToCall(payload: Record<string, unknown>): UiCallRecord {
  const ok = payload.parsed_ok === true;
  return normalizeCall({
    phase: "generation",
    timestamp: stringOrNull(payload.timestamp) ?? utcNowIso(),
    stage: stringOrNull(payload.mode),
    backend_category: classifyBackend({
      provider: payload.provider,
      apiBaseUrl: payload.api_base_url,
      backendId: payload.backend_id,
      modelId: payload.model_id
    }),
    backend_id: stringOrNull(payload.backend_id),
    model_id: stringOrNull(payload.model_id),
    profile: stringOrNull(payload.profile),
    dataset: stringOrNull(payload.dataset),
    mode: stringOrNull(payload.mode),
    row_id: primitiveOrNull(payload.row_id),
    attempt: numberOrNull(payload.attempt),
    duration_seconds: numberOrNull(payload.duration_seconds),
    ok,
    failure_kind: ok ? null : (stringOrNull(payload.error) ? "exception" : "parse"),
    error: stringOrNull(payload.error)
  }, null);
}

function rawJudgeToCall(payload: Record<string, unknown>): UiCallRecord {
  const ok = payload.parsed_ok === true;
  return normalizeCall({
    phase: "judge",
    timestamp: stringOrNull(payload.timestamp) ?? utcNowIso(),
    stage: stringOrNull(payload.mode) ?? "judge",
    backend_category: classifyBackend({
      provider: payload.provider,
      apiBaseUrl: payload.api_base_url,
      backendId: payload.backend_id,
      modelId: payload.model_id
    }),
    backend_id: stringOrNull(payload.backend_id),
    model_id: stringOrNull(payload.model_id),
    profile: stringOrNull(payload.profile),
    dataset: stringOrNull(payload.dataset),
    mode: stringOrNull(payload.mode),
    metric: stringOrNull(payload.metric),
    row_id: primitiveOrNull(payload.row_id),
    attempt: numberOrNull(payload.attempt),
    duration_seconds: numberOrNull(payload.duration_seconds),
    ok,
    failure_kind: ok ? null : (stringOrNull(payload.error) ? "exception" : "parse"),
    error: stringOrNull(payload.error)
  }, null);
}

function normalizeCall(call: Partial<UiCallRecord> & { phase: CallPhase; ok: boolean }, currentStage: string | null): UiCallRecord {
  const backendCategory = call.backend_category ?? classifyBackend({
    provider: null,
    apiBaseUrl: null,
    backendId: call.backend_id,
    modelId: call.model_id
  });
  return {
    timestamp: call.timestamp ?? utcNowIso(),
    phase: call.phase,
    stage: call.stage ?? currentStage,
    backend_category: backendCategory,
    backend_id: call.backend_id ?? null,
    model_id: call.model_id ?? null,
    profile: call.profile ?? null,
    dataset: call.dataset ?? null,
    mode: call.mode ?? null,
    metric: call.metric ?? null,
    row_id: call.row_id ?? null,
    attempt: call.attempt ?? null,
    duration_seconds: call.duration_seconds ?? null,
    ok: call.ok,
    failure_kind: call.failure_kind ?? (call.ok ? null : "unknown"),
    error: call.error ?? null
  };
}

function callKey(call: UiCallRecord): string {
  return [
    call.phase,
    call.backend_id,
    call.model_id,
    call.profile,
    call.dataset,
    call.mode,
    call.metric,
    call.row_id,
    call.attempt,
    call.duration_seconds,
    call.ok
  ].map((part) => String(part ?? "")).join("|");
}

function readJsonlRecords(filePath: string): Array<Record<string, unknown>> {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  return readTextFile(filePath)
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function formatStageName(event: Record<string, unknown>): string {
  return [event.profile, event.dataset, event.stage]
    .filter((part) => part !== undefined && part !== null && part !== "")
    .map(String)
    .join(":");
}

function completedUnitsFromEvents(events: Array<Record<string, unknown>>): { units: Set<string>; overall: number } {
  const units = new Set<string>();
  for (const event of events) {
    if (event.event === "generation_checkpoint_hit" || (event.event === "generation_attempt" && event.ok === true)) {
      units.add(["generation", event.profile, event.dataset, event.mode, event.row_id].map(String).join("::"));
    } else if (event.event === "judge_checkpoint_hit" || (event.event === "judge_attempt" && event.ok === true)) {
      units.add(["judge", event.profile, event.dataset, event.mode, event.row_id, event.metric].map(String).join("::"));
    }
  }
  return { units, overall: units.size };
}

function countStageUnits(units: Set<string>, stage: Record<string, unknown>): number {
  const profile = String(stage.profile ?? "");
  const dataset = String(stage.dataset ?? "");
  const mode = String(stage.mode ?? "");
  let count = 0;
  for (const unit of units) {
    if (unit.startsWith(`generation::${profile}::${dataset}::${mode}::`)) {
      count += 1;
    } else if (unit.startsWith(`judge::${profile}::${dataset}::${mode}::`)) {
      count += 1;
    }
  }
  return count;
}

function renderCallLine(call: UiCallRecord, width: number): string {
  const status = call.ok ? `${ANSI.green}OK${ANSI.reset}` : `${ANSI.red}FAIL${ANSI.reset}`;
  const row = call.row_id === null ? "" : ` row=${String(call.row_id)}`;
  const metric = call.metric ? ` ${call.metric}` : "";
  const attempt = call.attempt === null ? "" : ` try=${call.attempt}`;
  const duration = call.duration_seconds === null ? "" : ` ${formatSeconds(call.duration_seconds)}`;
  const error = call.error ? ` ${ANSI.dim}${call.error}${ANSI.reset}` : "";
  const text = `${timeFromIso(call.timestamp)} ${status} ${call.phase} ${call.backend_category} ${call.profile ?? "-"}:${call.dataset ?? "-"}:${call.mode ?? "-"}${metric}${row}${attempt}${duration}${error}`;
  return truncateAnsi(text, width);
}

function formatPayload(payload: Record<string, unknown>): string {
  const rendered = Object.entries(payload)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, 8)
    .map(([key, value]) => `${key}=${shortValue(value)}`)
    .join(" ");
  return rendered ? ` ${rendered}` : "";
}

function wrapLine(text: string, width: number, maxLines: number): string[] {
  const lines: string[] = [];
  let remaining = text;
  while (remaining.length > 0 && lines.length < maxLines) {
    lines.push(remaining.slice(0, width));
    remaining = remaining.slice(width);
  }
  return lines.length > 0 ? lines : [""];
}

function separator(width: number): string {
  return ANSI.dim + "-".repeat(Math.max(1, width)) + ANSI.reset;
}

function truncate(value: string, limit: number): string {
  const visibleLimit = Math.max(1, limit);
  return value.length <= visibleLimit ? value : `${value.slice(0, Math.max(1, visibleLimit - 1))}…`;
}

function truncateAnsi(value: string, limit: number): string {
  const visibleLimit = Math.max(1, limit);
  let output = "";
  let visible = 0;
  for (let index = 0; index < value.length && visible < visibleLimit; index += 1) {
    if (value[index] === "\u001b") {
      const match = value.slice(index).match(/^\u001b\[[0-9;?]*[A-Za-z]/);
      if (match) {
        output += match[0];
        index += match[0].length - 1;
        continue;
      }
    }
    output += value[index];
    visible += 1;
  }
  return visibleLength(value) <= visibleLimit ? value : `${output.slice(0, Math.max(0, output.length - 1))}…${ANSI.reset}`;
}

function visibleLength(value: string): number {
  return value.replace(/\u001b\[[0-9;?]*[A-Za-z]/g, "").length;
}

function padRight(value: string, width: number): string {
  return value.length >= width ? value : `${value}${" ".repeat(width - value.length)}`;
}

function formatCount(value: number, total: number): string {
  const percentage = Math.floor((Math.max(0, value) / Math.max(1, total)) * 100);
  return `${percentage}% | ${value}/${total}`;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) {
    return "--";
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours > 0) {
    return `${hours}h${String(minutes).padStart(2, "0")}m`;
  }
  return `${minutes}m${String(rest).padStart(2, "0")}s`;
}

function formatSeconds(seconds: number | null): string {
  return seconds === null ? "--" : `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
}

function averageOrNull(total: number, count: number): number | null {
  return count === 0 ? null : Math.round((total / count) * 1000) / 1000;
}

function terminalWidth(): number {
  return Math.max(80, process.stdout.columns || 100);
}

function terminalHeight(): number {
  return Math.max(20, process.stdout.rows || 30);
}

function localTime(): string {
  return new Date().toLocaleTimeString("pt-PT", { hour12: false });
}

function timeFromIso(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString("pt-PT", { hour12: false });
}

function shortValue(value: unknown, limit = 72): string {
  const rendered = String(value).replace(/\n/g, " ");
  return rendered.length <= limit ? rendered : `${rendered.slice(0, limit - 3)}...`;
}

function stringOrNull(value: unknown): string | null {
  return value === undefined || value === null || value === "" ? null : String(value);
}

function numberOrNull(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function primitiveOrNull(value: unknown): string | number | boolean | null {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return value === null || value === undefined ? null : String(value);
}

interface KeypressKey {
  name?: string;
  ctrl?: boolean;
}
