import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { classifyBackend, renderProgressBar, TerminalObserver } from "../src/runtime/terminalObserver.js";

describe("TerminalObserver runtime dashboard", () => {
  it("classifies human backend categories", () => {
    expect(classifyBackend({ provider: "openrouter", modelId: "openrouter_glm47_flash" })).toBe("OR");
    expect(classifyBackend({ provider: "minimax", modelId: "minimax-m27" })).toBe("MiniMax");
    expect(classifyBackend({ provider: "ollama", apiBaseUrl: "http://127.0.0.1:11434/v1" })).toBe("Ollama");
    expect(classifyBackend({ apiBaseUrl: "http://127.0.0.1:1234/v1", backendId: "lmstudio-local-openai-v1" })).toBe("LMS local");
    expect(classifyBackend({ apiBaseUrl: "http://203.0.113.10:1234/v1", backendId: "lmstudio-remote-openai-v1" })).toBe("LMS remoto");
    expect(classifyBackend({ provider: "custom" })).toBe("Outro");
  });

  it("renders progress with failures reserved from the right", () => {
    const bar = stripAnsi(renderProgressBar(6, 10, 2, 10));

    expect(bar).toBe("[====    !!]");
  });

  it("persists calls and UI state inside the run directory", () => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), "terminal-observer-"));
    const observer = new TerminalObserver(false);

    observer.configureRun(runDir);
    observer.start(10, "test run");
    observer.startStage("generation:test", 4);
    observer.recordCall({
      phase: "generation",
      backend_category: "OR",
      backend_id: "openrouter:0",
      model_id: "openrouter_glm47_flash",
      profile: "Felix-V",
      dataset: "oeq",
      mode: "responses",
      row_id: 1,
      attempt: 2,
      duration_seconds: 1.25,
      ok: true
    });
    observer.advanceStage("generation:test", 1);
    observer.advanceOverall(1);
    observer.stop();

    const calls = readJsonl(path.join(runDir, "ui_calls.jsonl"));
    const state = JSON.parse(fs.readFileSync(path.join(runDir, "ui_state.json"), "utf8"));

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      phase: "generation",
      backend_category: "OR",
      attempt: 2,
      ok: true
    });
    expect(state.overall.value).toBe(1);
    expect(state.stage.name).toBe("generation:test");
    expect(state.calls.total).toBe(1);
    expect(state.calls.retries).toBe(1);
    expect(state.backends.OR.answered_in_stage).toBe(1);
  });

  it("restores persisted calls when resuming a run", () => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), "terminal-observer-resume-"));
    const first = new TerminalObserver(false);

    first.configureRun(runDir);
    first.recordCall({
      phase: "judge",
      backend_category: "LMS remoto",
      backend_id: "lmstudio-remote-native-v1:0",
      model_id: "lmstudio_native_gemma4_26b_a4b",
      attempt: 1,
      duration_seconds: 2,
      ok: false,
      failure_kind: "parse",
      error: "invalid label"
    });

    const resumed = new TerminalObserver(false);
    resumed.configureRun(runDir);
    resumed.startStage("judge:test", 2);
    resumed.recordCall({
      phase: "judge",
      backend_category: "LMS remoto",
      backend_id: "lmstudio-remote-native-v1:0",
      model_id: "lmstudio_native_gemma4_26b_a4b",
      attempt: 2,
      duration_seconds: 3,
      ok: true
    });

    const state = JSON.parse(fs.readFileSync(path.join(runDir, "ui_state.json"), "utf8"));

    expect(readJsonl(path.join(runDir, "ui_calls.jsonl"))).toHaveLength(2);
    expect(state.calls.total).toBe(2);
    expect(state.calls.failures).toBe(1);
    expect(state.calls.retries).toBe(1);
    expect(state.backends["LMS remoto"].calls).toBe(2);
  });

  it("restores persisted progress when a run is resumed", () => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), "terminal-observer-progress-"));
    const first = new TerminalObserver(false);

    first.configureRun(runDir);
    first.start(100, "original");
    first.startStage("Felix-V:oeq:build-sample", 30);
    first.advanceStage("Felix-V:oeq:build-sample", 7);
    first.advanceOverall(11);
    first.stop();

    const resumed = new TerminalObserver(false);
    resumed.configureRun(runDir);
    resumed.start(100, "resumed");
    resumed.startStage("Felix-V:oeq:build-sample", 30);
    resumed.advanceStage("Felix-V:oeq:build-sample", 1);
    resumed.advanceOverall(1);

    const state = JSON.parse(fs.readFileSync(path.join(runDir, "ui_state.json"), "utf8"));

    expect(state.stage.value).toBe(8);
    expect(state.overall.value).toBe(12);
  });

  it("backfills UI calls and progress from legacy run artifacts on resume", () => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), "terminal-observer-backfill-"));
    writeJsonl(path.join(runDir, "run_events.jsonl"), [
      {
        timestamp: "2026-05-14T15:00:00.000Z",
        event: "run_started",
        overall_total: 100
      },
      {
        timestamp: "2026-05-14T15:00:01.000Z",
        event: "stage_started",
        profile: "Felix-V",
        dataset: "oeq",
        stage: "build-sample",
        mode: "responses",
        target_n: 30
      },
      {
        timestamp: "2026-05-14T15:00:10.000Z",
        event: "generation_attempt",
        profile: "Felix-V",
        dataset: "oeq",
        mode: "responses",
        row_id: 42,
        ok: true
      }
    ]);
    writeJsonl(path.join(runDir, "raw_generation.jsonl"), [
      {
        timestamp: "2026-05-14T15:00:10.000Z",
        provider: "lmstudio",
        api_base_url: "http://127.0.0.1:1234/v1",
        backend_id: "lmstudio-local-openai-v1:0",
        model_id: "lmstudio_openai_gemma4_26b",
        profile: "Felix-V",
        dataset: "oeq",
        mode: "responses",
        row_id: 42,
        attempt: 1,
        duration_seconds: 12.5,
        parsed_ok: true
      }
    ]);

    const resumed = new TerminalObserver(false);
    resumed.configureRun(runDir);
    resumed.start(100, "resumed");
    resumed.startStage("Felix-V:oeq:build-sample", 30);

    const calls = readJsonl(path.join(runDir, "ui_calls.jsonl"));
    const state = JSON.parse(fs.readFileSync(path.join(runDir, "ui_state.json"), "utf8"));

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      phase: "generation",
      backend_category: "LMS local",
      row_id: 42,
      ok: true
    });
    expect(state.stage.value).toBe(1);
    expect(state.overall.value).toBe(1);
    expect(state.backends["LMS local"].answered_in_stage).toBe(1);
  });
});

function readJsonl(filePath: string): Array<Record<string, unknown>> {
  return fs.readFileSync(filePath, "utf8").trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
}

function writeJsonl(filePath: string, rows: Array<Record<string, unknown>>): void {
  fs.writeFileSync(filePath, rows.map((row) => JSON.stringify(row)).join("\n").concat("\n"), "utf8");
}

function stripAnsi(value: string): string {
  return value.replace(/\u001b\[[0-9;?]*[A-Za-z]/g, "");
}
