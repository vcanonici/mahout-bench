import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  buildStatusReport,
  discoverRuns,
  parseStatusArgs,
  renderRunList,
  renderStatusReport
} from "../src/cli/status.js";

describe("mahout-bench status", () => {
  it("discovers and orders active, paused, then completed runs", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-status-runs-"));
    writeRun(root, "paused", "2026-05-16T10:00:00.000Z", 5, 10);
    writeRun(root, "active", "2026-05-16T11:59:00.000Z", 2, 10);
    writeRun(root, "completed", "2026-05-16T11:50:00.000Z", 10, 10);

    const runs = discoverRuns(root, new Date("2026-05-16T12:00:00.000Z"));

    expect(runs.map((run) => [run.number, run.name, run.status])).toEqual([
      [1, "active", "active"],
      [2, "paused", "paused"],
      [3, "completed", "completed"]
    ]);
  });

  it("renders run selector instructions and next command", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-status-list-"));
    writeRun(root, "active", "2026-05-16T11:59:00.000Z", 2, 10);

    const text = renderRunList(discoverRuns(root, new Date("2026-05-16T12:00:00.000Z")));

    expect(text).toContain("AGENTE: se nao sabe qual escolher");
    expect(text).toContain("[1] active");
    expect(text).toContain("mahout-bench status --run \"1\"");
  });

  it("builds detailed pretty reports with marked JSON", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-status-report-"));
    const runDir = writeRun(root, "active", "2026-05-16T11:59:00.000Z", 2, 10);
    writeJsonl(path.join(runDir, "run_events.jsonl"), [
      {
        timestamp: "2026-05-16T11:00:00.000Z",
        event: "generation_attempt",
        ok: true
      },
      {
        timestamp: "2026-05-16T11:30:00.000Z",
        event: "generation_attempt",
        ok: true
      }
    ]);

    const report = buildStatusReport(runDir, 1, 1);
    const rendered = renderStatusReport(report, "pretty");

    expect(rendered).toContain("Mahout Bench status report");
    expect(rendered).toContain("---BEGIN MAHOUT_STATUS_JSON---");
    expect(rendered).toContain("\"selected_run_number\": 1");
    expect(report.throughput.ok_per_hour).toBe(2);
  });

  it("prints pure JSON and pure human modes", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-status-modes-"));
    const runDir = writeRun(root, "active", "2026-05-16T11:59:00.000Z", 2, 10);
    const report = buildStatusReport(runDir, 2, 6);

    expect(renderStatusReport(report, "json").trim().startsWith("{")).toBe(true);
    expect(renderStatusReport(report, "json")).not.toContain("Mahout Bench status report");
    expect(renderStatusReport(report, "human")).toContain("Mahout Bench status report");
    expect(renderStatusReport(report, "human")).not.toContain("---BEGIN MAHOUT_STATUS_JSON---");
  });

  it("falls back to partial report without ui_state.json", () => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-status-partial-"));
    writeJsonl(path.join(runDir, "run_events.jsonl"), [
      {
        timestamp: "2026-05-16T11:00:00.000Z",
        event: "stage_started",
        profile: "Felix-V",
        dataset: "oeq",
        stage: "build-sample",
        target_n: 30
      },
      {
        timestamp: "2026-05-16T11:01:00.000Z",
        event: "generation_attempt",
        ok: true
      },
      {
        timestamp: "2026-05-16T11:02:00.000Z",
        event: "generation_attempt",
        ok: false
      }
    ]);

    const report = buildStatusReport(runDir, null, 1);

    expect(report.run.confidence).toBe("partial");
    expect(report.stage.name).toBe("Felix-V:oeq:build-sample");
    expect(report.overall.ok).toBe(1);
    expect(report.overall.failed).toBe(1);
  });

  it("validates incompatible output modes and run numbers", () => {
    expect(() => parseStatusArgs(["--json", "--human"])).toThrow("Use only one");
    expect(() => parseStatusArgs(["--run", "0"])).toThrow("positive integer");
  });
});

function writeRun(root: string, name: string, updatedAt: string, value: number, total: number): string {
  const runDir = path.join(root, name);
  fs.mkdirSync(runDir, { recursive: true });
  fs.writeFileSync(path.join(runDir, "ui_state.json"), JSON.stringify({
    updated_at: updatedAt,
    description: "Overall benchmark",
    stage: {
      name: "Felix-V:oeq:build-sample",
      total,
      value,
      ok: value,
      failed: 0
    },
    overall: {
      total,
      value,
      ok: value,
      failed: 0,
      eta_seconds: 3600
    },
    calls: {
      total: value,
      retries: 0,
      failures: 0,
      average_duration_seconds: 1
    },
    backends: {}
  }), "utf8");
  return runDir;
}

function writeJsonl(filePath: string, rows: Array<Record<string, unknown>>): void {
  fs.writeFileSync(filePath, rows.map((row) => JSON.stringify(row)).join("\n").concat("\n"), "utf8");
}
