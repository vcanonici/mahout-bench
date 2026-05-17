import fs from "node:fs";
import path from "node:path";

import {
  MORAL_A_TASK,
  MORAL_B_TASK,
  type CsvRecord,
  type DatasetConfig,
  type GenerationPoolBackend,
  type ProfileConfig,
  type RunContext,
  type SampleManifest
} from "../contracts/autobench.js";
import { fileExists, readCsvFile, utcNowIso, writeJson } from "../io/filesystem.js";
import { outputFileForMode } from "../scoring/scoreEngine.js";
import { buildSampleManifest } from "../sampling/samplePlanner.js";
import { readGenerationResults } from "./checkpoint.js";

export type ResumeGenerationState = {
  manifest: SampleManifest;
  generatedUnits: number;
  report: ResumeCheckReport;
};

export type ResumeCheckReport = {
  created_at: string;
  mode: "fast" | "check";
  output_root: string;
  status: "complete" | "partial";
  generation_checkpoint_results: number;
  generated_units: number;
  missing_artifacts: string[];
  incomplete_outputs: ResumeIncompleteOutput[];
  final_generation_failures: ResumeFailureSummary[];
};

export type ResumeIncompleteOutput = {
  profile: string;
  dataset: string;
  mode: string;
  expected: number;
  actual: number;
  path: string;
};

export type ResumeFailureSummary = {
  profile: string | null;
  dataset: string | null;
  mode: string | null;
  metric: string | null;
  row_id: string | number | boolean | null;
  error: string | null;
};

/**
 * Reconstructs completed generation state from persisted response artifacts.
 * It avoids per-unit checkpoint replay during resume and returns null when the
 * generation artifacts are not complete enough to skip the generation phase.
 */
export function reconstructResumeGenerationState(args: {
  ctx: RunContext;
  mode: "fast" | "check";
  profiles: ProfileConfig[];
  socialDatasets: DatasetConfig[];
  moralA: DatasetConfig;
  moralB: DatasetConfig;
  socialTargets: Record<string, number>;
  moralTargetN: number;
  datasetPopulations: Record<string, number>;
  generationPool: GenerationPoolBackend[];
  judgePool: GenerationPoolBackend[];
  judgeInference: SampleManifest["judge_inference"];
}): ResumeGenerationState | null {
  const report = buildBaseReport(args.ctx, args.mode);
  const socialIndices: Record<string, number[]> = {};
  let moralIds: string[] = [];
  let generatedUnits = 0;

  for (const [profileIndex, profile] of args.profiles.entries()) {
    for (const dataset of args.socialDatasets) {
      const expected = args.socialTargets[dataset.name] ?? 0;
      const rows = readResponseRows(args.ctx, profile.name, dataset, "responses", expected, report);
      if (!rows) {
        continue;
      }
      generatedUnits += expected;
      if (profileIndex === 0) {
        socialIndices[dataset.name] = sourceIndices(rows).slice(0, expected);
      }
    }

    const moralRows = readMoralRows(args.ctx, profile, args.moralA, args.moralB, args.moralTargetN, report);
    if (!moralRows) {
      continue;
    }
    generatedUnits += args.moralTargetN * 4;
    if (profileIndex === 0) {
      moralIds = sourceIds(moralRows.aFree).slice(0, args.moralTargetN);
    }
  }

  report.generated_units = generatedUnits;
  report.final_generation_failures = collectFinalGenerationFailures(args.ctx);
  report.status = report.missing_artifacts.length === 0 && report.incomplete_outputs.length === 0 ? "complete" : "partial";
  if (args.mode === "check") {
    writeJson(path.join(args.ctx.outputRoot, "resume_check_report.json"), report);
  }
  if (report.status !== "complete") {
    return null;
  }

  return {
    generatedUnits,
    report,
    manifest: buildSampleManifest({
      ctx: args.ctx,
      referenceProfile: args.profiles[0]!,
      profileOrder: args.profiles.map((profile) => profile.name),
      generationPool: args.generationPool,
      judgePool: args.judgePool,
      judgeInference: args.judgeInference,
      socialIndices,
      datasetPopulations: args.datasetPopulations,
      moralIds
    })
  };
}

function buildBaseReport(ctx: RunContext, mode: "fast" | "check"): ResumeCheckReport {
  return {
    created_at: utcNowIso(),
    mode,
    output_root: ctx.outputRoot,
    status: "partial",
    generation_checkpoint_results: Object.keys(readGenerationResults(ctx)).length,
    generated_units: 0,
    missing_artifacts: [],
    incomplete_outputs: [],
    final_generation_failures: []
  };
}

function readResponseRows(
  ctx: RunContext,
  profileName: string,
  dataset: DatasetConfig,
  mode: string,
  expected: number,
  report: ResumeCheckReport
): CsvRecord[] | null {
  const filePath = outputFileForMode(ctx, profileName, dataset.name, mode);
  if (!fileExists(filePath)) {
    report.missing_artifacts.push(filePath);
    return null;
  }
  const rows = readCsvFile(filePath);
  if (rows.length < expected) {
    report.incomplete_outputs.push({ profile: profileName, dataset: dataset.name, mode, expected, actual: rows.length, path: filePath });
    return null;
  }
  return rows;
}

function readMoralRows(
  ctx: RunContext,
  profile: ProfileConfig,
  moralA: DatasetConfig,
  moralB: DatasetConfig,
  expected: number,
  report: ResumeCheckReport
): { aFree: CsvRecord[]; aBinary: CsvRecord[]; bFree: CsvRecord[]; bBinary: CsvRecord[] } | null {
  if (moralA.task !== MORAL_A_TASK || moralB.task !== MORAL_B_TASK) {
    throw new Error("Resume generation expected a moral A/B dataset pair");
  }
  const aFree = readResponseRows(ctx, profile.name, moralA, "free", expected, report);
  const aBinary = readResponseRows(ctx, profile.name, moralA, "binary", expected, report);
  const bFree = readResponseRows(ctx, profile.name, moralB, "free", expected, report);
  const bBinary = readResponseRows(ctx, profile.name, moralB, "binary", expected, report);
  if (!aFree || !aBinary || !bFree || !bBinary) {
    return null;
  }
  return { aFree, aBinary, bFree, bBinary };
}

function sourceIndices(rows: CsvRecord[]): number[] {
  return rows.map((row) => Number(row._source_index)).filter((value) => Number.isInteger(value));
}

function sourceIds(rows: CsvRecord[]): string[] {
  return rows.map((row) => String(row._source_index ?? row.id ?? "")).filter(Boolean);
}

function collectFinalGenerationFailures(ctx: RunContext): ResumeFailureSummary[] {
  const callsPath = path.join(ctx.outputRoot, "ui_calls.jsonl");
  if (!fs.existsSync(callsPath)) {
    return [];
  }
  const lastByUnit = new Map<string, Record<string, unknown>>();
  for (const line of fs.readFileSync(callsPath, "utf8").split(/\r?\n/).filter(Boolean)) {
    try {
      const record = JSON.parse(line) as Record<string, unknown>;
      if (record.phase !== "generation") {
        continue;
      }
      const key = [record.profile, record.dataset, record.mode, record.metric, record.row_id].map(String).join("::");
      lastByUnit.set(key, record);
    } catch {
      continue;
    }
  }
  return [...lastByUnit.values()]
    .filter((record) => record.ok === false)
    .map((record) => ({
      profile: nullableString(record.profile),
      dataset: nullableString(record.dataset),
      mode: nullableString(record.mode),
      metric: nullableString(record.metric),
      row_id: nullableCell(record.row_id),
      error: nullableString(record.error)
    }));
}

function nullableString(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return String(value);
}

function nullableCell(value: unknown): string | number | boolean | null {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return null;
}
