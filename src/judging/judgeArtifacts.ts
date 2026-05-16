import fs from "node:fs";
import path from "node:path";

import {
  JUDGE_ARCHIVE_DIRNAME,
  type DatasetConfig,
  type ProfileConfig,
  type RunContext,
  type SampleManifest
} from "../contracts/autobench.js";
import { fileExists, listFiles, localRunStamp, readJsonFile } from "../io/filesystem.js";
import { logEvent } from "../pipeline/runContext.js";
import type { TerminalObserver } from "../runtime/terminalObserver.js";
import { outputFileForMode } from "../scoring/scoreEngine.js";
import { metricsForDataset } from "./judgePrompts.js";
import { readCsvFile } from "../io/filesystem.js";

/**
 * Validates that an existing run has all generation artifacts needed for judge-only mode.
 */
export function validateExistingJudgeInputs(
  outputRoot: string,
  profiles: ProfileConfig[],
  socialDatasets: DatasetConfig[],
  moralA: DatasetConfig,
  moralB: DatasetConfig
): SampleManifest {
  const manifestPath = path.join(outputRoot, "sample_manifest.json");
  if (!fileExists(manifestPath)) {
    throw new Error(`Missing sample manifest: ${manifestPath}`);
  }
  const manifest = readJsonFile<SampleManifest>(manifestPath);
  const missingPaths = collectMissingJudgeInputs(outputRoot, profiles, socialDatasets, moralA, moralB);
  if (missingPaths.length > 0) {
    throw new Error(`Missing generation artifacts required for --judge-only:\n${missingPaths.join("\n")}`);
  }
  return manifest;
}

/**
 * Counts judge work units for an existing generation run.
 */
export function plannedJudgeOnlyUnits(
  ctx: RunContext,
  profiles: ProfileConfig[],
  socialDatasets: DatasetConfig[],
  moralA: DatasetConfig,
  moralB: DatasetConfig
): number {
  let total = 3;
  for (const profile of profiles) {
    for (const dataset of socialDatasets) {
      total += readCsvFile(outputFileForMode(ctx, profile.name, dataset.name, "responses")).length * metricsForDataset(dataset.name).length;
    }
    for (const dataset of [moralA, moralB]) {
      total += readCsvFile(outputFileForMode(ctx, profile.name, dataset.name, "free")).length * metricsForDataset(dataset.name, true).length;
    }
  }
  return total;
}

/**
 * Returns judge artifacts that should be archived before rerunning judge-only mode.
 */
export function judgeArtifactPaths(ctx: RunContext): string[] {
  const candidates = [ctx.rawJudgePath, path.join(ctx.outputRoot, "audit_consolidated.json"), path.join(ctx.outputRoot, "RESULTS.md")].filter(fileExists);
  const scoreArtifacts = listFiles(ctx.outputRoot, (entryPath) => {
    if (entryPath.includes(`${path.sep}${JUDGE_ARCHIVE_DIRNAME}${path.sep}`)) {
      return false;
    }
    return /scores.*\.(csv|jsonl)$/i.test(entryPath);
  });
  return [...new Set([...candidates, ...scoreArtifacts])].sort();
}

/**
 * Moves prior judge artifacts under `judge_archives/<timestamp>` without deleting them.
 */
export function archiveExistingJudgeArtifacts(ctx: RunContext, observer: TerminalObserver): string | null {
  const artifactPaths = judgeArtifactPaths(ctx);
  logEvent(ctx, "judge_reset_started", observer, {
    output_root: ctx.outputRoot,
    artifact_count: artifactPaths.length
  });
  if (artifactPaths.length === 0) {
    logEvent(ctx, "judge_archive_completed", observer, { archive_path: null, archived_paths: [] });
    return null;
  }
  const archiveRoot = path.join(ctx.outputRoot, JUDGE_ARCHIVE_DIRNAME, localRunStamp());
  const archivedPaths: string[] = [];
  for (const source of artifactPaths) {
    const relative = path.relative(ctx.outputRoot, source);
    const destination = path.join(archiveRoot, relative);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.renameSync(source, destination);
    archivedPaths.push(relative);
  }
  logEvent(ctx, "judge_archive_completed", observer, {
    archive_path: archiveRoot,
    archived_paths: archivedPaths
  });
  return archiveRoot;
}

function collectMissingJudgeInputs(
  outputRoot: string,
  profiles: ProfileConfig[],
  socialDatasets: DatasetConfig[],
  moralA: DatasetConfig,
  moralB: DatasetConfig
): string[] {
  const missingPaths: string[] = [];
  for (const profile of profiles) {
    for (const dataset of socialDatasets) {
      pushMissingArtifacts(missingPaths, outputRoot, profile.name, dataset.name, ["responses.csv", "responses.jsonl"]);
    }
    for (const dataset of [moralA, moralB]) {
      pushMissingArtifacts(missingPaths, outputRoot, profile.name, dataset.name, [
        "responses_free.csv",
        "responses_free.jsonl",
        "responses_binary.csv",
        "responses_binary.jsonl"
      ]);
    }
  }
  return missingPaths;
}

function pushMissingArtifacts(
  missingPaths: string[],
  outputRoot: string,
  profileName: string,
  datasetName: string,
  suffixes: string[]
): void {
  for (const suffix of suffixes) {
    const artifactPath = path.join(outputRoot, profileName, datasetName, suffix);
    if (!fileExists(artifactPath)) {
      missingPaths.push(artifactPath);
    }
  }
}
