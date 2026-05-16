import fs from "node:fs";
import path from "node:path";

import seedrandom from "seedrandom";

import {
  DEFAULT_CONFIDENCE,
  JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR,
  JUDGE_AFFERITION_MARGIN_DIR,
  JUDGE_AFFERITION_STRATIFIED_1000_NAME,
  JUDGE_AFFERITION_TEST_SETS_DIR,
  type CsvRecord,
  type JudgeAfferitionSamplingSummary
} from "../contracts/autobench.js";
import { readJsonFile, stableSeedOffset, utcNowIso, writeCsvFile, writeJson } from "../io/filesystem.js";
import { resolveDataRootForRepo } from "../runtime/paths.js";
import { sampleTargetN } from "../sampling/samplePlanner.js";
import { loadElephantReference, type ValidationReferenceRow } from "./loadElephantReference.js";

export const JUDGE_AFFERITION_SAMPLE_BY = "dataset_metric";
export const JUDGE_AFFERITION_SAMPLE_SEED = 20260513;

const FORMULA = "binary_proportion_worst_case_fpc";
const MANIFEST_VERSION = 1;
const TEST_SET_MANIFEST_VERSION = 1;
const TEST_SET_TOTAL = 1000;
const REQUIRED_SAMPLE_FILES = ["all.csv", "oeq.csv", "aita_yta.csv", "ss.csv", "aita_nta_og.csv", "aita_nta_flip.csv", "manifest.json"];
const CSV_FIELDS = ["row_id", "dataset", "metric", "source_row_index", "prompt", "response_model", "response", "reference_judge", "reference_label"];
const DATASET_FILES: Record<string, string> = {
  OEQ: "oeq.csv",
  "AITA-YTA": "aita_yta.csv",
  SS: "ss.csv",
  "AITA-NTA-OG": "aita_nta_og.csv",
  "AITA-NTA-FLIP": "aita_nta_flip.csv"
};

export interface JudgeAfferitionMarginDataset {
  dataDir: string;
  manifestPath: string;
  manifest: JudgeAfferitionMarginManifest;
  sampling: JudgeAfferitionSamplingSummary;
}

export interface JudgeAfferitionTestSetDataset {
  dataDir: string;
  manifestPath: string;
  manifest: JudgeAfferitionTestSetManifest;
  sampling: JudgeAfferitionSamplingSummary;
}

export interface JudgeAfferitionMarginManifest {
  version: 1;
  created_at: string;
  reference: "ELEPHANT GPT-4o labels over Claude responses";
  confidence: number;
  margin_of_error: number;
  margin_label: string;
  sample_by: typeof JUDGE_AFFERITION_SAMPLE_BY;
  seed: number;
  formula: typeof FORMULA;
  source_dir: string;
  source_fingerprint: string;
  full_total: number;
  sample_total: number;
  datasets: Record<string, {
    file: string;
    full_rows: number;
    sample_rows: number;
    metrics: Record<string, {
      full: number;
      target: number;
      selected: number;
    }>;
  }>;
}

export interface JudgeAfferitionTestSetManifest {
  version: 1;
  created_at: string;
  name: typeof JUDGE_AFFERITION_STRATIFIED_1000_NAME;
  reference: "ELEPHANT GPT-4o labels over Claude responses";
  sample_by: "dataset_metric_fixed_total";
  seed: number;
  source_dir: string;
  source_fingerprint: string;
  full_total: number;
  sample_total: number;
  datasets: Record<string, {
    file: string;
    full_rows: number;
    target_rows: number;
    sample_rows: number;
    metrics: Record<string, {
      full: number;
      target: number;
      selected: number;
    }>;
  }>;
}

export function ensureJudgeAfferitionMarginDataset(
  repoRoot: string,
  marginOfError: number,
  confidence = DEFAULT_CONFIDENCE,
  seed = JUDGE_AFFERITION_SAMPLE_SEED
): JudgeAfferitionMarginDataset {
  validateJudgeAfferitionMargin(marginOfError);
  const dataRoot = resolveDataRootForRepo(repoRoot);
  const sourceDir = path.join(dataRoot, JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR);
  const reference = loadElephantReference(repoRoot, sourceDir);
  const marginLabel = formatMarginLabel(marginOfError);
  const dataDir = path.join(dataRoot, JUDGE_AFFERITION_MARGIN_DIR, marginLabel);
  const manifestPath = path.join(dataDir, "manifest.json");
  const existing = readReusableManifest(dataDir, manifestPath, reference.fingerprint, marginOfError, confidence, seed);
  if (existing) {
    return {
      dataDir,
      manifestPath,
      manifest: existing,
      sampling: buildMarginSamplingSummary(repoRoot, dataDir, manifestPath, existing)
    };
  }

  const selectedRows = sampleRowsByDatasetMetric(reference.rows, marginOfError, confidence, seed, marginLabel);
  const manifest = buildManifest(repoRoot, sourceDir, reference.fingerprint, reference.rows, selectedRows, marginOfError, confidence, seed, marginLabel);
  writeSampleCsvs(dataDir, selectedRows);
  writeJson(manifestPath, manifest);
  return {
    dataDir,
    manifestPath,
    manifest,
    sampling: buildMarginSamplingSummary(repoRoot, dataDir, manifestPath, manifest)
  };
}

export function buildFullJudgeAfferitionSamplingSummary(
  repoRoot: string,
  dataDir: string,
  fullTotal: number,
  sourceFingerprint: string
): JudgeAfferitionSamplingSummary {
  const dataRoot = resolveDataRootForRepo(repoRoot);
  return {
    kind: "full",
    confidence: DEFAULT_CONFIDENCE,
    marginOfError: null,
    marginLabel: "full",
    sampleBy: "full",
    seed: null,
    datasetPath: path.relative(dataRoot, dataDir),
    manifestPath: fs.existsSync(path.join(dataDir, "manifest.json")) ? path.relative(dataRoot, path.join(dataDir, "manifest.json")) : null,
    sourcePath: path.relative(dataRoot, dataDir),
    sourceFingerprint,
    fullTotal,
    sampleTotal: fullTotal
  };
}

export function ensureJudgeAfferitionStratifiedTestSet1000(
  repoRoot: string,
  seed = JUDGE_AFFERITION_SAMPLE_SEED
): JudgeAfferitionTestSetDataset {
  const dataRoot = resolveDataRootForRepo(repoRoot);
  const sourceDir = path.join(dataRoot, JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR);
  const reference = loadElephantReference(repoRoot, sourceDir);
  const dataDir = path.join(dataRoot, JUDGE_AFFERITION_TEST_SETS_DIR, JUDGE_AFFERITION_STRATIFIED_1000_NAME);
  const manifestPath = path.join(dataDir, "manifest.json");
  const existing = readReusableTestSetManifest(dataDir, manifestPath, reference.fingerprint, seed);
  if (existing) {
    return {
      dataDir,
      manifestPath,
      manifest: existing,
      sampling: buildTestSetSamplingSummary(repoRoot, dataDir, manifestPath, existing)
    };
  }

  const selectedRows = sampleFixedRowsByDatasetMetric(reference.rows, TEST_SET_TOTAL, seed);
  const manifest = buildTestSetManifest(repoRoot, dataDir, sourceDir, reference.fingerprint, reference.rows, selectedRows, seed);
  writeSampleCsvs(dataDir, selectedRows);
  writeJson(manifestPath, manifest);
  return {
    dataDir,
    manifestPath,
    manifest,
    sampling: buildTestSetSamplingSummary(repoRoot, dataDir, manifestPath, manifest)
  };
}

export function listJudgeAfferitionMarginDatasets(repoRoot: string): JudgeAfferitionMarginManifest[] {
  void repoRoot;
  const root = path.join(resolveDataRootForRepo(repoRoot), JUDGE_AFFERITION_MARGIN_DIR);
  if (!fs.existsSync(root)) {
    return [];
  }
  return fs.readdirSync(root)
    .map((entry) => path.join(root, entry, "manifest.json"))
    .filter((entryPath) => fs.existsSync(entryPath))
    .map((entryPath) => readJsonFile<JudgeAfferitionMarginManifest>(entryPath))
    .filter(isMarginManifest)
    .sort((left, right) => left.margin_of_error - right.margin_of_error);
}

export function parseJudgeAfferitionTestSetInput(value: string): string {
  const normalized = value.trim();
  if (normalized !== JUDGE_AFFERITION_STRATIFIED_1000_NAME) {
    throw new Error(`Unsupported judge afferition test set: ${value}`);
  }
  return normalized;
}

export function parseJudgeAfferitionMarginInput(value: string): number {
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    throw new Error("Margin of error is required");
  }
  const numeric = normalized.endsWith("pp") ? Number(normalized.replace("pp", "")) / 100 : Number(normalized);
  const margin = numeric > 1 ? numeric / 100 : numeric;
  validateJudgeAfferitionMargin(margin);
  return margin;
}

export function validateJudgeAfferitionMargin(marginOfError: number): void {
  if (!Number.isFinite(marginOfError) || !(marginOfError > 0 && marginOfError < 1)) {
    throw new Error(`Judge afferition margin must be > 0 and < 100pp, got: ${String(marginOfError)}`);
  }
}

export function formatMarginLabel(marginOfError: number): string {
  validateJudgeAfferitionMargin(marginOfError);
  const pp = trimTrailingZeros((marginOfError * 100).toFixed(6));
  return `${pp.replace(".", "_")}pp`;
}

function readReusableManifest(
  dataDir: string,
  manifestPath: string,
  sourceFingerprint: string,
  marginOfError: number,
  confidence: number,
  seed: number
): JudgeAfferitionMarginManifest | null {
  if (!REQUIRED_SAMPLE_FILES.every((fileName) => fs.existsSync(path.join(dataDir, fileName)))) {
    return null;
  }
  const manifest = readJsonFile<JudgeAfferitionMarginManifest>(manifestPath);
  if (!isMarginManifest(manifest)) {
    return null;
  }
  const matches =
    manifest.source_fingerprint === sourceFingerprint &&
    manifest.margin_of_error === marginOfError &&
    manifest.confidence === confidence &&
    manifest.seed === seed &&
    manifest.sample_by === JUDGE_AFFERITION_SAMPLE_BY &&
    manifest.formula === FORMULA;
  return matches ? manifest : null;
}

function readReusableTestSetManifest(
  dataDir: string,
  manifestPath: string,
  sourceFingerprint: string,
  seed: number
): JudgeAfferitionTestSetManifest | null {
  if (!REQUIRED_SAMPLE_FILES.every((fileName) => fs.existsSync(path.join(dataDir, fileName)))) {
    return null;
  }
  const manifest = readJsonFile<JudgeAfferitionTestSetManifest>(manifestPath);
  if (!isTestSetManifest(manifest)) {
    return null;
  }
  const matches =
    manifest.name === JUDGE_AFFERITION_STRATIFIED_1000_NAME &&
    manifest.source_fingerprint === sourceFingerprint &&
    manifest.seed === seed &&
    manifest.sample_by === "dataset_metric_fixed_total" &&
    manifest.sample_total === TEST_SET_TOTAL;
  return matches ? manifest : null;
}

function sampleRowsByDatasetMetric(
  rows: ValidationReferenceRow[],
  marginOfError: number,
  confidence: number,
  seed: number,
  marginLabel: string
): ValidationReferenceRow[] {
  const groups = groupRowsByDatasetMetric(rows);
  const selectedIds = new Set<string>();
  for (const [key, groupRows] of groups) {
    const targetN = sampleTargetN(groupRows.length, { confidence, marginOfError });
    const shuffled = shuffleRows(groupRows, seed + stableSeedOffset(`${marginLabel}:${key}`));
    for (const row of shuffled.slice(0, targetN)) {
      selectedIds.add(row.rowId);
    }
  }
  return rows.filter((row) => selectedIds.has(row.rowId));
}

function sampleFixedRowsByDatasetMetric(rows: ValidationReferenceRow[], targetTotal: number, seed: number): ValidationReferenceRow[] {
  const datasetTargets = allocateEvenly(Object.keys(DATASET_FILES), targetTotal);
  const selectedIds = new Set<string>();
  for (const [dataset, datasetTarget] of datasetTargets) {
    const datasetRows = rows.filter((row) => row.dataset === dataset);
    const metricGroups = groupRowsByDatasetMetric(datasetRows);
    const metricTargets = allocateProportionally(metricGroups, datasetTarget);
    for (const [key, target] of metricTargets) {
      const shuffled = shuffleRows(metricGroups.get(key) ?? [], seed + stableSeedOffset(`${JUDGE_AFFERITION_STRATIFIED_1000_NAME}:${key}`));
      for (const row of shuffled.slice(0, target)) {
        selectedIds.add(row.rowId);
      }
    }
  }
  return rows.filter((row) => selectedIds.has(row.rowId));
}

function allocateEvenly(keys: string[], targetTotal: number): Map<string, number> {
  const base = Math.floor(targetTotal / keys.length);
  let remaining = targetTotal - base * keys.length;
  const targets = new Map<string, number>();
  for (const key of keys) {
    const increment = remaining > 0 ? 1 : 0;
    targets.set(key, base + increment);
    remaining -= increment;
  }
  return targets;
}

function allocateProportionally(groups: Map<string, ValidationReferenceRow[]>, targetTotal: number): Map<string, number> {
  const entries = [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
  const population = entries.reduce((total, [, rows]) => total + rows.length, 0);
  const allocated = entries.map(([key, rows]) => {
    const exact = population === 0 ? 0 : (rows.length / population) * targetTotal;
    return { key, rows: rows.length, base: Math.floor(exact), remainder: exact - Math.floor(exact) };
  });
  let remaining = targetTotal - allocated.reduce((total, item) => total + item.base, 0);
  allocated.sort((left, right) => right.remainder - left.remainder || left.key.localeCompare(right.key));
  for (const item of allocated) {
    if (remaining <= 0) {
      break;
    }
    item.base += 1;
    remaining -= 1;
  }
  return new Map(allocated.map((item) => [item.key, Math.min(item.base, item.rows)]));
}

function buildManifest(
  repoRoot: string,
  sourceDir: string,
  sourceFingerprint: string,
  fullRows: ValidationReferenceRow[],
  sampleRows: ValidationReferenceRow[],
  marginOfError: number,
  confidence: number,
  seed: number,
  marginLabel: string
): JudgeAfferitionMarginManifest {
  void repoRoot;
  const fullGroups = groupRowsByDatasetMetric(fullRows);
  const sampleGroups = groupRowsByDatasetMetric(sampleRows);
  const datasets: JudgeAfferitionMarginManifest["datasets"] = {};

  for (const [dataset, file] of Object.entries(DATASET_FILES)) {
    const fullDatasetRows = fullRows.filter((row) => row.dataset === dataset);
    const sampleDatasetRows = sampleRows.filter((row) => row.dataset === dataset);
    const metrics: JudgeAfferitionMarginManifest["datasets"][string]["metrics"] = {};
    for (const [key, rows] of fullGroups) {
      const [groupDataset, metric] = key.split(":") as [string, string];
      if (groupDataset !== dataset) {
        continue;
      }
      metrics[metric] = {
        full: rows.length,
        target: sampleTargetN(rows.length, { confidence, marginOfError }),
        selected: sampleGroups.get(key)?.length ?? 0
      };
    }
    datasets[dataset] = {
      file,
      full_rows: fullDatasetRows.length,
      sample_rows: sampleDatasetRows.length,
      metrics
    };
  }

  return {
    version: MANIFEST_VERSION,
    created_at: utcNowIso(),
    reference: "ELEPHANT GPT-4o labels over Claude responses",
    confidence,
    margin_of_error: marginOfError,
    margin_label: marginLabel,
    sample_by: JUDGE_AFFERITION_SAMPLE_BY,
    seed,
    formula: FORMULA,
    source_dir: path.relative(path.join(resolveDataRootForRepo(repoRoot), JUDGE_AFFERITION_MARGIN_DIR, marginLabel), sourceDir),
    source_fingerprint: sourceFingerprint,
    full_total: fullRows.length,
    sample_total: sampleRows.length,
    datasets
  };
}

function buildTestSetManifest(
  repoRoot: string,
  dataDir: string,
  sourceDir: string,
  sourceFingerprint: string,
  fullRows: ValidationReferenceRow[],
  sampleRows: ValidationReferenceRow[],
  seed: number
): JudgeAfferitionTestSetManifest {
  void repoRoot;
  const fullGroups = groupRowsByDatasetMetric(fullRows);
  const sampleGroups = groupRowsByDatasetMetric(sampleRows);
  const datasets: JudgeAfferitionTestSetManifest["datasets"] = {};

  for (const [dataset, file] of Object.entries(DATASET_FILES)) {
    const fullDatasetRows = fullRows.filter((row) => row.dataset === dataset);
    const sampleDatasetRows = sampleRows.filter((row) => row.dataset === dataset);
    const metrics: JudgeAfferitionTestSetManifest["datasets"][string]["metrics"] = {};
    for (const [key, rows] of fullGroups) {
      const [groupDataset, metric] = key.split(":") as [string, string];
      if (groupDataset !== dataset) {
        continue;
      }
      metrics[metric] = {
        full: rows.length,
        target: sampleGroups.get(key)?.length ?? 0,
        selected: sampleGroups.get(key)?.length ?? 0
      };
    }
    datasets[dataset] = {
      file,
      full_rows: fullDatasetRows.length,
      target_rows: sampleDatasetRows.length,
      sample_rows: sampleDatasetRows.length,
      metrics
    };
  }

  return {
    version: TEST_SET_MANIFEST_VERSION,
    created_at: utcNowIso(),
    name: JUDGE_AFFERITION_STRATIFIED_1000_NAME,
    reference: "ELEPHANT GPT-4o labels over Claude responses",
    sample_by: "dataset_metric_fixed_total",
    seed,
    source_dir: path.relative(dataDir, sourceDir),
    source_fingerprint: sourceFingerprint,
    full_total: fullRows.length,
    sample_total: sampleRows.length,
    datasets
  };
}

function writeSampleCsvs(dataDir: string, rows: ValidationReferenceRow[]): void {
  writeCsvFile(rows.map(rowToCsvRecord), path.join(dataDir, "all.csv"), CSV_FIELDS);
  for (const [dataset, fileName] of Object.entries(DATASET_FILES)) {
    writeCsvFile(rows.filter((row) => row.dataset === dataset).map(rowToCsvRecord), path.join(dataDir, fileName), CSV_FIELDS);
  }
}

function buildMarginSamplingSummary(
  repoRoot: string,
  dataDir: string,
  manifestPath: string,
  manifest: JudgeAfferitionMarginManifest
): JudgeAfferitionSamplingSummary {
  void repoRoot;
  const dataRoot = resolveDataRootForRepo(repoRoot);
  return {
    kind: "margin",
    confidence: manifest.confidence,
    marginOfError: manifest.margin_of_error,
    marginLabel: manifest.margin_label,
    sampleBy: manifest.sample_by,
    seed: manifest.seed,
    datasetPath: path.relative(dataRoot, dataDir),
    manifestPath: path.relative(dataRoot, manifestPath),
    sourcePath: path.relative(dataRoot, path.resolve(dataDir, manifest.source_dir)),
    sourceFingerprint: manifest.source_fingerprint,
    fullTotal: manifest.full_total,
    sampleTotal: manifest.sample_total
  };
}

function buildTestSetSamplingSummary(
  repoRoot: string,
  dataDir: string,
  manifestPath: string,
  manifest: JudgeAfferitionTestSetManifest
): JudgeAfferitionSamplingSummary {
  void repoRoot;
  const dataRoot = resolveDataRootForRepo(repoRoot);
  return {
    kind: "test_set",
    confidence: DEFAULT_CONFIDENCE,
    marginOfError: null,
    marginLabel: manifest.name,
    sampleBy: manifest.sample_by,
    seed: manifest.seed,
    datasetPath: path.relative(dataRoot, dataDir),
    manifestPath: path.relative(dataRoot, manifestPath),
    sourcePath: path.relative(dataRoot, path.resolve(dataDir, manifest.source_dir)),
    sourceFingerprint: manifest.source_fingerprint,
    fullTotal: manifest.full_total,
    sampleTotal: manifest.sample_total
  };
}

function groupRowsByDatasetMetric(rows: ValidationReferenceRow[]): Map<string, ValidationReferenceRow[]> {
  const groups = new Map<string, ValidationReferenceRow[]>();
  for (const row of rows) {
    const key = `${row.dataset}:${row.metric}`;
    const current = groups.get(key) ?? [];
    current.push(row);
    groups.set(key, current);
  }
  return groups;
}

function shuffleRows(rows: ValidationReferenceRow[], seed: number): ValidationReferenceRow[] {
  const rng = seedrandom(String(seed));
  const cloned = [...rows];
  for (let index = cloned.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    const current = cloned[index]!;
    cloned[index] = cloned[swapIndex]!;
    cloned[swapIndex] = current;
  }
  return cloned;
}

function rowToCsvRecord(row: ValidationReferenceRow): CsvRecord {
  return {
    row_id: row.rowId,
    dataset: row.dataset,
    metric: row.metric,
    source_row_index: row.sourceRowIndex,
    prompt: row.prompt,
    response_model: row.responseModel,
    response: row.response,
    reference_judge: row.referenceJudge,
    reference_label: row.referenceLabel
  };
}

function isMarginManifest(manifest: JudgeAfferitionMarginManifest): manifest is JudgeAfferitionMarginManifest {
  return manifest?.version === MANIFEST_VERSION &&
    manifest.sample_by === JUDGE_AFFERITION_SAMPLE_BY &&
    manifest.formula === FORMULA &&
    typeof manifest.margin_label === "string" &&
    typeof manifest.sample_total === "number";
}

function isTestSetManifest(manifest: JudgeAfferitionTestSetManifest): manifest is JudgeAfferitionTestSetManifest {
  return manifest?.version === TEST_SET_MANIFEST_VERSION &&
    manifest.name === JUDGE_AFFERITION_STRATIFIED_1000_NAME &&
    manifest.sample_by === "dataset_metric_fixed_total" &&
    typeof manifest.sample_total === "number";
}

function trimTrailingZeros(value: string): string {
  return value.replace(/\.?0+$/, "");
}
