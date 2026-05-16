import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import Papa from "papaparse";

import { ELEPHANT_FULL_RESULTS_DIR, JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR, METRICS, type CsvRecord } from "../contracts/autobench.js";
import { ensureDir, readCsvFile, sha256Text, utcNowIso, writeJson } from "../io/filesystem.js";
import { resolveDataRootForRepo } from "../runtime/paths.js";
import { requiredReferenceFiles } from "./loadElephantReference.js";
import { normalizeReferenceLabel, type BinaryLabel } from "./parseBinaryLabel.js";

const OSF_FULL_RESULTS_URL = "https://osf.io/download/cjwfe/?view_only=37ee66a8020a45c29a38bd704ca61067";
const OSF_DATASETS_URL = "https://osf.io/download/4chzv/?view_only=37ee66a8020a45c29a38bd704ca61067";

const FULL_RESULT_FILES = [
  "OEQ_full_results.csv",
  "AITA-YTA_full_results.csv",
  "SS_full_results.csv",
  path.join("elephant_full_results", "AITA-NTA-OG_full_results.csv"),
  path.join("elephant_full_results", "AITA-NTA-FLIP_flipped_full_results.csv")
] as const;

const PROMPT_DATASET_FILES = [
  "OEQ.csv",
  "AITA-YTA.csv",
  "SS.csv",
  "AITA-NTA-OG.csv",
  "AITA-NTA-FLIP.csv"
] as const;

type CuratedRow = {
  row_id: string;
  dataset: string;
  metric: string;
  source_row_index: number;
  prompt: string;
  response_model: "Claude";
  response: string;
  reference_judge: "GPT-4o";
  reference_label: BinaryLabel;
};

const CURATED_SPECS = [
  {
    dataset: "OEQ",
    fileName: "oeq.csv",
    fullResultFile: "OEQ_full_results.csv",
    promptDatasetFile: "OEQ.csv",
    promptAliases: ["prompt"],
    metrics: [...METRICS]
  },
  {
    dataset: "AITA-YTA",
    fileName: "aita_yta.csv",
    fullResultFile: "AITA-YTA_full_results.csv",
    promptDatasetFile: "AITA-YTA.csv",
    promptAliases: ["prompt", "original_post", "question"],
    metrics: [...METRICS]
  },
  {
    dataset: "SS",
    fileName: "ss.csv",
    fullResultFile: "SS_full_results.csv",
    promptDatasetFile: "SS.csv",
    promptAliases: ["prompt", "sentence"],
    metrics: ["framing"]
  },
  {
    dataset: "AITA-NTA-OG",
    fileName: "aita_nta_og.csv",
    fullResultFile: path.join("elephant_full_results", "AITA-NTA-OG_full_results.csv"),
    promptDatasetFile: "AITA-NTA-OG.csv",
    promptAliases: ["original_post", "prompt"],
    metrics: [...METRICS]
  },
  {
    dataset: "AITA-NTA-FLIP",
    fileName: "aita_nta_flip.csv",
    fullResultFile: path.join("elephant_full_results", "AITA-NTA-FLIP_flipped_full_results.csv"),
    promptDatasetFile: "AITA-NTA-FLIP.csv",
    promptAliases: ["flipped_story", "prompt"],
    metrics: [...METRICS]
  }
] as const;

/**
 * Ensures official ELEPHANT full results are present under this package.
 */
export async function ensureElephantFullResults(repoRoot: string): Promise<string> {
  const dataDir = path.join(resolveDataRootForRepo(repoRoot), ELEPHANT_FULL_RESULTS_DIR);
  ensureDir(dataDir);
  await ensurePromptDatasets(dataDir);
  if (hasAllFullResultFiles(dataDir)) {
    return dataDir;
  }
  const zipPath = path.join(dataDir, "elephant_full_results.zip");
  if (!fs.existsSync(zipPath)) {
    await downloadFile(OSF_FULL_RESULTS_URL, zipPath);
  }
  extractZip(zipPath, dataDir);
  await ensurePromptDatasets(dataDir);
  if (!hasAllFullResultFiles(dataDir)) {
    throw new Error(`ELEPHANT full results are incomplete after download/extract: ${missingFullResultFiles(dataDir).join(", ")}`);
  }
  return dataDir;
}

/**
 * Builds or verifies the local curated Claude social afferition dataset.
 */
export async function ensureClaudeSocialJudgeAfferition(repoRoot: string): Promise<string> {
  const fullResultsDir = await ensureElephantFullResults(repoRoot);
  const dataDir = path.join(resolveDataRootForRepo(repoRoot), JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR);
  if (hasAllReferenceFiles(dataDir)) {
    return dataDir;
  }
  ensureDir(dataDir);
  buildClaudeSocialAfferition(fullResultsDir, dataDir);
  if (!hasAllReferenceFiles(dataDir)) {
    throw new Error(`Curated Claude judge afferition dataset is incomplete: ${missingReferenceFiles(dataDir).join(", ")}`);
  }
  return dataDir;
}

export function hasAllReferenceFiles(dataDir: string): boolean {
  return missingReferenceFiles(dataDir).length === 0;
}

export function missingReferenceFiles(dataDir: string): string[] {
  return requiredReferenceFiles(dataDir).filter((filePath) => !fs.existsSync(filePath));
}

function hasAllFullResultFiles(dataDir: string): boolean {
  return missingFullResultFiles(dataDir).length === 0;
}

function missingFullResultFiles(dataDir: string): string[] {
  return [...FULL_RESULT_FILES, ...PROMPT_DATASET_FILES].map((filePath) => path.join(dataDir, filePath)).filter((filePath) => !fs.existsSync(filePath));
}

function buildClaudeSocialAfferition(fullResultsDir: string, dataDir: string): void {
  const allRows: CuratedRow[] = [];
  const manifestDatasets: Record<string, Record<string, unknown>> = {};

  for (const spec of CURATED_SPECS) {
    const rows = buildCuratedRowsForSpec(fullResultsDir, spec);
    allRows.push(...rows);
    writeCsvRows(rows, path.join(dataDir, spec.fileName));
    manifestDatasets[spec.dataset] = {
      file: spec.fileName,
      rows: rows.length,
      metrics: Object.fromEntries(spec.metrics.map((metric) => [metric, rows.filter((row) => row.metric === metric).length])),
      full_result_file: spec.fullResultFile,
      prompt_dataset_file: spec.promptDatasetFile
    };
  }

  writeCsvRows(allRows, path.join(dataDir, "all.csv"));
  writeJson(path.join(dataDir, "manifest.json"), {
    created_at: utcNowIso(),
    reference: "ELEPHANT GPT-4o labels over Claude responses",
    response_model: "Claude",
    reference_judge: "GPT-4o",
    source_dir: path.relative(path.dirname(dataDir), fullResultsDir),
    total_rows: allRows.length,
    datasets: manifestDatasets,
    source_fingerprint: buildFullResultsFingerprint(fullResultsDir)
  });
}

function buildCuratedRowsForSpec(fullResultsDir: string, spec: (typeof CURATED_SPECS)[number]): CuratedRow[] {
  const fullRows = readCsvFile(path.join(fullResultsDir, spec.fullResultFile));
  const promptRows = readCsvFile(path.join(fullResultsDir, spec.promptDatasetFile));
  const curated: CuratedRow[] = [];
  for (const [index, record] of fullRows.entries()) {
    const prompt = aliasValue(record, spec.promptAliases) ?? aliasValue(promptRows[index] ?? {}, spec.promptAliases);
    const response = record.Claude;
    if (!prompt || response === null || response === undefined || String(response).trim() === "") {
      continue;
    }
    for (const metric of spec.metrics) {
      const referenceLabel = normalizeReferenceLabel(record[`${metric}_Claude`]);
      if (referenceLabel === null) {
        continue;
      }
      curated.push({
        row_id: `${spec.dataset}:${String(index).padStart(6, "0")}:${metric}:Claude`,
        dataset: spec.dataset,
        metric,
        source_row_index: index,
        prompt,
        response_model: "Claude",
        response: String(response),
        reference_judge: "GPT-4o",
        reference_label: referenceLabel
      });
    }
  }
  return curated;
}

async function ensurePromptDatasets(dataDir: string): Promise<void> {
  if (PROMPT_DATASET_FILES.every((fileName) => fs.existsSync(path.join(dataDir, fileName)))) {
    return;
  }
  const zipPath = path.join(dataDir, "datasets.zip");
  if (!fs.existsSync(zipPath)) {
    await downloadFile(OSF_DATASETS_URL, zipPath);
  }
  extractZip(zipPath, dataDir);
  flattenKnownDatasetFiles(dataDir);
}

async function downloadFile(url: string, destination: string): Promise<void> {
  ensureDir(path.dirname(destination));
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ELEPHANT data: ${response.status} ${response.statusText}`);
  }
  fs.writeFileSync(destination, Buffer.from(await response.arrayBuffer()));
}

function extractZip(zipPath: string, dataDir: string): void {
  const result = spawnSync("unzip", ["-o", zipPath, "-d", dataDir], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`Failed to extract ${zipPath}: ${result.stderr || result.stdout}`);
  }
  flattenKnownFiles(dataDir);
}

function flattenKnownFiles(dataDir: string): void {
  for (const name of [...FULL_RESULT_FILES, ...PROMPT_DATASET_FILES]) {
    const directPath = path.join(dataDir, name);
    if (fs.existsSync(directPath)) {
      continue;
    }
    const found = findFirst(dataDir, path.basename(name));
    if (found) {
      ensureDir(path.dirname(directPath));
      fs.copyFileSync(found, directPath);
    }
  }
}

function flattenKnownDatasetFiles(dataDir: string): void {
  for (const name of PROMPT_DATASET_FILES) {
    const directPath = path.join(dataDir, name);
    if (fs.existsSync(directPath)) {
      continue;
    }
    const found = findFirst(dataDir, name);
    if (found) {
      fs.copyFileSync(found, directPath);
    }
  }
}

function findFirst(root: string, fileName: string): string | null {
  for (const entry of fs.readdirSync(root)) {
    const entryPath = path.join(root, entry);
    const stat = fs.statSync(entryPath);
    if (stat.isDirectory()) {
      const child = findFirst(entryPath, fileName);
      if (child) {
        return child;
      }
    } else if (entry === fileName) {
      return entryPath;
    }
  }
  return null;
}

function aliasValue(record: CsvRecord, aliases: readonly string[]): string | null {
  for (const alias of aliases) {
    const value = record[alias];
    if (value !== null && value !== undefined && String(value).trim()) {
      return String(value);
    }
  }
  return null;
}

function buildFullResultsFingerprint(dataDir: string): string {
  const parts = [...FULL_RESULT_FILES, ...PROMPT_DATASET_FILES].map((filePath) => {
    const absolutePath = path.join(dataDir, filePath);
    const stat = fs.statSync(absolutePath);
    return `${filePath}:${stat.size}:${sha256Text(fs.readFileSync(absolutePath, "utf8"))}`;
  });
  return sha256Text(parts.join("\n"));
}

function writeCsvRows(rows: CuratedRow[], filePath: string): void {
  ensureDir(path.dirname(filePath));
  const csv = Papa.unparse({
    fields: ["row_id", "dataset", "metric", "source_row_index", "prompt", "response_model", "response", "reference_judge", "reference_label"],
    data: rows
  });
  fs.writeFileSync(filePath, `${csv}\n`, "utf8");
}
