import fs from "node:fs";
import path from "node:path";

import { JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR, type CsvRecord } from "../contracts/autobench.js";
import { readCsvFile, sha256Text } from "../io/filesystem.js";
import { resolveDataRootForRepo } from "../runtime/paths.js";
import { normalizeReferenceLabel, type BinaryLabel } from "./parseBinaryLabel.js";

export interface ValidationReferenceRow {
  rowId: string;
  dataset: string;
  metric: string;
  sourceRowIndex: number;
  prompt: string;
  responseModel: "Claude";
  response: string;
  referenceJudge: "GPT-4o";
  referenceLabel: BinaryLabel;
}

export interface ElephantReference {
  rows: ValidationReferenceRow[];
  fingerprint: string;
  dataDir: string;
}

export const CURATED_REFERENCE_FILES = [
  "oeq.csv",
  "aita_yta.csv",
  "ss.csv",
  "aita_nta_og.csv",
  "aita_nta_flip.csv",
  "all.csv",
  "manifest.json"
] as const;

/**
 * Loads the curated Claude social afferition dataset.
 */
export function loadElephantReference(
  repoRoot: string,
  dataDir = path.join(resolveDataRootForRepo(repoRoot), JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR)
): ElephantReference {
  const allPath = path.join(dataDir, "all.csv");
  if (!fs.existsSync(allPath)) {
    throw new Error(`Missing curated Claude judge afferition CSV: ${allPath}`);
  }
  const rows = readCsvFile(allPath).map(parseReferenceRow);
  if (rows.length === 0) {
    throw new Error(`No curated Claude judge afferition rows loaded from ${allPath}`);
  }
  return {
    rows,
    fingerprint: buildDataFingerprint(dataDir),
    dataDir
  };
}

export function requiredReferenceFiles(dataDir: string): string[] {
  return CURATED_REFERENCE_FILES.map((fileName) => path.join(dataDir, fileName));
}

function parseReferenceRow(record: CsvRecord): ValidationReferenceRow {
  const referenceLabel = normalizeReferenceLabel(record.reference_label);
  if (referenceLabel === null) {
    throw new Error(`Invalid reference_label for row ${String(record.row_id ?? "")}`);
  }
  const responseModel = String(record.response_model ?? "");
  const referenceJudge = String(record.reference_judge ?? "");
  if (responseModel !== "Claude" || referenceJudge !== "GPT-4o") {
    throw new Error(`Invalid reference row model/judge for row ${String(record.row_id ?? "")}`);
  }
  return {
    rowId: requireString(record, "row_id"),
    dataset: requireString(record, "dataset"),
    metric: requireString(record, "metric"),
    sourceRowIndex: Number(requireString(record, "source_row_index")),
    prompt: requireString(record, "prompt"),
    responseModel,
    response: requireString(record, "response"),
    referenceJudge,
    referenceLabel
  };
}

function requireString(record: CsvRecord, key: string): string {
  const value = record[key];
  if (value === null || value === undefined || String(value).trim() === "") {
    throw new Error(`Missing required curated reference field: ${key}`);
  }
  return String(value);
}

function buildDataFingerprint(dataDir: string): string {
  const parts = requiredReferenceFiles(dataDir)
    .filter((filePath) => fs.existsSync(filePath))
    .map((filePath) => {
      const stat = fs.statSync(filePath);
      return `${path.basename(filePath)}:${stat.size}:${sha256Text(fs.readFileSync(filePath, "utf8"))}`;
    });
  return sha256Text(parts.join("\n"));
}
