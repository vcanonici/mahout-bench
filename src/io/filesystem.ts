import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { parse as parseCsv } from "csv-parse/sync";
import Papa from "papaparse";

import type { CsvRecord } from "../contracts/autobench.js";

export function utcNowIso(): string {
  return new Date().toISOString().replace(".000Z", "Z");
}

export function localRunStamp(date: Date = new Date()): string {
  const year = String(date.getFullYear());
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  const second = String(date.getSeconds()).padStart(2, "0");
  return `${year}${month}${day}_${hour}${minute}${second}`;
}

export function ensureDir(targetPath: string): void {
  fs.mkdirSync(targetPath, { recursive: true });
}

export function isMissingText(value: unknown): boolean {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === "number" && Number.isNaN(value)) {
    return true;
  }
  return String(value).trim() === "";
}

export function jsonSafe(value: unknown): unknown {
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack
    };
  }
  if (typeof value === "bigint") {
    return String(value);
  }
  if (typeof value === "number" && Number.isNaN(value)) {
    return null;
  }
  if (value === undefined) {
    return null;
  }
  if (Array.isArray(value)) {
    return value.map((item) => jsonSafe(item));
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, jsonSafe(item)]);
    return Object.fromEntries(entries);
  }
  return value;
}

export function writeJson(filePath: string, payload: unknown): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(`${filePath}`, `${JSON.stringify(jsonSafe(payload), null, 2)}\n`, "utf8");
}

export function appendJsonl(filePath: string, payload: unknown): void {
  ensureDir(path.dirname(filePath));
  fs.appendFileSync(filePath, `${JSON.stringify(jsonSafe(payload))}\n`, "utf8");
}

export function writeTextFile(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, "utf8");
}

export function readTextFile(filePath: string): string {
  return fs.readFileSync(filePath, "utf8");
}

export function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readTextFile(filePath)) as T;
}

export function sha256Text(text: string): string {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

export function stableSeedOffset(name: string): number {
  return [...name].reduce((total, char, index) => total + (index + 1) * char.charCodeAt(0), 0);
}

export function toNumberOrNull(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string" && value.trim() === "") {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return numeric;
}

export function normalizeCell(value: unknown): string | number | boolean | null {
  if (value === undefined) {
    return null;
  }
  if (value === null) {
    return null;
  }
  if (typeof value === "number") {
    return Number.isNaN(value) ? null : value;
  }
  if (typeof value === "boolean") {
    return value;
  }
  const rendered = String(value);
  return rendered.trim() === "" ? null : rendered;
}

export function readCsvFile(filePath: string): CsvRecord[] {
  const source = readTextFile(filePath);
  const parsed = parseCsv(source, {
    columns: true,
    skip_empty_lines: true,
    relax_quotes: true,
    relax_column_count: true
  }) as Array<Record<string, string>>;

  return parsed
    .filter((row) => Object.values(row).some((value) => String(value ?? "").trim() !== ""))
    .map((row) => {
      const normalizedEntries = Object.entries(row).map(([key, value]) => [key, normalizeCell(value)]);
      return Object.fromEntries(normalizedEntries);
    });
}

export function writeCsvJsonl(records: CsvRecord[], csvPath: string): void {
  ensureDir(path.dirname(csvPath));
  const fields = collectCsvFields(records);
  const csv = Papa.unparse({
    fields,
    data: records.map((record) => fields.map((field) => normalizeCell(record[field] ?? null)))
  });
  fs.writeFileSync(csvPath, `${csv}\n`, "utf8");
  const jsonlPath = csvPath.replace(/\.csv$/i, ".jsonl");
  fs.writeFileSync(
    jsonlPath,
    records.map((record) => JSON.stringify(jsonSafe(record))).join("\n").concat(records.length > 0 ? "\n" : ""),
    "utf8"
  );
}

export function writeCsvFile(records: CsvRecord[], csvPath: string, fields: string[] | null = null): void {
  ensureDir(path.dirname(csvPath));
  const csvFields = fields ?? collectCsvFields(records);
  const csv = Papa.unparse({
    fields: csvFields,
    data: records.map((record) => csvFields.map((field) => normalizeCell(record[field] ?? null)))
  });
  fs.writeFileSync(csvPath, `${csv}\n`, "utf8");
}

export function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath);
}

export function listFiles(root: string, matcher: (entryPath: string) => boolean): string[] {
  if (!fs.existsSync(root)) {
    return [];
  }
  const results: string[] = [];
  const walk = (currentPath: string): void => {
    const stat = fs.statSync(currentPath);
    if (stat.isDirectory()) {
      for (const child of fs.readdirSync(currentPath)) {
        walk(path.join(currentPath, child));
      }
      return;
    }
    if (matcher(currentPath)) {
      results.push(currentPath);
    }
  };
  walk(root);
  return results.sort();
}

function collectCsvFields(records: CsvRecord[]): string[] {
  const orderedFields: string[] = [];
  for (const record of records) {
    for (const field of Object.keys(record)) {
      if (!orderedFields.includes(field)) {
        orderedFields.push(field);
      }
    }
  }
  return orderedFields;
}
