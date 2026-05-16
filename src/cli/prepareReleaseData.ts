#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { ensureDir, writeJson } from "../io/filesystem.js";
import { resolveDataRoot } from "../runtime/paths.js";
import { DEFAULT_ARCHIVE_URL, REQUIRED_DATA_PATHS, type SetupDataManifest } from "./setup.js";

const ARCHIVE_NAME = "mahout-bench-data-v0.0.5.zip";
const MANIFEST_NAME = "mahout-bench-data-v0.0.5.manifest.json";

/**
 * Creates the v0.0.5 GitHub Release data archive and manifest.
 */
export function main(argv = process.argv.slice(2)): number {
  const args = parseArgs(argv);
  verifyRequiredPaths(args.dataRoot);
  ensureDir(args.outputDir);
  const archivePath = path.join(args.outputDir, ARCHIVE_NAME);
  createZip(args.dataRoot, archivePath);
  const stat = fs.statSync(archivePath);
  const manifest: SetupDataManifest = {
    version: "0.0.5",
    createdAt: new Date().toISOString(),
    archive: {
      fileName: ARCHIVE_NAME,
      url: DEFAULT_ARCHIVE_URL,
      sha256: sha256File(archivePath),
      sizeBytes: stat.size
    },
    requiredPaths: [...REQUIRED_DATA_PATHS],
    source: {
      name: "Mahout Bench data bundle",
      citation: "Mahout Bench v0.0.5 data bundle, distributed by vcanonici/mahout-bench GitHub Releases.",
      url: DEFAULT_ARCHIVE_URL,
      license: "CC0-1.0 upstream data; MIT package code"
    },
    upstream: {
      name: "ELEPHANT / Social Sycophancy",
      citation: "Cheng, Yu, Lee, Khadpe, Ibrahim, and Jurafsky. ELEPHANT: Measuring and understanding social sycophancy in LLMs.",
      url: "https://github.com/myracheng/elephant",
      license: "CC0-1.0"
    }
  };
  writeJson(path.join(args.outputDir, MANIFEST_NAME), manifest);
  process.stdout.write(`Release data archive: ${archivePath}\n`);
  process.stdout.write(`Release data manifest: ${path.join(args.outputDir, MANIFEST_NAME)}\n`);
  return 0;
}

function parseArgs(argv: string[]): { dataRoot: string; outputDir: string } {
  const args = {
    dataRoot: resolveDataRoot(),
    outputDir: path.resolve("release-assets")
  };
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    if (current === "--") {
      continue;
    } else if (current === "--data-root") {
      args.dataRoot = path.resolve(requireValue(argv, ++index, current));
    } else if (current === "--output-dir") {
      args.outputDir = path.resolve(requireValue(argv, ++index, current));
    } else {
      throw new Error(`Unknown release data argument: ${current}`);
    }
  }
  return args;
}

function verifyRequiredPaths(dataRoot: string): void {
  const missing = REQUIRED_DATA_PATHS.filter((entry) => !fs.existsSync(path.join(dataRoot, entry)));
  if (missing.length > 0) {
    throw new Error(`Cannot build release data archive. Missing paths: ${missing.join(", ")}`);
  }
}

function createZip(dataRoot: string, archivePath: string): void {
  const stagingDir = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-release-"));
  try {
    copyRequiredPaths(dataRoot, stagingDir);
    const args = ["-qr", archivePath, ...topLevelEntries()];
    const result = spawnSync("zip", args, { cwd: stagingDir, encoding: "utf8" });
    if (result.status !== 0) {
      throw new Error(`Failed to create ${archivePath}: ${result.stderr || result.stdout}`);
    }
  } finally {
    fs.rmSync(stagingDir, { recursive: true, force: true });
  }
}

function copyRequiredPaths(dataRoot: string, stagingDir: string): void {
  for (const entry of REQUIRED_DATA_PATHS) {
    const sourcePath = path.join(dataRoot, entry);
    const targetPath = path.join(stagingDir, entry);
    ensureDir(path.dirname(targetPath));
    fs.copyFileSync(sourcePath, targetPath);
  }
}

function topLevelEntries(): string[] {
  return [...new Set(REQUIRED_DATA_PATHS.map((entry) => entry.split(path.sep)[0] ?? entry))];
}

function sha256File(filePath: string): string {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

if (path.resolve(process.argv[1] ?? "") === path.resolve(fileURLToPath(import.meta.url))) {
  try {
    process.exit(main());
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  }
}
