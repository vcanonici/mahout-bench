#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  ELEPHANT_FULL_RESULTS_DIR,
  JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR,
  JUDGE_AFFERITION_MARGIN_DIR,
  JUDGE_AFFERITION_STRATIFIED_1000_NAME,
  JUDGE_AFFERITION_TEST_SETS_DIR
} from "../contracts/autobench.js";
import { ensureDir, writeJson } from "../io/filesystem.js";
import { DATA_ROOT_ENV, resolveDataRoot } from "../runtime/paths.js";

const RELEASE_TAG = "v0.0.5";
const RELEASE_BASE_URL = `https://github.com/vcanonici/mahout-bench/releases/download/${RELEASE_TAG}`;
const DEFAULT_MANIFEST_URL = `${RELEASE_BASE_URL}/mahout-bench-data-v0.0.5.manifest.json`;
const DEFAULT_ARCHIVE_URL = `${RELEASE_BASE_URL}/mahout-bench-data-v0.0.5.zip`;
const LOCAL_MANIFEST_PATH = "mahout-bench-data.manifest.json";

const REQUIRED_DATA_PATHS = [
  `${ELEPHANT_FULL_RESULTS_DIR}/OEQ.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/AITA-YTA.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/AITA-NTA-OG.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/AITA-NTA-FLIP.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/SS.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/OEQ_full_results.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/AITA-YTA_full_results.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/SS_full_results.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/elephant_full_results/AITA-NTA-OG_full_results.csv`,
  `${ELEPHANT_FULL_RESULTS_DIR}/elephant_full_results/AITA-NTA-FLIP_flipped_full_results.csv`,
  `${JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR}/all.csv`,
  `${JUDGE_AFFERITION_MARGIN_DIR}/8pp/all.csv`,
  `${JUDGE_AFFERITION_TEST_SETS_DIR}/${JUDGE_AFFERITION_STRATIFIED_1000_NAME}/all.csv`
] as const;

export interface SetupDataManifest {
  version: "0.0.5";
  createdAt: string;
  archive: {
    fileName: string;
    url: string;
    sha256: string;
    sizeBytes: number;
  };
  requiredPaths: string[];
  source: {
    name: string;
    citation: string;
    url: string;
    license?: string;
  };
}

interface SetupArgs {
  dataRoot: string;
  manifestUrl: string;
  archiveUrl: string;
  force: boolean;
}

/**
 * Downloads, verifies, and installs the public data bundle.
 */
export async function setup(argv = process.argv.slice(2)): Promise<number> {
  const args = parseArgs(argv);
  ensureDir(args.dataRoot);
  const localManifest = path.join(args.dataRoot, LOCAL_MANIFEST_PATH);
  if (!args.force && isDataInstalled(args.dataRoot, localManifest)) {
    process.stdout.write(`Mahout Bench data already installed at ${args.dataRoot}\n`);
    process.stdout.write(`Set ${DATA_ROOT_ENV} to use a different data root.\n`);
    return 0;
  }

  const manifest = await loadRemoteManifest(args.manifestUrl);
  const archiveUrl = args.archiveUrl || manifest.archive.url || DEFAULT_ARCHIVE_URL;
  const archivePath = path.join(args.dataRoot, manifest.archive.fileName);
  await downloadToFile(archiveUrl, archivePath);
  verifyArchive(archivePath, manifest);
  extractZip(archivePath, args.dataRoot);
  verifyRequiredPaths(args.dataRoot, manifest.requiredPaths);
  writeJson(localManifest, manifest);
  process.stdout.write(`Mahout Bench data installed at ${args.dataRoot}\n`);
  process.stdout.write(`Source: ${manifest.source.citation}\n`);
  return 0;
}

function parseArgs(argv: string[]): SetupArgs {
  const args: SetupArgs = {
    dataRoot: resolveDataRoot(),
    manifestUrl: DEFAULT_MANIFEST_URL,
    archiveUrl: DEFAULT_ARCHIVE_URL,
    force: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    if (current === "--data-root") {
      args.dataRoot = path.resolve(requireValue(argv, ++index, current));
    } else if (current === "--manifest-url") {
      args.manifestUrl = requireValue(argv, ++index, current);
    } else if (current === "--archive-url") {
      args.archiveUrl = requireValue(argv, ++index, current);
    } else if (current === "--force") {
      args.force = true;
    } else if (current === "--help" || current === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown setup argument: ${current}`);
    }
  }
  return args;
}

function isDataInstalled(dataRoot: string, manifestPath: string): boolean {
  return fs.existsSync(manifestPath) && REQUIRED_DATA_PATHS.every((entry) => fs.existsSync(path.join(dataRoot, entry)));
}

async function loadRemoteManifest(url: string): Promise<SetupDataManifest> {
  const payload = await readUrlOrFile(url);
  const manifest = JSON.parse(payload) as SetupDataManifest;
  validateManifest(manifest);
  return manifest;
}

function validateManifest(manifest: SetupDataManifest): void {
  if (manifest.version !== "0.0.5") {
    throw new Error(`Unsupported data manifest version: ${manifest.version}`);
  }
  if (!manifest.archive?.fileName || !manifest.archive.sha256 || !Number.isInteger(manifest.archive.sizeBytes)) {
    throw new Error("Invalid data manifest archive contract");
  }
  const missing = REQUIRED_DATA_PATHS.filter((entry) => !manifest.requiredPaths.includes(entry));
  if (missing.length > 0) {
    throw new Error(`Data manifest is missing required paths: ${missing.join(", ")}`);
  }
}

async function downloadToFile(url: string, destination: string): Promise<void> {
  ensureDir(path.dirname(destination));
  if (url.startsWith("file://")) {
    fs.copyFileSync(fileURLToPath(url), destination);
    return;
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  fs.writeFileSync(destination, Buffer.from(await response.arrayBuffer()));
}

async function readUrlOrFile(url: string): Promise<string> {
  if (url.startsWith("file://")) {
    return fs.readFileSync(fileURLToPath(url), "utf8");
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function verifyArchive(archivePath: string, manifest: SetupDataManifest): void {
  const stat = fs.statSync(archivePath);
  if (stat.size !== manifest.archive.sizeBytes) {
    throw new Error(`Data archive size mismatch: expected ${manifest.archive.sizeBytes}, got ${stat.size}`);
  }
  const digest = crypto.createHash("sha256").update(fs.readFileSync(archivePath)).digest("hex");
  if (digest !== manifest.archive.sha256) {
    throw new Error(`Data archive SHA256 mismatch: expected ${manifest.archive.sha256}, got ${digest}`);
  }
}

function extractZip(zipPath: string, dataRoot: string): void {
  const result = spawnSync("unzip", ["-o", zipPath, "-d", dataRoot], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`Failed to extract ${zipPath}: ${result.stderr || result.stdout}`);
  }
}

function verifyRequiredPaths(dataRoot: string, requiredPaths: string[]): void {
  const missing = requiredPaths.filter((entry) => !fs.existsSync(path.join(dataRoot, entry)));
  if (missing.length > 0) {
    throw new Error(`Data bundle is incomplete after extraction: ${missing.join(", ")}`);
  }
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function printHelp(): void {
  process.stdout.write(`mahout-bench setup\n\n`);
  process.stdout.write(`Downloads the v0.0.5 data bundle into ${DATA_ROOT_ENV} or ./.mahout-bench.\n\n`);
  process.stdout.write(`--data-root <path>      Override the data install directory.\n`);
  process.stdout.write(`--manifest-url <url>   Override the release manifest URL.\n`);
  process.stdout.write(`--archive-url <url>    Override the release archive URL.\n`);
  process.stdout.write(`--force                Redownload and reinstall even when data is present.\n`);
}

if (path.resolve(process.argv[1] ?? "") === path.resolve(fileURLToPath(import.meta.url))) {
  setup().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}

export { REQUIRED_DATA_PATHS, DEFAULT_ARCHIVE_URL, DEFAULT_MANIFEST_URL };
