import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DATA_ROOT_ENV = "MAHOUT_BENCH_HOME";

/**
 * Resolves the installed package root from any compiled file inside build/src.
 */
export function packageRootFromMeta(metaUrl: string): string {
  const currentDir = path.dirname(fileURLToPath(metaUrl));
  const twoLevelsUp = path.resolve(currentDir, "..", "..");
  return path.basename(twoLevelsUp) === "build" ? path.resolve(twoLevelsUp, "..") : twoLevelsUp;
}

/**
 * Returns the default package root for runtime modules.
 */
export function defaultPackageRoot(): string {
  return packageRootFromMeta(import.meta.url);
}

/**
 * Resolves where user-owned mutable data lives.
 */
export function resolveDataRoot(cwd = process.cwd(), env = process.env): string {
  const configured = env[DATA_ROOT_ENV];
  if (configured && configured.trim()) {
    return path.resolve(configured);
  }
  return path.resolve(cwd, ".mahout-bench");
}

/**
 * Keeps test fixtures and source checkouts working when they carry a local datasets directory.
 */
export function resolveDataRootForRepo(repoRoot: string): string {
  return fs.existsSync(path.join(repoRoot, "datasets")) ? repoRoot : resolveDataRoot();
}

/**
 * Resolves an output directory under the user data root.
 */
export function resolveOutputBase(dataRoot = resolveDataRoot()): string {
  return path.join(dataRoot, "outputs");
}

/**
 * Renders a concise runtime path summary for users and logs.
 */
export function runtimePathSummary(packageRoot: string, dataRoot: string): string {
  return `packageRoot=${packageRoot}\ndataRoot=${dataRoot}`;
}
