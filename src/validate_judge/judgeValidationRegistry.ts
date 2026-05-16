import fs from "node:fs";
import path from "node:path";

import {
  DEFAULT_JUDGE_VALIDATIONS_REGISTRY,
  type JudgeValidationRegistry,
  type JudgeValidationRegistryEntry,
  type ModelCatalogEntry
} from "../contracts/autobench.js";
import { readJsonFile, utcNowIso, writeJson } from "../io/filesystem.js";

/**
 * Loads the judge validation registry, creating an empty in-memory registry when absent.
 */
export function loadJudgeValidationRegistry(
  repoRoot: string,
  registryPath = DEFAULT_JUDGE_VALIDATIONS_REGISTRY
): JudgeValidationRegistry {
  const absolutePath = path.join(repoRoot, registryPath);
  if (!fs.existsSync(absolutePath)) {
    return { version: 1, updatedAt: utcNowIso(), validations: [] };
  }
  const registry = readJsonFile<JudgeValidationRegistry>(absolutePath);
  validateRegistry(registry, registryPath);
  return registry;
}

export function saveJudgeValidationRegistry(
  repoRoot: string,
  registry: JudgeValidationRegistry,
  registryPath = DEFAULT_JUDGE_VALIDATIONS_REGISTRY
): void {
  writeJson(path.join(repoRoot, registryPath), { ...registry, updatedAt: utcNowIso() });
}

export function upsertJudgeValidation(
  registry: JudgeValidationRegistry,
  entry: JudgeValidationRegistryEntry
): JudgeValidationRegistry {
  return {
    version: 1,
    updatedAt: utcNowIso(),
    validations: [
      entry,
      ...registry.validations.filter((existing) => existing.modelId !== entry.modelId)
    ]
  };
}

export function findUsableJudgeValidation(
  registry: JudgeValidationRegistry,
  model: ModelCatalogEntry
): JudgeValidationRegistryEntry | null {
  const entry = registry.validations.find((validation) => validation.modelId === model.id);
  return entry ?? null;
}

export function formatValidationSummary(entry: JudgeValidationRegistryEntry): string {
  return [
    `status=${formatAfferitionSampling(entry)}`,
    `validated=${entry.validatedAt}`,
    `similarity=${formatPercent(entry.overallSimilarity)}`,
    `reference=${entry.reference}`,
    `report=${entry.outputPath}`
  ].join(" | ");
}

function formatAfferitionSampling(entry: JudgeValidationRegistryEntry): string {
  if (!entry.afferitionSampling || entry.afferitionSampling.kind === "full") {
    return "aferido (full)";
  }
  if (entry.afferitionSampling.kind === "test_set") {
    return `aferido (${entry.afferitionSampling.marginLabel})`;
  }
  return `aferido (${entry.afferitionSampling.marginLabel} margem de erro)`;
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

function validateRegistry(registry: JudgeValidationRegistry, registryPath: string): void {
  if (registry.version !== 1 || !Array.isArray(registry.validations)) {
    throw new Error(`Invalid judge validation registry: ${registryPath}`);
  }
}
