#!/usr/bin/env node

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { loadModelCatalog } from "../config/loadConfig.js";
import { DEFAULT_JUDGE_VALIDATIONS_REGISTRY, type JudgeValidationRegistryEntry } from "../contracts/autobench.js";
import { ensureDir } from "../io/filesystem.js";
import {
  ensureJudgeAfferitionStratifiedTestSet1000,
  ensureJudgeAfferitionMarginDataset,
  parseJudgeAfferitionMarginInput,
  parseJudgeAfferitionTestSetInput
} from "../validate_judge/judgeAfferitionSampling.js";
import {
  loadJudgeValidationRegistry,
  saveJudgeValidationRegistry,
  upsertJudgeValidation
} from "../validate_judge/judgeValidationRegistry.js";
import { TerminalObserver } from "../runtime/terminalObserver.js";
import { defaultPackageRoot, resolveOutputBase } from "../runtime/paths.js";
import { ensureClaudeSocialJudgeAfferition } from "../validate_judge/prepareElephantFullResults.js";
import { runJudgeValidation } from "../validate_judge/runJudgeValidation.js";

const repoRoot = defaultPackageRoot();
const outputBase = path.join(resolveOutputBase(), "judge_afferition");

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const args = parseArgs(argv);
  const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === args.modelId);
  if (!model) {
    throw new Error(`Unknown judge model id: ${args.modelId}`);
  }
  ensureDir(outputBase);
  if (args.marginOfError !== null && args.testSet !== null) {
    throw new Error("--margin-of-error and --test-set are mutually exclusive");
  }
  let dataDir = await ensureClaudeSocialJudgeAfferition(repoRoot);
  const samplingDataset = args.marginOfError === null ? null : ensureJudgeAfferitionMarginDataset(repoRoot, args.marginOfError);
  const testSetDataset = args.testSet === null ? null : ensureJudgeAfferitionStratifiedTestSet1000(repoRoot);
  if (samplingDataset) {
    dataDir = samplingDataset.dataDir;
    process.stdout.write(
      `Judge afferition sample: ${samplingDataset.sampling.datasetPath} ` +
        `(${samplingDataset.manifest.sample_total}/${samplingDataset.manifest.full_total}, ${samplingDataset.manifest.margin_label})\n`
    );
  }
  if (testSetDataset) {
    dataDir = testSetDataset.dataDir;
    process.stdout.write(
      `Judge afferition test set: ${testSetDataset.sampling.datasetPath} ` +
        `(${testSetDataset.manifest.sample_total}/${testSetDataset.manifest.full_total})\n`
    );
  }
  const observer = new TerminalObserver(true);
  const entry = await runJudgeValidation({
    repoRoot,
    outputBase,
    dataDir,
    model,
    judgeConfigPath: args.judgeConfig,
    limit: args.limit,
    observer,
    outputPath: args.outputPath,
    afferitionSampling: samplingDataset?.sampling ?? testSetDataset?.sampling ?? null
  }).finally(() => observer.stop());
  if (args.limit === null && args.testSet === null) {
    const registry = upsertJudgeValidation(loadJudgeValidationRegistry(repoRoot), registryEntry(entry));
    saveJudgeValidationRegistry(repoRoot, registry, DEFAULT_JUDGE_VALIDATIONS_REGISTRY);
  }
  process.stdout.write(`Judge afferition complete: similarity=${formatPercent(entry.overallSimilarity)}\n`);
  process.stdout.write(`Report: ${entry.outputPath}\n`);
  if (args.limit !== null || args.testSet !== null) {
    process.stdout.write("Limited smoke run: registry not updated.\n");
    return 0;
  }
  return 0;
}

function parseArgs(argv: string[]): {
  modelId: string;
  judgeConfig: string;
  limit: number | null;
  outputPath: string | null;
  marginOfError: number | null;
  testSet: string | null;
} {
  const args = {
    modelId: "lmstudio_native_liquid_lfm25_12b",
    judgeConfig: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
    limit: null as number | null,
    outputPath: null as string | null,
    marginOfError: null as number | null,
    testSet: null as string | null
  };
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    if (current === "--model-id") {
      args.modelId = requireValue(argv, ++index, current);
    } else if (current === "--judge-config") {
      args.judgeConfig = requireValue(argv, ++index, current);
    } else if (current === "--limit") {
      args.limit = parsePositiveInteger(requireValue(argv, ++index, current), current);
    } else if (current === "--resume-output") {
      args.outputPath = path.resolve(requireValue(argv, ++index, current));
    } else if (current === "--margin-of-error") {
      args.marginOfError = parseJudgeAfferitionMarginInput(requireValue(argv, ++index, current));
    } else if (current === "--test-set") {
      args.testSet = parseJudgeAfferitionTestSetInput(requireValue(argv, ++index, current));
    } else {
      throw new Error(`Unknown argument: ${current}`);
    }
  }
  return args;
}

function registryEntry(entry: JudgeValidationRegistryEntry): JudgeValidationRegistryEntry {
  return {
    ...entry,
    outputPath: path.relative(repoRoot, entry.outputPath)
  };
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

function parsePositiveInteger(value: string, flag: string): number {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric < 1) {
    throw new Error(`Expected positive integer for ${flag}, got: ${value}`);
  }
  return numeric;
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

if (path.resolve(process.argv[1] ?? "") === path.resolve(fileURLToPath(import.meta.url))) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}
