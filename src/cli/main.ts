#!/usr/bin/env node

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_JUDGE_CONFIG,
  DEFAULT_PROFILES_ROOT,
  type BenchmarkArgs
} from "../contracts/autobench.js";
import { parseGenerationPoolJson } from "../config/generationPool.js";
import { runBenchmark, runDrySmoke, runSelfTest, validateConfigCli } from "../pipeline/benchmarkRunner.js";

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const args = parseArgs(argv);
  if (args.selfTest) {
    return runSelfTest();
  }
  if (args.drySmoke) {
    return runDrySmoke();
  }
  if (args.validateConfig) {
    return validateConfigCli(args);
  }
  return runBenchmark(args);
}

function parseArgs(argv: string[]): BenchmarkArgs {
  const args: BenchmarkArgs = {
    selfTest: false,
    drySmoke: false,
    validateConfig: false,
    judgeOnly: false,
    profilesRoot: DEFAULT_PROFILES_ROOT,
    profiles: [],
    judgeConfig: DEFAULT_JUDGE_CONFIG,
    outputRoot: "",
    skipLms: false,
    generationModelId: "",
    generationPool: [],
    judgeModelId: "",
    judgePool: [],
    benchmarkName: "",
    marginOfError: null
  };

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    switch (current) {
      case "--self-test":
        args.selfTest = true;
        break;
      case "--dry-smoke":
        args.drySmoke = true;
        break;
      case "--validate-config":
        args.validateConfig = true;
        break;
      case "--judge-only":
        args.judgeOnly = true;
        break;
      case "--skip-lms":
        args.skipLms = true;
        break;
      case "--profiles-root":
        args.profilesRoot = requireValue(argv, ++index, current);
        break;
      case "--profiles":
        args.profiles = requireValue(argv, ++index, current).split(",").map((value) => value.trim()).filter(Boolean);
        break;
      case "--judge-config":
        args.judgeConfig = requireValue(argv, ++index, current);
        break;
      case "--output-root":
        args.outputRoot = requireValue(argv, ++index, current);
        break;
      case "--generation-model-id":
        args.generationModelId = requireValue(argv, ++index, current);
        break;
      case "--generation-pool":
        args.generationPool = parseGenerationPoolJson(requireValue(argv, ++index, current));
        break;
      case "--judge-model-id":
        args.judgeModelId = requireValue(argv, ++index, current);
        break;
      case "--judge-pool":
        args.judgePool = parseGenerationPoolJson(requireValue(argv, ++index, current));
        break;
      case "--benchmark-name":
        args.benchmarkName = requireValue(argv, ++index, current);
        break;
      case "--margin-of-error":
        args.marginOfError = parseMarginOfError(requireValue(argv, ++index, current));
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${current}`);
    }
  }

  return args;
}

function parseMarginOfError(value: string): number {
  const normalized = value.trim().toLowerCase();
  if (normalized === "full" || normalized === "fullbench") {
    return 0;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0 || numeric >= 1) {
    throw new Error(`Expected --margin-of-error between 0 and 1 or fullbench, got: ${value}`);
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

function printHelp(): void {
  process.stdout.write(`Mahout Bench: sycophancy benchmark runner.\n\n`);
  process.stdout.write(`Run through "mahout-bench run". Use "mahout-bench setup" before config validation or real benchmarks.\n\n`);
  process.stdout.write(`--self-test        Run parser and sampling self-tests only.\n`);
  process.stdout.write(`--dry-smoke        Run fake smoke check without network calls.\n`);
  process.stdout.write(`--validate-config  Validate TOMLs, datasets, and sample sizes only.\n`);
  process.stdout.write(`--judge-only       Reuse existing generation artifacts and rerun only judge + consolidation.\n`);
  process.stdout.write(`--profiles-root    Profiles root relative to the installed package.\n`);
  process.stdout.write(`--profiles         Comma-separated profile names from profiles root.\n`);
  process.stdout.write(`--judge-config     Judge TOML relative to the installed package.\n`);
  process.stdout.write(`--output-root      Optional output root override.\n`);
  process.stdout.write(`--generation-model-id  Model catalog id for generation inference.\n`);
  process.stdout.write(`--generation-pool      JSON array of generation backends with modelId, workers, timeoutSeconds.\n`);
  process.stdout.write(`--judge-model-id       Model catalog id for judge inference.\n`);
  process.stdout.write(`--judge-pool           JSON array of judge backends with modelId, workers, timeoutSeconds.\n`);
  process.stdout.write(`--benchmark-name       Human-readable benchmark name stored in artifacts.\n`);
  process.stdout.write(`--margin-of-error      Sampling margin override, e.g. 0.10, 0, full, or fullbench.\n`);
  process.stdout.write(`--skip-lms         Compatibility no-op; LM Studio is always provider-managed.\n`);
}

function isDirectCliExecution(): boolean {
  const entryPath = process.argv[1];
  if (!entryPath) {
    return false;
  }
  return path.resolve(entryPath) === fileURLToPath(import.meta.url);
}

if (isDirectCliExecution()) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}
