#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { loadModelCatalog, parseJudge, resolveInferenceFromModelCatalog } from "../config/loadConfig.js";
import {
  JUDGE_AFFERITION_STRATIFIED_1000_NAME,
  type JudgeValidationRegistryEntry,
  type ModelCatalogEntry
} from "../contracts/autobench.js";
import { buildJudgeResponseFormat } from "../inference/chatClient.js";
import { ensureDir, localRunStamp, readJsonFile, readTextFile, utcNowIso, writeJson, writeTextFile } from "../io/filesystem.js";
import { TerminalObserver } from "../runtime/terminalObserver.js";
import { defaultPackageRoot, resolveOutputBase } from "../runtime/paths.js";
import { ensureJudgeAfferitionStratifiedTestSet1000 } from "../validate_judge/judgeAfferitionSampling.js";
import { ensureClaudeSocialJudgeAfferition } from "../validate_judge/prepareElephantFullResults.js";
import { runJudgeValidation } from "../validate_judge/runJudgeValidation.js";

const repoRoot = defaultPackageRoot();
const outputBase = path.join(resolveOutputBase(), "judge_afferition");

const MODES = [
  {
    label: "Liquid LM Studio Native text",
    modelId: "lmstudio_native_liquid_lfm25_12b",
    judgeConfig: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml"
  },
  {
    label: "Liquid LM Studio OpenAI JSON",
    modelId: "lmstudio_openai_liquid_lfm25_12b",
    judgeConfig: "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml"
  }
] as const;

interface ModeSummary {
  label: string;
  modelId: string;
  judgeConfig: string;
  outputPath: string;
  completed: number;
  total: number;
  invalidRate: number;
  similarity: number | null;
  averageAttempts: number | null;
  provider: string;
  apiMode: string;
  responseFormatUsed: boolean;
}

export async function main(): Promise<number> {
  ensureDir(outputBase);
  await ensureClaudeSocialJudgeAfferition(repoRoot);
  const testSet = ensureJudgeAfferitionStratifiedTestSet1000(repoRoot);
  const runRoot = path.join(outputBase, `compare_liquid_modes_${localRunStamp()}`);
  ensureDir(runRoot);
  const summaries: ModeSummary[] = [];

  for (const mode of MODES) {
    const model = requireModel(mode.modelId);
    const observer = new TerminalObserver(true);
    const outputPath = path.join(runRoot, mode.modelId);
    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir: testSet.dataDir,
      model,
      judgeConfigPath: mode.judgeConfig,
      observer,
      outputPath,
      afferitionSampling: testSet.sampling
    }).finally(() => observer.stop());
    summaries.push(buildModeSummary(mode.label, model, mode.judgeConfig, entry));
  }

  const payload = {
    created_at: utcNowIso(),
    test_set: JUDGE_AFFERITION_STRATIFIED_1000_NAME,
    gate: "lmstudio_openai_liquid_lfm25_12b approved only when invalidRate is 0",
    openai_json_approved: summaries.find((summary) => summary.modelId === "lmstudio_openai_liquid_lfm25_12b")?.invalidRate === 0,
    modes: summaries
  };
  writeJson(path.join(runRoot, "liquid_modes_comparison.json"), payload);
  writeTextFile(path.join(runRoot, "liquid_modes_comparison.md"), renderReport(payload));
  process.stdout.write(`Liquid mode comparison complete: ${runRoot}\n`);
  return 0;
}

function requireModel(modelId: string): ModelCatalogEntry {
  const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === modelId);
  if (!model) {
    throw new Error(`Missing required model catalog id: ${modelId}`);
  }
  return model;
}

function buildModeSummary(
  label: string,
  model: ModelCatalogEntry,
  judgeConfig: string,
  entry: JudgeValidationRegistryEntry
): ModeSummary {
  const judge = parseJudge(repoRoot, judgeConfig);
  const resolvedInference = resolveInferenceFromModelCatalog({ repoRoot, base: judge.inference, modelId: model.id });
  const labels = readCandidateArtifacts(entry.outputPath);
  const attempts = labels.map((artifact) => Number(artifact.attempts)).filter((value) => Number.isFinite(value));
  return {
    label,
    modelId: model.id,
    judgeConfig,
    outputPath: entry.outputPath,
    completed: labels.length,
    total: entry.afferitionSampling?.sampleTotal ?? labels.length,
    invalidRate: invalidRate(entry),
    similarity: entry.overallSimilarity,
    averageAttempts: attempts.length === 0 ? null : attempts.reduce((total, value) => total + value, 0) / attempts.length,
    provider: resolvedInference.provider,
    apiMode: resolvedInference.apiMode,
    responseFormatUsed: buildJudgeResponseFormat({ ...judge, inference: resolvedInference }) !== null
  };
}

function readCandidateArtifacts(outputPath: string): Array<{ attempts: number }> {
  const labelsPath = path.join(outputPath, "labels_candidate.jsonl");
  if (!fs.existsSync(labelsPath)) {
    return [];
  }
  return readTextFile(labelsPath)
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as { attempts: number });
}

function invalidRate(entry: JudgeValidationRegistryEntry): number {
  const metricsPath = path.join(entry.outputPath, "judge_afferition_metrics.json");
  const metrics = readJsonFile<{ metrics: Array<{ dataset: string; total: number; invalidN: number }> }>(metricsPath).metrics;
  const global = metrics.filter((metric) => metric.dataset === "global");
  const totals = global.reduce((current, metric) => ({
    total: current.total + metric.total,
    invalidN: current.invalidN + metric.invalidN
  }), { total: 0, invalidN: 0 });
  return totals.total === 0 ? 1 : totals.invalidN / totals.total;
}

function renderReport(payload: {
  created_at: string;
  test_set: string;
  gate: string;
  openai_json_approved: boolean;
  modes: ModeSummary[];
}): string {
  const lines = [
    "# Liquid Judge Mode Comparison",
    "",
    `Created: ${payload.created_at}`,
    `Test set: ${payload.test_set}`,
    `Gate: ${payload.gate}`,
    `OpenAI JSON approved: ${payload.openai_json_approved ? "yes" : "no"}`,
    "",
    "| Mode | Completed | Invalid | Similarity | Avg Attempts | Provider | API Mode | Response Format | Output |",
    "|---|---:|---:|---:|---:|---|---|---|---|"
  ];
  for (const mode of payload.modes) {
    lines.push([
      mode.modelId,
      `${mode.completed}/${mode.total}`,
      formatPercent(mode.invalidRate),
      formatPercent(mode.similarity),
      mode.averageAttempts === null ? "n/a" : mode.averageAttempts.toFixed(2),
      mode.provider,
      mode.apiMode,
      mode.responseFormatUsed ? "yes" : "no",
      mode.outputPath
    ].join(" | ").replace(/^/, "| ").replace(/$/, " |"));
  }
  lines.push("");
  return lines.join("\n");
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
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
