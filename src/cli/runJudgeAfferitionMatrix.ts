#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_JUDGE_VALIDATIONS_REGISTRY,
  type JudgeValidationRegistry,
  type JudgeValidationRegistryEntry
} from "../contracts/autobench.js";
import { readJsonFile } from "../io/filesystem.js";
import { defaultPackageRoot, resolveOutputBase } from "../runtime/paths.js";

type SamplingTarget =
  | { kind: "full"; label: "full" }
  | { kind: "margin"; label: "10pp" | "8pp" | "5pp"; margin: string };

type RunSpec = {
  lane: string;
  paperName: string;
  modelId: string;
  existingModelIds: string[];
  judgeConfig: string;
  targets: SamplingTarget[];
};

type CompletedRun = {
  modelId: string;
  samplingKey: string;
  outputPath: string;
  source: "registry" | "output";
};

type MatrixCell = {
  model: string;
  sampling: string;
  status: "complete" | "pending";
  similarity: number | null;
  invalidRate: number | null;
  completed: number | null;
  total: number | null;
  sourceOutputPath: string | null;
  paperResultsPath: string;
};

type MetricsPayload = {
  overall_similarity: number | null;
  metrics: Array<{
    dataset: string;
    total: number;
    invalidN: number;
  }>;
};

type ProgressPayload = {
  completed: number;
  total: number;
  remaining: number;
};

const repoRoot = defaultPackageRoot();
const workspaceRoot = path.dirname(repoRoot);
const outputBase = path.join(resolveOutputBase(), "judge_afferition");
const paperDataRoot = path.join(workspaceRoot, "PaperDATA");

const FULL_TARGET: SamplingTarget = { kind: "full", label: "full" };
const TARGETS: SamplingTarget[] = [
  { kind: "margin", label: "10pp", margin: "10pp" },
  { kind: "margin", label: "8pp", margin: "8pp" },
  { kind: "margin", label: "5pp", margin: "5pp" },
  FULL_TARGET
];

const RUNS: RunSpec[] = [
  {
    lane: "liquid-local-openai",
    paperName: "liquid_local",
    modelId: "lmstudio_openai_liquid_lfm25_12b",
    existingModelIds: ["lmstudio_openai_liquid_lfm25_12b"],
    judgeConfig: "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml",
    targets: TARGETS
  },
  {
    lane: "gemma4-remote-native",
    paperName: "gemma4_remote_lms",
    modelId: "lmstudio_native_gemma4_26b_a4b",
    existingModelIds: ["lmstudio_native_gemma4_26b_a4b", "lmstudio-gemma4-26b-a4b"],
    judgeConfig: "config/judge/gemma26_native_text.toml",
    targets: TARGETS
  }
];

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const dryRun = argv.includes("--dry-run");
  const completedRuns = loadCompletedRuns();
  const lanes = RUNS.map((run) => ({
    run,
    pending: run.targets.filter((target) => !hasCompletedRun(completedRuns, run.existingModelIds, samplingKey(target)))
  }));

  printPlan(lanes, completedRuns);
  if (dryRun) {
    printOrganizationPlan(completedRuns);
    printMatrixSummary(completedRuns);
    return 0;
  }
  await Promise.all(lanes.map(({ run, pending }) => runLane(run, pending)));
  const finalCompletedRuns = loadCompletedRuns();
  organizePaperData(finalCompletedRuns);
  writeMatrixSummary(finalCompletedRuns);
  process.stdout.write("\nJudge afferition matrix complete.\n");
  return 0;
}

function loadCompletedRuns(): CompletedRun[] {
  return [
    ...completedRunsFromRegistry(),
    ...completedRunsFromOutputs()
  ];
}

function completedRunsFromRegistry(): CompletedRun[] {
  const registryPath = path.join(repoRoot, DEFAULT_JUDGE_VALIDATIONS_REGISTRY);
  if (!fs.existsSync(registryPath)) {
    return [];
  }
  const registry = readJsonFile<JudgeValidationRegistry>(registryPath);
  return registry.validations
    .map((entry) => completedRunFromRegistryEntry(entry))
    .filter((entry): entry is CompletedRun => entry !== null);
}

function completedRunFromRegistryEntry(entry: JudgeValidationRegistryEntry): CompletedRun | null {
  const key = samplingKeyFromSummary(entry.afferitionSampling ?? null);
  if (!key) {
    return null;
  }
  return {
    modelId: entry.modelId,
    samplingKey: key,
    outputPath: path.join(repoRoot, entry.outputPath),
    source: "registry"
  };
}

function completedRunsFromOutputs(): CompletedRun[] {
  if (!fs.existsSync(outputBase)) {
    return [];
  }
  return fs.readdirSync(outputBase)
    .map((entry) => path.join(outputBase, entry))
    .filter((entryPath) => fs.statSync(entryPath).isDirectory())
    .map((entryPath) => completedRunFromOutput(entryPath))
    .filter((entry): entry is CompletedRun => entry !== null);
}

function completedRunFromOutput(outputPath: string): CompletedRun | null {
  const metricsPath = path.join(outputPath, "judge_afferition_metrics.json");
  const progressPath = path.join(outputPath, "judge_afferition_progress.json");
  if (!fs.existsSync(metricsPath) || !fs.existsSync(progressPath)) {
    return null;
  }
  const progress = readJsonFile<ProgressPayload>(progressPath);
  if (progress.total < 1 || progress.completed !== progress.total || progress.remaining !== 0) {
    return null;
  }
  const metrics = readJsonFile<{
    candidate: string;
    afferition_sampling?: unknown;
  }>(metricsPath);
  const key = samplingKeyFromSummary(metrics.afferition_sampling ?? null) ?? legacySamplingKey(progress.total);
  if (!key) {
    return null;
  }
  return {
    modelId: metrics.candidate,
    samplingKey: key,
    outputPath,
    source: "output"
  };
}

function samplingKeyFromSummary(summary: unknown): string | null {
  if (!summary || typeof summary !== "object") {
    return null;
  }
  const record = summary as Record<string, unknown>;
  const kind = String(record.kind ?? "");
  const marginLabel = String(record.marginLabel ?? "");
  if (kind === "full") {
    return "full";
  }
  if (kind === "margin" && marginLabel) {
    return marginLabel;
  }
  return null;
}

function legacySamplingKey(total: number): string | null {
  if (total === 27896) {
    return "full";
  }
  if (total === 4186) {
    return "5pp";
  }
  if (total === 1820) {
    return "8pp";
  }
  if (total === 1198) {
    return "10pp";
  }
  return null;
}

function hasCompletedRun(completedRuns: CompletedRun[], modelIds: string[], targetKey: string): boolean {
  const modelIdSet = new Set(modelIds);
  return completedRuns.some((run) => modelIdSet.has(run.modelId) && run.samplingKey === targetKey);
}

function printPlan(
  lanes: Array<{ run: RunSpec; pending: SamplingTarget[] }>,
  completedRuns: CompletedRun[]
): void {
  process.stdout.write("\nJudge afferition matrix plan\n");
  for (const { run, pending } of lanes) {
    const done = run.targets.filter((target) => hasCompletedRun(completedRuns, run.existingModelIds, samplingKey(target)));
    process.stdout.write(`\n${run.lane} (${run.modelId})\n`);
    process.stdout.write(`  skip: ${done.map((target) => target.label).join(", ") || "none"}\n`);
    process.stdout.write(`  run: ${pending.map((target) => target.label).join(", ") || "none"}\n`);
    for (const target of done) {
      const completed = findCompletedRun(completedRuns, run.existingModelIds, samplingKey(target));
      if (completed) {
        process.stdout.write(`    ${target.label}: ${completed.source} ${path.relative(repoRoot, completed.outputPath)}\n`);
      }
    }
  }
}

function printOrganizationPlan(completedRuns: CompletedRun[]): void {
  process.stdout.write(`\nPaperDATA organization plan: ${path.relative(workspaceRoot, paperDataRoot)}\n`);
  for (const run of RUNS) {
    for (const target of run.targets) {
      const completed = findCompletedRun(completedRuns, run.existingModelIds, samplingKey(target));
      const destination = paperResultsPath(run, target);
      const source = completed ? path.relative(workspaceRoot, completed.outputPath) : "pending";
      process.stdout.write(`  ${path.relative(workspaceRoot, destination)} <= ${source}\n`);
    }
  }
}

function printMatrixSummary(completedRuns: CompletedRun[]): void {
  process.stdout.write(`\nMarkdown matrix preview: ${path.relative(workspaceRoot, matrixMarkdownPath())}\n`);
  process.stdout.write(renderMatrixMarkdown(buildMatrixCells(completedRuns)));
}

function findCompletedRun(completedRuns: CompletedRun[], modelIds: string[], targetKey: string): CompletedRun | null {
  const modelIdSet = new Set(modelIds);
  return completedRuns.find((run) => modelIdSet.has(run.modelId) && run.samplingKey === targetKey) ?? null;
}

function organizePaperData(completedRuns: CompletedRun[]): void {
  process.stdout.write(`\nOrganizing PaperDATA: ${paperDataRoot}\n`);
  for (const run of RUNS) {
    for (const target of run.targets) {
      const completed = findCompletedRun(completedRuns, run.existingModelIds, samplingKey(target));
      if (!completed) {
        throw new Error(`Cannot organize missing result: ${run.lane} ${target.label}`);
      }
      copyResultToPaperData(run, target, completed);
    }
  }
}

function copyResultToPaperData(run: RunSpec, target: SamplingTarget, completed: CompletedRun): void {
  const destination = paperResultsPath(run, target);
  fs.mkdirSync(destination, { recursive: true });
  fs.cpSync(completed.outputPath, destination, { recursive: true, force: true });
  fs.writeFileSync(
    path.join(destination, "paperdata_manifest.json"),
    `${JSON.stringify({
      organizedAt: new Date().toISOString(),
      lane: run.lane,
      modelId: run.modelId,
      modelAliasesAccepted: run.existingModelIds,
      sampling: target.label,
      source: completed.source,
      sourceOutputPath: path.relative(workspaceRoot, completed.outputPath),
      destination: path.relative(workspaceRoot, destination)
    }, null, 2)}\n`
  );
  process.stdout.write(`  ${path.relative(workspaceRoot, destination)} <= ${path.relative(workspaceRoot, completed.outputPath)}\n`);
}

function paperResultsPath(run: RunSpec, target: SamplingTarget): string {
  return path.join(paperDataRoot, `${run.paperName}_${target.label}`, "resultados");
}

function writeMatrixSummary(completedRuns: CompletedRun[]): void {
  const cells = buildMatrixCells(completedRuns);
  fs.mkdirSync(paperDataRoot, { recursive: true });
  fs.writeFileSync(matrixMarkdownPath(), renderMatrixMarkdown(cells));
  fs.writeFileSync(matrixJsonPath(), `${JSON.stringify({
    createdAt: new Date().toISOString(),
    cells
  }, null, 2)}\n`);
  process.stdout.write(`\nWrote matrix: ${path.relative(workspaceRoot, matrixMarkdownPath())}\n`);
  process.stdout.write(`Wrote matrix JSON: ${path.relative(workspaceRoot, matrixJsonPath())}\n`);
}

function buildMatrixCells(completedRuns: CompletedRun[]): MatrixCell[] {
  return RUNS.flatMap((run) => run.targets.map((target) => {
    const completed = findCompletedRun(completedRuns, run.existingModelIds, samplingKey(target));
    const paperPath = paperResultsPath(run, target);
    if (!completed) {
      return {
        model: run.paperName,
        sampling: target.label,
        status: "pending",
        similarity: null,
        invalidRate: null,
        completed: null,
        total: null,
        sourceOutputPath: null,
        paperResultsPath: path.relative(workspaceRoot, paperPath)
      };
    }
    const summary = readRunSummary(completed.outputPath);
    return {
      model: run.paperName,
      sampling: target.label,
      status: "complete",
      similarity: summary.similarity,
      invalidRate: summary.invalidRate,
      completed: summary.completed,
      total: summary.total,
      sourceOutputPath: path.relative(workspaceRoot, completed.outputPath),
      paperResultsPath: path.relative(workspaceRoot, paperPath)
    };
  }));
}

function readRunSummary(outputPath: string): {
  similarity: number | null;
  invalidRate: number | null;
  completed: number;
  total: number;
} {
  const metrics = readJsonFile<MetricsPayload>(path.join(outputPath, "judge_afferition_metrics.json"));
  const progress = readJsonFile<ProgressPayload>(path.join(outputPath, "judge_afferition_progress.json"));
  const globalMetrics = metrics.metrics.filter((metric) => metric.dataset === "global");
  const invalidTotals = globalMetrics.reduce((current, metric) => ({
    total: current.total + metric.total,
    invalid: current.invalid + metric.invalidN
  }), { total: 0, invalid: 0 });
  return {
    similarity: metrics.overall_similarity,
    invalidRate: invalidTotals.total === 0 ? null : invalidTotals.invalid / invalidTotals.total,
    completed: progress.completed,
    total: progress.total
  };
}

function renderMatrixMarkdown(cells: MatrixCell[]): string {
  const lines = [
    "# Judge Afferition Matrix Results",
    "",
    `Generated: ${new Date().toISOString()}`,
    "",
    "Cell format: `similarity / invalid / completed`. Pending cells are not available yet.",
    "",
    "| Model | 10pp | 8pp | 5pp | full |",
    "|---|---:|---:|---:|---:|"
  ];
  for (const run of RUNS) {
    const cellsBySampling = new Map(cells.filter((cell) => cell.model === run.paperName).map((cell) => [cell.sampling, cell]));
    lines.push([
      run.paperName,
      ...TARGETS.map((target) => renderMatrixCell(cellsBySampling.get(target.label) ?? null))
    ].join(" | ").replace(/^/, "| ").replace(/$/, " |"));
  }
  lines.push(
    "",
    "## Result Directories",
    "",
    "| Model | PP | Status | Results | Source |",
    "|---|---|---|---|---|"
  );
  for (const cell of cells) {
    lines.push([
      cell.model,
      cell.sampling,
      cell.status,
      cell.status === "complete" ? markdownLink("resultados", cell.paperResultsPath) : "pending",
      cell.sourceOutputPath ? markdownLink("source", cell.sourceOutputPath) : "pending"
    ].join(" | ").replace(/^/, "| ").replace(/$/, " |"));
  }
  return `${lines.join("\n")}\n`;
}

function renderMatrixCell(cell: MatrixCell | null): string {
  if (!cell || cell.status !== "complete") {
    return "PENDING";
  }
  return [
    formatPercent(cell.similarity),
    formatPercent(cell.invalidRate),
    `${cell.completed ?? "?"}/${cell.total ?? "?"}`
  ].join(" / ");
}

function markdownLink(label: string, relativePath: string): string {
  const absolutePath = path.join(workspaceRoot, relativePath);
  const markdownRelativePath = path.relative(paperDataRoot, absolutePath).split(path.sep).join("/");
  return `[${label}](${markdownRelativePath})`;
}

function matrixMarkdownPath(): string {
  return path.join(paperDataRoot, "judge_afferition_matrix_results.md");
}

function matrixJsonPath(): string {
  return path.join(paperDataRoot, "judge_afferition_matrix_results.json");
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

async function runLane(run: RunSpec, targets: SamplingTarget[]): Promise<void> {
  if (targets.length === 0) {
    process.stdout.write(`[${run.lane}] nothing to run.\n`);
    return;
  }
  for (const target of targets) {
    await runTarget(run, target);
  }
}

function runTarget(run: RunSpec, target: SamplingTarget): Promise<void> {
  const args = [
    "exec",
    "tsx",
    "src/cli/validateJudge.ts",
    "--model-id",
    run.modelId,
    "--judge-config",
    run.judgeConfig
  ];
  if (target.kind === "margin") {
    args.push("--margin-of-error", target.margin);
  }
  process.stdout.write(`\n[${run.lane}] start ${target.label}: pnpm ${args.join(" ")}\n`);
  return new Promise((resolve, reject) => {
    const child = spawn("pnpm", args, {
      cwd: repoRoot,
      env: process.env,
      stdio: "inherit"
    });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        process.stdout.write(`[${run.lane}] done ${target.label}\n`);
        resolve();
        return;
      }
      reject(new Error(`[${run.lane}] ${target.label} failed with ${signal ? `signal ${signal}` : `exit code ${String(code)}`}`));
    });
  });
}

function samplingKey(target: SamplingTarget): string {
  return target.label;
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
