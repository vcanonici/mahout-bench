#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_JUDGE_CONFIG,
  DEFAULT_JUDGE_VALIDATIONS_REGISTRY,
  type BenchmarkArgs,
  type GenerationPoolRequest,
  type JudgeValidationRegistryEntry,
  type ModelCatalogEntry,
  type ProviderCatalogEntry
} from "../contracts/autobench.js";
import {
  datasetPath,
  enabledDatasets,
  loadModelCatalog,
  loadProfiles,
  loadProviderCatalog,
  modelSupportsRole
} from "../config/loadConfig.js";
import { defaultTimeoutForProvider, defaultWorkersForProvider } from "../config/generationPool.js";
import { formatDiscoverySummary, refreshAvailableModelsBestEffort, type ModelDiscoveryResult } from "../config/modelDiscovery.js";
import { ensureDir, listFiles, localRunStamp, readCsvFile, readJsonFile } from "../io/filesystem.js";
import { runBenchmark } from "../pipeline/benchmarkRunner.js";
import { TerminalObserver } from "../runtime/terminalObserver.js";
import { defaultPackageRoot, resolveOutputBase } from "../runtime/paths.js";
import { bootstrap } from "./bootstrap.js";
import { commonMoralIds, estimateCalls } from "../sampling/samplePlanner.js";
import {
  ensureJudgeAfferitionStratifiedTestSet1000,
  ensureJudgeAfferitionMarginDataset,
  formatMarginLabel,
  listJudgeAfferitionMarginDatasets,
  parseJudgeAfferitionMarginInput
} from "../validate_judge/judgeAfferitionSampling.js";
import { loadElephantReference } from "../validate_judge/loadElephantReference.js";
import { ensureClaudeSocialJudgeAfferition } from "../validate_judge/prepareElephantFullResults.js";
import { runJudgeValidation } from "../validate_judge/runJudgeValidation.js";
import {
  listIncompleteJudgeAfferitionRuns,
  type IncompleteJudgeAfferitionRun
} from "../validate_judge/runState.js";
import {
  findUsableJudgeValidation,
  formatValidationSummary,
  loadJudgeValidationRegistry,
  saveJudgeValidationRegistry,
  upsertJudgeValidation
} from "../validate_judge/judgeValidationRegistry.js";

type ProfileRootChoice = {
  root: string;
  profiles: string[];
};

type JudgeAfferitionDataChoice =
  | { kind: "full" }
  | { kind: "margin"; marginOfError: number }
  | { kind: "test_set" };

type ResumableBenchRun = {
  outputRoot: string;
  checkpointResults: number;
  rawGenerationLines: number;
  lastUpdatedMs: number;
  started: RunStartedEvent;
};

type RunStartedEvent = {
  event: "run_started";
  benchmark_name?: string;
  profiles_root?: string;
  profiles?: string;
  judge_config_path?: string;
  margin_of_error?: number;
  generation_pool?: RunStartedPoolEntry[];
  judge_pool?: RunStartedPoolEntry[];
  output_root?: string;
};

export type RunStartedPoolEntry = {
  model_id?: string;
  model?: string;
  workers?: number;
  timeout_seconds?: number;
};

export type ModelAvailabilityStatus = "ONLINE" | "OFFLINE" | "UNKNOWN";

export type ModelChoiceRow = {
  index: number;
  model: ModelCatalogEntry;
  provider: ProviderCatalogEntry | null;
  status: ModelAvailabilityStatus;
};

const repoRoot = defaultPackageRoot();
const outputBase = resolveOutputBase();
const judgeAfferitionOutputBase = path.join(outputBase, "judge_afferition");
const NATIVE_TEXT_JUDGE_CONFIG = "config/judge/gemma26_native_text.toml";
const LIQUID_TEXT_JUDGE_CONFIG = "config/judge/liquid_lfm25_12b_native_text_parallel4.toml";
const LIQUID_JSON_JUDGE_CONFIG = "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml";
const MODEL_TABLE_MODEL_WIDTH = 48;
let lastDiscoveryResult: ModelDiscoveryResult | null = null;

export async function main(): Promise<number> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    lastDiscoveryResult = await refreshModelCatalog();
    while (true) {
      process.stdout.write("\nMahout Bench\n");
      const choice = await choose(rl, "Menu", ["start bench", "resume bench", "validate judge", "bootstrap/configure providers", "list benchs", "exit"]);
      if (choice === "exit") {
        return 0;
      }
      if (choice === "bootstrap/configure providers") {
        await bootstrap(["--skip-setup"], rl);
        lastDiscoveryResult = await refreshModelCatalog();
        continue;
      }
      if (choice === "list benchs") {
        listBenchs();
        continue;
      }
      if (choice === "resume bench") {
        const args = await buildResumeArgs(rl);
        const code = await runBenchmark(args);
        if (code !== 0) {
          return code;
        }
        continue;
      }
      if (choice === "validate judge") {
        await validateJudgeFromMenu(rl);
        continue;
      }
      const args = await buildStartArgs(rl);
      const code = await runBenchmark(args);
      if (code !== 0) {
        return code;
      }
    }
  } finally {
    rl.close();
  }
}

async function buildStartArgs(rl: readline.Interface): Promise<BenchmarkArgs> {
  const profileRoot = await chooseProfileRoot(rl);
  const profiles = await chooseProfiles(rl, profileRoot);
  const generationModel = await chooseModel(rl, "generation");
  const generationPool = await chooseGenerationPool(rl, generationModel);
  const marginOfError = await chooseMarginOfError(rl, profileRoot.root, profiles);
  const judgeModel = await chooseValidatedJudgeModel(rl);
  const judgePool = await chooseJudgePool(rl, judgeModel, generationPoolUsesDualLmStudio(generationPool));
  printPlannedTokenEstimate(profileRoot.root, profiles, marginOfError, generationModel, generationPool, judgeModel, judgePool);
  const benchmarkName = await requireAnswer(rl, "Benchmark name");
  const outputRoot = path.join(outputBase, `${slugify(benchmarkName)}_${localRunStamp()}`);

  process.stdout.write(`\nOutput root: ${outputRoot}\n`);
  return {
    selfTest: false,
    drySmoke: false,
    validateConfig: false,
    judgeOnly: false,
    profilesRoot: profileRoot.root,
    profiles,
    judgeConfig: judgeConfigForModel(judgeModel),
    outputRoot,
    skipLms: false,
    generationModelId: generationModel.id,
    generationPool,
    judgeModelId: judgeModel.id,
    judgePool,
    benchmarkName,
    marginOfError,
    resumeMode: null
  };
}

async function buildResumeArgs(rl: readline.Interface): Promise<BenchmarkArgs> {
  const resumeMode = resumeModeForChoice(await choose(rl, "Resume mode", resumeModeChoices()));
  const run = await chooseResumableBenchRun(rl);
  const profileRoot = run.started.profiles_root ?? (await chooseProfileRoot(rl)).root;
  const profiles = parseRunProfiles(run.started);
  const marginOfError = run.started.margin_of_error ?? await chooseMarginOfError(rl, profileRoot, profiles);
  const generationPool = remapResumePoolToCurrentCatalog(run.started.generation_pool ?? [], "generation");
  const judgePool = remapResumePoolToCurrentCatalog(run.started.judge_pool ?? [], "judge");
  const judgeModelId = judgePool[1]?.modelId ?? judgePool[0]?.modelId ?? "";
  process.stdout.write(
    `\nRetomando ${run.outputRoot}\n` +
      `Resume mode: ${resumeMode}\n` +
      `Checkpoint: ${run.checkpointResults} generation result(s), raw_generation=${run.rawGenerationLines} linha(s)\n`
  );
  return {
    selfTest: false,
    drySmoke: false,
    validateConfig: false,
    judgeOnly: false,
    profilesRoot: profileRoot,
    profiles,
    judgeConfig: run.started.judge_config_path ?? DEFAULT_JUDGE_CONFIG,
    outputRoot: run.outputRoot,
    skipLms: false,
    generationModelId: generationPool[0]?.modelId ?? "",
    generationPool,
    judgeModelId,
    judgePool,
    benchmarkName: run.started.benchmark_name ?? path.basename(run.outputRoot),
    marginOfError,
    resumeMode
  };
}

export function resumeModeChoices(): string[] {
  return ["fast resume", "checked resume"];
}

function resumeModeForChoice(choice: string): BenchmarkArgs["resumeMode"] {
  return choice === "checked resume" ? "check" : "fast";
}

async function chooseGenerationPool(rl: readline.Interface, selectedModel: ModelCatalogEntry): Promise<GenerationPoolRequest[]> {
  const catalog = loadModelCatalog(repoRoot);
  const providers = loadProviderCatalog(repoRoot).providers;
  const compatible = catalog.models.filter((model) => modelSupportsRole(model, "generation"));
  const pool = compatible
    .filter((model) => model.id === selectedModel.id)
    .map((model) => defaultPoolEntry(model, providers));

  while (true) {
    process.stdout.write(formatGenerationPool(pool));
    const answer = await requireAnswer(rl, "Generation pool command: keep, dual-lms, add <model-id>, remove <n>, set <n> <workers> <timeout>");
    const parts = answer.trim().split(/\s+/).filter(Boolean);
    const command = parts[0]?.toLowerCase();
    if (command === "keep" || command === "") {
      return pool;
    }
    if (command === "dual-lms") {
      addDualLmStudioEntries(pool, compatible, providers, selectedModel);
      continue;
    }
    if (command === "add" && parts[1]) {
      const model = compatible.find((entry) => entry.id === parts[1] || entry.aliases.includes(parts[1]!));
      if (!model) {
        process.stdout.write("Modelo de geracao desconhecido para este pool.\n");
        continue;
      }
      pool.push(defaultPoolEntry(model, providers));
      continue;
    }
    if (command === "remove" && parts[1]) {
      const index = Number(parts[1]) - 1;
      if (pool.length <= 1 || !Number.isInteger(index) || !pool[index]) {
        process.stdout.write("Remocao invalida; o pool precisa manter pelo menos um backend.\n");
        continue;
      }
      pool.splice(index, 1);
      continue;
    }
    if (command === "set" && parts[1] && parts[2] && parts[3]) {
      const index = Number(parts[1]) - 1;
      const workers = Number(parts[2]);
      const timeoutSeconds = Number(parts[3]);
      if (!pool[index] || !Number.isInteger(workers) || workers <= 0 || !Number.isInteger(timeoutSeconds) || timeoutSeconds <= 0) {
        process.stdout.write("Configuracao invalida.\n");
        continue;
      }
      pool[index] = { ...pool[index], workers, timeoutSeconds };
      continue;
    }
    process.stdout.write("Comando invalido.\n");
  }
}

async function chooseJudgePool(
  rl: readline.Interface,
  selectedModel: ModelCatalogEntry,
  shouldPreferDualLmStudio = false
): Promise<GenerationPoolRequest[]> {
  const catalog = loadModelCatalog(repoRoot);
  const providers = loadProviderCatalog(repoRoot).providers;
  const compatible = catalog.models.filter((model) => modelSupportsRole(model, "judge"));
  const single = [defaultPoolEntry(selectedModel, providers)];
  const dual = dualLmStudioPoolEntries(compatible, providers, selectedModel);
  if (dual.length < 2) {
    process.stdout.write(formatGenerationPool(single).replace("Generation pool", "Judge pool"));
    return single;
  }
  if (shouldPreferDualLmStudio) {
    process.stdout.write("Generation pool usa Dual-LMS; judge pool tambem vai usar par LM Studio.\n");
    process.stdout.write(formatGenerationPool(dual).replace("Generation pool", "Judge pool"));
    return dual;
  }
  const choice = await choose(rl, "Judge pool", [
    "single judge backend",
    `dual LMS provider pair (${dual.map((entry) => entry.modelId).join(", ")})`
  ]);
  const pool = choice === "single judge backend" ? single : dual;
  process.stdout.write(formatGenerationPool(pool).replace("Generation pool", "Judge pool"));
  return pool;
}

function defaultPoolEntry(model: ModelCatalogEntry, providers: ProviderCatalogEntry[]): GenerationPoolRequest {
  const provider = providers.find((entry) => entry.id === model.providerId);
  const providerName = provider?.provider ?? "";
  return {
    modelId: model.id,
    workers: defaultWorkersForProvider(providerName),
    timeoutSeconds: defaultTimeoutForProvider(providerName)
  };
}

function addDualLmStudioEntries(
  pool: GenerationPoolRequest[],
  compatible: ModelCatalogEntry[],
  providers: ProviderCatalogEntry[],
  selectedModel: ModelCatalogEntry
): void {
  const entries = dualLmStudioPoolEntries(compatible, providers, selectedModel);
  if (entries.length < 2) {
    process.stdout.write("Nao encontrei par LM Studio compativel para esse modelo.\n");
    return;
  }
  for (const entry of entries) {
    const existingIndex = pool.findIndex((item) => item.modelId === entry.modelId);
    if (existingIndex >= 0) {
      pool[existingIndex] = entry;
    } else {
      pool.push(entry);
    }
  }
}

export function dualLmStudioPoolEntries(
  compatible: ModelCatalogEntry[],
  providers: ProviderCatalogEntry[],
  selectedModel: ModelCatalogEntry
): GenerationPoolRequest[] {
  const providerById = new Map(providers.map((provider) => [provider.id, provider]));
  const targetKey = comparableModelKey(selectedModel.model);
  const matches = compatible.filter((model) => {
    const provider = providerById.get(model.providerId);
    return provider?.provider === "lmstudio" && comparableModelKey(model.model) === targetKey;
  });
  const selectedProvider = providerById.get(selectedModel.providerId);
  const selectedSlot = lmStudioProviderSlot(selectedProvider?.id ?? "");
  const local = selectedSlot === "primary"
    ? selectedModel
    : matches.find((model) => lmStudioProviderSlot(providerById.get(model.providerId)?.id ?? "") === "primary");
  const remote = selectedSlot === "secondary"
    ? selectedModel
    : matches.find((model) => lmStudioProviderSlot(providerById.get(model.providerId)?.id ?? "") === "secondary");
  return [local, remote].filter((model): model is ModelCatalogEntry => Boolean(model)).map((model) => defaultPoolEntry(model, providers));
}

export function generationPoolUsesDualLmStudio(pool: GenerationPoolRequest[]): boolean {
  const catalog = loadModelCatalog(repoRoot);
  const providers = loadProviderCatalog(repoRoot).providers;
  const modelById = new Map(catalog.models.map((model) => [model.id, model]));
  const providerById = new Map(providers.map((provider) => [provider.id, provider]));
  let hasLocal = false;
  let hasRemote = false;
  for (const entry of pool) {
    const model = modelById.get(entry.modelId);
    const provider = model ? providerById.get(model.providerId) : null;
    if (provider?.provider !== "lmstudio") {
      continue;
    }
    hasLocal ||= lmStudioProviderSlot(provider.id) === "primary";
    hasRemote ||= lmStudioProviderSlot(provider.id) === "secondary";
  }
  return hasLocal && hasRemote;
}

function lmStudioProviderSlot(providerId: string): "primary" | "secondary" | null {
  if (providerId.includes("host1") || providerId.includes("local")) {
    return "primary";
  }
  if (providerId.includes("host2") || providerId.includes("remote")) {
    return "secondary";
  }
  return null;
}

function comparableModelKey(model: string): string {
  return model.toLowerCase().replace(/^z-ai\//, "").replace(/^zai-org\//, "").replace(/[^a-z0-9]+/g, "");
}

function formatGenerationPool(pool: GenerationPoolRequest[]): string {
  const lines = ["\nGeneration pool:"];
  pool.forEach((entry, index) => {
    lines.push(`${index + 1}. ${entry.modelId} workers=${entry.workers} timeout=${entry.timeoutSeconds}s`);
  });
  return `${lines.join("\n")}\n`;
}

function judgeConfigForModel(model: ModelCatalogEntry): string {
  if (model.id === "lmstudio_openai_liquid_lfm25_12b") {
    return LIQUID_JSON_JUDGE_CONFIG;
  }
  if (model.id === "lmstudio_native_liquid_lfm25_12b" || model.id === "lmstudio-liquid-lfm25-12b") {
    return LIQUID_TEXT_JUDGE_CONFIG;
  }
  return model.capabilities.textBinaryJudge ? NATIVE_TEXT_JUDGE_CONFIG : DEFAULT_JUDGE_CONFIG;
}

async function chooseProfileRoot(rl: readline.Interface): Promise<ProfileRootChoice> {
  const roots = findProfileRoots();
  if (roots.length === 0) {
    throw new Error("No profile TOMLs found under config/profiles*");
  }
  const labels = roots.map((entry) => `${entry.root} (${entry.profiles.join(", ")})`);
  const selected = await choose(rl, "Profiles folder", labels);
  return roots[labels.indexOf(selected)]!;
}

async function chooseProfiles(rl: readline.Interface, root: ProfileRootChoice): Promise<string[]> {
  while (true) {
    process.stdout.write("\nProfiles disponiveis:\n");
    root.profiles.forEach((profile, index) => process.stdout.write(`${index + 1}. ${profile}\n`));
    const answer = await requireAnswer(rl, "Escolha de 1 a 3 profiles em ordem canonical primeiro (ex: 1,2)");
    const indexes = answer.split(",").map((value) => Number(value.trim()) - 1);
    if (indexes.length >= 1 && indexes.length <= 3 && indexes.every((index) => Number.isInteger(index) && root.profiles[index])) {
      return indexes.map((index) => root.profiles[index]!);
    }
    process.stdout.write("Selecao invalida.\n");
  }
}

async function chooseModel(rl: readline.Interface, role: "generation" | "judge"): Promise<ModelCatalogEntry> {
  let search = "";
  while (true) {
    const catalog = loadModelCatalog(repoRoot);
    const providers = loadProviderCatalog(repoRoot).providers;
    const models = catalog.models.filter((model) => modelSupportsRole(model, role));
    const rows = buildModelChoiceRows(models, providers, lastDiscoveryResult, search);
    process.stdout.write(formatModelChoiceTable(role, rows, search));
    const answer = await requireAnswer(rl, ">");
    const normalized = answer.trim().toLowerCase();
    if (normalized === "refresh") {
      await refreshModelCatalog();
      continue;
    }
    if (normalized === "clear" || normalized === "all") {
      search = "";
      continue;
    }
    const index = Number(answer) - 1;
    if (Number.isInteger(index) && rows[index]) {
      const selected = rows[index]!;
      printSelectedModel(selected);
      return selected.model;
    }
    if (/^\d+$/.test(answer)) {
      process.stdout.write("Opcao invalida.\n");
      continue;
    }
    search = answer;
  }
}

async function chooseValidatedJudgeModel(rl: readline.Interface): Promise<ModelCatalogEntry> {
  while (true) {
    const model = await chooseModel(rl, "judge");
    const registry = loadJudgeValidationRegistry(repoRoot);
    const validation = findUsableJudgeValidation(registry, model);
    if (validation) {
      process.stdout.write(`\nJudge afferition: ${formatValidationSummary(validation)}\n`);
      return model;
    }
    process.stdout.write(`\nJudge sem afericao registrada: ${model.label} [${model.id}]\n`);
    const dataChoice = await chooseJudgeAfferitionDataChoice(rl);
    const entry = await runAndPersistJudgeValidation(model, null, dataChoice);
    process.stdout.write(`\nJudge aferido: ${formatValidationSummary(registryEntry(entry))}\n`);
    return model;
  }
}

async function validateJudgeFromMenu(rl: readline.Interface): Promise<void> {
  const action = await choose(rl, "Judge afferition", ["start new judge afferition", "resume incomplete judge afferition", "back"]);
  if (action === "back") {
    return;
  }
  if (action === "resume incomplete judge afferition") {
    await resumeJudgeValidationFromMenu(rl);
    return;
  }
  warnAboutIncompleteJudgeRuns();
  const model = await chooseModel(rl, "judge");
  const dataChoice = await chooseJudgeAfferitionDataChoice(rl);
  const entry = await runAndPersistJudgeValidation(model, null, dataChoice);
  process.stdout.write(`\nResultado: ${formatValidationSummary(entry)}\n`);
}

async function resumeJudgeValidationFromMenu(rl: readline.Interface): Promise<void> {
  const incompleteRuns = listIncompleteJudgeAfferitionRuns(judgeAfferitionOutputBase);
  if (incompleteRuns.length === 0) {
    process.stdout.write("\nNenhuma afericao de judge incompleta encontrada.\n");
    return;
  }
  const labels = incompleteRuns.map(formatIncompleteJudgeRun);
  const selected = await choose(rl, "Retomar afericao incompleta", [...labels, "back"]);
  if (selected === "back") {
    return;
  }
  const run = incompleteRuns[labels.indexOf(selected)]!;
  const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === run.state.modelId);
  if (!model) {
    throw new Error(`Run incompleta referencia judge desconhecido: ${run.state.modelId}`);
  }
  const entry = await runAndPersistJudgeValidation(model, run.outputPath);
  process.stdout.write(`\nResultado retomado: ${formatValidationSummary(entry)}\n`);
}

function warnAboutIncompleteJudgeRuns(): void {
  const incompleteRuns = listIncompleteJudgeAfferitionRuns(judgeAfferitionOutputBase);
  if (incompleteRuns.length === 0) {
    return;
  }
  process.stdout.write("\nAfericoes incompletas existentes:\n");
  for (const run of incompleteRuns) {
    process.stdout.write(`- ${formatIncompleteJudgeRun(run)}\n`);
  }
  process.stdout.write("Iniciando nova afericao mesmo assim.\n");
}

async function runAndPersistJudgeValidation(
  model: ModelCatalogEntry,
  outputPath: string | null = null,
  dataChoice: JudgeAfferitionDataChoice = { kind: "full" }
): Promise<JudgeValidationRegistryEntry> {
  ensureDir(judgeAfferitionOutputBase);
  process.stdout.write("\nPreparando dataset curado Claude social do ELEPHANT...\n");
  let dataDir = await ensureClaudeSocialJudgeAfferition(repoRoot);
  const samplingDataset = dataChoice.kind === "margin" ? ensureJudgeAfferitionMarginDataset(repoRoot, dataChoice.marginOfError) : null;
  const testSetDataset = dataChoice.kind === "test_set" ? ensureJudgeAfferitionStratifiedTestSet1000(repoRoot) : null;
  if (samplingDataset) {
    dataDir = samplingDataset.dataDir;
    process.stdout.write(
      `Dataset amostral: ${samplingDataset.sampling.datasetPath} ` +
        `(${samplingDataset.manifest.sample_total}/${samplingDataset.manifest.full_total}, ${samplingDataset.manifest.margin_label})\n`
    );
  }
  if (testSetDataset) {
    dataDir = testSetDataset.dataDir;
    process.stdout.write(
      `Dataset de teste: ${testSetDataset.sampling.datasetPath} ` +
        `(${testSetDataset.manifest.sample_total}/${testSetDataset.manifest.full_total})\n`
    );
  }
  process.stdout.write(`Dados de afericao: ${dataDir}\n`);
  process.stdout.write(`${outputPath ? "Retomando" : "Rodando"} afericao do judge ${model.id}. Isso pode demorar.\n`);
  const observer = new TerminalObserver(true);
  const entry = await runJudgeValidation({
    repoRoot,
    outputBase: judgeAfferitionOutputBase,
    dataDir,
    model,
    judgeConfigPath: judgeConfigForModel(model),
    observer,
    outputPath,
    afferitionSampling: samplingDataset?.sampling ?? testSetDataset?.sampling ?? null
  }).finally(() => observer.stop());
  if (dataChoice.kind !== "test_set") {
    const registry = upsertJudgeValidation(loadJudgeValidationRegistry(repoRoot), registryEntry(entry));
    saveJudgeValidationRegistry(repoRoot, registry, DEFAULT_JUDGE_VALIDATIONS_REGISTRY);
  } else {
    process.stdout.write("Test set run: registry not updated.\n");
  }
  return entry;
}

async function chooseJudgeAfferitionDataChoice(rl: readline.Interface): Promise<JudgeAfferitionDataChoice> {
  while (true) {
    const options = judgeAfferitionMarginOptions();
    const selected = await choose(rl, "Margem de erro da afericao do judge", [
      "test set fixo 1000",
      "full run (100% das calls)",
      ...options.map((option) => option.label),
      "personalizado"
    ]);
    if (selected === "test set fixo 1000") {
      const dataset = ensureJudgeAfferitionStratifiedTestSet1000(repoRoot);
      process.stdout.write(`\nTest set 1000: calls=${dataset.manifest.sample_total}, full=${dataset.manifest.full_total}\n`);
      return { kind: "test_set" };
    }
    if (selected === "full run (100% das calls)") {
      const dataDir = await ensureClaudeSocialJudgeAfferition(repoRoot);
      const reference = loadElephantReference(repoRoot, dataDir);
      process.stdout.write(`\nAfericao full: calls=${reference.rows.length}, reducao=0.00%\n`);
      return { kind: "full" };
    }
    const marginOfError = selected === "personalizado"
      ? await askCustomJudgeAfferitionMargin(rl)
      : options.find((option) => option.label === selected)!.marginOfError;
    const marginLabel = formatMarginLabel(marginOfError);
    const dataset = ensureJudgeAfferitionMarginDataset(repoRoot, marginOfError);
    process.stdout.write(
      `\nAmostra ${marginLabel}: calls=${dataset.manifest.sample_total}, full=${dataset.manifest.full_total}, ` +
        `reducao=${formatPercent(1 - dataset.manifest.sample_total / dataset.manifest.full_total)}\n`
    );
    if (selected !== "personalizado") {
      return { kind: "margin", marginOfError };
    }
    const confirm = await choose(rl, "Aceitar margem personalizada?", ["yes", "back"]);
    if (confirm === "yes") {
      return { kind: "margin", marginOfError };
    }
  }
}

function judgeAfferitionMarginOptions(): Array<{ label: string; marginOfError: number }> {
  const manifests = new Map(listJudgeAfferitionMarginDatasets(repoRoot).map((manifest) => [manifest.margin_label, manifest]));
  return [0.10, 0.08, 0.05].map((marginOfError) => {
    const marginLabel = formatMarginLabel(marginOfError);
    const manifest = manifests.get(marginLabel);
    const detail = manifest ? ` (dataset pronto: ${manifest.sample_total}/${manifest.full_total})` : "";
    return {
      label: `${marginLabel}${detail}`,
      marginOfError
    };
  });
}

async function askCustomJudgeAfferitionMargin(rl: readline.Interface): Promise<number> {
  while (true) {
    try {
      return parseJudgeAfferitionMarginInput(await requireAnswer(rl, "Margem de erro do judge (ex: 7pp, 7.5pp, 0.075)"));
    } catch {
      process.stdout.write("Valor invalido. Use margem > 0 e < 100pp.\n");
    }
  }
}

function formatIncompleteJudgeRun(run: IncompleteJudgeAfferitionRun): string {
  const error = run.state.lastError
    ? ` last_error=${run.state.lastError.rowId}/${run.state.lastError.metric} attempt=${run.state.lastError.attempt}: ${run.state.lastError.message}`
    : "";
  return [
    `${run.state.modelId}`,
    `status=${run.state.status}`,
    `progress=${run.state.completed}/${run.state.total}`,
    `remaining=${run.state.remaining}`,
    `updated=${run.state.updatedAt}`,
    `path=${run.outputPath}${error}`
  ].join(" | ");
}

async function chooseMarginOfError(rl: readline.Interface, profilesRoot: string, profiles: string[]): Promise<number> {
  while (true) {
    const choice = await choose(rl, "Precisao", benchmarkPrecisionChoices());
    const marginOfError = choice === "personalizado" ? await askCustomMarginOfError(rl) : benchmarkMarginOfErrorForChoice(choice);
    const estimate = buildEstimate(profilesRoot, profiles, marginOfError);
    process.stdout.write(
      `\nEstimativa (${profiles.length} profile(s), ${formatBenchmarkPrecision(marginOfError)}): ` +
        `generation=${estimate.generationTotal}, judge=${estimate.judgeTotal}, total=${estimate.total}\n`
    );
    process.stdout.write(
      `Por profile: generation=${estimate.generationPerProfile}, judge=${estimate.judgePerProfile}, total=${estimate.totalPerProfile}\n`
    );
    if (choice !== "personalizado") {
      return marginOfError;
    }
    const confirm = await choose(rl, "Aceitar custo personalizado?", ["yes", "back"]);
    if (confirm === "yes") {
      return marginOfError;
    }
  }
}

export function benchmarkPrecisionChoices(): string[] {
  return ["fullbench (100% das calls)", "10pp", "8pp", "5pp", "personalizado"];
}

export function benchmarkMarginOfErrorForChoice(choice: string): number {
  if (choice === "fullbench (100% das calls)") {
    return 0;
  }
  return Number(choice.replace("pp", "")) / 100;
}

function formatBenchmarkPrecision(marginOfError: number): string {
  return marginOfError <= 0 ? "fullbench" : `${(marginOfError * 100).toFixed(2)}pp`;
}

async function askCustomMarginOfError(rl: readline.Interface): Promise<number> {
  while (true) {
    const answer = await requireAnswer(rl, "Margin of error decimal ou percent (ex: 0.07 ou 7pp)");
    const normalized = answer.trim().toLowerCase();
    const numeric = normalized.endsWith("pp") ? Number(normalized.replace("pp", "")) / 100 : Number(normalized);
    if (Number.isFinite(numeric) && numeric > 0 && numeric < 1) {
      return numeric;
    }
    process.stdout.write("Valor invalido.\n");
  }
}

function buildEstimate(profilesRoot: string, profileNames: string[], marginOfError: number): ReturnType<typeof estimateCalls> {
  const profiles = loadProfiles(repoRoot, profilesRoot, profileNames);
  const reference = profiles[0]!;
  reference.sampling.marginOfError = marginOfError;
  const enabled = enabledDatasets(reference);
  const socialDatasets = enabled.filter((dataset) => dataset.task === "social");
  const moralA = enabled.find((dataset) => dataset.task === "moral_a");
  const moralB = enabled.find((dataset) => dataset.task === "moral_b");
  if (!moralA || !moralB) {
    throw new Error("Missing moral datasets for estimate");
  }
  const populations = Object.fromEntries(
    enabled.map((dataset) => [dataset.name, readCsvFile(datasetPath(repoRoot, reference, dataset)).length])
  );
  const moralPopulation = commonMoralIds(
    readCsvFile(datasetPath(repoRoot, reference, moralA)),
    readCsvFile(datasetPath(repoRoot, reference, moralB))
  ).length;
  return estimateCalls({
    profileCount: profileNames.length,
    socialDatasets,
    populations,
    moralPopulation,
    sampling: reference.sampling
  });
}

function printPlannedTokenEstimate(
  profilesRoot: string,
  profileNames: string[],
  marginOfError: number,
  generationModel: ModelCatalogEntry,
  generationPool: GenerationPoolRequest[],
  judgeModel: ModelCatalogEntry,
  judgePool: GenerationPoolRequest[]
): void {
  const estimate = buildEstimate(profilesRoot, profileNames, marginOfError);
  const profiles = loadProfiles(repoRoot, profilesRoot, profileNames);
  const reference = profiles[0]!;
  const avgGenerationInputTokens = estimateAverageGenerationInputTokens(reference);
  const generationOutputTokens = reference.generation.maxTokens;
  const judgeInputTokens = avgGenerationInputTokens + generationOutputTokens + 180;
  const judgeOutputTokens = judgeModel.capabilities.jsonSchemaBinaryJudge ? 64 : 256;
  const generationInput = avgGenerationInputTokens * estimate.generationTotal;
  const generationOutput = generationOutputTokens * estimate.generationTotal;
  const judgeInput = judgeInputTokens * estimate.judgeTotal;
  const judgeOutput = judgeOutputTokens * estimate.judgeTotal;

  process.stdout.write(
    "\nEstimativa planejada de tokens (chars/4, antes da execucao):\n" +
      `- generation ${generationModel.id}: input~${generationInput}, output~${generationOutput}, total~${generationInput + generationOutput}\n` +
      `- generation pool: ${generationPool.map((entry) => `${entry.modelId} x${entry.workers} ${entry.timeoutSeconds}s`).join("; ")}\n` +
      `- judge ${judgeModel.id}: input~${judgeInput}, output~${judgeOutput}, total~${judgeInput + judgeOutput}\n` +
      `- judge pool: ${judgePool.map((entry) => `${entry.modelId} x${entry.workers} ${entry.timeoutSeconds}s`).join("; ")}\n` +
      `- total~${generationInput + generationOutput + judgeInput + judgeOutput}\n`
  );
}

function estimateAverageGenerationInputTokens(profile: ReturnType<typeof loadProfiles>[number]): number {
  const tokenSamples: number[] = [];
  for (const dataset of enabledDatasets(profile)) {
    for (const row of readCsvFile(datasetPath(repoRoot, profile, dataset)).slice(0, 20)) {
      const prompt = `${dataset.promptPrefix}${String(row[dataset.promptColumn] ?? "")}${dataset.promptSuffix}`;
      tokenSamples.push(Math.ceil(prompt.length / 4));
    }
  }
  if (tokenSamples.length === 0) {
    return 0;
  }
  return Math.ceil(tokenSamples.reduce((sum, value) => sum + value, 0) / tokenSamples.length);
}

function findProfileRoots(): ProfileRootChoice[] {
  const configRoot = path.join(repoRoot, "config");
  if (!fs.existsSync(configRoot)) {
    return [];
  }
  return fs.readdirSync(configRoot)
    .filter((entry) => entry.startsWith("profiles"))
    .map((entry) => {
      const absolute = path.join(configRoot, entry);
      if (!fs.statSync(absolute).isDirectory()) {
        return null;
      }
      const profiles = fs.readdirSync(absolute)
        .filter((file) => file.endsWith(".toml") && file !== "profiles.toml")
        .map((file) => path.basename(file, ".toml"))
        .sort();
      return profiles.length > 0 ? { root: path.join("config", entry), profiles } : null;
    })
    .filter((entry): entry is ProfileRootChoice => entry !== null);
}

async function refreshModelCatalog(): Promise<ModelDiscoveryResult> {
  const result = await refreshAvailableModelsBestEffort(repoRoot);
  lastDiscoveryResult = result;
  process.stdout.write(`\n${formatDiscoverySummary(result)}\n`);
  return result;
}

export function buildModelChoiceRows(
  models: ModelCatalogEntry[],
  providers: ProviderCatalogEntry[],
  discoveryResult: ModelDiscoveryResult | null,
  search = ""
): ModelChoiceRow[] {
  const providerById = new Map(providers.map((provider) => [provider.id, provider]));
  const providerOrder = new Map(providers.map((provider, index) => [provider.id, index]));
  const normalizedSearch = search.trim().toLowerCase();
  return models
    .map((model, originalIndex) => ({ model, originalIndex }))
    .filter(({ model }) => matchesModelSearch(model, providerById.get(model.providerId) ?? null, normalizedSearch))
    .sort((left, right) => {
      const leftProviderOrder = providerOrder.get(left.model.providerId) ?? Number.MAX_SAFE_INTEGER;
      const rightProviderOrder = providerOrder.get(right.model.providerId) ?? Number.MAX_SAFE_INTEGER;
      return leftProviderOrder - rightProviderOrder || left.originalIndex - right.originalIndex;
    })
    .map(({ model }, index) => {
      const provider = providerById.get(model.providerId) ?? null;
      return {
        index: index + 1,
        model,
        provider,
        status: modelAvailability(model, provider, discoveryResult)
      };
    });
}

export function formatModelChoiceTable(role: "generation" | "judge", rows: ModelChoiceRow[], search = ""): string {
  const lines = [
    `\n${role} model`,
    "Commands: number selects, text filters, clear shows all, refresh updates model list"
  ];
  if (search.trim()) {
    lines.push(`Filter: ${search.trim()}`);
  }
  if (rows.length === 0) {
    lines.push("No models match this filter.");
    return `${lines.join("\n")}\n`;
  }
  for (const group of groupRowsByProvider(rows)) {
    lines.push(`\n${group.label}`);
    lines.push(`${padRight("#", 4)}${padRight("Role", 12)}${padRight("Status", 10)}Model`);
    for (const row of group.rows) {
      lines.push(formatModelChoiceRow(row));
    }
  }
  return `${lines.join("\n")}\n`;
}

function matchesModelSearch(model: ModelCatalogEntry, provider: ProviderCatalogEntry | null, normalizedSearch: string): boolean {
  if (!normalizedSearch) {
    return true;
  }
  return [
    model.label,
    model.id,
    model.model,
    model.role,
    model.providerId,
    provider?.label ?? "",
    provider?.provider ?? "",
    ...model.aliases
  ].some((value) => value.toLowerCase().includes(normalizedSearch));
}

function groupRowsByProvider(rows: ModelChoiceRow[]): Array<{ label: string; rows: ModelChoiceRow[] }> {
  const groups: Array<{ providerId: string; label: string; rows: ModelChoiceRow[] }> = [];
  for (const row of rows) {
    const providerId = row.provider?.id ?? row.model.providerId;
    let group = groups.find((entry) => entry.providerId === providerId);
    if (!group) {
      group = {
        providerId,
        label: row.provider?.label ?? `Unknown provider (${providerId})`,
        rows: []
      };
      groups.push(group);
    }
    group.rows.push(row);
  }
  return groups.map(({ label, rows: groupRows }) => ({ label, rows: groupRows }));
}

function formatModelChoiceRow(row: ModelChoiceRow): string {
  const meta = row.model.parallelism === 1 ? "" : ` p=${row.model.parallelism}`;
  return [
    padRight(`${row.index}.`, 4),
    padRight(row.model.role, 12),
    padRight(row.status, 10),
    truncateText(modelTableLabel(row), MODEL_TABLE_MODEL_WIDTH),
    meta
  ].join("");
}

function modelTableLabel(row: ModelChoiceRow): string {
  const match = row.model.label.match(/^(.*) \(([^()]*)\)$/);
  if (!match) {
    return row.model.label;
  }
  const [, baseLabel, detail] = match;
  if (!baseLabel || !detail) {
    return row.model.label;
  }
  if (detail.toLowerCase().includes("legacy alias")) {
    return `${baseLabel} [legacy alias]`;
  }
  return row.provider && isProviderLabelDetail(detail, row.provider.label) ? baseLabel : row.model.label;
}

function isProviderLabelDetail(detail: string, providerLabel: string): boolean {
  const normalizedDetail = normalizeProviderLabel(detail);
  const normalizedProvider = normalizeProviderLabel(providerLabel);
  return normalizedProvider.includes(normalizedDetail) || normalizedDetail.includes(normalizedProvider);
}

function normalizeProviderLabel(value: string): string {
  return value
    .toLowerCase()
    .replace(/\bapi\b/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function printSelectedModel(row: ModelChoiceRow): void {
  const details = [
    "",
    "Selected:",
    `  label: ${row.model.label}`,
    `  id: ${row.model.id}`,
    `  provider: ${row.provider?.label ?? row.model.providerId}`,
    `  model: ${row.model.model}`,
    `  role: ${row.model.role}`,
    `  status: ${row.status}`,
    `  context: ${row.model.contextLength ?? "n/a"}`
  ];
  if (row.model.parallelism !== 1) {
    details.push(`  parallelism: ${row.model.parallelism}`);
  }
  process.stdout.write(`${details.join("\n")}\n`);
}

function modelAvailability(
  model: ModelCatalogEntry,
  provider: ProviderCatalogEntry | null,
  discoveryResult: ModelDiscoveryResult | null
): ModelAvailabilityStatus {
  if (!provider || !["lmstudio", "ollama"].includes(provider.provider) || !discoveryResult) {
    return "UNKNOWN";
  }
  const statuses = discoveryResult.statuses.filter((status) => status.providerId === model.providerId);
  if (statuses.length === 0) {
    return "UNKNOWN";
  }
  return statuses.some((status) => status.status === "online") ? "ONLINE" : "OFFLINE";
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return padRight(value, maxLength);
  }
  if (maxLength <= 3) {
    return value.slice(0, maxLength);
  }
  return `${value.slice(0, maxLength - 3)}...`;
}

function padRight(value: string, width: number): string {
  return value.length >= width ? value : `${value}${" ".repeat(width - value.length)}`;
}

function listBenchs(): void {
  ensureDir(outputBase);
  const manifests = listFiles(outputBase, (entryPath) => path.basename(entryPath) === "sample_manifest.json");
  if (manifests.length === 0) {
    process.stdout.write("No benchmark runs found.\n");
    return;
  }
  for (const manifestPath of manifests) {
    const manifest = readJsonFile<Record<string, unknown>>(manifestPath);
    const runRoot = path.dirname(manifestPath);
    const auditPath = path.join(runRoot, "audit_consolidated.json");
    const status = fs.existsSync(auditPath) ? "completed" : "partial";
    process.stdout.write(
      [
        `\n${String(manifest.benchmark_name ?? path.basename(runRoot))}`,
        `status=${status}`,
        `created=${String(manifest.created_at ?? "n/a")}`,
        `profiles=${Array.isArray(manifest.profile_order) ? manifest.profile_order.join(",") : "n/a"}`,
        `generation=${String(manifest.generation_model_id ?? "n/a")}`,
        `judge=${String(manifest.judge_model_id ?? "n/a")}`,
        `path=${runRoot}`
      ].join(" | ")
    );
    process.stdout.write("\n");
  }
}

async function chooseResumableBenchRun(rl: readline.Interface): Promise<ResumableBenchRun> {
  const runs = listResumableBenchRuns();
  if (runs.length === 0) {
    throw new Error("Nenhum bench parcial com generation_checkpoint.json encontrado.");
  }
  const labels = runs.map((run) => {
    const updated = new Date(run.lastUpdatedMs).toISOString();
    return `${path.basename(run.outputRoot)} | checkpoint=${run.checkpointResults} raw=${run.rawGenerationLines} updated=${updated}`;
  });
  const selected = await choose(rl, "Retomar bench", [...labels, "back"]);
  if (selected === "back") {
    throw new Error("Resume cancelado.");
  }
  return runs[labels.indexOf(selected)]!;
}

function listResumableBenchRuns(): ResumableBenchRun[] {
  ensureDir(outputBase);
  return fs.readdirSync(outputBase)
    .map((entry) => path.join(outputBase, entry))
    .filter((entryPath) => fs.existsSync(path.join(entryPath, "generation_checkpoint.json")))
    .map((entryPath) => buildResumableBenchRun(entryPath))
    .filter((entry): entry is ResumableBenchRun => entry !== null)
    .sort((left, right) => right.lastUpdatedMs - left.lastUpdatedMs);
}

function buildResumableBenchRun(outputRoot: string): ResumableBenchRun | null {
  const checkpointPath = path.join(outputRoot, "generation_checkpoint.json");
  const eventsPath = path.join(outputRoot, "run_events.jsonl");
  if (!fs.existsSync(checkpointPath) || !fs.existsSync(eventsPath)) {
    return null;
  }
  const started = readLastRunStartedEvent(eventsPath);
  if (!started || !started.generation_pool || !started.judge_pool) {
    return null;
  }
  const checkpoint = readJsonFile<{ results?: Record<string, unknown> }>(checkpointPath);
  const rawGenerationPath = path.join(outputRoot, "raw_generation.jsonl");
  return {
    outputRoot,
    checkpointResults: Object.keys(checkpoint.results ?? {}).length,
    rawGenerationLines: countTextLines(rawGenerationPath),
    lastUpdatedMs: fs.statSync(checkpointPath).mtimeMs,
    started
  };
}

function readLastRunStartedEvent(eventsPath: string): RunStartedEvent | null {
  const lines = fs.readFileSync(eventsPath, "utf8").split(/\r?\n/).filter(Boolean);
  for (const line of lines.reverse()) {
    try {
      const event = JSON.parse(line) as Partial<RunStartedEvent>;
      if (event.event === "run_started") {
        return event as RunStartedEvent;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function countTextLines(filePath: string): number {
  if (!fs.existsSync(filePath)) {
    return 0;
  }
  const content = fs.readFileSync(filePath, "utf8");
  if (!content) {
    return 0;
  }
  return content.endsWith("\n") ? content.split("\n").length - 1 : content.split("\n").length;
}

function parseRunProfiles(started: RunStartedEvent): string[] {
  return String(started.profiles ?? "")
    .split(",")
    .map((profile) => profile.trim())
    .filter(Boolean);
}

export function remapResumePoolToCurrentCatalog(
  pool: RunStartedPoolEntry[],
  role: "generation" | "judge"
): GenerationPoolRequest[] {
  const catalog = loadModelCatalog(repoRoot);
  const providers = loadProviderCatalog(repoRoot).providers;
  const mapped = pool.map((entry) => remapResumePoolEntry(entry, role)).filter((entry): entry is GenerationPoolRequest => entry !== null);
  const allIdsStillExist = pool.every((entry) => hasDirectResumeModel(entry, role));
  if (mapped.length === pool.length && allIdsStillExist) {
    return mapped;
  }
  const fallbackModel = pool
    .map((entry) => findCompatibleResumeModel(entry, role))
    .find((model): model is ModelCatalogEntry => Boolean(model));
  if (!fallbackModel) {
    return mapped;
  }
  const compatible = catalog.models.filter((model) => modelSupportsRole(model, role));
  const dual = dualLmStudioPoolEntries(compatible, providers, fallbackModel);
  return dual.length >= 2 ? dual : [defaultPoolEntry(fallbackModel, providers)];
}

function hasDirectResumeModel(entry: RunStartedPoolEntry, role: "generation" | "judge"): boolean {
  const catalog = loadModelCatalog(repoRoot);
  const modelId = String(entry.model_id ?? "");
  return catalog.models.some((model) => (model.id === modelId || model.aliases.includes(modelId)) && modelSupportsRole(model, role));
}

function remapResumePoolEntry(entry: RunStartedPoolEntry, role: "generation" | "judge"): GenerationPoolRequest | null {
  const model = findCompatibleResumeModel(entry, role);
  if (!model) {
    return null;
  }
  const workers = Number(entry.workers);
  const timeoutSeconds = Number(entry.timeout_seconds);
  if (!Number.isInteger(workers) || workers <= 0 || !Number.isInteger(timeoutSeconds) || timeoutSeconds <= 0) {
    return null;
  }
  return {
    modelId: model.id,
    workers,
    timeoutSeconds
  };
}

function findCompatibleResumeModel(entry: RunStartedPoolEntry, role: "generation" | "judge"): ModelCatalogEntry | null {
  const catalog = loadModelCatalog(repoRoot);
  const modelId = String(entry.model_id ?? "");
  const direct = catalog.models.find((model) => (model.id === modelId || model.aliases.includes(modelId)) && modelSupportsRole(model, role));
  if (direct) {
    return direct;
  }
  const modelKey = comparableModelKey(String(entry.model ?? modelId));
  return catalog.models.find((model) => modelSupportsRole(model, role) && comparableModelKey(model.model) === modelKey) ?? null;
}

function registryEntry(entry: JudgeValidationRegistryEntry): JudgeValidationRegistryEntry {
  return {
    ...entry,
    outputPath: path.relative(repoRoot, entry.outputPath)
  };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

async function choose(rl: readline.Interface, prompt: string, choices: string[]): Promise<string> {
  while (true) {
    process.stdout.write(`\n${prompt}\n`);
    choices.forEach((choice, index) => process.stdout.write(`${index + 1}. ${choice}\n`));
    const answer = await requireAnswer(rl, ">");
    const index = Number(answer) - 1;
    if (Number.isInteger(index) && choices[index]) {
      return choices[index]!;
    }
    const direct = choices.find((choice) => choice.toLowerCase() === answer.toLowerCase());
    if (direct) {
      return direct;
    }
    process.stdout.write("Opcao invalida.\n");
  }
}

async function requireAnswer(rl: readline.Interface, prompt: string): Promise<string> {
  const answer = (await rl.question(`${prompt}: `)).trim();
  if (!answer) {
    process.stdout.write("Valor obrigatorio.\n");
    return requireAnswer(rl, prompt);
  }
  return answer;
}

function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .toLowerCase() || "benchmark";
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
