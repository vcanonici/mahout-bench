import fs from "node:fs";
import path from "node:path";

import toml from "toml";

import {
  DEFAULT_JUDGE_CONFIG,
  DEFAULT_PROFILES_ROOT,
  API_MODE_LMSTUDIO_NATIVE_CHAT,
  API_MODE_OPENAI_CHAT_COMPLETIONS,
  DEFAULT_CONFIDENCE,
  DEFAULT_MARGIN_OF_ERROR,
  DEFAULT_MODELS_CATALOG,
  DEFAULT_PROVIDERS_CATALOG,
  JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL,
  JUDGE_OUTPUT_TEXT_BINARY_LABEL,
  MORAL_A_TASK,
  MORAL_B_TASK,
  PROVIDER_LMSTUDIO,
  PROVIDER_MINIMAX,
  PROVIDER_OLLAMA,
  PROVIDER_OPENROUTER,
  PROVIDER_OPENAI_COMPATIBLE,
  PROFILE_ORDER,
  SOCIAL_TASK,
  type CsvRecord,
  type DatasetConfig,
  type InferenceConfig,
  type JudgeConfig,
  type ModelCatalog,
  type ModelCatalogEntry,
  type ProfileConfig,
  type ProviderCatalog,
  type ProviderCatalogEntry
} from "../contracts/autobench.js";
import { readCsvFile, readJsonFile, readTextFile } from "../io/filesystem.js";
import { resolveDataRoot, resolveDataRootForRepo } from "../runtime/paths.js";

type TomlTable = Record<string, unknown>;

export function loadProfiles(repoRoot: string, profilesRoot = DEFAULT_PROFILES_ROOT, profileNames: string[] = []): ProfileConfig[] {
  const orderedNames = profileNames.length > 0 ? profileNames : PROFILE_ORDER.map((name) => String(name));
  return orderedNames.map((profileName) => {
    const filePath = path.join(repoRoot, profilesRoot, `${profileName}.toml`);
    const profile = parseProfile(filePath);
    if (profile.name !== profileName) {
      throw new Error(`Profile name mismatch in ${filePath}: expected ${profileName}, got ${profile.name}`);
    }
    return profile;
  });
}

export function parseProfile(profilePath: string): ProfileConfig {
  const data = parseTomlTable(profilePath);
  const profileSection = table(data, "profile");
  const generationSection = table(data, "generation");
  const pathsSection = table(data, "paths");
  const samplingSection = table(data, "sampling");
  const promptsSection = table(data, "prompts");
  const datasetsSection = table(data, "datasets");

  const profileName = stringValue(profileSection.name, path.basename(profilePath, ".toml"));
  const benchmarkModel = stringValue(profileSection.benchmark_model, "");

  const generation: InferenceConfig = {
    provider: providerValue(generationSection.provider, "generation.provider", generationSection.api_mode),
    apiBaseUrl: stringValue(generationSection.api_base_url, "http://127.0.0.1:1234/v1"),
    apiMode: apiModeValue(generationSection.api_mode, "generation.api_mode"),
    apiKey: apiKeyValue(generationSection, profilePath, "lm-studio"),
    apiKeyFile: stringValue(generationSection.api_key_file, ""),
    model: benchmarkModel,
    temperature: floatValue(generationSection.temperature, "generation.temperature", 0.7),
    topP: floatValue(generationSection.top_p, "generation.top_p", 1),
    maxTokens: intValue(generationSection.max_tokens, "generation.max_tokens", 4096),
    contextLength: intValue(generationSection.context_length, "generation.context_length", 16384),
    parallelism: positiveIntValue(generationSection.parallelism, "generation.parallelism", 1),
    thinkingEnabled: boolValue(generationSection.thinking_enabled, "generation.thinking_enabled", true),
    reasoningEffort: stringValue(generationSection.reasoning_effort, "low"),
    includeReasoningParameter: boolValue(generationSection.include_reasoning_parameter, "generation.include_reasoning_parameter", true),
    systemPrompt: stringValue(generationSection.system_prompt, ""),
    quotaLabel: stringValue(generationSection.quota_label, ""),
    quotaMaxRequests: optionalIntValue(generationSection.quota_max_requests, "generation.quota_max_requests"),
    quotaWindowSeconds: optionalIntValue(generationSection.quota_window_seconds, "generation.quota_window_seconds")
  };

  const globalPrefix = stringValue(promptsSection.prefix, "");
  const globalSuffix = stringValue(promptsSection.suffix, "");
  const datasets: Record<string, DatasetConfig> = {};
  for (const [name, value] of Object.entries(datasetsSection)) {
    if (!isRecord(value)) {
      throw new Error(`datasets.${name} must be a table`);
    }
    datasets[name] = {
      name,
      enabled: boolValue(value.enabled, `datasets.${name}.enabled`, true),
      file: stringValue(value.file, ""),
      promptColumn: stringValue(value.prompt_column, "prompt"),
      task: stringValue(value.task, SOCIAL_TASK),
      aitaBinary: boolValue(value.aita_binary, `datasets.${name}.aita_binary`, false),
      baseline: value.baseline === undefined || value.baseline === null ? null : floatValue(value.baseline, `datasets.${name}.baseline`, 0),
      promptPrefix: stringValue(value.prefix, globalPrefix),
      promptSuffix: stringValue(value.suffix, globalSuffix)
    };
  }

  return {
    name: profileName,
    description: stringValue(profileSection.description, ""),
    sourcePath: profilePath,
    generation,
    sampling: {
      confidence: floatValue(samplingSection.confidence, "sampling.confidence", DEFAULT_CONFIDENCE),
      marginOfError: floatValue(samplingSection.margin_of_error, "sampling.margin_of_error", DEFAULT_MARGIN_OF_ERROR)
    },
    datasetsDir: stringValue(pathsSection.datasets_dir, "datasets"),
    seed: intValue(samplingSection.seed, "sampling.seed", 42),
    datasets
  };
}

export function parseProfileOrder(repoRoot: string, profilesRoot: string): { order: string[]; canonical: string } {
  const profileOrderPath = path.join(repoRoot, profilesRoot, "profiles.toml");
  try {
    const data = parseTomlTable(profileOrderPath);
    const order = arrayOfStrings(data.order, "profiles.order");
    const canonical = stringValue(data.canonical, order[0] ?? "");
    if (order.length === 0) {
      throw new Error(`profiles.toml must define a non-empty order in ${profileOrderPath}`);
    }
    if (!order.includes(canonical)) {
      throw new Error(`profiles.toml canonical must be present in order: ${canonical}`);
    }
    return { order, canonical };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
    return { order: PROFILE_ORDER.map((name) => String(name)), canonical: String(PROFILE_ORDER[0]) };
  }
}

export function loadProviderCatalog(repoRoot: string, catalogPath = DEFAULT_PROVIDERS_CATALOG): ProviderCatalog {
  const catalog = readJsonFile<ProviderCatalog>(path.join(repoRoot, catalogPath));
  const localCatalogPath = path.join(resolveDataRoot(), "config", "providers.local.json");
  if (fs.existsSync(localCatalogPath)) {
    const localCatalog = readJsonFile<ProviderCatalog>(localCatalogPath);
    catalog.providers = mergeProviderEntries(catalog.providers, localCatalog.providers);
  }
  validateProviderCatalog(catalog, catalogPath);
  return catalog;
}

export function loadModelCatalog(repoRoot: string, catalogPath = DEFAULT_MODELS_CATALOG): ModelCatalog {
  const catalog = readJsonFile<ModelCatalog>(path.join(repoRoot, catalogPath));
  const localCatalogPath = path.join(resolveDataRoot(), "config", "models.local.json");
  if (fs.existsSync(localCatalogPath)) {
    const localCatalog = readJsonFile<ModelCatalog>(localCatalogPath);
    catalog.models.push(...localCatalog.models.filter((entry) => !catalog.models.some((model) => model.id === entry.id)));
  }
  validateModelCatalog(catalog, catalogPath);
  return catalog;
}

export function resolveInferenceFromModelCatalog(args: {
  repoRoot: string;
  base: InferenceConfig;
  modelId: string;
}): InferenceConfig {
  if (!args.modelId) {
    return args.base;
  }
  const providers = loadProviderCatalog(args.repoRoot);
  const models = loadModelCatalog(args.repoRoot);
  const model = models.models.find((entry) => entry.id === args.modelId || entry.aliases.includes(args.modelId));
  if (!model) {
    throw new Error(`Unknown model catalog id: ${args.modelId}`);
  }
  const provider = providers.providers.find((entry) => entry.id === model.providerId);
  if (!provider) {
    throw new Error(`Model ${model.id} references missing provider ${model.providerId}`);
  }
  const apiKeyFile = model.apiKeyFile ?? provider.apiKeyFile;
  return {
    ...args.base,
    provider: provider.provider,
    apiBaseUrl: model.apiBaseUrl ?? provider.apiBaseUrl,
    apiMode: model.apiMode ?? provider.apiMode,
    apiKey: catalogApiKeyValue(args.repoRoot, model.apiKey ?? provider.apiKey, apiKeyFile, model.id),
    apiKeyFile,
    model: model.model,
    parallelism: model.parallelism,
    contextLength: model.contextLength ?? args.base.contextLength,
    includeReasoningParameter: model.capabilities.reasoningParameter === false ? false : args.base.includeReasoningParameter
  };
}

export function modelSupportsRole(model: ModelCatalogEntry, role: "generation" | "judge"): boolean {
  return model.role === "both" || model.role === role;
}

export function parseJudge(repoRoot: string, judgeConfigPath = DEFAULT_JUDGE_CONFIG): JudgeConfig {
  const filePath = path.join(repoRoot, judgeConfigPath);
  const data = parseTomlTable(filePath);
  const judgeSection = table(data, "judge");
  const promptsSection = table(data, "judge_prompts");
  const model = stringValue(judgeSection.model, "");
  if (!model) {
    throw new Error(`Missing judge.model in ${filePath}`);
  }

  const outputMode = stringValue(judgeSection.output_mode, JUDGE_OUTPUT_TEXT_BINARY_LABEL);
  if (outputMode !== JUDGE_OUTPUT_TEXT_BINARY_LABEL && outputMode !== JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL) {
    throw new Error(`Unsupported judge.output_mode in ${filePath}: ${outputMode}`);
  }

  return {
    inference: {
      provider: providerValue(judgeSection.provider, "judge.provider", judgeSection.api_mode),
      apiBaseUrl: stringValue(judgeSection.api_base_url, "http://127.0.0.1:1234/v1"),
      apiMode: apiModeValue(judgeSection.api_mode, "judge.api_mode"),
      apiKey: apiKeyValue(judgeSection, filePath, "lm-studio"),
      apiKeyFile: stringValue(judgeSection.api_key_file, ""),
      model,
      temperature: floatValue(judgeSection.temperature, "judge.temperature", 0.7),
      topP: floatValue(judgeSection.top_p, "judge.top_p", 1),
      maxTokens: intValue(judgeSection.max_tokens, "judge.max_tokens", 4096),
      contextLength: intValue(judgeSection.context_length, "judge.context_length", 16384),
      parallelism: positiveIntValue(judgeSection.parallelism, "judge.parallelism", 1),
      thinkingEnabled: boolValue(judgeSection.thinking_enabled, "judge.thinking_enabled", false),
      reasoningEffort: stringValue(judgeSection.reasoning_effort, "low"),
      includeReasoningParameter: boolValue(judgeSection.include_reasoning_parameter, "judge.include_reasoning_parameter", true),
      systemPrompt: stringValue(judgeSection.system_prompt, ""),
      quotaLabel: stringValue(judgeSection.quota_label, ""),
      quotaMaxRequests: optionalIntValue(judgeSection.quota_max_requests, "judge.quota_max_requests"),
      quotaWindowSeconds: optionalIntValue(judgeSection.quota_window_seconds, "judge.quota_window_seconds")
    },
    promptPrefix: stringValue(promptsSection.prefix, ""),
    promptSuffix: stringValue(promptsSection.suffix, ""),
    outputMode
  };
}

export function enabledDatasets(profile: ProfileConfig): DatasetConfig[] {
  return Object.values(profile.datasets).filter((dataset) => dataset.enabled);
}

export function datasetPath(repoRoot: string, profile: ProfileConfig, dataset: DatasetConfig): string {
  return path.join(resolveDataRootForRepo(repoRoot), profile.datasetsDir, dataset.file);
}

export function validateContract(repoRoot: string, profiles: ProfileConfig[]): void {
  if (profiles.length === 0) {
    throw new Error("No profiles loaded");
  }

  const reference = profiles[0]!;
  const referenceModel = reference.generation.model;
  if (!referenceModel) {
    throw new Error("Missing benchmark model; set profile.benchmark_model or pass --generation-model-id");
  }
  const referenceDatasets = enabledDatasets(reference).map((dataset) => [
    dataset.name,
    dataset.file,
    dataset.promptColumn,
    dataset.task,
    dataset.aitaBinary
  ]);

  for (const profile of profiles) {
    if (referenceModel && profile.generation.model && profile.generation.model !== referenceModel) {
      throw new Error("All Felix profiles must use the same benchmark model");
    }
    if (profile.datasetsDir !== reference.datasetsDir) {
      throw new Error("All Felix profiles must use the same datasets_dir");
    }
    const currentDatasets = enabledDatasets(profile).map((dataset) => [
      dataset.name,
      dataset.file,
      dataset.promptColumn,
      dataset.task,
      dataset.aitaBinary
    ]);
    if (JSON.stringify(currentDatasets) !== JSON.stringify(referenceDatasets)) {
      throw new Error(`Dataset contract mismatch for profile ${profile.name}`);
    }
  }

  for (const dataset of enabledDatasets(reference)) {
    const rows = readCsvFile(datasetPath(repoRoot, reference, dataset)).slice(0, 5);
    requireColumn(rows, dataset.promptColumn, dataset.file);
  }

  const moralA = enabledDatasets(reference).filter((dataset) => dataset.task === MORAL_A_TASK);
  const moralB = enabledDatasets(reference).filter((dataset) => dataset.task === MORAL_B_TASK);
  if (moralA.length !== 1 || moralB.length !== 1) {
    throw new Error("Expected exactly one moral_a and one moral_b dataset");
  }

  for (const dataset of [moralA[0]!, moralB[0]!]) {
    const rows = readCsvFile(datasetPath(repoRoot, reference, dataset)).slice(0, 5);
    requireColumn(rows, "id", dataset.file);
  }
}

function parseTomlTable(filePath: string): TomlTable {
  const parsed = toml.parse(readTextFile(filePath));
  if (!isRecord(parsed)) {
    throw new Error(`Invalid TOML root in ${filePath}`);
  }
  return parsed;
}

function table(root: TomlTable, key: string): TomlTable {
  const value = root[key];
  if (!isRecord(value)) {
    return {};
  }
  return value;
}

function stringValue(value: unknown, fallback: string): string {
  if (value === undefined || value === null) {
    return fallback;
  }
  return String(value).trim();
}

function arrayOfStrings(value: unknown, fieldName: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} must be an array`);
  }
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function validateProviderCatalog(catalog: ProviderCatalog, catalogPath: string): void {
  if (!Array.isArray(catalog.providers) || catalog.providers.length === 0) {
    throw new Error(`Provider catalog must contain providers: ${catalogPath}`);
  }
  const ids = new Set<string>();
  for (const provider of catalog.providers) {
    if (!provider.id || ids.has(provider.id)) {
      throw new Error(`Provider catalog has missing or duplicate provider id: ${provider.id}`);
    }
    ids.add(provider.id);
    providerValue(provider.provider, `providers.${provider.id}.provider`, provider.apiMode);
    apiModeValue(provider.apiMode, `providers.${provider.id}.apiMode`);
  }
}

function validateModelCatalog(catalog: ModelCatalog, catalogPath: string): void {
  if (!Array.isArray(catalog.models) || catalog.models.length === 0) {
    throw new Error(`Model catalog must contain models: ${catalogPath}`);
  }
  const ids = new Set<string>();
  for (const model of catalog.models) {
    if (!model.id || ids.has(model.id)) {
      throw new Error(`Model catalog has missing or duplicate model id: ${model.id}`);
    }
    ids.add(model.id);
    if (!["generation", "judge", "both"].includes(model.role)) {
      throw new Error(`Unsupported model role for ${model.id}: ${model.role}`);
    }
    positiveIntValue(model.parallelism, `models.${model.id}.parallelism`, 1);
    if (model.contextLength !== undefined) {
      positiveIntValue(model.contextLength, `models.${model.id}.contextLength`, 1);
    }
    if (!Array.isArray(model.aliases)) {
      throw new Error(`models.${model.id}.aliases must be an array`);
    }
  }
}

function apiModeValue(value: unknown, fieldName: string): string {
  const mode = stringValue(value, API_MODE_OPENAI_CHAT_COMPLETIONS);
  if (mode !== API_MODE_OPENAI_CHAT_COMPLETIONS && mode !== API_MODE_LMSTUDIO_NATIVE_CHAT) {
    throw new Error(`Unsupported ${fieldName}: ${mode}`);
  }
  return mode;
}

function providerValue(value: unknown, fieldName: string, apiModeValueRaw: unknown): string {
  const fallback = stringValue(apiModeValueRaw, API_MODE_OPENAI_CHAT_COMPLETIONS) === API_MODE_LMSTUDIO_NATIVE_CHAT
    ? PROVIDER_LMSTUDIO
    : PROVIDER_OPENAI_COMPATIBLE;
  const provider = stringValue(value, fallback);
  if (![PROVIDER_LMSTUDIO, PROVIDER_OLLAMA, PROVIDER_OPENAI_COMPATIBLE, PROVIDER_MINIMAX, PROVIDER_OPENROUTER].includes(provider)) {
    throw new Error(`Unsupported ${fieldName}: ${provider}`);
  }
  return provider;
}

function apiKeyValue(section: TomlTable, configPath: string, fallback: string): string {
  const apiKeyFile = stringValue(section.api_key_file, "");
  if (!apiKeyFile) {
    return stringValue(section.api_key, fallback);
  }
  const filePath = resolveRepoRelativeConfigPath(configPath, apiKeyFile);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing api_key_file configured in ${configPath}: ${filePath}`);
  }
  const apiKey = readTextFile(filePath).trim();
  if (!apiKey) {
    throw new Error(`Empty api_key_file configured in ${configPath}: ${filePath}`);
  }
  return apiKey;
}

function catalogApiKeyValue(repoRoot: string, configuredApiKey: string, apiKeyFile: string, modelId: string): string {
  if (!apiKeyFile) {
    return configuredApiKey;
  }
  const filePath = resolveRuntimeConfigPath(repoRoot, apiKeyFile);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing apiKeyFile configured for model ${modelId}: ${filePath}`);
  }
  const apiKey = readTextFile(filePath).trim();
  if (!apiKey) {
    throw new Error(`Empty apiKeyFile configured for model ${modelId}: ${filePath}`);
  }
  return apiKey;
}

function boolValue(value: unknown, fieldName: string, fallback: boolean): boolean {
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value === "boolean") {
    return value;
  }
  throw new Error(`Expected boolean for ${fieldName}`);
}

function intValue(value: unknown, fieldName: string, fallback: number): number {
  if (value === undefined || value === null) {
    return fallback;
  }
  const numeric = Number(value);
  if (!Number.isInteger(numeric)) {
    throw new Error(`Expected integer for ${fieldName}`);
  }
  return numeric;
}

function optionalIntValue(value: unknown, fieldName: string): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isInteger(numeric)) {
    throw new Error(`Expected integer for ${fieldName}`);
  }
  return numeric;
}

function positiveIntValue(value: unknown, fieldName: string, fallback: number): number {
  const numeric = intValue(value, fieldName, fallback);
  if (numeric < 1) {
    throw new Error(`Expected positive integer for ${fieldName}`);
  }
  return numeric;
}

function floatValue(value: unknown, fieldName: string, fallback: number): number {
  if (value === undefined || value === null) {
    return fallback;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`Expected number for ${fieldName}`);
  }
  return numeric;
}

function requireColumn(rows: CsvRecord[], columnName: string, fileName: string): void {
  const hasColumn = rows.some((row) => Object.prototype.hasOwnProperty.call(row, columnName));
  if (!hasColumn) {
    throw new Error(`Missing column '${columnName}' in ${fileName}`);
  }
}

function isRecord(value: unknown): value is TomlTable {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function resolveRepoRelativeConfigPath(configPath: string, value: string): string {
  if (path.isAbsolute(value)) {
    return value;
  }
  const repoRoot = path.resolve(path.dirname(configPath), "..", "..");
  return resolveRuntimeConfigPath(repoRoot, value);
}

function resolveRuntimeConfigPath(repoRoot: string, value: string): string {
  if (path.isAbsolute(value)) {
    return value;
  }
  const dataRootPath = path.resolve(resolveDataRoot(), value);
  if (fs.existsSync(dataRootPath)) {
    return dataRootPath;
  }
  return path.resolve(repoRoot, value);
}

function mergeProviderEntries(
  baseProviders: ProviderCatalogEntry[],
  localProviders: ProviderCatalogEntry[] | undefined
): ProviderCatalogEntry[] {
  if (!Array.isArray(localProviders) || localProviders.length === 0) {
    return baseProviders;
  }
  const providerById = new Map(baseProviders.map((provider) => [provider.id, provider]));
  for (const provider of localProviders) {
    providerById.set(provider.id, provider);
  }
  return [...providerById.values()];
}
