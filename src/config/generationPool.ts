import {
  PROVIDER_OPENROUTER,
  type GenerationPoolBackend,
  type GenerationPoolRequest,
  type InferenceConfig
} from "../contracts/autobench.js";
import { loadModelCatalog, loadProviderCatalog, modelSupportsRole, resolveInferenceFromModelCatalog } from "./loadConfig.js";

export const DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 180;
export const DEFAULT_LMSTUDIO_TIMEOUT_SECONDS = 900;

/**
 * Parses the run-specific generation pool JSON passed by the CLI.
 */
export function parseGenerationPoolJson(value: string): GenerationPoolRequest[] {
  let payload: unknown;
  try {
    payload = JSON.parse(value);
  } catch (error) {
    throw new Error(`Invalid --generation-pool JSON: ${(error as Error).message}`);
  }
  if (!Array.isArray(payload)) {
    throw new Error("--generation-pool must be a JSON array");
  }
  return payload.map((entry, index) => parseGenerationPoolEntry(entry, index));
}

export function defaultTimeoutForProvider(provider: string): number {
  return provider === PROVIDER_OPENROUTER ? DEFAULT_OPENROUTER_TIMEOUT_SECONDS : DEFAULT_LMSTUDIO_TIMEOUT_SECONDS;
}

export function defaultWorkersForProvider(provider: string): number {
  return provider === PROVIDER_OPENROUTER ? 50 : 1;
}

export function resolveGenerationPool(args: {
  repoRoot: string;
  base: InferenceConfig;
  generationModelId: string;
  generationPool: GenerationPoolRequest[];
}): GenerationPoolBackend[] {
  if (args.generationPool.length > 0) {
    return args.generationPool.map((entry, index) => resolveGenerationPoolEntry(args.repoRoot, args.base, entry, index));
  }
  const inference = args.generationModelId
    ? resolveInferenceFromModelCatalog({ repoRoot: args.repoRoot, base: args.base, modelId: args.generationModelId })
    : args.base;
  return [{
    backendId: `${args.generationModelId || inference.model}:0`,
    modelId: args.generationModelId || inference.model,
    workers: Math.max(1, inference.parallelism),
    timeoutSeconds: 600,
    inference
  }];
}

export function resolveJudgePool(args: {
  repoRoot: string;
  base: InferenceConfig;
  judgeModelId: string;
  judgePool: GenerationPoolRequest[];
}): GenerationPoolBackend[] {
  if (args.judgePool.length > 0) {
    return args.judgePool.map((entry, index) => resolvePoolEntry(args.repoRoot, args.base, entry, index, "judge"));
  }
  const inference = args.judgeModelId
    ? resolveInferenceFromModelCatalog({ repoRoot: args.repoRoot, base: args.base, modelId: args.judgeModelId })
    : args.base;
  return [{
    backendId: `${args.judgeModelId || inference.model}:0`,
    modelId: args.judgeModelId || inference.model,
    workers: Math.max(1, inference.parallelism),
    timeoutSeconds: 600,
    inference
  }];
}

function parseGenerationPoolEntry(entry: unknown, index: number): GenerationPoolRequest {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    throw new Error(`--generation-pool[${index}] must be an object`);
  }
  const record = entry as Record<string, unknown>;
  const modelId = stringField(record.modelId, `--generation-pool[${index}].modelId`);
  const workers = positiveIntegerField(record.workers, `--generation-pool[${index}].workers`);
  const timeoutSeconds = positiveIntegerField(record.timeoutSeconds, `--generation-pool[${index}].timeoutSeconds`);
  return { modelId, workers, timeoutSeconds };
}

function resolveGenerationPoolEntry(
  repoRoot: string,
  base: InferenceConfig,
  entry: GenerationPoolRequest,
  index: number
): GenerationPoolBackend {
  return resolvePoolEntry(repoRoot, base, entry, index, "generation");
}

function resolvePoolEntry(
  repoRoot: string,
  base: InferenceConfig,
  entry: GenerationPoolRequest,
  index: number,
  role: "generation" | "judge"
): GenerationPoolBackend {
  const models = loadModelCatalog(repoRoot);
  const model = models.models.find((candidate) => candidate.id === entry.modelId || candidate.aliases.includes(entry.modelId));
  if (!model) {
    throw new Error(`Unknown ${role} pool model id: ${entry.modelId}`);
  }
  if (!modelSupportsRole(model, role)) {
    throw new Error(`${role} pool model does not support ${role} role: ${entry.modelId}`);
  }
  const providers = loadProviderCatalog(repoRoot);
  const provider = providers.providers.find((candidate) => candidate.id === model.providerId);
  if (!provider) {
    throw new Error(`Generation pool model ${model.id} references missing provider ${model.providerId}`);
  }
  return {
    backendId: `${model.id}:${index}`,
    modelId: model.id,
    workers: entry.workers,
    timeoutSeconds: entry.timeoutSeconds,
    inference: resolveInferenceFromModelCatalog({ repoRoot, base, modelId: model.id })
  };
}

function stringField(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${field} must be a non-empty string`);
  }
  return value.trim();
}

function positiveIntegerField(value: unknown, field: string): number {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric <= 0) {
    throw new Error(`${field} must be a positive integer`);
  }
  return numeric;
}
