import path from "node:path";

import {
  API_MODE_LMSTUDIO_NATIVE_CHAT,
  API_MODE_OPENAI_CHAT_COMPLETIONS,
  DEFAULT_MODELS_CATALOG,
  PROVIDER_LMSTUDIO,
  PROVIDER_OLLAMA,
  type ModelCatalog,
  type ModelCatalogEntry,
  type ProviderCatalogEntry
} from "../contracts/autobench.js";
import { loadProviderCatalog } from "./loadConfig.js";
import { readJsonFile, writeJson } from "../io/filesystem.js";
import { resolveDataRoot } from "../runtime/paths.js";

const DISCOVERY_TIMEOUT_MS = 3000;
const DEFAULT_DISCOVERED_CONTEXT_LENGTH = 128000;

export interface ProviderDiscoveryStatus {
  providerId: string;
  label: string;
  url: string;
  status: "online" | "offline";
  discovered: number;
  added: number;
  error: string | null;
}

export interface ModelDiscoveryResult {
  added: number;
  statuses: ProviderDiscoveryStatus[];
}

/**
 * Refreshes local runtime availability and appends only new discovered models.
 */
export async function refreshAvailableModelsBestEffort(repoRoot: string, fetchImpl: typeof fetch = fetch): Promise<ModelDiscoveryResult> {
  const providers = loadProviderCatalog(repoRoot).providers.filter((provider) => isDiscoverableProvider(provider.provider));
  const catalogPath = path.join(repoRoot, DEFAULT_MODELS_CATALOG);
  const baseCatalog = readJsonFile<ModelCatalog>(catalogPath);
  const localCatalogPath = path.join(resolveDataRoot(), "config", "models.local.json");
  const localCatalog = readLocalModelCatalog(localCatalogPath);
  const knownKeys = new Set([...baseCatalog.models, ...localCatalog.models].map((model) => modelKey(model.providerId, model.model)));
  const statuses: ProviderDiscoveryStatus[] = [];
  let totalAdded = 0;

  for (const provider of providers) {
    for (const url of provider.discoveryUrls) {
      const status = await discoverProviderModels(provider, url, fetchImpl);
      for (const discovered of status.models) {
        const key = modelKey(provider.id, discovered.model);
        if (knownKeys.has(key)) {
          continue;
        }
        knownKeys.add(key);
        localCatalog.models.push(buildDiscoveredModel(provider, discovered));
        status.publicStatus.added += 1;
        totalAdded += 1;
      }
      statuses.push(status.publicStatus);
    }
  }

  if (totalAdded > 0) {
    writeJson(localCatalogPath, localCatalog);
  }
  return { added: totalAdded, statuses };
}

export function formatDiscoverySummary(result: ModelDiscoveryResult): string {
  const online = result.statuses.filter((status) => status.status === "online").length;
  const offline = result.statuses.length - online;
  const lines = [
    `Model discovery: backends=${result.statuses.length}, online=${online}, offline=${offline}, added=${result.added}`
  ];
  for (const status of result.statuses) {
    const suffix = status.error ? ` error=${status.error}` : "";
    lines.push(`- ${status.providerId}: ${status.status}, discovered=${status.discovered}, added=${status.added}${suffix}`);
  }
  return lines.join("\n");
}

function isDiscoverableProvider(provider: string): boolean {
  return provider === PROVIDER_LMSTUDIO || provider === PROVIDER_OLLAMA;
}

function readLocalModelCatalog(catalogPath: string): ModelCatalog {
  try {
    return readJsonFile<ModelCatalog>(catalogPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
    return { models: [] };
  }
}

async function discoverProviderModels(
  provider: ProviderCatalogEntry,
  url: string,
  fetchImpl: typeof fetch
): Promise<{ publicStatus: ProviderDiscoveryStatus; models: DiscoveredModel[] }> {
  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), DISCOVERY_TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, { signal: controller.signal });
    if (!response.ok) {
      return failedStatus(provider, url, `${response.status} ${response.statusText}`);
    }
    const payload = (await response.json()) as Record<string, unknown>;
    const models = extractDiscoveredModels(payload);
    return {
      publicStatus: {
        providerId: provider.id,
        label: provider.label,
        url,
        status: "online",
        discovered: models.length,
        added: 0,
        error: null
      },
      models
    };
  } catch (error) {
    return failedStatus(provider, url, error instanceof Error ? error.message : String(error));
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function failedStatus(
  provider: ProviderCatalogEntry,
  url: string,
  error: string
): { publicStatus: ProviderDiscoveryStatus; models: DiscoveredModel[] } {
  return {
    publicStatus: {
      providerId: provider.id,
      label: provider.label,
      url,
      status: "offline",
      discovered: 0,
      added: 0,
      error
    },
    models: []
  };
}

type DiscoveredModel = {
  model: string;
  contextLength: number;
};

function extractDiscoveredModels(payload: Record<string, unknown>): DiscoveredModel[] {
  const data = Array.isArray(payload.data) ? payload.data : Array.isArray(payload.models) ? payload.models : [];
  return data
    .map((entry) => {
      if (typeof entry === "string") {
        return { model: entry, contextLength: DEFAULT_DISCOVERED_CONTEXT_LENGTH };
      }
      if (!entry || typeof entry !== "object") {
        return null;
      }
      const record = entry as Record<string, unknown>;
      const model = typeof record.id === "string" ? record.id : typeof record.model === "string" ? record.model : "";
      if (!model.trim()) {
        return null;
      }
      return {
        model,
        contextLength: maxContextLength(record) ?? DEFAULT_DISCOVERED_CONTEXT_LENGTH
      };
    })
    .filter((entry): entry is DiscoveredModel => entry !== null);
}

function buildDiscoveredModel(provider: ProviderCatalogEntry, discovered: DiscoveredModel): ModelCatalogEntry {
  return {
    id: `${provider.id}-${slugify(discovered.model)}`,
    label: `${discovered.model} (${provider.label})`,
    model: discovered.model,
    providerId: provider.id,
    role: "both",
    parallelism: 1,
    contextLength: discovered.contextLength,
    aliases: [discovered.model],
    capabilities: {
      ...(provider.apiMode === API_MODE_LMSTUDIO_NATIVE_CHAT ? { nativeChat: true } : {}),
      ...(provider.apiMode === API_MODE_OPENAI_CHAT_COMPLETIONS ? { openAiChatCompletions: true } : {}),
      reasoningParameter: true
    }
  };
}

function maxContextLength(record: Record<string, unknown>): number | null {
  const candidates = [
    record.context_length,
    record.contextLength,
    record.max_context_length,
    record.maxContextLength,
    record.n_ctx,
    record.max_tokens,
    record.context_window
  ].map((value) => Number(value)).filter((value) => Number.isInteger(value) && value > 0);
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

function modelKey(providerId: string, model: string): string {
  return `${providerId}:${model}`;
}

function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .toLowerCase() || "model";
}
