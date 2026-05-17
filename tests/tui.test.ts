import { describe, expect, it } from "vitest";

import type { ModelCatalogEntry, ProviderCatalogEntry } from "../src/contracts/autobench.js";
import {
  benchmarkMarginOfErrorForChoice,
  benchmarkPrecisionChoices,
  buildModelChoiceRows,
  dualLmStudioPoolEntries,
  formatModelChoiceTable,
  remapResumePoolToCurrentCatalog,
  resumeModeChoices
} from "../src/cli/tui.js";
import type { ModelDiscoveryResult } from "../src/config/modelDiscovery.js";

const providers: ProviderCatalogEntry[] = [
  {
    id: "lmstudio-local-openai-v1",
    label: "LM Studio local OpenAI-compatible API",
    provider: "lmstudio",
    apiMode: "openai_chat_completions",
    apiBaseUrl: "http://127.0.0.1:1234/v1",
    apiKey: "lm-studio",
    apiKeyFile: "",
    discoveryUrls: ["http://127.0.0.1:1234/v1/models"]
  },
  {
    id: "lmstudio-remote-native-v1",
    label: "LM Studio remote Native API",
    provider: "lmstudio",
    apiMode: "lmstudio_native_chat",
    apiBaseUrl: "http://203.0.113.10:1234",
    apiKey: "lm-studio",
    apiKeyFile: "",
    discoveryUrls: ["http://203.0.113.10:1234/api/v1/models"]
  },
  {
    id: "lmstudio-remote-openai-v1",
    label: "LM Studio remote OpenAI-compatible API",
    provider: "lmstudio",
    apiMode: "openai_chat_completions",
    apiBaseUrl: "http://203.0.113.10:1234/v1",
    apiKey: "lm-studio",
    apiKeyFile: "",
    discoveryUrls: ["http://203.0.113.10:1234/v1/models"]
  },
  {
    id: "ollama-local-openai-v1",
    label: "Ollama local OpenAI-compatible API",
    provider: "ollama",
    apiMode: "openai_chat_completions",
    apiBaseUrl: "http://127.0.0.1:11434/v1",
    apiKey: "ollama",
    apiKeyFile: "",
    discoveryUrls: ["http://127.0.0.1:11434/v1/models"]
  },
  {
    id: "openrouter",
    label: "OpenRouter API",
    provider: "openrouter",
    apiMode: "openai_chat_completions",
    apiBaseUrl: "https://openrouter.ai/api/v1",
    apiKey: "",
    apiKeyFile: "secrets/openrouter_api_key.txt",
    discoveryUrls: []
  }
];

const models: ModelCatalogEntry[] = [
  {
    id: "lmstudio_openai_liquid_lfm25_12b",
    label: "Liquid LFM2.5 1.2B (LM Studio local OpenAI-compatible API)",
    model: "liquid/lfm2.5-1.2b",
    providerId: "lmstudio-local-openai-v1",
    role: "judge",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["liquid/lfm2.5-1.2b"],
    capabilities: { openAiChatCompletions: true }
  },
  {
    id: "ollama-local-openai-v1-llama33-latest",
    label: "llama3.3:latest (Ollama local OpenAI-compatible API)",
    model: "llama3.3:latest",
    providerId: "ollama-local-openai-v1",
    role: "both",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["llama3.3:latest"],
    capabilities: { openAiChatCompletions: true }
  },
  {
    id: "openrouter_glm47_flash",
    label: "GLM 4.7 Flash",
    model: "z-ai/glm-4.7-flash",
    providerId: "openrouter",
    role: "both",
    parallelism: 4,
    contextLength: 202752,
    aliases: ["z-ai/glm-4.7-flash"],
    capabilities: { openAiChatCompletions: true }
  },
  {
    id: "lmstudio-local-openai-v1-zai-orgglm-47-flash",
    label: "zai-org/glm-4.7-flash (LM Studio local OpenAI-compatible API)",
    model: "zai-org/glm-4.7-flash",
    providerId: "lmstudio-local-openai-v1",
    role: "both",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["zai-org/glm-4.7-flash"],
    capabilities: { openAiChatCompletions: true }
  },
  {
    id: "lmstudio_native_glm47_flash",
    label: "GLM 4.7 Flash (LM Studio remote native)",
    model: "zai-org/glm-4.7-flash",
    providerId: "lmstudio-remote-native-v1",
    role: "generation",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["zai-org/glm-4.7-flash"],
    capabilities: { nativeChat: true }
  },
  {
    id: "lmstudio-local-openai-v1-googlegemma-4-26b-a4b",
    label: "google/gemma-4-26b-a4b (LM Studio local OpenAI-compatible API)",
    model: "google/gemma-4-26b-a4b",
    providerId: "lmstudio-local-openai-v1",
    role: "both",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["google/gemma-4-26b-a4b"],
    capabilities: { openAiChatCompletions: true }
  },
  {
    id: "lmstudio_native_gemma4_26b_a4b",
    label: "Gemma 4 26B A4B (LM Studio remote native)",
    model: "google/gemma-4-26b-a4b",
    providerId: "lmstudio-remote-native-v1",
    role: "judge",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["google/gemma-4-26b-a4b"],
    capabilities: { nativeChat: true }
  },
  {
    id: "lmstudio_openai_gemma4_26b_a4b",
    label: "Gemma 4 26B A4B (LM Studio remote OpenAI-compatible)",
    model: "google/gemma-4-26b-a4b",
    providerId: "lmstudio-remote-openai-v1",
    role: "judge",
    parallelism: 1,
    contextLength: 128000,
    aliases: ["google/gemma-4-26b-a4b"],
    capabilities: { openAiChatCompletions: true }
  }
];

const discoveryResult: ModelDiscoveryResult = {
  added: 0,
  statuses: [
    {
      providerId: "ollama-local-openai-v1",
      label: "Ollama local OpenAI-compatible API",
      url: "http://127.0.0.1:11434/v1/models",
      status: "online",
      discovered: 1,
      added: 0,
      error: null
    },
    {
      providerId: "lmstudio-local-openai-v1",
      label: "LM Studio local OpenAI-compatible API",
      url: "http://127.0.0.1:1234/v1/models",
      status: "online",
      discovered: 1,
      added: 0,
      error: null
    }
  ]
};

describe("TUI model choice formatting", () => {
  it("renders compact provider groups without ids or default parallelism", () => {
    const rows = buildModelChoiceRows(models, providers, discoveryResult);
    const table = formatModelChoiceTable("judge", rows);

    expect(table).toContain("LM Studio local OpenAI-compatible API");
    expect(table).toContain("Ollama local OpenAI-compatible API");
    expect(table).toContain("OpenRouter API");
    expect(table).toContain("ONLINE");
    expect(table).toContain("UNKNOWN");
    expect(table).toContain("Liquid LFM2.5 1.2B");
    expect(table).toContain("llama3.3:latest");
    expect(table).toContain("GLM 4.7 Flash");
    expect(table).not.toContain("Liquid LFM2.5 1.2B (LM Studio local OpenAI-compatible API)");
    expect(table).not.toContain("lmstudio_openai_liquid_lfm25_12b");
    expect(table).not.toContain("parallelism=1");
    expect(table).toContain("p=4");
  });

  it("filters by hidden id, provider and model text", () => {
    expect(buildModelChoiceRows(models, providers, discoveryResult, "lmstudio_openai_liquid").map((row) => row.model.id))
      .toEqual(["lmstudio_openai_liquid_lfm25_12b"]);
    expect(buildModelChoiceRows(models, providers, discoveryResult, "openrouter").map((row) => row.model.id))
      .toEqual(["openrouter_glm47_flash"]);
    expect(buildModelChoiceRows(models, providers, discoveryResult, "lfm2.5").map((row) => row.model.id))
      .toEqual(["lmstudio_openai_liquid_lfm25_12b"]);
  });

  it("offers fullbench as a benchmark precision option", () => {
    expect(benchmarkPrecisionChoices()[0]).toBe("fullbench (100% das calls)");
    expect(benchmarkMarginOfErrorForChoice("fullbench (100% das calls)")).toBe(0);
    expect(benchmarkMarginOfErrorForChoice("10pp")).toBe(0.10);
  });

  it("offers resume mode before selecting a run", () => {
    expect(resumeModeChoices()).toEqual(["fast resume", "checked resume"]);
  });

  it("builds a dual LMS pool for matching local and remote models", () => {
    const selected = models.find((model) => model.id === "openrouter_glm47_flash")!;
    const pool = dualLmStudioPoolEntries(models, providers, selected);

    expect(pool.map((entry) => entry.modelId)).toEqual([
      "lmstudio-local-openai-v1-zai-orgglm-47-flash",
      "lmstudio_native_glm47_flash"
    ]);
    expect(pool.every((entry) => entry.workers === 1 && entry.timeoutSeconds === 900)).toBe(true);
  });

  it("keeps the selected remote transport when building a dual LMS judge pool", () => {
    const selected = models.find((model) => model.id === "lmstudio_openai_gemma4_26b_a4b")!;
    const pool = dualLmStudioPoolEntries(models, providers, selected);

    expect(pool.map((entry) => entry.modelId)).toEqual([
      "lmstudio-local-openai-v1-googlegemma-4-26b-a4b",
      "lmstudio_openai_gemma4_26b_a4b"
    ]);
  });

  it("remaps resume judge pools through the current catalog", () => {
    const pool = remapResumePoolToCurrentCatalog([
      { model_id: "lmstudio-local-openai-v1-googlegemma-4-26b-a4b", model: "google/gemma-4-26b-a4b", workers: 1, timeout_seconds: 900 },
      { model_id: "lmstudio_openai_gemma4_26b_a4b", model: "google/gemma-4-26b-a4b", workers: 1, timeout_seconds: 900 }
    ], "judge");

    expect(pool.map((entry) => entry.modelId)).toEqual([
      "lmstudio-local-openai-v1-googlegemma-4-26b-a4b",
      "lmstudio_openai_gemma4_26b_a4b"
    ]);
  });

  it("remaps missing resume judge ids to a dual LMS model pair", () => {
    const pool = remapResumePoolToCurrentCatalog([
      { model_id: "old-gemma-id", model: "google/gemma-4-26b-a4b", workers: 1, timeout_seconds: 900 }
    ], "judge");

    expect(pool).toHaveLength(2);
    expect(new Set(pool.map((entry) => entry.modelId))).toEqual(new Set([
      "lmstudio-local-openai-v1-googlegemma-4-26b-a4b",
      "lmstudio_native_gemma4_26b_a4b"
    ]));
  });
});
