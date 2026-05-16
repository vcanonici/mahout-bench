import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { main as cliMain } from "../src/cli/index.js";
import {
  bootstrap,
  bootstrapMarkerPath,
  defaultBootstrapProviders,
  shouldAutoBootstrap,
  validateBootstrapProviders,
  writeBootstrapProviderCatalog,
  writeBootstrapSecret
} from "../src/cli/bootstrap.js";
import { loadModelCatalog, loadProviderCatalog, resolveInferenceFromModelCatalog } from "../src/config/loadConfig.js";
import { writeJson } from "../src/io/filesystem.js";

const repoRoot = path.resolve(path.join(import.meta.dirname, ".."));

describe("mahout-bench bootstrap", () => {
  it("accepts one local backend and rejects remote-only providers", () => {
    const [local] = defaultBootstrapProviders();
    expect(() => validateBootstrapProviders([local!])).not.toThrow();
    expect(() => validateBootstrapProviders([{
      ...local!,
      id: "remote-only",
      apiBaseUrl: "http://203.0.113.10:1234/v1",
      discoveryUrls: ["http://203.0.113.10:1234/v1/models"]
    }])).toThrow(/local LM Studio or Ollama/);
  });

  it("writes local providers and merges them without duplicate ids", () => {
    const dataRoot = makeDataRoot();
    const previousDataRoot = process.env.MAHOUT_BENCH_HOME;
    process.env.MAHOUT_BENCH_HOME = dataRoot;
    try {
      const [local] = defaultBootstrapProviders();
      writeBootstrapProviderCatalog(dataRoot, [
        local!,
        {
          ...local!,
          id: "openrouter",
          label: "OpenRouter local override",
          provider: "openrouter",
          apiBaseUrl: "https://openrouter.ai/api/v1",
          apiKey: "",
          apiKeyFile: "secrets/openrouter_api_key.txt",
          discoveryUrls: ["https://openrouter.ai/api/v1/models"]
        }
      ]);

      const catalog = loadProviderCatalog(repoRoot);
      const ids = catalog.providers.map((provider) => provider.id);
      expect(ids.filter((id) => id === "openrouter")).toHaveLength(1);
      expect(catalog.providers.some((provider) => provider.id === local!.id)).toBe(true);
      expect(catalog.providers.find((provider) => provider.id === "openrouter")?.label).toBe("OpenRouter local override");
    } finally {
      restoreDataRoot(previousDataRoot);
    }
  });

  it("resolves model apiKeyFile relative to the data root", () => {
    const dataRoot = makeDataRoot();
    const previousDataRoot = process.env.MAHOUT_BENCH_HOME;
    process.env.MAHOUT_BENCH_HOME = dataRoot;
    try {
      const [local] = defaultBootstrapProviders();
      writeBootstrapProviderCatalog(dataRoot, [local!]);
      writeBootstrapSecret(dataRoot, "local_key.txt", "secret-a", false);
      writeJson(path.join(dataRoot, "config", "models.local.json"), {
        models: [{
          id: "bootstrap-secret-model",
          label: "Bootstrap Secret Model",
          model: "secret/model",
          providerId: local!.id,
          role: "both",
          parallelism: 1,
          contextLength: 8192,
          apiKeyFile: "secrets/local_key.txt",
          aliases: ["secret/model"],
          capabilities: { openAiChatCompletions: true }
        }]
      });
      const base = {
        provider: "lmstudio",
        apiBaseUrl: "http://127.0.0.1:1234/v1",
        apiMode: "openai_chat_completions",
        apiKey: "",
        apiKeyFile: "",
        model: "",
        temperature: 0.7,
        topP: 1,
        maxTokens: 32,
        contextLength: 4096,
        parallelism: 1,
        thinkingEnabled: false,
        reasoningEffort: "low",
        includeReasoningParameter: true,
        systemPrompt: "",
        quotaLabel: "",
        quotaMaxRequests: null,
        quotaWindowSeconds: null
      };

      expect(loadModelCatalog(repoRoot).models.some((model) => model.id === "bootstrap-secret-model")).toBe(true);
      expect(resolveInferenceFromModelCatalog({ repoRoot, base, modelId: "bootstrap-secret-model" }).apiKey).toBe("secret-a");
    } finally {
      restoreDataRoot(previousDataRoot);
    }
  });

  it("does not overwrite generated secrets unless forced", () => {
    const dataRoot = makeDataRoot();
    writeBootstrapSecret(dataRoot, "openrouter_api_key.txt", "first", false);
    writeBootstrapSecret(dataRoot, "openrouter_api_key.txt", "second", false);
    expect(fs.readFileSync(path.join(dataRoot, "secrets", "openrouter_api_key.txt"), "utf8").trim()).toBe("first");
    writeBootstrapSecret(dataRoot, "openrouter_api_key.txt", "second", true);
    expect(fs.readFileSync(path.join(dataRoot, "secrets", "openrouter_api_key.txt"), "utf8").trim()).toBe("second");
  });

  it("supports non-interactive defaults and first-run gating", async () => {
    const dataRoot = makeDataRoot();
    expect(shouldAutoBootstrap(["run"], true, dataRoot)).toBe(true);
    expect(shouldAutoBootstrap(["run", "--self-test"], true, dataRoot)).toBe(false);
    expect(shouldAutoBootstrap(["status"], true, dataRoot)).toBe(false);
    expect(shouldAutoBootstrap(["--no-bootstrap", "run"], true, dataRoot)).toBe(false);
    expect(await bootstrap(["--data-root", dataRoot, "--skip-setup", "--skip-discovery", "--defaults"])).toBe(0);
    expect(fs.existsSync(bootstrapMarkerPath(dataRoot))).toBe(true);
    expect(shouldAutoBootstrap(["run"], true, dataRoot)).toBe(false);
  });

  it("exposes bootstrap help through the public CLI", async () => {
    await expect(cliMain(["bootstrap", "--help"])).resolves.toBe(0);
  });
});

function makeDataRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bootstrap-"));
}

function restoreDataRoot(previousDataRoot: string | undefined): void {
  if (previousDataRoot === undefined) {
    delete process.env.MAHOUT_BENCH_HOME;
  } else {
    process.env.MAHOUT_BENCH_HOME = previousDataRoot;
  }
}
