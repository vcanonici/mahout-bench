import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  COMPARISON_ORDER,
  DOUBLE_SIDED_SECTION_KEYS,
  JUDGE_AFFERITION_STRATIFIED_1000_NAME,
  PAPER_SECTION_KEYS,
  PROVIDER_LMSTUDIO,
  PROVIDER_OLLAMA,
  PROVIDER_OPENROUTER,
  PROFILE_ORDER,
  type AuditConsolidated,
  type DatasetConfig,
  type RunContext
} from "../src/contracts/autobench.js";
import {
  loadModelCatalog,
  enabledDatasets,
  loadProfiles,
  loadProviderCatalog,
  parseJudge,
  parseProfileOrder,
  resolveInferenceFromModelCatalog
} from "../src/config/loadConfig.js";
import { parseGenerationPoolJson, resolveGenerationPool } from "../src/config/generationPool.js";
import { runGenerationQueue } from "../src/generation/generationScheduler.js";
import { nextSocialCandidateBatch } from "../src/generation/socialGeneration.js";
import { archiveExistingJudgeArtifacts, judgeArtifactPaths, validateExistingJudgeInputs } from "../src/judging/judgeArtifacts.js";
import { generationUnitKey, judgeUnitKey, readGenerationResult, readJudgeLabel, writeGenerationResult, writeJudgeLabel } from "../src/pipeline/checkpoint.js";
import { metricsForDataset } from "../src/judging/judgePrompts.js";
import { buildExtraBody, buildJudgeResponseFormat, buildLmStudioNativeBody, isProviderLimitResponse } from "../src/inference/chatClient.js";
import { runDrySmoke, runSelfTest, validateConfigCli } from "../src/pipeline/benchmarkRunner.js";
import { writeResults } from "../src/reporting/resultsReport.js";
import {
  isLocalLmStudioBaseUrl,
  lmStudioSdkBaseUrl,
  loadedModelContextLength,
  loadedModelMatches,
  localLmStudioBackendsNotReadyForSkipLms,
  localLmStudioLoadConfig
} from "../src/runtime/lmsLifecycle.js";
import { TerminalObserver } from "../src/runtime/terminalObserver.js";
import { computeDoubleSidedScores, computeMoralScores, outputFileForMode, summaryForScoreFile } from "../src/scoring/scoreEngine.js";
import { buildSampleManifest, estimateCalls, plannedOverallUnits, sampleTargetN } from "../src/sampling/samplePlanner.js";
import { readCsvFile, readJsonFile, writeCsvJsonl, writeJson } from "../src/io/filesystem.js";
import { computeMetricSummary, computeSimilarityCounts } from "../src/validate_judge/metrics.js";
import { loadElephantReference } from "../src/validate_judge/loadElephantReference.js";
import {
  ensureJudgeAfferitionStratifiedTestSet1000,
  ensureJudgeAfferitionMarginDataset,
  formatMarginLabel,
  parseJudgeAfferitionMarginInput
} from "../src/validate_judge/judgeAfferitionSampling.js";
import { normalizeReferenceLabel, parseStrictBinaryJson } from "../src/validate_judge/parseBinaryLabel.js";
import { ensureClaudeSocialJudgeAfferition } from "../src/validate_judge/prepareElephantFullResults.js";
import { buildValidationJudge, runJudgeValidation } from "../src/validate_judge/runJudgeValidation.js";
import {
  buildInitialJudgeAfferitionRunState,
  listIncompleteJudgeAfferitionRuns,
  readJudgeAfferitionRunState,
  writeJudgeAfferitionRunState
} from "../src/validate_judge/runState.js";
import {
  findUsableJudgeValidation,
  formatValidationSummary,
  loadJudgeValidationRegistry,
  saveJudgeValidationRegistry,
  upsertJudgeValidation
} from "../src/validate_judge/judgeValidationRegistry.js";
import { refreshAvailableModelsBestEffort } from "../src/config/modelDiscovery.js";
import {
  aggregateTokenUsageFromJsonl,
  combineTokenUsageSummaries,
  estimateTokenUsage,
  tokenUsageForChat
} from "../src/inference/tokenUsage.js";

const repoRoot = path.resolve(path.join(import.meta.dirname, ".."));
const dataRoot = path.resolve(process.env.MAHOUT_BENCH_HOME ?? repoRoot);
process.env.MAHOUT_BENCH_HOME = dataRoot;
const hasInstalledDataBundle = fs.existsSync(path.join(dataRoot, "datasets", "full_results", "OEQ.csv")) &&
  fs.existsSync(path.join(dataRoot, "datasets", "judge_afferition", "claude_social", "all.csv"));

describe("mahout-bench", () => {
  it("loads profiles and judge config from the self-contained folder", () => {
    const profiles = loadProfiles(repoRoot);
    const judge = parseJudge(repoRoot);

    expect(profiles.map((profile) => profile.name)).toEqual([...PROFILE_ORDER]);
    expect(judge.inference.model).toBe("google/gemma-4-26b-a4b");
  });

  it("loads model/provider catalogs and resolves inference without changing profile hyperparameters", () => {
    const providers = loadProviderCatalog(repoRoot);
    const models = loadModelCatalog(repoRoot);
    const profile = loadProfiles(repoRoot)[0]!;
    const resolved = resolveInferenceFromModelCatalog({
      repoRoot,
      base: profile.generation,
      modelId: "lmstudio_native_qwen36_35b_a3b"
    });

    expect(providers.providers.some((provider) => provider.id === "minimax")).toBe(true);
    expect(providers.providers.some((provider) => provider.id === "lmstudio-local-native-v1")).toBe(true);
    expect(providers.providers.some((provider) => provider.id === "lmstudio-local-openai-v1")).toBe(true);
    expect(providers.providers.some((provider) => provider.id === "ollama-local-openai-v1" && provider.provider === PROVIDER_OLLAMA)).toBe(false);
    expect(providers.providers.some((provider) => provider.id === "openrouter" && provider.provider === PROVIDER_OPENROUTER)).toBe(true);
    expect(models.models.some((model) => model.id === "minimax-m27")).toBe(true);
    expect(models.models.some((model) => model.id === "lmstudio_native_liquid_lfm25_12b")).toBe(true);
    expect(models.models.some((model) => model.id === "lmstudio_openai_liquid_lfm25_12b")).toBe(true);
    expect(models.models.some((model) => model.id === "lmstudio-liquid-lfm25-12b" && model.capabilities.legacyId === true)).toBe(true);
    expect(resolved.model).toBe("qwen/qwen3.6-35b-a3b");
    expect(resolved.provider).toBe("lmstudio");
    expect(resolved.parallelism).toBe(1);
    expect(resolved.temperature).toBe(profile.generation.temperature);
    expect(resolved.systemPrompt).toBe(profile.generation.systemPrompt);
  });

  it("resolves model catalog context length overrides", () => {
    const profile = loadProfiles(repoRoot)[0]!;
    const resolved = resolveInferenceFromModelCatalog({
      repoRoot,
      base: { ...profile.generation, contextLength: 4096 },
      modelId: "lmstudio_native_liquid_lfm25_12b"
    });

    expect(resolved.contextLength).toBe(128000);
  });

  it("registers OpenRouter GLM 4.7 Flash as a both-role model", () => {
    const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "openrouter_glm47_flash");

    expect(model).toMatchObject({
      model: "z-ai/glm-4.7-flash",
      providerId: "openrouter",
      role: "both",
      contextLength: 202752
    });
  });

  it("parses and validates run-specific generation pools", () => {
    const parsed = parseGenerationPoolJson('[{"modelId":"openrouter_glm47_flash","workers":50,"timeoutSeconds":180}]');
    expect(parsed).toEqual([{ modelId: "openrouter_glm47_flash", workers: 50, timeoutSeconds: 180 }]);
    expect(() => parseGenerationPoolJson('{"modelId":"openrouter_glm47_flash"}')).toThrow(/JSON array/);
    expect(() => parseGenerationPoolJson('[{"modelId":"openrouter_glm47_flash","workers":0,"timeoutSeconds":180}]')).toThrow(/workers/);
    expect(() => parseGenerationPoolJson('[{"modelId":"openrouter_glm47_flash","workers":1,"timeoutSeconds":0}]')).toThrow(/timeoutSeconds/);

    const [reference] = loadProfiles(repoRoot);
    expect(() => resolveGenerationPool({
      repoRoot,
      base: reference!.generation,
      generationModelId: "",
      generationPool: [{ modelId: "missing-model", workers: 1, timeoutSeconds: 180 }]
    })).toThrow(/Unknown generation pool model id/);
  });

  it("runs generation queues through FIFO backend workers", async () => {
    const [reference] = loadProfiles(repoRoot);
    const pool = resolveGenerationPool({
      repoRoot,
      base: reference!.generation,
      generationModelId: "",
      generationPool: [
        { modelId: "lmstudio-local-openai-v1-zai-orgglm-47-flash", workers: 2, timeoutSeconds: 180 },
        { modelId: "lmstudio_native_glm47_flash", workers: 1, timeoutSeconds: 900 }
      ]
    });
    const seen: string[] = [];
    const results = await runGenerationQueue([1, 2, 3, 4], pool, async (item, backend) => {
      seen.push(`${item}:${backend.modelId}:${backend.timeoutSeconds}`);
      return `${item}:${backend.backendId}`;
    });

    expect(results.map((entry) => entry.item)).toEqual([1, 2, 3, 4]);
    expect(results.map((entry) => entry.result)).toHaveLength(4);
    expect(seen.some((entry) => entry.includes("lmstudio-local-openai-v1-zai-orgglm-47-flash:180"))).toBe(true);
    expect(seen.some((entry) => entry.includes("lmstudio_native_glm47_flash:900"))).toBe(true);
  });

  it("keeps every LM Studio catalog model serialized by default", () => {
    const providers = loadProviderCatalog(repoRoot);
    const models = loadModelCatalog(repoRoot);
    const lmStudioProviderIds = new Set(
      providers.providers.filter((provider) => provider.provider === "lmstudio").map((provider) => provider.id)
    );
    const lmStudioModels = models.models.filter((model) => lmStudioProviderIds.has(model.providerId));

    expect(lmStudioModels.length).toBeGreaterThan(0);
    expect(lmStudioModels.every((model) => model.parallelism === 1)).toBe(true);
  });

  it("uses SDK-only local LM Studio load defaults for flash attention and q4 KV cache", () => {
    const config = localLmStudioLoadConfig(128000);

    expect(config).toMatchObject({
      contextLength: 128000,
      gpu: { ratio: "max" },
      gpuStrictVramCap: true,
      flashAttention: true,
      offloadKVCacheToGpu: true,
      llamaKCacheQuantizationType: "q4_0",
      llamaVCacheQuantizationType: "q4_0"
    });
    expect(lmStudioSdkBaseUrl("http://127.0.0.1:1234/v1")).toBe("ws://127.0.0.1:1234");
    expect(lmStudioSdkBaseUrl("https://localhost:8080/api/v1/models")).toBe("wss://localhost:8080");
  });

  it("discovers new LM Studio models without mutating existing catalog entries", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-discovery-root-"));
    fs.mkdirSync(path.join(fixtureRoot, "config"), { recursive: true });
    writeJson(path.join(fixtureRoot, "config", "providers.json"), {
      providers: [
        {
          id: "lmstudio-test-native",
          label: "LM Studio test native",
          provider: "lmstudio",
          apiMode: "lmstudio_native_chat",
          apiBaseUrl: "http://127.0.0.1:1234",
          apiKey: "lm-studio",
          apiKeyFile: "",
          discoveryUrls: ["http://ok/models", "http://offline/models"]
        }
      ]
    });
    writeJson(path.join(fixtureRoot, "config", "models.json"), {
      models: [
        {
          id: "existing",
          label: "Existing",
          model: "existing/model",
          providerId: "lmstudio-test-native",
          role: "judge",
          parallelism: 1,
          contextLength: 32000,
          aliases: ["existing/model"],
          capabilities: { nativeChat: true }
        }
      ]
    });
    const fetchImpl = async (url: string | URL | Request) => {
      const textUrl = String(url);
      if (textUrl.includes("offline")) {
        throw new Error("offline");
      }
      return new Response(JSON.stringify({
        data: [
          { id: "existing/model", context_length: 64000 },
          { id: "new/model", max_context_length: 99000 }
        ]
      }), { status: 200 });
    };

    const previousDataRoot = process.env.MAHOUT_BENCH_HOME;
    process.env.MAHOUT_BENCH_HOME = fixtureRoot;
    const result = await refreshAvailableModelsBestEffort(fixtureRoot, fetchImpl as typeof fetch);
    if (previousDataRoot === undefined) {
      delete process.env.MAHOUT_BENCH_HOME;
    } else {
      process.env.MAHOUT_BENCH_HOME = previousDataRoot;
    }
    const baseCatalog = readJsonFile<{ models: Array<Record<string, unknown>> }>(path.join(fixtureRoot, "config", "models.json"));
    const catalog = readJsonFile<{ models: Array<Record<string, unknown>> }>(path.join(fixtureRoot, "config", "models.local.json"));
    const existing = baseCatalog.models.find((entry) => entry.id === "existing");
    const added = catalog.models.find((entry) => entry.model === "new/model");

    expect(result.added).toBe(1);
    expect(result.statuses.map((status) => status.status)).toEqual(["online", "offline"]);
    expect(existing?.contextLength).toBe(32000);
    expect(added).toMatchObject({
      providerId: "lmstudio-test-native",
      role: "both",
      parallelism: 1,
      contextLength: 99000,
      capabilities: { nativeChat: true, reasoningParameter: true }
    });
  });

  it("discovers new Ollama relay models from OpenAI-compatible model lists", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-ollama-discovery-root-"));
    fs.mkdirSync(path.join(fixtureRoot, "config"), { recursive: true });
    writeJson(path.join(fixtureRoot, "config", "providers.json"), {
      providers: [
        {
          id: "ollama-test",
          label: "Ollama test relay",
          provider: "ollama",
          apiMode: "openai_chat_completions",
          apiBaseUrl: "http://127.0.0.1:11434/v1",
          apiKey: "ollama",
          apiKeyFile: "",
          discoveryUrls: ["http://ok/v1/models"]
        }
      ]
    });
    writeJson(path.join(fixtureRoot, "config", "models.json"), {
      models: [
        {
          id: "existing",
          label: "Existing",
          model: "existing/model",
          providerId: "ollama-test",
          role: "generation",
          parallelism: 1,
          contextLength: 32000,
          aliases: ["existing/model"],
          capabilities: { openAiChatCompletions: true }
        }
      ]
    });
    const fetchImpl = async () => new Response(JSON.stringify({
      data: [
        { id: "llama3.3:latest" }
      ]
    }), { status: 200 });

    const previousDataRoot = process.env.MAHOUT_BENCH_HOME;
    process.env.MAHOUT_BENCH_HOME = fixtureRoot;
    const result = await refreshAvailableModelsBestEffort(fixtureRoot, fetchImpl as typeof fetch);
    if (previousDataRoot === undefined) {
      delete process.env.MAHOUT_BENCH_HOME;
    } else {
      process.env.MAHOUT_BENCH_HOME = previousDataRoot;
    }
    const catalog = readJsonFile<{ models: Array<Record<string, unknown>> }>(path.join(fixtureRoot, "config", "models.local.json"));

    expect(result.added).toBe(1);
    expect(result.statuses[0]).toMatchObject({ providerId: "ollama-test", status: "online", discovered: 1 });
    const added = catalog.models.find((entry) => entry.model === "llama3.3:latest");
    expect(added).toMatchObject({
      id: "ollama-test-llama33latest",
      providerId: "ollama-test",
      model: "llama3.3:latest",
      role: "both",
      parallelism: 1,
      contextLength: 128000,
      capabilities: { openAiChatCompletions: true, reasoningParameter: true }
    });
  });

  it("extracts provider token usage and falls back to deterministic estimates", () => {
    const inference = loadProfiles(repoRoot)[0]!.generation;
    const providerUsage = tokenUsageForChat({
      inference,
      prompt: "hello",
      outputText: "world",
      responseDump: {
        usage: {
          prompt_tokens: 10,
          completion_tokens: 3,
          total_tokens: 13,
          completion_tokens_details: { reasoning_tokens: 2 }
        }
      }
    });
    const estimated = estimateTokenUsage(inference, "12345678", "1234");

    expect(providerUsage).toEqual({
      inputTokens: 10,
      outputTokens: 3,
      totalTokens: 13,
      reasoningTokens: 2,
      source: "provider_usage"
    });
    expect(estimated).toMatchObject({
      inputTokens: expect.any(Number),
      outputTokens: 1,
      source: "estimated"
    });
  });

  it("aggregates token usage from JSONL traces", () => {
    const outputPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "elephant-token-usage-")), "raw.jsonl");
    fs.writeFileSync(outputPath, [
      JSON.stringify({ token_usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12, reasoningTokens: 1, source: "provider_usage" } }),
      JSON.stringify({ token_usage: { inputTokens: 5, outputTokens: 1, totalTokens: 6, reasoningTokens: null, source: "estimated" } })
    ].join("\n"));

    const summary = aggregateTokenUsageFromJsonl(outputPath);
    const total = combineTokenUsageSummaries(summary);

    expect(summary).toMatchObject({
      calls: 2,
      inputTokens: 15,
      outputTokens: 3,
      totalTokens: 18,
      reasoningTokens: 1,
      providerUsageCalls: 1,
      estimatedCalls: 1
    });
    expect(total.totalTokens).toBe(18);
  });

  it("loads the MiniMax identity two-profile order", () => {
    const order = parseProfileOrder(repoRoot, "config/profiles_minimax_m27_identity");
    const profiles = loadProfiles(repoRoot, "config/profiles_minimax_m27_identity", order.order);

    expect(order).toEqual({ order: ["M2V", "M2ID"], canonical: "M2V" });
    expect(profiles.map((profile) => profile.name)).toEqual(["M2V", "M2ID"]);
    expect(profiles[0]!.generation.model).toBe("");
    expect(profiles[1]!.generation.systemPrompt).toContain("Identity prompt placeholder");
  });

  it("keeps the sample size formula aligned with the Python runner", () => {
    expect(sampleTargetN(3027)).toBe(144);
    expect(sampleTargetN(2000)).toBe(140);
    expect(sampleTargetN(3777)).toBe(145);
    expect(sampleTargetN(3027, { confidence: 0.95, marginOfError: 0.10 })).toBe(94);
    expect(sampleTargetN(3027, { confidence: 0.95, marginOfError: 0 })).toBe(3027);
  });

  it("counts planned benchmark progress in provider calls", () => {
    const socialTargets = { oeq: 10, ss: 5 };
    const moralTargetN = 7;
    const profileCount = 2;
    const socialGeneration = 15;
    const moralGeneration = moralTargetN * 4;
    const judgeSocial = 10 * 3 + 5;
    const judgeMoral = moralTargetN * 2 * 3;
    const judgePreflight = 3;

    expect(plannedOverallUnits(profileCount, socialTargets, moralTargetN)).toBe(
      profileCount * (socialGeneration + moralGeneration + judgeSocial + judgeMoral) + judgePreflight
    );
  });

  it("does not batch-throttle canonical social fullbench candidates", () => {
    const shuffled = Array.from({ length: 20 }, (_, index) => index);
    const pool = [
      { workers: 1 },
      { workers: 1 }
    ] as Array<{ workers: number }>;

    expect(nextSocialCandidateBatch(shuffled, 3, 17, pool as never, true)).toEqual(shuffled.slice(3));
    expect(nextSocialCandidateBatch(shuffled, 3, 17, pool as never, false)).toHaveLength(8);
  });

  it("passes the offline self-test and config validation", () => {
    expect(runSelfTest()).toBe(0);
    if (!hasInstalledDataBundle) {
      return;
    }
    expect(
      validateConfigCli({
        selfTest: false,
        drySmoke: false,
        validateConfig: true,
        judgeOnly: false,
        profilesRoot: "config/profiles",
        profiles: [],
        judgeConfig: "config/judge/juiz.toml",
        outputRoot: "",
        skipLms: true,
        generationModelId: "",
        generationPool: [],
        judgeModelId: "",
        judgePool: [],
        benchmarkName: "",
        marginOfError: null
      })
    ).toBe(0);
  });

  it("runs the dry smoke flow without network calls", () => {
    expect(runDrySmoke()).toBe(0);
  });

  it("renders RESULTS.md with the expected top-level sections", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-report-"));
    const ctx: RunContext = {
      repoRoot,
      dataRoot: repoRoot,
      outputRoot,
      eventsPath: path.join(outputRoot, "run_events.jsonl"),
      rawGenerationPath: path.join(outputRoot, "raw_generation.jsonl"),
      rawJudgePath: path.join(outputRoot, "raw_judge.jsonl"),
      quarantinePath: path.join(outputRoot, "quarantine.jsonl"),
      providerEventsPath: path.join(outputRoot, "provider_events.jsonl"),
      generationCheckpointPath: path.join(outputRoot, "generation_checkpoint.json"),
      judgeCheckpointPath: path.join(outputRoot, "judge_checkpoint.json"),
      profilesRoot: "config/profiles",
      judgeConfigPath: "config/judge/juiz.toml",
      profileNames: [],
      benchmarkName: "report fixture",
      generationModelId: "",
      generationPool: [],
      judgeModelId: "",
      judgePool: [],
      marginOfError: null
    };

    const audit = buildAuditFixture(outputRoot);
    writeResults(ctx, audit);

    const report = fs.readFileSync(path.join(outputRoot, "RESULTS.md"), "utf8");
    expect(report).toContain("# METRICAS PRINCIPAIS");
    expect(report).toContain("# METRICAS AUXILIARES");
    expect(report).toContain("# PROFILES NA INTEGRA");
    expect(report).toContain("# EXPLICACAO DOS RESULTADOS");
    for (const [, , title] of COMPARISON_ORDER) {
      expect(report).toContain(`## ${title}`);
    }
  });

  it("renders RESULTS.md with dynamic profile comparisons", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-report-dynamic-"));
    const ctx = buildRunContext(outputRoot);
    const audit = buildAuditFixture(outputRoot);
    audit.sample_manifest = {
      created_at: new Date().toISOString(),
      benchmark_name: "dynamic report",
      canonical_profile: "Felix-V",
      profile_order: ["Felix-V", "Felix-P"],
      generation_model_id: "model",
      generation_pool: [],
      judge_model_id: "judge",
      judge_pool: [],
      generation_inference: null,
      judge_inference: null,
      confidence: 0.95,
      margin_of_error: 0.1,
      datasets: {},
      moral_pair_ids: []
    };
    audit.social_summaries = audit.social_summaries.filter((summary) => summary.profile !== "Felix-A");
    audit.moral_summaries = audit.moral_summaries.filter((summary) => summary.profile !== "Felix-A");
    audit.double_sided_summaries = audit.double_sided_summaries.filter((summary) => summary.profile !== "Felix-A");

    writeResults(ctx, audit);

    const report = fs.readFileSync(path.join(outputRoot, "RESULTS.md"), "utf8");
    expect(report).toContain("## Felix-P vs Felix-V");
    expect(report).not.toContain("## Assertivo vs Vanilla");
    expect(report).toContain("| Variante | Felix-V | Felix-P |");
  });

  it("renders RESULTS.md with MiniMax identity comparison", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-report-m2-"));
    const ctx = {
      ...buildRunContext(outputRoot),
      profilesRoot: "config/profiles_minimax_m27_identity",
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml"
    };
    const audit = buildAuditFixtureForProfiles(outputRoot, ["M2V", "M2ID"]);
    audit.sample_manifest = {
      created_at: new Date().toISOString(),
      benchmark_name: "minimax identity",
      canonical_profile: "M2V",
      profile_order: ["M2V", "M2ID"],
      generation_model_id: "minimax-m27",
      generation_pool: [],
      judge_model_id: "lmstudio_native_liquid_lfm25_12b",
      judge_pool: [],
      generation_inference: null,
      judge_inference: null,
      confidence: 0.95,
      margin_of_error: 0.1,
      datasets: {},
      moral_pair_ids: []
    };

    writeResults(ctx, audit);

    const report = fs.readFileSync(path.join(outputRoot, "RESULTS.md"), "utf8");
    expect(report).toContain("## M2ID vs M2V");
    expect(report).toContain("## PROFILE M2V");
    expect(report).toContain("## PROFILE M2ID");
  });

  it("builds sample manifests with real per-dataset populations", () => {
    if (!hasInstalledDataBundle) {
      return;
    }
    const [reference] = loadProfiles(repoRoot);
    const datasetPopulations = Object.fromEntries(
      Object.values(reference!.datasets).map((dataset) => [dataset.name, readCsvFile(path.join(dataRoot, reference!.datasetsDir, dataset.file)).length])
    );

    const manifest = buildSampleManifest({
      ctx: buildRunContext(fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-manifest-"))),
      referenceProfile: reference!,
      profileOrder: ["Felix-V", "Felix-P", "Felix-A"],
      generationPool: [],
      judgePool: [],
      judgeInference: null,
      socialIndices: { oeq: [1, 2, 3] },
      datasetPopulations,
      moralIds: ["pair-a", "pair-b"]
    });

    expect(manifest.datasets.oeq?.acceptedIndices).toEqual([1, 2, 3]);
    expect(manifest.datasets.aita_nta_og?.population).toBe(datasetPopulations.aita_nta_og);
    expect(manifest.datasets.aita_nta_og?.acceptedIndices).toEqual([]);
    expect(manifest.moral_pair_ids).toEqual(["pair-a", "pair-b"]);
    expect(manifest.canonical_profile).toBe(reference!.name);
    expect(manifest.profile_order).toEqual(["Felix-V", "Felix-P", "Felix-A"]);
  });

  it("estimates 10pp call counts from current datasets", () => {
    if (!hasInstalledDataBundle) {
      return;
    }
    const [reference] = loadProfiles(repoRoot);
    const enabled = enabledDatasets(reference!);
    const socialDatasets = enabled.filter((dataset) => dataset.task === "social");
    const moralA = enabled.find((dataset) => dataset.task === "moral_a")!;
    const moralB = enabled.find((dataset) => dataset.task === "moral_b")!;
    const populations = Object.fromEntries(
      enabled.map((dataset) => [dataset.name, readCsvFile(path.join(dataRoot, reference!.datasetsDir, dataset.file)).length])
    );
    const moralPopulation = commonIdsForTest(
      readCsvFile(path.join(dataRoot, reference!.datasetsDir, moralA.file)),
      readCsvFile(path.join(dataRoot, reference!.datasetsDir, moralB.file))
    );

    const estimate = estimateCalls({
      profileCount: 2,
      socialDatasets,
      populations,
      moralPopulation,
      sampling: { confidence: 0.95, marginOfError: 0.10 }
    });

    expect(estimate.generationPerProfile).toBe(644);
    expect(estimate.judgePerProfile).toBe(1198);
    expect(estimate.judgeTotal).toBe(2399);
    expect(estimate.total).toBe(3687);
  });

  it("estimates fullbench call counts from current datasets", () => {
    if (!hasInstalledDataBundle) {
      return;
    }
    const [reference] = loadProfiles(repoRoot);
    const enabled = enabledDatasets(reference!);
    const socialDatasets = enabled.filter((dataset) => dataset.task === "social");
    const moralA = enabled.find((dataset) => dataset.task === "moral_a")!;
    const moralB = enabled.find((dataset) => dataset.task === "moral_b")!;
    const populations = Object.fromEntries(
      enabled.map((dataset) => [dataset.name, readCsvFile(path.join(dataRoot, reference!.datasetsDir, dataset.file)).length])
    );
    const moralPopulation = commonIdsForTest(
      readCsvFile(path.join(dataRoot, reference!.datasetsDir, moralA.file)),
      readCsvFile(path.join(dataRoot, reference!.datasetsDir, moralB.file))
    );

    const estimate = estimateCalls({
      profileCount: 1,
      socialDatasets,
      populations,
      moralPopulation,
      sampling: { confidence: 0.95, marginOfError: 0 }
    });

    expect(estimate.socialTargets).toEqual({
      oeq: populations.oeq,
      aita_yta: populations.aita_yta,
      ss: populations.ss
    });
    expect(estimate.moralTargetN).toBe(moralPopulation);
  });

  it("normalizes judge afferition margin labels and inputs", () => {
    expect(formatMarginLabel(0.10)).toBe("10pp");
    expect(formatMarginLabel(0.08)).toBe("8pp");
    expect(formatMarginLabel(0.075)).toBe("7_5pp");
    expect(parseJudgeAfferitionMarginInput("10pp")).toBe(0.10);
    expect(parseJudgeAfferitionMarginInput("10")).toBe(0.10);
    expect(parseJudgeAfferitionMarginInput("0.08")).toBe(0.08);
    expect(() => parseJudgeAfferitionMarginInput("0pp")).toThrow(/margin/i);
  });

  it("builds judge afferition margin samples by dataset and metric", () => {
    if (!hasInstalledDataBundle) {
      return;
    }
    const tenPp = ensureJudgeAfferitionMarginDataset(repoRoot, 0.10);
    const eightPp = ensureJudgeAfferitionMarginDataset(repoRoot, 0.08);
    const fivePp = ensureJudgeAfferitionMarginDataset(repoRoot, 0.05);

    expect(tenPp.manifest.sample_total).toBe(1198);
    expect(eightPp.manifest.sample_total).toBe(1820);
    expect(fivePp.manifest.sample_total).toBe(4186);
    expect(tenPp.manifest.datasets.OEQ?.metrics.framing).toEqual({ full: 2985, target: 94, selected: 94 });
    expect(tenPp.manifest.datasets["AITA-NTA-FLIP"]?.metrics.validation).toEqual({ full: 1591, target: 91, selected: 91 });
    expect(loadElephantReference(repoRoot, tenPp.dataDir).rows).toHaveLength(1198);
  });

  it("builds the fixed 1000-row judge afferition test set by dataset and metric", () => {
    if (!hasInstalledDataBundle) {
      return;
    }
    const testSet = ensureJudgeAfferitionStratifiedTestSet1000(repoRoot);
    const rows = loadElephantReference(repoRoot, testSet.dataDir).rows;

    expect(testSet.manifest.name).toBe(JUDGE_AFFERITION_STRATIFIED_1000_NAME);
    expect(testSet.manifest.sample_total).toBe(1000);
    expect(testSet.sampling.kind).toBe("test_set");
    expect(testSet.sampling.sampleBy).toBe("dataset_metric_fixed_total");
    expect(rows).toHaveLength(1000);
    expect(testSet.manifest.datasets.OEQ?.sample_rows).toBe(200);
    expect(testSet.manifest.datasets["AITA-YTA"]?.sample_rows).toBe(200);
    expect(testSet.manifest.datasets.SS?.metrics.framing).toEqual({ full: 3759, target: 200, selected: 200 });
    expect(testSet.manifest.datasets["AITA-NTA-FLIP"]?.sample_rows).toBe(200);
  });

  it("reuses valid judge afferition sample manifests and regenerates stale ones", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-sampling-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    await ensureClaudeSocialJudgeAfferition(fixtureRoot);

    const first = ensureJudgeAfferitionMarginDataset(fixtureRoot, 0.20);
    const second = ensureJudgeAfferitionMarginDataset(fixtureRoot, 0.20);
    expect(second.manifest.created_at).toBe(first.manifest.created_at);

    writeJson(first.manifestPath, { ...first.manifest, source_fingerprint: "stale" });
    const regenerated = ensureJudgeAfferitionMarginDataset(fixtureRoot, 0.20);
    expect(regenerated.manifest.source_fingerprint).not.toBe("stale");
    expect(regenerated.manifest.sample_total).toBe(first.manifest.sample_total);
  });

  it("keeps metric routing and score summaries aligned with ELEPHANT", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-scoring-"));
    const ctx = buildRunContext(outputRoot);
    const profile = loadProfiles(repoRoot)[0]!;
    const socialDataset = datasetConfig("oeq", "prompt", "social");
    const moralA = datasetConfig("aita_nta_og", "original_post", "moral_a");
    const moralB = datasetConfig("aita_nta_flip", "flipped_story", "moral_b");

    expect(metricsForDataset("ss")).toEqual(["framing"]);
    expect(metricsForDataset("ss", true)).toEqual(["validation", "indirectness", "framing"]);

    writeCsvJsonl(
      [
        { prompt: "q1", [`validation_${profile.name}`]: 1, [`indirectness_${profile.name}`]: 0, [`framing_${profile.name}`]: 1, validation_human: 0 },
        { prompt: "q2", [`validation_${profile.name}`]: 0, [`indirectness_${profile.name}`]: 1, [`framing_${profile.name}`]: null, validation_human: 1 }
      ],
      path.join(outputRoot, profile.name, socialDataset.name, "scores.csv")
    );
    const summary = summaryForScoreFile(path.join(outputRoot, profile.name, socialDataset.name, "scores.csv"), profile, socialDataset);
    expect(summary.metrics.validation?.positive_n).toBe(1);
    expect(summary.metrics.validation?.baseline).toBe(0.5);

    writeCsvJsonl(
      [
        { id: "1", [`${profile.name}_label`]: "NTA" },
        { id: "2", [`${profile.name}_label`]: "YTA" }
      ],
      outputFileForMode(ctx, profile.name, moralA.name, "binary")
    );
    writeCsvJsonl(
      [
        { id: "1", [`${profile.name}_label`]: "NTA" },
        { id: "2", [`${profile.name}_label`]: "NTA" }
      ],
      outputFileForMode(ctx, profile.name, moralB.name, "binary")
    );
    const moral = computeMoralScores(ctx, profile, moralA, moralB);
    expect(moral.metrics.both_NTA_rate).toBe(0.5);

    writeCsvJsonl(
      [
        { id: "1", [`validation_${profile.name}`]: 1, [`indirectness_${profile.name}`]: 0, [`framing_${profile.name}`]: 1 },
        { id: "2", [`validation_${profile.name}`]: 1, [`indirectness_${profile.name}`]: 1, [`framing_${profile.name}`]: null }
      ],
      outputFileForMode(ctx, profile.name, moralA.name, "free").replace("responses_free.csv", "scores_free.csv")
    );
    writeCsvJsonl(
      [
        { id: "1", [`validation_${profile.name}`]: 1, [`indirectness_${profile.name}`]: 1, [`framing_${profile.name}`]: 1 },
        { id: "2", [`validation_${profile.name}`]: 0, [`indirectness_${profile.name}`]: 1, [`framing_${profile.name}`]: 1 }
      ],
      outputFileForMode(ctx, profile.name, moralB.name, "free").replace("responses_free.csv", "scores_free.csv")
    );
    const double = computeDoubleSidedScores(ctx, profile, moralA, moralB);
    expect(double.metrics.validation!.both_1_count).toBe(1);
    expect(double.metrics.framing!.valid_n_pairs).toBe(1);
  });

  it("validates and archives judge-only artifacts without deleting them", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-judge-artifacts-"));
    const ctx = buildRunContext(outputRoot);
    const profiles = loadProfiles(repoRoot);
    const reference = profiles[0]!;
    const socialDatasets = [datasetConfig("oeq", "prompt", "social")];
    const moralA = datasetConfig("aita_nta_og", "original_post", "moral_a");
    const moralB = datasetConfig("aita_nta_flip", "flipped_story", "moral_b");

    writeJson(path.join(outputRoot, "sample_manifest.json"), {
      created_at: new Date().toISOString(),
      canonical_profile: reference.name,
      profile_order: [...PROFILE_ORDER],
      confidence: 0.95,
      margin_of_error: 0.08,
      datasets: {},
      moral_pair_ids: []
    });
    for (const profile of profiles) {
      writeCsvJsonl([{ prompt: "q", [`${profile.name}_response`]: "a" }], path.join(outputRoot, profile.name, "oeq", "responses.csv"));
      writeCsvJsonl([{ id: "1", original_post: "q", [`${profile.name}_response`]: "a" }], path.join(outputRoot, profile.name, "aita_nta_og", "responses_free.csv"));
      writeCsvJsonl([{ id: "1", original_post: "q", [`${profile.name}_label`]: "NTA" }], path.join(outputRoot, profile.name, "aita_nta_og", "responses_binary.csv"));
      writeCsvJsonl([{ id: "1", flipped_story: "q", [`${profile.name}_response`]: "a" }], path.join(outputRoot, profile.name, "aita_nta_flip", "responses_free.csv"));
      writeCsvJsonl([{ id: "1", flipped_story: "q", [`${profile.name}_label`]: "NTA" }], path.join(outputRoot, profile.name, "aita_nta_flip", "responses_binary.csv"));
    }

    expect(validateExistingJudgeInputs(outputRoot, profiles, socialDatasets, moralA, moralB).profile_order).toEqual([...PROFILE_ORDER]);
    writeCsvJsonl([{ prompt: "q", [`validation_${reference.name}`]: 1 }], path.join(outputRoot, reference.name, "oeq", "scores.csv"));
    writeJson(path.join(outputRoot, "audit_consolidated.json"), { ok: true });
    expect(judgeArtifactPaths(ctx).some((entry) => entry.endsWith("scores.csv"))).toBe(true);

    const archiveRoot = archiveExistingJudgeArtifacts(ctx, new TerminalObserver(false));
    expect(archiveRoot).toBeTruthy();
    expect(fs.existsSync(path.join(archiveRoot!, reference.name, "oeq", "scores.csv"))).toBe(true);
  });

  it("keeps LMS model helpers deterministic", () => {
    const entry = {
      identifier: "model-a",
      modelKey: "other",
      contextLength: "8192"
    };
    expect(loadedModelMatches(entry, "model-a")).toBe(true);
    expect(loadedModelMatches(entry, "missing")).toBe(false);
    expect(loadedModelContextLength(entry)).toBe(8192);
    expect(loadedModelContextLength({ contextLength: "bad" })).toBeNull();
    expect(isLocalLmStudioBaseUrl("http://127.0.0.1:1234/v1")).toBe(true);
    expect(isLocalLmStudioBaseUrl("http://203.0.113.10:1234/v1")).toBe(false);
  });

  it("detects unloaded local LMS backends before skip-lms runs", () => {
    const profile = loadProfiles(repoRoot, "config/profiles", ["Felix-V"])[0]!;
    const [backend] = resolveGenerationPool({
      repoRoot,
      base: profile.generation,
      generationModelId: "",
      generationPool: [{ modelId: "lmstudio-local-openai-v1-zai-orgglm-47-flash", workers: 1, timeoutSeconds: 900 }]
    });
    expect(localLmStudioBackendsNotReadyForSkipLms([backend!], [{ identifier: "zai-org/glm-4.7-flash", contextLength: 128000 }])).toEqual([]);
    expect(localLmStudioBackendsNotReadyForSkipLms([backend!], [{ identifier: "other-model" }])).toEqual([
      { backend, reason: "model is not loaded" }
    ]);
    expect(localLmStudioBackendsNotReadyForSkipLms([backend!], [{ identifier: "zai-org/glm-4.7-flash", contextLength: 4096 }])).toEqual([
      { backend, reason: "context_length 4096 < expected 128000" }
    ]);
  });

  it("builds LM Studio native chat bodies without changing inference hyperparameters", () => {
    const body = buildLmStudioNativeBody(
      {
        apiBaseUrl: "http://203.0.113.10:1234",
        provider: PROVIDER_LMSTUDIO,
        apiMode: "lmstudio_native_chat",
        apiKey: "lm-studio",
        apiKeyFile: "",
        model: "qwen/qwen3.6-35b-a3b",
        temperature: 0.8,
        topP: 1,
        maxTokens: 4096,
        contextLength: 16384,
        parallelism: 1,
        thinkingEnabled: false,
        reasoningEffort: "low",
        includeReasoningParameter: true,
        systemPrompt: "system",
        quotaLabel: "",
        quotaMaxRequests: 1500,
        quotaWindowSeconds: 18000
      },
      "hello"
    );

    expect(body).toMatchObject({
      model: "qwen/qwen3.6-35b-a3b",
      input: "hello",
      system_prompt: "system",
      temperature: 0.8,
      top_p: 1,
      max_output_tokens: 4096,
      context_length: 16384,
      reasoning: "off"
    });
  });

  it("omits LM Studio reasoning parameter for models that reject it", () => {
    const profile = loadProfiles(repoRoot)[0]!;
    const resolved = resolveInferenceFromModelCatalog({
      repoRoot,
      base: profile.generation,
      modelId: "lmstudio_native_liquid_lfm25_12b"
    });
    const body = buildLmStudioNativeBody(resolved, "hello");

    expect(resolved.includeReasoningParameter).toBe(false);
    expect(body).not.toHaveProperty("reasoning");
  });

  it("keeps GLM generation reasoning enabled for LM Studio backends", () => {
    const profile = loadProfiles(repoRoot)[0]!;
    const localOpenAi = resolveInferenceFromModelCatalog({
      repoRoot,
      base: profile.generation,
      modelId: "lmstudio-local-openai-v1-zai-orgglm-47-flash"
    });
    const remoteNative = resolveInferenceFromModelCatalog({
      repoRoot,
      base: profile.generation,
      modelId: "lmstudio_native_glm47_flash"
    });

    expect(localOpenAi.thinkingEnabled).toBe(true);
    expect(buildExtraBody(localOpenAi)).toMatchObject({ reasoning: { effort: "low" } });
    expect(buildLmStudioNativeBody(remoteNative, "hello")).toMatchObject({ reasoning: "on" });
  });

  it("keeps native and OpenAI judge response contracts separate", () => {
    const nativeJudge = parseJudge(repoRoot, "config/judge/liquid_lfm25_12b_native_text_parallel4.toml");
    const openAiJudge = parseJudge(repoRoot, "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml");
    const openAiModel = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_openai_liquid_lfm25_12b")!;
    const resolvedOpenAi = resolveInferenceFromModelCatalog({ repoRoot, base: openAiJudge.inference, modelId: openAiModel.id });

    expect(nativeJudge.outputMode).toBe("text_binary_label");
    expect(buildJudgeResponseFormat(nativeJudge)).toBeNull();
    expect(openAiJudge.outputMode).toBe("json_schema_binary_label");
    expect(resolvedOpenAi.apiMode).toBe("openai_chat_completions");
    expect(buildJudgeResponseFormat({ ...openAiJudge, inference: resolvedOpenAi })).toMatchObject({
      type: "json_schema"
    });
  });

  it("preserves configured judge context length for full afferition rows", () => {
    const judge = buildValidationJudge(
      repoRoot,
      "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml",
      "lmstudio_openai_liquid_lfm25_12b"
    );

    expect(judge.inference.contextLength).toBe(128000);
    expect(judge.inference.maxTokens).toBe(64);
    expect(judge.inference.temperature).toBe(0);
  });

  it("treats provider quota signals as authoritative and keeps local quota metadata informational", () => {
    expect(isProviderLimitResponse(429, "")).toBe(true);
    expect(isProviderLimitResponse(400, '{"error":"token plan exhausted"}')).toBe(true);
    expect(isProviderLimitResponse(400, '{"error":"insufficient_quota"}')).toBe(true);
    expect(isProviderLimitResponse(200, '{"usage":1501}')).toBe(false);
  });

  it("persists generation and judge checkpoints by stable unit keys", () => {
    const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mahout-bench-checkpoint-"));
    const ctx = buildRunContext(outputRoot);
    const generationKey = generationUnitKey({ profileName: "Felix-V", datasetName: "oeq", mode: "responses", rowIndex: 7 });
    const judgeKey = judgeUnitKey({ profileName: "Felix-V", datasetName: "oeq", mode: "responses", rowId: 7, metric: "validation" });

    writeGenerationResult(ctx, generationKey, { ok: true, text: "answer", label: null, attempts: 1, error: null });
    writeJudgeLabel(ctx, judgeKey, "1");

    expect(readGenerationResult(ctx, generationKey)?.text).toBe("answer");
    expect(readJudgeLabel(ctx, judgeKey)).toBe("1");
    expect(readJudgeLabel(ctx, `${judgeKey}:missing`)).toBeUndefined();
  });

  it("parses strict binary validation labels and rejects non-contract outputs", () => {
    expect(normalizeReferenceLabel("False")).toBe(0);
    expect(normalizeReferenceLabel("0.0")).toBe(0);
    expect(normalizeReferenceLabel("1.0")).toBe(1);
    expect(normalizeReferenceLabel(true)).toBe(1);
    expect(normalizeReferenceLabel("maybe")).toBeNull();
    expect(parseStrictBinaryJson('{"label":0}')).toEqual({ ok: true, label: 0, error: null });
    expect(parseStrictBinaryJson('{"label":"1"}')).toEqual({ ok: true, label: 1, error: null });
    expect(parseStrictBinaryJson("1").ok).toBe(false);
    expect(parseStrictBinaryJson('{"label":1,"why":"extra"}').ok).toBe(false);
    expect(parseStrictBinaryJson("The answer is 1").ok).toBe(false);
  });

  it("computes judge afferition similarity from valid candidate labels", () => {
    const rows = [
      { rowId: "1", dataset: "OEQ", metric: "validation", referenceLabel: 1 as const, candidateLabel: 1 as const, validParse: true },
      { rowId: "2", dataset: "OEQ", metric: "validation", referenceLabel: 0 as const, candidateLabel: 0 as const, validParse: true },
      { rowId: "3", dataset: "OEQ", metric: "validation", referenceLabel: 0 as const, candidateLabel: 1 as const, validParse: true },
      { rowId: "4", dataset: "OEQ", metric: "validation", referenceLabel: 1 as const, candidateLabel: null, validParse: false }
    ];
    const counts = computeSimilarityCounts(rows);
    const summary = computeMetricSummary("OEQ", "validation", counts);

    expect(counts).toMatchObject({ matching: 2, mismatching: 1, invalid: 1, total: 4 });
    expect(summary.validN).toBe(3);
    expect(summary.invalidRate).toBe(0.25);
    expect(summary.matchingN).toBe(2);
    expect(summary.similarity).toBeCloseTo(2 / 3);
  });

  it("builds curated Claude afferition fixtures and runs offline judge afferition artifacts", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-curated-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    const dataDir = await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const reference = loadElephantReference(repoRoot, dataDir);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-judge-validation-"));
    const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;

    expect(reference.rows.length).toBe(26);
    expect(reference.rows.every((row) => row.responseModel === "Claude" && row.referenceJudge === "GPT-4o")).toBe(true);
    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      judgeCall: async (row) => String(row.referenceLabel),
      observer: new TerminalObserver(false)
    });

    expect(entry.overallSimilarity).toBe(1);
    expect(fs.existsSync(path.join(entry.outputPath, "labels_candidate.jsonl"))).toBe(true);
    expect(fs.existsSync(path.join(entry.outputPath, "judge_afferition_metrics.json"))).toBe(true);
    expect(fs.existsSync(path.join(entry.outputPath, "judge_afferition_report.md"))).toBe(true);
    expect(readJudgeAfferitionRunState(entry.outputPath)?.status).toBe("completed");
  });

  it("runs judge afferition from a fixed margin sample and records sampling metadata", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-sampled-validation-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const sample = ensureJudgeAfferitionMarginDataset(fixtureRoot, 0.20);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-sampled-validation-"));
    const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;

    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir: sample.dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      judgeCall: async (row) => String(row.referenceLabel),
      observer: new TerminalObserver(false),
      afferitionSampling: sample.sampling
    });

    const metrics = readJsonFile<Record<string, unknown>>(path.join(entry.outputPath, "judge_afferition_metrics.json"));
    const state = readJudgeAfferitionRunState(entry.outputPath);
    const report = fs.readFileSync(path.join(entry.outputPath, "judge_afferition_report.md"), "utf8");
    expect(entry.afferitionSampling?.kind).toBe("margin");
    expect(entry.afferitionSampling?.marginLabel).toBe("20pp");
    expect((metrics.afferition_sampling as { marginLabel: string }).marginLabel).toBe("20pp");
    expect(state?.afferitionSampling?.marginLabel).toBe("20pp");
    expect(report).toContain("Sampling: 20pp margem de erro");
  });

  it("retries judge afferition labels until success and records attempts", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-retry-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    const dataDir = await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-judge-retry-"));
    const baseModel = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;
    const model = { ...baseModel, parallelism: 1 };
    let attempts = 0;

    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      limit: 1,
      judgeCall: async (row) => {
        attempts += 1;
        if (attempts < 10) {
          throw new Error("temporary provider failure");
        }
        return String(row.referenceLabel);
      },
      retryDelayMs: () => 0,
      observer: new TerminalObserver(false)
    });

    const labels = fs.readFileSync(path.join(entry.outputPath, "labels_candidate.jsonl"), "utf8").trim().split("\n").map((line) => JSON.parse(line));
    expect(attempts).toBe(10);
    expect(labels).toHaveLength(1);
    expect(labels[0].attempts).toBe(10);
    expect(readJudgeAfferitionRunState(entry.outputPath)?.status).toBe("completed");
  });

  it("records persistent judge afferition parse failures as invalid rows", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-invalid-parse-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    const dataDir = await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-judge-invalid-parse-"));
    const baseModel = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;
    const model = { ...baseModel, parallelism: 1 };
    let attempts = 0;

    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      limit: 1,
      judgeCall: async () => {
        attempts += 1;
        return "score: 1";
      },
      retryDelayMs: () => 0,
      observer: new TerminalObserver(false)
    });

    const labels = fs.readFileSync(path.join(entry.outputPath, "labels_candidate.jsonl"), "utf8").trim().split("\n").map((line) => JSON.parse(line));
    const failures = fs.readFileSync(path.join(entry.outputPath, "failures_invalid_outputs.jsonl"), "utf8").trim().split("\n").map((line) => JSON.parse(line));
    expect(attempts).toBe(10);
    expect(labels).toHaveLength(1);
    expect(labels[0]).toMatchObject({
      candidateLabel: null,
      validParse: false,
      candidateRawOutput: "score: 1",
      parseError: "missing leading 0/1 judge label",
      attempts: 10
    });
    expect(failures).toHaveLength(1);
    expect(entry.overallSimilarity).toBeNull();
    expect(readJudgeAfferitionRunState(entry.outputPath)?.status).toBe("completed");
  });

  it("records judge afferition context overflow as invalid rows", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-context-overflow-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    const dataDir = await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-judge-context-overflow-"));
    const baseModel = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_openai_liquid_lfm25_12b")!;
    const model = { ...baseModel, parallelism: 1 };
    let attempts = 0;

    const entry = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_openai_json_parallel4.toml",
      limit: 1,
      judgeCall: async () => {
        attempts += 1;
        throw new Error('chat completion failed: 400 Bad Request: {"error":"Context size has been exceeded."}');
      },
      retryDelayMs: () => 0,
      observer: new TerminalObserver(false)
    });

    const labels = fs.readFileSync(path.join(entry.outputPath, "labels_candidate.jsonl"), "utf8").trim().split("\n").map((line) => JSON.parse(line));
    expect(attempts).toBe(1);
    expect(labels).toHaveLength(1);
    expect(labels[0]).toMatchObject({
      candidateLabel: null,
      validParse: false,
      candidateRawOutput: "",
      attempts: 1
    });
    expect(labels[0].parseError).toMatch(/Context size has been exceeded/);
    expect(entry.overallSimilarity).toBeNull();
    expect(readJudgeAfferitionRunState(entry.outputPath)?.status).toBe("completed");
  });

  it("marks judge afferition failed after retries and resumes from processed labels", async () => {
    const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-resume-root-"));
    const fullResultsDir = path.join(fixtureRoot, "datasets", "full_results");
    writeFullResultFixtures(fullResultsDir);
    const dataDir = await ensureClaudeSocialJudgeAfferition(fixtureRoot);
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-judge-resume-"));
    const outputPath = path.join(outputBase, "judge_afferition_resume_case");
    const baseModel = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;
    const model = { ...baseModel, parallelism: 1 };
    let firstRunCalls = 0;

    await expect(runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      limit: 3,
      outputPath,
      judgeCall: async (row) => {
        firstRunCalls += 1;
        if (firstRunCalls === 1) {
          return String(row.referenceLabel);
        }
        throw new Error("persistent provider failure");
      },
      retryDelayMs: () => 0,
      observer: new TerminalObserver(false)
    })).rejects.toThrow(/Judge afferition failed/);

    const failedState = readJudgeAfferitionRunState(outputPath);
    expect(failedState?.status).toBe("failed");
    expect(failedState?.completed).toBe(1);
    expect(failedState?.lastError?.attempt).toBe(10);

    const resumed = await runJudgeValidation({
      repoRoot,
      outputBase,
      dataDir,
      model,
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      limit: 3,
      outputPath,
      judgeCall: async (row) => String(row.referenceLabel),
      retryDelayMs: () => 0,
      observer: new TerminalObserver(false)
    });

    const labels = fs.readFileSync(path.join(resumed.outputPath, "labels_candidate.jsonl"), "utf8").trim().split("\n").map((line) => JSON.parse(line));
    const rowIds = labels.map((row) => row.rowId);
    expect(labels).toHaveLength(3);
    expect(new Set(rowIds).size).toBe(3);
    expect(readJudgeAfferitionRunState(outputPath)?.status).toBe("completed");
  });

  it("lists only incomplete judge afferition run states", () => {
    const outputBase = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-incomplete-runs-"));
    const failedPath = path.join(outputBase, "failed");
    const runningPath = path.join(outputBase, "running");
    const completedPath = path.join(outputBase, "completed");

    const failed = buildInitialJudgeAfferitionRunState({
      modelId: "judge-a",
      model: "model-a",
      judgeConfigPath: "config/judge/a.toml",
      outputPath: failedPath,
      total: 10,
      completed: 4,
      lastProcessedRowId: "row-4"
    });
    writeJudgeAfferitionRunState(failedPath, {
      ...failed,
      status: "failed",
      lastError: { rowId: "row-5", metric: "framing", attempt: 10, message: "boom" }
    });
    writeJudgeAfferitionRunState(runningPath, buildInitialJudgeAfferitionRunState({
      modelId: "judge-b",
      model: "model-b",
      judgeConfigPath: "config/judge/b.toml",
      outputPath: runningPath,
      total: 10,
      completed: 2,
      lastProcessedRowId: "row-2"
    }));
    writeJudgeAfferitionRunState(completedPath, {
      ...buildInitialJudgeAfferitionRunState({
        modelId: "judge-c",
        model: "model-c",
        judgeConfigPath: "config/judge/c.toml",
        outputPath: completedPath,
        total: 10,
        completed: 10,
        lastProcessedRowId: "row-10"
      }),
      status: "completed",
      remaining: 0,
      completedAt: new Date().toISOString()
    });

    const incomplete = listIncompleteJudgeAfferitionRuns(outputBase);
    expect(incomplete.map((run) => run.state.modelId).sort()).toEqual(["judge-a", "judge-b"]);
  });

  it("persists judge afferition registry entries without PASS/FAIL gating", () => {
    const registryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "elephant-registry-"));
    const model = loadModelCatalog(repoRoot).models.find((entry) => entry.id === "lmstudio_native_liquid_lfm25_12b")!;
    const empty = loadJudgeValidationRegistry(registryRoot);
    const entry = {
      modelId: model.id,
      model: model.model,
      label: model.label,
      reference: "ELEPHANT GPT-4o labels over Claude responses",
      validatedAt: new Date().toISOString(),
      judgeConfigPath: "config/judge/liquid_lfm25_12b_native_text_parallel4.toml",
      outputPath: "outputs/judge_afferition/report",
      dataFingerprint: "abc",
      overallSimilarity: 0.82,
      metrics: []
    };

    const updated = upsertJudgeValidation(empty, entry);
    saveJudgeValidationRegistry(registryRoot, updated);

    expect(findUsableJudgeValidation(loadJudgeValidationRegistry(registryRoot), model)?.overallSimilarity).toBe(0.82);
    expect(formatValidationSummary(entry)).toContain("status=aferido (full)");
    expect(formatValidationSummary({
      ...entry,
      afferitionSampling: {
        kind: "margin",
        confidence: 0.95,
        marginOfError: 0.08,
        marginLabel: "8pp",
        sampleBy: "dataset_metric",
        seed: 20260513,
        datasetPath: "datasets/judge_afferition/ME/8pp",
        manifestPath: "datasets/judge_afferition/ME/8pp/manifest.json",
        sourcePath: "datasets/judge_afferition/claude_social",
        sourceFingerprint: "abc",
        fullTotal: 27896,
        sampleTotal: 1820
      }
    })).toContain("status=aferido (8pp margem de erro)");
  });
});

function buildRunContext(outputRoot: string): RunContext {
  return {
    repoRoot,
    dataRoot: repoRoot,
    outputRoot,
    eventsPath: path.join(outputRoot, "run_events.jsonl"),
    rawGenerationPath: path.join(outputRoot, "raw_generation.jsonl"),
    rawJudgePath: path.join(outputRoot, "raw_judge.jsonl"),
    quarantinePath: path.join(outputRoot, "quarantine.jsonl"),
    providerEventsPath: path.join(outputRoot, "provider_events.jsonl"),
    generationCheckpointPath: path.join(outputRoot, "generation_checkpoint.json"),
    judgeCheckpointPath: path.join(outputRoot, "judge_checkpoint.json"),
    profilesRoot: "config/profiles",
    judgeConfigPath: "config/judge/juiz.toml",
    profileNames: [],
    benchmarkName: "test benchmark",
    generationModelId: "",
    generationPool: [],
    judgeModelId: "",
    judgePool: [],
    marginOfError: null
  };
}

function datasetConfig(name: string, promptColumn: string, task: string): DatasetConfig {
  return {
    name,
    enabled: true,
    file: `${name}.csv`,
    promptColumn,
    task,
    aitaBinary: false,
    baseline: 0.5,
    promptPrefix: "",
    promptSuffix: ""
  };
}

function commonIdsForTest(left: Array<Record<string, unknown>>, right: Array<Record<string, unknown>>): number {
  const rightIds = new Set(right.map((row) => String(row.id ?? "")));
  return left.map((row) => String(row.id ?? "")).filter((id) => rightIds.has(id)).length;
}

function buildAuditFixture(outputRoot: string): AuditConsolidated {
  return buildAuditFixtureForProfiles(outputRoot, [...PROFILE_ORDER]);
}

function buildAuditFixtureForProfiles(outputRoot: string, profiles: string[]): AuditConsolidated {
  const socialSummaries: AuditConsolidated["social_summaries"] = [];
  for (const profile of profiles) {
    for (const [title, [datasetName, metric]] of Object.entries(PAPER_SECTION_KEYS)) {
      void title;
      let summary = socialSummaries.find((entry) => entry.profile === profile && entry.dataset === datasetName);
      if (!summary) {
        summary = {
          profile,
          dataset: datasetName,
          file: path.join(outputRoot, profile, datasetName, "scores.csv"),
          rows: 10,
          metrics: {}
        };
        socialSummaries.push(summary);
      }
      summary.metrics[metric] = {
        total_n: 10,
        positive_n: 5,
        valid_n: 10,
        invalid_n: 0,
        rate_valid_only: 0.5,
        paper_rate: 0.5,
        ci95: 0.1,
        baseline: datasetName === "ss" ? 0.5 : 0.2,
        paper_score: datasetName === "ss" ? 0 : 0.3
      };
    }
  }

  const moralSummaries: AuditConsolidated["moral_summaries"] = profiles.map((profile) => ({
    created_at: new Date().toISOString(),
    profile,
    side_datasets: {
      og: "aita_nta_og",
      flip: "aita_nta_flip"
    },
    metrics: {
      both_YTA_rate: 0.2,
      both_NTA_rate: 0.4,
      flipped_NTA_og_YTA_rate: 0.1,
      flipped_YTA_og_NTA_rate: 0.1,
      refused_rate: 0.2,
      counts: {
        both_YTA: 2,
        both_NTA: 4,
        flipped_NTA_og_YTA: 1,
        flipped_YTA_og_NTA: 1,
        refused: 2,
        denominator_pairs: 10
      }
    }
  }));

  const doubleSidedSummaries: AuditConsolidated["double_sided_summaries"] = profiles.map((profile) => ({
    profile,
    metrics: Object.fromEntries(
      ["validation", "indirectness", "framing"].map((metric) => [
        metric,
        {
          pair_population: 10,
          valid_n_pairs: 10,
          invalid_n_pairs: 0,
          both_1_count: 4,
          both_1_rate: 0.4,
          both_1_rate_valid: 0.4
        }
      ])
    )
  }));

  for (const title of Object.keys(DOUBLE_SIDED_SECTION_KEYS)) {
    expect(title).toBeTruthy();
  }

  return {
    created_at: new Date().toISOString(),
    output_root: outputRoot,
    sample_manifest: null,
    social_summaries: socialSummaries,
    moral_summaries: moralSummaries,
    double_sided_summaries: doubleSidedSummaries
  };
}

function writeFullResultFixtures(dataDir: string): void {
  const socialRows = [
    {
      prompt: "p1",
      Claude: "claude r1",
      validation_Claude: 1,
      indirectness_Claude: 0,
      framing_Claude: 1
    },
    {
      prompt: "p2",
      Claude: "claude r2",
      validation_Claude: 0,
      indirectness_Claude: 1,
      framing_Claude: 0
    }
  ];
  writeCsvJsonl(socialRows, path.join(dataDir, "OEQ_full_results.csv"));
  writeCsvJsonl(
    socialRows.map((row) => ({ Claude: row.Claude, validation_Claude: row.validation_Claude, indirectness_Claude: row.indirectness_Claude, framing_Claude: row.framing_Claude })),
    path.join(dataDir, "AITA-YTA_full_results.csv")
  );
  writeCsvJsonl(
    socialRows.map((row) => ({ sentence: row.prompt, Claude: row.Claude, validation_Claude: row.validation_Claude, indirectness_Claude: row.indirectness_Claude, framing_Claude: row.framing_Claude })),
    path.join(dataDir, "SS_full_results.csv")
  );
  writeCsvJsonl(
    socialRows.map((row) => ({ Claude: row.Claude, validation_Claude: row.validation_Claude, indirectness_Claude: row.indirectness_Claude, framing_Claude: row.framing_Claude })),
    path.join(dataDir, "elephant_full_results", "AITA-NTA-OG_full_results.csv")
  );
  writeCsvJsonl(
    socialRows.map((row) => ({ Claude: row.Claude, validation_Claude: row.validation_Claude, indirectness_Claude: row.indirectness_Claude, framing_Claude: row.framing_Claude })),
    path.join(dataDir, "elephant_full_results", "AITA-NTA-FLIP_flipped_full_results.csv")
  );
  writeCsvJsonl(socialRows.map((row) => ({ prompt: row.prompt })), path.join(dataDir, "OEQ.csv"));
  writeCsvJsonl(socialRows.map((row) => ({ prompt: row.prompt })), path.join(dataDir, "AITA-YTA.csv"));
  writeCsvJsonl(socialRows.map((row) => ({ prompt: row.prompt })), path.join(dataDir, "SS.csv"));
  writeCsvJsonl(socialRows.map((row) => ({ original_post: row.prompt })), path.join(dataDir, "AITA-NTA-OG.csv"));
  writeCsvJsonl(socialRows.map((row) => ({ original_post: row.prompt, flipped_story: row.prompt })), path.join(dataDir, "AITA-NTA-FLIP.csv"));
}
