import path from "node:path";

import {
  API_MODE_OPENAI_CHAT_COMPLETIONS,
  BINARY_SCORE_PARSE,
  JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL,
  MORAL_A_TASK,
  MORAL_B_TASK,
  PROVIDER_OPENAI_COMPATIBLE,
  SOCIAL_TASK,
  YTA_NTA_PARSE,
  type AuditConsolidated,
  type BenchmarkArgs,
  type DatasetConfig,
  type GenerationPoolBackend,
  type InferenceConfig,
  type JudgeConfig,
  type ProfileConfig,
  type RunContext,
  type SampleManifest
} from "../contracts/autobench.js";
import {
  datasetPath,
  enabledDatasets,
  loadProfiles,
  parseJudge,
  parseProfileOrder,
  resolveInferenceFromModelCatalog,
  validateContract
} from "../config/loadConfig.js";
import { resolveGenerationPool, resolveJudgePool } from "../config/generationPool.js";
import { fixedMoralGeneration, canonicalMoralGeneration } from "../generation/moralGeneration.js";
import { canonicalSocialGeneration, fixedSocialGeneration } from "../generation/socialGeneration.js";
import {
  buildExtraBody,
  buildJudgeResponseFormat,
  extractAssistantText,
  parseJsonSchemaBinaryLabel,
  parseJudgeOutput,
  parseOutput
} from "../inference/chatClient.js";
import { aggregateTokenUsageFromJsonl, combineTokenUsageSummaries } from "../inference/tokenUsage.js";
import { archiveExistingJudgeArtifacts, plannedJudgeOnlyUnits, validateExistingJudgeInputs } from "../judging/judgeArtifacts.js";
import { executeJudgePhase } from "../judging/judgeScoring.js";
import { ensureDir, appendJsonl, localRunStamp, readCsvFile, utcNowIso, writeJson, writeTextFile } from "../io/filesystem.js";
import { writeResults } from "../reporting/resultsReport.js";
import { commonMoralIds, buildSampleManifest, plannedOverallUnits, sampleTargetN, shuffleIds, socialTargetMap } from "../sampling/samplePlanner.js";
import { defaultPackageRoot, resolveDataRoot } from "../runtime/paths.js";
import { TerminalObserver, requireInteractiveTty } from "../runtime/terminalObserver.js";
import { BenchmarkAbort } from "./benchmarkAbort.js";
import { createRunContext, logEvent, renderError } from "./runContext.js";
import { reconstructResumeGenerationState } from "./resumeState.js";

export const ELEPHANT_ACKNOWLEDGEMENT =
  "Thanks to Myra Cheng and the ELEPHANT team for the data and procedure.";

/**
 * Runs the complete benchmark or judge-only flow from parsed CLI arguments.
 */
export async function runBenchmark(args: BenchmarkArgs): Promise<number> {
  requireInteractiveTty();
  const ctx = createRunContext(args);
  process.stdout.write(`${ELEPHANT_ACKNOWLEDGEMENT}\n`);
  if (!args.judgeOnly) {
    ensureDir(ctx.outputRoot);
  }

  const setup = loadBenchmarkSetup(ctx);
  const plan = buildExecutionPlan(ctx, setup.profiles, setup.socialDatasets, setup.moralA, setup.moralB);
  let manifest: SampleManifest | null = null;
  let overallTotal = plannedOverallUnits(setup.profiles.length, plan.socialTargets, plan.moralTargetN);
  if (args.judgeOnly) {
    manifest = validateExistingJudgeInputs(ctx.outputRoot, setup.profiles, setup.socialDatasets, setup.moralA, setup.moralB);
    overallTotal = plannedJudgeOnlyUnits(ctx, setup.profiles, setup.socialDatasets, setup.moralA, setup.moralB);
  }

  const observer = new TerminalObserver(true);
  if (!observer.isEnabled()) {
    throw new Error("Terminal observer requires an interactive TTY for real benchmark runs.");
  }
  observer.configureRun(ctx.outputRoot);
  observer.start(overallTotal);
  logRunStarted(ctx, args, setup, overallTotal, observer);

  try {
    manifest = await runGenerationIfNeeded(args, ctx, setup, plan, manifest, observer);
    const audit = await runJudgeAndReport(args, ctx, setup, manifest, observer);
    logEvent(ctx, "run_completed", observer);
    observer.stop();
    process.stdout.write(`Autobench complete: ${ctx.outputRoot}\n`);
    void audit;
    return 0;
  } catch (error) {
    return handleRunError(ctx, observer, error);
  } finally {
    observer.stop();
  }
}

/**
 * Validates TOMLs, dataset contracts, and planned sample sizes without network calls.
 */
export function validateConfigCli(args: BenchmarkArgs): number {
  const ctx = createRunContext({ ...args, judgeOnly: false });
  const profiles = loadProfiles(ctx.repoRoot, args.profilesRoot, resolveProfileNames(ctx.repoRoot, args.profilesRoot, args.profiles));
  const judge = resolveJudge(ctx, parseJudge(ctx.repoRoot, args.judgeConfig));
  applyRuntimeGenerationConfig(ctx, profiles, args.marginOfError);
  validateContract(ctx.repoRoot, profiles);
  const reference = profiles[0]!;
  const generationPool = resolveGenerationPool({
    repoRoot: ctx.repoRoot,
    base: reference.generation,
    generationModelId: ctx.generationModelId,
    generationPool: ctx.generationPool
  });
  const judgePool = resolveJudgePool({
    repoRoot: ctx.repoRoot,
    base: judge.inference,
    judgeModelId: ctx.judgeModelId,
    judgePool: ctx.judgePool
  });

  process.stdout.write("Config validation passed.\n");
  process.stdout.write(`Benchmark model: ${reference.generation.model}\n`);
  process.stdout.write(`Generation pool: ${generationPool.map((backend) => `${backend.modelId}x${backend.workers}@${backend.timeoutSeconds}s`).join(", ")}\n`);
  process.stdout.write(`Judge model: ${judge.inference.model}\n`);
  process.stdout.write(`Judge pool: ${judgePool.map((backend) => `${backend.modelId}x${backend.workers}@${backend.timeoutSeconds}s`).join(", ")}\n`);
  process.stdout.write(`Judge output mode: ${judge.outputMode}\n`);
  process.stdout.write(`Profiles: ${profiles.map((profile) => profile.name).join(", ")}\n`);

  for (const dataset of enabledDatasets(reference)) {
    const population = readCsvFile(datasetPath(ctx.repoRoot, reference, dataset)).length;
    process.stdout.write(
      `Dataset ${dataset.name}: task=${dataset.task}, population=${population}, target_n=${sampleTargetN(population, reference.sampling)}\n`
    );
  }

  const { moralA, moralB } = requireMoralDatasets(reference);
  const commonIds = commonMoralIds(
    readCsvFile(datasetPath(ctx.repoRoot, reference, moralA)),
    readCsvFile(datasetPath(ctx.repoRoot, reference, moralB))
  );
  process.stdout.write(`Moral common id pairs: ${commonIds.length}, target_n=${sampleTargetN(commonIds.length, reference.sampling)}\n`);
  return 0;
}

/**
 * Runs parser, extraction, sampling, and judge-format assertions without network calls.
 */
export function runSelfTest(): number {
  const assert = (condition: boolean, message: string): void => {
    if (!condition) {
      throw new Error(message);
    }
  };

  assert(parseOutput("NTA. A repair shop is a workplace", YTA_NTA_PARSE).label === "NTA", "YTA/NTA parse failed");
  assert(parseOutput("**YTA**", YTA_NTA_PARSE).label === "YTA", "Markdown YTA parse failed");
  assert(parseOutput("I think NTA", YTA_NTA_PARSE).ok === false, "Loose YTA/NTA parse should fail");
  assert(parseOutput("1 - emotionally validating", BINARY_SCORE_PARSE).label === "1", "Binary parse failed");
  assert(parseOutput("score: 1", BINARY_SCORE_PARSE).ok === false, "Loose binary parse should fail");
  assert(parseJudgeOutput('{"label":"1"}', "json_schema_binary_label").label === "1", "JSON schema 1 parse failed");
  assert(parseJudgeOutput('{"label":"0"}', "json_schema_binary_label").label === "0", "JSON schema 0 parse failed");
  assert(parseJudgeOutput('{"label":"2"}', "json_schema_binary_label").ok === false, "JSON schema invalid label should fail");
  assert(parseJsonSchemaBinaryLabel('{"nope":"1"}').ok === false, "JSON schema missing label should fail");
  assert(buildJudgeResponseFormat(buildSelfTestJudge()) !== null, "JSON schema response format should be enabled");
  assert(!Object.prototype.hasOwnProperty.call(buildExtraBody(buildSelfTestInference(false)), "reasoning"), "Judge extra body should not enable reasoning");

  const extracted = extractAssistantText({
    content: "",
    model_extra: {
      reasoning_content: "thinking...\nFinal answer: **NTA**"
    }
  });
  assert(extracted.thinkingDetected, "Reasoning field should be detected");
  assert(parseOutput(extracted.text, YTA_NTA_PARSE).label === "NTA", "Final answer extraction failed");
  assert(sampleTargetN(3027) === 144, "sampleTargetN mismatch");
  assert([...shuffleIds(["a", "b", "c"], 42)].sort().join(",") === "a,b,c", "shuffleIds set changed");
  process.stdout.write("Self-test passed.\n");
  return 0;
}

/**
 * Writes minimal local artifacts to verify filesystem/report plumbing without network calls.
 */
export function runDrySmoke(): number {
  runSelfTest();
  const smokeRoot = path.join("/tmp", `autobench_smoke_${localRunStamp()}`);
  ensureDir(smokeRoot);
  const ctx: RunContext = {
    repoRoot: defaultPackageRoot(),
    dataRoot: resolveDataRoot(),
    outputRoot: smokeRoot,
    eventsPath: path.join(smokeRoot, "run_events.jsonl"),
    rawGenerationPath: path.join(smokeRoot, "raw_generation.jsonl"),
    rawJudgePath: path.join(smokeRoot, "raw_judge.jsonl"),
    quarantinePath: path.join(smokeRoot, "quarantine.jsonl"),
    providerEventsPath: path.join(smokeRoot, "provider_events.jsonl"),
    generationCheckpointPath: path.join(smokeRoot, "generation_checkpoint.json"),
    judgeCheckpointPath: path.join(smokeRoot, "judge_checkpoint.json"),
    profilesRoot: "config/profiles",
    judgeConfigPath: "config/judge/juiz.toml",
    profileNames: [],
    benchmarkName: "dry_smoke",
    generationModelId: "",
    generationPool: [],
    judgeModelId: "",
    judgePool: [],
    marginOfError: null,
    resumeMode: null
  };
  appendJsonl(ctx.eventsPath, { timestamp: utcNowIso(), event: "dry_smoke" });
  writeJson(path.join(smokeRoot, "audit_consolidated.json"), {
    created_at: utcNowIso(),
    mode: "dry_smoke",
    network: "not used"
  });
  writeTextFile(path.join(smokeRoot, "RESULTS.md"), "# Dry Smoke\n\nNo network calls were made.\n");
  process.stdout.write(`Dry smoke passed: ${smokeRoot}\n`);
  return 0;
}

function loadBenchmarkSetup(ctx: RunContext): {
  profiles: ProfileConfig[];
  generationPool: GenerationPoolBackend[];
  judgePool: GenerationPoolBackend[];
  judge: JudgeConfig;
  reference: ProfileConfig;
  socialDatasets: DatasetConfig[];
  moralA: DatasetConfig;
  moralB: DatasetConfig;
} {
  const profiles = loadProfiles(ctx.repoRoot, ctx.profilesRoot, resolveProfileNames(ctx.repoRoot, ctx.profilesRoot, ctx.profileNames));
  const judge = resolveJudge(ctx, parseJudge(ctx.repoRoot, ctx.judgeConfigPath));
  applyRuntimeGenerationConfig(ctx, profiles, ctx.marginOfError);
  validateContract(ctx.repoRoot, profiles);
  const reference = profiles[0]!;
  const generationPool = resolveGenerationPool({
    repoRoot: ctx.repoRoot,
    base: reference.generation,
    generationModelId: ctx.generationModelId,
    generationPool: ctx.generationPool
  });
  const judgePool = resolveJudgePool({
    repoRoot: ctx.repoRoot,
    base: judge.inference,
    judgeModelId: ctx.judgeModelId,
    judgePool: ctx.judgePool
  });
  judge.inference = judgePool[0]!.inference;
  for (const profile of profiles) {
    profile.generation = generationPool[0]!.inference;
  }
  const socialDatasets = enabledDatasets(reference).filter((dataset) => dataset.task === SOCIAL_TASK);
  const { moralA, moralB } = requireMoralDatasets(reference);
  return { profiles, generationPool, judgePool, judge, reference, socialDatasets, moralA, moralB };
}

function buildExecutionPlan(
  ctx: RunContext,
  profiles: ProfileConfig[],
  socialDatasets: DatasetConfig[],
  moralA: DatasetConfig,
  moralB: DatasetConfig
): {
  datasetPopulations: Record<string, number>;
  socialTargets: Record<string, number>;
  moralPopulation: number;
  moralTargetN: number;
} {
  const reference = profiles[0]!;
  const enabled = enabledDatasets(reference);
  const datasetPopulations = Object.fromEntries(
    enabled.map((dataset) => [dataset.name, readCsvFile(datasetPath(ctx.repoRoot, reference, dataset)).length])
  );
  const socialTargets = socialTargetMap(reference, socialDatasets, datasetPopulations);
  const moralPopulation = commonMoralIds(
    readCsvFile(datasetPath(ctx.repoRoot, reference, moralA)),
    readCsvFile(datasetPath(ctx.repoRoot, reference, moralB))
  ).length;
  return {
    datasetPopulations,
    socialTargets,
    moralPopulation,
    moralTargetN: sampleTargetN(moralPopulation, reference.sampling)
  };
}

async function runGenerationIfNeeded(
  args: BenchmarkArgs,
  ctx: RunContext,
  setup: ReturnType<typeof loadBenchmarkSetup>,
  plan: ReturnType<typeof buildExecutionPlan>,
  manifest: SampleManifest | null,
  observer: TerminalObserver
): Promise<SampleManifest | null> {
  if (args.judgeOnly) {
    archiveExistingJudgeArtifacts(ctx, observer);
    return manifest;
  }
  if (ctx.resumeMode) {
    const resumed = reconstructResumeGenerationState({
      ctx,
      mode: ctx.resumeMode,
      profiles: setup.profiles,
      socialDatasets: setup.socialDatasets,
      moralA: setup.moralA,
      moralB: setup.moralB,
      socialTargets: plan.socialTargets,
      moralTargetN: plan.moralTargetN,
      datasetPopulations: plan.datasetPopulations,
      generationPool: setup.generationPool,
      judgePool: setup.judgePool,
      judgeInference: setup.judge.inference
    });
    if (resumed) {
      observer.advanceOverall(resumed.generatedUnits);
      logEvent(ctx, "resume_generation_reconstructed", observer, {
        mode: ctx.resumeMode,
        generated_units: resumed.generatedUnits,
        generation_checkpoint_results: resumed.report.generation_checkpoint_results,
        final_generation_failures: resumed.report.final_generation_failures.length
      });
      writeJson(path.join(ctx.outputRoot, "sample_manifest.json"), resumed.manifest);
      return resumed.manifest;
    }
    logEvent(ctx, "resume_generation_partial_fallback", observer, { mode: ctx.resumeMode });
  }
  const generationResult = await runGenerationPhase(ctx, setup, plan, observer);
  const generatedManifest = buildSampleManifest({
    ctx,
    referenceProfile: setup.reference,
    profileOrder: setup.profiles.map((profile) => profile.name),
    generationPool: setup.generationPool,
    judgePool: setup.judgePool,
    judgeInference: setup.judge.inference,
    socialIndices: generationResult.socialIndices,
    datasetPopulations: plan.datasetPopulations,
    moralIds: generationResult.moralIds
  });
  writeJson(path.join(ctx.outputRoot, "sample_manifest.json"), generatedManifest);
  return generatedManifest;
}

async function runGenerationPhase(
  ctx: RunContext,
  setup: ReturnType<typeof loadBenchmarkSetup>,
  plan: ReturnType<typeof buildExecutionPlan>,
  observer: TerminalObserver
): Promise<{ socialIndices: Record<string, number[]>; moralIds: string[] }> {
  const socialIndices: Record<string, number[]> = {};
  for (const dataset of setup.socialDatasets) {
    socialIndices[dataset.name] = (
      await canonicalSocialGeneration({
        ctx,
        profile: setup.reference,
        dataset,
        targetN: plan.socialTargets[dataset.name] ?? 0,
        generationPool: setup.generationPool,
        observer
      })
    ).acceptedIndices;
  }
  const moralIds = await canonicalMoralGeneration({
    ctx,
    profile: setup.reference,
    aDataset: setup.moralA,
    bDataset: setup.moralB,
    targetN: plan.moralTargetN,
    generationPool: setup.generationPool,
    observer
  });
  for (const profile of setup.profiles.slice(1)) {
    await runFixedProfileGeneration(ctx, setup, profile, socialIndices, moralIds, observer);
  }
  return { socialIndices, moralIds };
}

async function runFixedProfileGeneration(
  ctx: RunContext,
  setup: ReturnType<typeof loadBenchmarkSetup>,
  profile: ProfileConfig,
  socialIndices: Record<string, number[]>,
  moralIds: string[],
  observer: TerminalObserver
): Promise<void> {
  for (const dataset of setup.socialDatasets) {
    await fixedSocialGeneration({
      ctx,
      profile,
      dataset,
      acceptedIndices: socialIndices[dataset.name] ?? [],
      generationPool: setup.generationPool,
      observer
    });
  }
  await fixedMoralGeneration({
    ctx,
    profile,
    aDataset: setup.moralA,
    bDataset: setup.moralB,
    acceptedIds: moralIds,
    generationPool: setup.generationPool,
    observer
  });
}

async function runJudgeAndReport(
  _args: BenchmarkArgs,
  ctx: RunContext,
  setup: ReturnType<typeof loadBenchmarkSetup>,
  manifest: SampleManifest | null,
  observer: TerminalObserver
): Promise<AuditConsolidated> {
  const { socialSummaries, moralSummaries, doubleSidedSummaries } = await executeJudgePhase({
    ctx,
    judge: setup.judge,
    judgePool: setup.judgePool,
    profiles: setup.profiles,
    socialDatasets: setup.socialDatasets,
    moralA: setup.moralA,
    moralB: setup.moralB,
    observer
  });
  const audit: AuditConsolidated = {
    created_at: utcNowIso(),
    output_root: ctx.outputRoot,
    sample_manifest: manifest,
    social_summaries: socialSummaries,
    moral_summaries: moralSummaries,
    double_sided_summaries: doubleSidedSummaries,
    token_usage: buildBenchmarkTokenUsage(ctx)
  };
  writeJson(path.join(ctx.outputRoot, "audit_consolidated.json"), audit);
  writeResults(ctx, audit);
  return audit;
}

function buildBenchmarkTokenUsage(ctx: RunContext): AuditConsolidated["token_usage"] {
  const generation = aggregateTokenUsageFromJsonl(ctx.rawGenerationPath);
  const judge = aggregateTokenUsageFromJsonl(ctx.rawJudgePath);
  return {
    generation,
    judge,
    total: combineTokenUsageSummaries(generation, judge)
  };
}

function requireMoralDatasets(profile: ProfileConfig): { moralA: DatasetConfig; moralB: DatasetConfig } {
  const moralA = enabledDatasets(profile).find((dataset) => dataset.task === MORAL_A_TASK);
  const moralB = enabledDatasets(profile).find((dataset) => dataset.task === MORAL_B_TASK);
  if (!moralA || !moralB) {
    throw new Error("Missing moral datasets in config");
  }
  return { moralA, moralB };
}

function logRunStarted(
  ctx: RunContext,
  args: BenchmarkArgs,
  setup: ReturnType<typeof loadBenchmarkSetup>,
  overallTotal: number,
  observer: TerminalObserver
): void {
  logEvent(ctx, "run_started", observer, {
    output_root: ctx.outputRoot,
    benchmark_name: ctx.benchmarkName,
    run_mode: args.judgeOnly ? "judge_only" : "full",
    profiles_root: ctx.profilesRoot,
    profiles: setup.profiles.map((profile) => profile.name).join(","),
    judge_config_path: ctx.judgeConfigPath,
    margin_of_error: ctx.marginOfError,
    benchmark_model: setup.reference.generation.model,
    generation_pool: setup.generationPool.map((backend) => ({
      backend_id: backend.backendId,
      model_id: backend.modelId,
      workers: backend.workers,
      timeout_seconds: backend.timeoutSeconds,
      provider: backend.inference.provider,
      model: backend.inference.model
    })),
    judge_model: setup.judge.inference.model,
    judge_pool: setup.judgePool.map((backend) => ({
      backend_id: backend.backendId,
      model_id: backend.modelId,
      workers: backend.workers,
      timeout_seconds: backend.timeoutSeconds,
      provider: backend.inference.provider,
      model: backend.inference.model
    })),
    judge_output_mode: setup.judge.outputMode,
    overall_total: overallTotal
  });
}

function resolveProfileNames(repoRoot: string, profilesRoot: string, selectedProfiles: string[]): string[] {
  if (selectedProfiles.length > 0) {
    return selectedProfiles;
  }
  return parseProfileOrder(repoRoot, profilesRoot).order;
}

function applyRuntimeGenerationConfig(ctx: RunContext, profiles: ProfileConfig[], marginOfError: number | null): void {
  for (const profile of profiles) {
    if (ctx.generationModelId) {
      profile.generation = resolveInferenceFromModelCatalog({
        repoRoot: ctx.repoRoot,
        base: profile.generation,
        modelId: ctx.generationModelId
      });
    }
    if (marginOfError !== null) {
      profile.sampling.marginOfError = marginOfError;
    }
  }
}

function resolveJudge(ctx: RunContext, judge: JudgeConfig): JudgeConfig {
  if (!ctx.judgeModelId) {
    return judge;
  }
  return {
    ...judge,
    inference: resolveInferenceFromModelCatalog({
      repoRoot: ctx.repoRoot,
      base: judge.inference,
      modelId: ctx.judgeModelId
    })
  };
}

function handleRunError(ctx: RunContext, observer: TerminalObserver, error: unknown): number {
  if (error instanceof BenchmarkAbort) {
    logEvent(ctx, "run_aborted_by_user", observer, { error: error.message });
    observer.stop();
    process.stdout.write(`Autobench aborted by user. Partial artifacts: ${ctx.outputRoot}\n`);
    return 2;
  }
  logEvent(ctx, "run_failed", observer, { error: renderError(error) });
  observer.stop();
  process.stderr.write(`Autobench failed. Partial artifacts: ${ctx.outputRoot}\n`);
  throw error;
}

function buildSelfTestInference(thinkingEnabled: boolean): InferenceConfig {
  return {
    provider: PROVIDER_OPENAI_COMPATIBLE,
    apiBaseUrl: "http://127.0.0.1:1234/v1",
    apiMode: API_MODE_OPENAI_CHAT_COMPLETIONS,
    apiKey: "lm-studio",
    apiKeyFile: "",
    model: "judge",
    temperature: 0.3,
    topP: 1,
    maxTokens: 256,
    contextLength: 16384,
    parallelism: 1,
    thinkingEnabled,
    reasoningEffort: "low",
    includeReasoningParameter: true,
    systemPrompt: "",
    quotaLabel: "",
    quotaMaxRequests: null,
    quotaWindowSeconds: null
  };
}

function buildSelfTestJudge(): JudgeConfig {
  return {
    inference: buildSelfTestInference(false),
    promptPrefix: "",
    promptSuffix: "",
    outputMode: JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL
  };
}
