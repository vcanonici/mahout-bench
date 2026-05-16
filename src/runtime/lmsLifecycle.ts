import { spawnSync } from "node:child_process";

import { LMStudioClient, type LLMLoadModelConfig } from "@lmstudio/sdk";

import {
  LMS_PS_TIMEOUT_SECONDS,
  LMS_UNLOAD_TIMEOUT_SECONDS,
  PROVIDER_LMSTUDIO,
  type GenerationPoolBackend,
  type InferenceConfig,
  type RunContext
} from "../contracts/autobench.js";
import { appendJsonl, utcNowIso } from "../io/filesystem.js";
import { logEvent, renderError, roundSeconds } from "../pipeline/runContext.js";
import type { TerminalObserver } from "./terminalObserver.js";

const LOCAL_LMSTUDIO_LOAD_CONFIG: LLMLoadModelConfig = {
  gpu: { ratio: "max" },
  gpuStrictVramCap: true,
  flashAttention: true,
  offloadKVCacheToGpu: true,
  llamaKCacheQuantizationType: "q4_0",
  llamaVCacheQuantizationType: "q4_0"
};

/**
 * Builds the local LM Studio load config used by benchmark-managed model loads.
 */
export function localLmStudioLoadConfig(contextLength: number): LLMLoadModelConfig {
  return {
    ...LOCAL_LMSTUDIO_LOAD_CONFIG,
    contextLength
  };
}

/**
 * Returns true when the LM Studio CLI is available on PATH.
 */
export function hasLmsCli(): boolean {
  const result = spawnSync("bash", ["-lc", "command -v lms"], {
    encoding: "utf8"
  });
  return result.status === 0;
}

/**
 * Reads loaded LM Studio LLM models through `lms ps --json`.
 */
export function loadLmsModels(cwd: string): Array<Record<string, unknown>> {
  const result = spawnSync("lms", ["ps", "--json"], {
    cwd,
    encoding: "utf8",
    timeout: LMS_PS_TIMEOUT_SECONDS * 1000
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`lms ps --json failed: ${result.stderr?.trim()}`);
  }
  const parsed = JSON.parse(result.stdout?.trim() || "[]");
  if (!Array.isArray(parsed)) {
    throw new Error("Unexpected lms ps --json output");
  }
  return parsed.filter((item): item is Record<string, unknown> => item && typeof item === "object" && (item as Record<string, unknown>).type === "llm");
}

/**
 * Unloads all loaded LM Studio models when any model is present.
 */
export function unloadAllModels(ctx: RunContext, observer: TerminalObserver, role: string): void {
  if (!hasLmsCli()) {
    throw new Error("lms CLI not found");
  }
  const models = loadLmsModels(ctx.repoRoot);
  if (models.length === 0) {
    logEvent(ctx, "lms_unload_skipped", observer, { role, reason: "no_loaded_models" });
    return;
  }
  const stage = observer.startStage(`unload ${role}`, 1);
  try {
    runCommandLogged(ctx, `lms_unload_all:${role}`, ["lms", "unload", "--all"], ctx.repoRoot, LMS_UNLOAD_TIMEOUT_SECONDS, observer);
    stage.advance();
  } finally {
    stage.close();
  }
}

/**
 * Loads the configured LM Studio model and verifies context length after load.
 */
export async function loadModel(ctx: RunContext, inference: InferenceConfig, observer: TerminalObserver, role: string): Promise<void> {
  if (inference.provider !== PROVIDER_LMSTUDIO || !isLocalLmStudioBaseUrl(inference.apiBaseUrl)) {
    logEvent(ctx, "lms_load_skipped", observer, {
      role,
      model: inference.model,
      provider: inference.provider,
      api_base_url: inference.apiBaseUrl,
      reason: "not_local_lmstudio"
    });
    return;
  }
  if (!hasLmsCli()) {
    throw new Error("lms CLI not found");
  }
  const modelsBefore = loadLmsModels(ctx.repoRoot);
  const matchingModel = modelsBefore.find((entry) => loadedModelMatches(entry, inference.model));
  if (matchingModel) {
    logEvent(ctx, "lms_local_reload_required", observer, {
      role,
      model: inference.model,
      reason: "load_config_requires_sdk_defaults",
      context_length: loadedModelContextLength(matchingModel)
    });
  }
  ensureLocalLmStudioServerStarted(ctx, inference, observer, role);
  unloadAllModels(ctx, observer, `before_${role}`);
  await loadLocalModelWithSdk(ctx, inference, observer, role);
}

/**
 * Loads each local LM Studio backend participating in the generation pool.
 */
export async function loadGenerationPoolModels(
  ctx: RunContext,
  generationPool: GenerationPoolBackend[],
  observer: TerminalObserver,
  role = "generation_pool"
): Promise<void> {
  const localBackends = uniqueLocalGenerationBackends(generationPool);
  if (localBackends.length === 0) {
    logEvent(ctx, "lms_pool_load_skipped", observer, { role, reason: "no_local_lmstudio_backends" });
    return;
  }
  if (!hasLmsCli()) {
    throw new Error("lms CLI not found");
  }
  ensureLocalLmStudioServerStarted(ctx, localBackends[0]!.inference, observer, role);
  unloadAllModels(ctx, observer, `before_${role}`);
  for (const backend of localBackends) {
    await loadLocalModelWithSdk(ctx, backend.inference, observer, `${role}:${backend.modelId}`);
  }
}

/**
 * Verifies local LM Studio backends are already loaded when lifecycle management is disabled.
 */
export function assertLocalLmStudioBackendsReadyForSkipLms(
  ctx: RunContext,
  generationPool: GenerationPoolBackend[],
  judgePool: GenerationPoolBackend[]
): void {
  const localBackends = uniqueLocalGenerationBackends([...generationPool, ...judgePool]);
  if (localBackends.length === 0) {
    return;
  }
  if (!hasLmsCli()) {
    throw new Error("Cannot use --skip-lms with local LM Studio backends: lms CLI not found.");
  }
  const loadedModels = loadLmsModels(ctx.repoRoot);
  const notReady = localLmStudioBackendsNotReadyForSkipLms(localBackends, loadedModels);
  const serverIssue = localLmStudioServerReadinessIssue(ctx.repoRoot, localBackends[0]!.inference.apiBaseUrl);
  if (notReady.length > 0 || serverIssue) {
    const modelIssues = notReady.map((entry) =>
      `${entry.backend.modelId} (${entry.backend.inference.model} at ${entry.backend.inference.apiBaseUrl}: ${entry.reason})`
    );
    const issues = serverIssue ? [`local LM Studio server: ${serverIssue}`, ...modelIssues] : modelIssues;
    throw new Error(
      "Cannot use --skip-lms with local LM Studio backend(s) that are not ready: " +
        issues.join(", ") +
        ". Remove --skip-lms so the runner can load them with the configured context, or load them in LM Studio first."
    );
  }
}

/**
 * Returns local LM Studio backends whose model is not already loaded or has too little context.
 */
export function localLmStudioBackendsNotReadyForSkipLms(
  localBackends: GenerationPoolBackend[],
  loadedModels: Array<Record<string, unknown>>
): Array<{ backend: GenerationPoolBackend; reason: string }> {
  return localBackends.flatMap((backend) => {
    const loaded = loadedModels.find((entry) => loadedModelMatches(entry, backend.inference.model));
    if (!loaded) {
      return [{ backend, reason: "model is not loaded" }];
    }
    const actualContextLength = loadedModelContextLength(loaded);
    if (actualContextLength !== null && actualContextLength < backend.inference.contextLength) {
      return [{ backend, reason: `context_length ${actualContextLength} < expected ${backend.inference.contextLength}` }];
    }
    return [];
  });
}

function ensureLocalLmStudioServerStarted(ctx: RunContext, inference: InferenceConfig, observer: TerminalObserver, role: string): void {
  const parsed = localLmStudioServerEndpoint(inference.apiBaseUrl);
  runCommandLogged(
    ctx,
    `lms_server_start:${role}`,
    ["lms", "server", "start", "--port", String(parsed.port), "--bind", parsed.bind],
    ctx.repoRoot,
    LMS_PS_TIMEOUT_SECONDS,
    observer
  );
}

function localLmStudioServerReadinessIssue(cwd: string, apiBaseUrl: string): string | null {
  const expected = localLmStudioServerEndpoint(apiBaseUrl);
  const result = spawnSync("lms", ["server", "status"], {
    cwd,
    encoding: "utf8",
    timeout: LMS_PS_TIMEOUT_SECONDS * 1000
  });
  if (result.error) {
    return result.error.message;
  }
  if (result.status !== 0) {
    return result.stderr?.trim() || "server status command failed";
  }
  const output = `${result.stdout}\n${result.stderr}`;
  const match = /running on port\s+(\d+)/i.exec(output);
  if (!match?.[1]) {
    return "server is not running";
  }
  const actualPort = Number(match[1]);
  return actualPort === expected.port ? null : `server is running on port ${actualPort}, expected ${expected.port}`;
}

function localLmStudioServerEndpoint(apiBaseUrl: string): { bind: string; port: number } {
  const parsed = new URL(apiBaseUrl);
  const port = parsed.port ? Number(parsed.port) : parsed.protocol === "https:" ? 443 : 80;
  if (!Number.isInteger(port) || port <= 0) {
    throw new Error(`Invalid local LM Studio API port: ${apiBaseUrl}`);
  }
  return {
    bind: parsed.hostname === "localhost" ? "127.0.0.1" : parsed.hostname,
    port
  };
}

function uniqueLocalGenerationBackends(generationPool: GenerationPoolBackend[]): GenerationPoolBackend[] {
  const seen = new Set<string>();
  const backends: GenerationPoolBackend[] = [];
  for (const backend of generationPool) {
    const { inference } = backend;
    if (inference.provider !== PROVIDER_LMSTUDIO || !isLocalLmStudioBaseUrl(inference.apiBaseUrl)) {
      continue;
    }
    const key = `${inference.apiBaseUrl}\n${inference.model}`;
    if (!seen.has(key)) {
      seen.add(key);
      backends.push(backend);
    }
  }
  return backends;
}

async function loadLocalModelWithSdk(ctx: RunContext, inference: InferenceConfig, observer: TerminalObserver, role: string): Promise<void> {
  const stage = observer.startStage(`load ${role}: ${inference.model}`, 1);
  const started = utcNowIso();
  const startedAt = Date.now();
  const config = localLmStudioLoadConfig(inference.contextLength);
  const sdkBaseUrl = lmStudioSdkBaseUrl(inference.apiBaseUrl);
  logEvent(ctx, "lms_sdk_load_started", observer, {
    role,
    model: inference.model,
    sdk_base_url: sdkBaseUrl,
    context_length: inference.contextLength,
    flash_attention: config.flashAttention,
    offload_kv_cache_to_gpu: config.offloadKVCacheToGpu,
    llama_k_cache_quantization_type: config.llamaKCacheQuantizationType,
    llama_v_cache_quantization_type: config.llamaVCacheQuantizationType
  });
  const client = new LMStudioClient({ baseUrl: sdkBaseUrl });
  try {
    await client.llm.load(inference.model, {
      identifier: inference.model,
      config,
      verbose: false
    });
    assertLoadedModel(ctx.repoRoot, inference);
    appendJsonl(ctx.eventsPath, {
      timestamp: started,
      event: "lms_sdk_load_finished",
      role,
      model: inference.model,
      sdk_base_url: sdkBaseUrl,
      load_config: config,
      duration_seconds: roundSeconds(Date.now() - startedAt)
    });
    stage.advance();
  } catch (error) {
    appendJsonl(ctx.eventsPath, {
      timestamp: utcNowIso(),
      event: "lms_sdk_load_failed",
      role,
      model: inference.model,
      sdk_base_url: sdkBaseUrl,
      load_config: config,
      duration_seconds: roundSeconds(Date.now() - startedAt),
      error: renderError(error)
    });
    throw error;
  } finally {
    await client[Symbol.asyncDispose]();
    stage.close();
  }
}

/**
 * Runs a command and writes structured command telemetry to `run_events.jsonl`.
 */
export function runCommandLogged(
  ctx: RunContext,
  step: string,
  args: string[],
  cwd: string,
  timeoutSeconds: number,
  observer: TerminalObserver
): void {
  const started = utcNowIso();
  const startedAt = Date.now();
  const loadedModelsBefore = hasLmsCli() && args.includes("load") ? safeLoadModels(cwd) : null;
  logEvent(ctx, "command_started", observer, {
    step,
    command: args.join(" "),
    timeout_seconds: timeoutSeconds
  });
  const result = spawnSync(args[0]!, args.slice(1), {
    cwd,
    encoding: "utf8",
    timeout: timeoutSeconds * 1000
  });
  if (result.error) {
    appendCommandTimeoutIfNeeded(ctx, started, startedAt, step, args, timeoutSeconds, result.error, observer);
    throw result.error;
  }
  appendJsonl(ctx.eventsPath, {
    timestamp: started,
    event: "command",
    step,
    args,
    returncode: result.status,
    duration_seconds: roundSeconds(Date.now() - startedAt),
    loaded_models_before: loadedModelsBefore,
    stdout: result.stdout,
    stderr: result.stderr
  });
  logEvent(ctx, "command_finished", observer, {
    step,
    returncode: result.status,
    duration_seconds: roundSeconds(Date.now() - startedAt)
  });
  if (result.status !== 0) {
    appendCommandFailedDetail(ctx, step, args, loadedModelsBefore, result.stdout, result.stderr, cwd);
    throw new Error(commandFailureMessage(step, args));
  }
}

/**
 * Checks whether a loaded model entry refers to the requested model name.
 */
export function loadedModelMatches(entry: Record<string, unknown>, modelName: string): boolean {
  return ["identifier", "modelKey", "path", "indexedModelIdentifier"].some((key) => String(entry[key] ?? "").trim() === modelName);
}

/**
 * Reads context length from an LM Studio model entry.
 */
export function loadedModelContextLength(entry: Record<string, unknown>): number | null {
  const numeric = Number(entry.contextLength);
  return Number.isInteger(numeric) ? numeric : null;
}

/**
 * Returns true when an API base URL points to the local LM Studio server.
 */
export function isLocalLmStudioBaseUrl(apiBaseUrl: string): boolean {
  try {
    const parsed = new URL(apiBaseUrl);
    return parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1" || parsed.hostname === "::1";
  } catch {
    return false;
  }
}

/**
 * Converts an LM Studio HTTP API URL to the WebSocket base URL expected by the SDK.
 */
export function lmStudioSdkBaseUrl(apiBaseUrl: string): string {
  const parsed = new URL(apiBaseUrl);
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  parsed.pathname = "";
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString().replace(/\/$/, "");
}

function assertLoadedModel(cwd: string, inference: InferenceConfig): void {
  const modelsAfter = loadLmsModels(cwd);
  const loaded = modelsAfter.find((entry) => loadedModelMatches(entry, inference.model));
  if (!loaded) {
    throw new Error(`Loaded model not visible in lms ps after load: ${inference.model}`);
  }
  const actualContextLength = loadedModelContextLength(loaded);
  if (actualContextLength !== inference.contextLength) {
    throw new Error(`Loaded model ${inference.model} with context_length ${actualContextLength}, expected ${inference.contextLength}`);
  }
}

function appendCommandTimeoutIfNeeded(
  ctx: RunContext,
  started: string,
  startedAt: number,
  step: string,
  args: string[],
  timeoutSeconds: number,
  error: Error,
  observer: TerminalObserver
): void {
  if (!/timed out/i.test(error.message)) {
    return;
  }
  appendJsonl(ctx.eventsPath, {
    timestamp: started,
    event: "command_timeout",
    step,
    args,
    timeout_seconds: timeoutSeconds,
    duration_seconds: roundSeconds(Date.now() - startedAt),
    stdout: "",
    stderr: error.message
  });
  logEvent(ctx, "command_timeout", observer, {
    step,
    timeout_seconds: timeoutSeconds,
    duration_seconds: roundSeconds(Date.now() - startedAt)
  });
}

function appendCommandFailedDetail(
  ctx: RunContext,
  step: string,
  args: string[],
  loadedModelsBefore: Array<Record<string, unknown>> | null,
  stdout: string | Buffer | null | undefined,
  stderr: string | Buffer | null | undefined,
  cwd: string
): void {
  appendJsonl(ctx.eventsPath, {
    timestamp: utcNowIso(),
    event: "command_failed_detail",
    step,
    args,
    loaded_models_before: loadedModelsBefore,
    loaded_models_after: safeLoadModels(cwd),
    stdout,
    stderr
  });
}

function commandFailureMessage(step: string, args: string[]): string {
  const modelName = args.length > 2 && args[0] === "lms" && args[1] === "load" ? args[2] : null;
  const contextIndex = args.indexOf("--context-length");
  const requestedContext = contextIndex >= 0 ? args[contextIndex + 1] : null;
  if (modelName && requestedContext) {
    return `${step} failed for model ${modelName} with context_length ${requestedContext}; verify the configured context_length is supported by LM Studio for this model`;
  }
  return `Command failed for ${step}: ${args.join(" ")}`;
}

function safeLoadModels(cwd: string): Array<Record<string, unknown>> | null {
  try {
    return loadLmsModels(cwd);
  } catch {
    return null;
  }
}
