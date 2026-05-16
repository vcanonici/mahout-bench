import {
  MAX_ATTEMPTS_PER_ROW,
  YTA_NTA_PARSE,
  type CsvRecord,
  type DatasetConfig,
  type GenerationPoolBackend,
  type GenerationResult,
  type ProfileConfig,
  type RunContext,
  type TokenUsage
} from "../contracts/autobench.js";
import { callChat, formatGenerationPrompt, isProviderLimitError, parseOutput } from "../inference/chatClient.js";
import { appendJsonl, sha256Text, utcNowIso } from "../io/filesystem.js";
import type { StageHandle, TerminalObserver } from "../runtime/terminalObserver.js";
import { logEvent, renderError, roundSeconds, sleep } from "../pipeline/runContext.js";
import { generationUnitKey, readGenerationResult, writeGenerationResult } from "../pipeline/checkpoint.js";
import { handleProviderLimit } from "../pipeline/providerLimit.js";

/**
 * Generates and parses one model response with bounded retries and raw JSONL traces.
 */
export async function generateOne(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  dataset: DatasetConfig;
  row: CsvRecord;
  rowIndex: number | string;
  mode: string;
  parser: string;
  backend?: GenerationPoolBackend;
  observer: TerminalObserver;
  progress?: StageHandle;
}): Promise<GenerationResult> {
  const { ctx, profile, dataset, row, rowIndex, mode, parser, backend, observer, progress } = args;
  const inference = backend?.inference ?? profile.generation;
  const timeoutSeconds = backend?.timeoutSeconds ?? 600;
  const prompt = formatGenerationPrompt(dataset, row, parser === YTA_NTA_PARSE);
  const rowId = row.id ?? rowIndex;
  const promptHash = sha256Text(prompt);
  const checkpointKey = generationUnitKey({ profileName: profile.name, datasetName: dataset.name, mode, rowIndex });
  const checkpointResult = readGenerationResult(ctx, checkpointKey);
  if (checkpointResult) {
    logEvent(ctx, "generation_checkpoint_hit", observer, {
      profile: profile.name,
      dataset: dataset.name,
      mode,
      row_id: rowId
    });
    completePlannedGenerationCall(observer, progress, true);
    return checkpointResult;
  }
  let lastError: string | null = null;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS_PER_ROW; attempt += 1) {
    const attemptStarted = Date.now();
    let responseDump: Record<string, unknown> | null = null;
    try {
      const response = await callChat({ inference, prompt, timeoutSeconds });
      responseDump = response.responseDump;
      const parsed = parseOutput(response.extracted.text, parser);
      lastError = parsed.error;
      appendGenerationTrace(ctx, {
        profileName: profile.name,
        datasetName: dataset.name,
        mode,
        rowIndex,
        rowId,
        promptHash,
        attempt,
        attemptStarted,
        backend,
        timeoutSeconds,
        parser,
        textSource: response.extracted.source,
        thinkingDetected: response.extracted.thinkingDetected,
        extractedText: response.extracted.text,
        parsed,
        responseDump,
        tokenUsage: response.tokenUsage
      });
      logEvent(ctx, "generation_attempt", observer, {
        profile: profile.name,
        dataset: dataset.name,
        mode,
        parser,
        row_id: rowId,
        backend_id: backend?.backendId ?? null,
        model_id: backend?.modelId ?? null,
        api_base_url: backend?.inference.apiBaseUrl ?? null,
        attempt,
        duration_seconds: roundSeconds(Date.now() - attemptStarted),
        ok: parsed.ok,
        label: parsed.label,
        source: response.extracted.source,
        thinking: response.extracted.thinkingDetected,
        error: parsed.error,
        preview: parsed.ok ? "" : response.extracted.text.slice(0, 160)
      });
      if (parsed.ok) {
        const result = {
          ok: true,
          text: parsed.text,
          label: parsed.label,
          attempts: attempt,
          error: null
        };
        writeGenerationResult(ctx, checkpointKey, result);
        completePlannedGenerationCall(observer, progress, true);
        return result;
      }
    } catch (error) {
      if (isProviderLimitError(error)) {
        await handleProviderLimit({
          ctx,
          observer,
          error,
          unit: {
            stage: "generation",
            profile: profile.name,
            dataset: dataset.name,
            mode,
            parser,
            row_id: rowId,
            attempt
          }
        });
        attempt -= 1;
        continue;
      }
      lastError = renderError(error);
      appendGenerationException(ctx, {
        profileName: profile.name,
        datasetName: dataset.name,
        mode,
        rowIndex,
        rowId,
        promptHash,
        attempt,
        attemptStarted,
        backend,
        timeoutSeconds,
        parser,
        error: lastError,
        responseDump
      });
      logEvent(ctx, "generation_exception", observer, {
        profile: profile.name,
        dataset: dataset.name,
        mode,
        parser,
        row_id: rowId,
        backend_id: backend?.backendId ?? null,
        model_id: backend?.modelId ?? null,
        api_base_url: backend?.inference.apiBaseUrl ?? null,
        attempt,
        duration_seconds: roundSeconds(Date.now() - attemptStarted),
        error: lastError
      });
      await sleep(Math.min(5000, attempt * 1000));
    }
  }

  const failedResult = {
    ok: false,
    text: "",
    label: null,
    attempts: MAX_ATTEMPTS_PER_ROW,
    error: lastError
  };
  completePlannedGenerationCall(observer, progress, false);
  return failedResult;
}

function completePlannedGenerationCall(observer: TerminalObserver, progress: StageHandle | undefined, ok: boolean): void {
  if (!progress) {
    return;
  }
  const outcome = ok ? {} : { ok: false, failureKind: "generation_failed" };
  progress.advance(1, outcome);
  observer.advanceOverall(1, outcome);
}

function appendGenerationTrace(
  ctx: RunContext,
  args: {
    profileName: string;
    datasetName: string;
    mode: string;
    rowIndex: number | string;
    rowId: unknown;
    promptHash: string;
    attempt: number;
    attemptStarted: number;
    backend: GenerationPoolBackend | undefined;
    timeoutSeconds: number;
    parser: string;
    textSource: string;
    thinkingDetected: boolean;
    extractedText: string;
    parsed: { ok: boolean; label: string | null; error: string | null };
    responseDump: Record<string, unknown> | null;
    tokenUsage: TokenUsage;
  }
): void {
  appendJsonl(ctx.rawGenerationPath, {
    timestamp: utcNowIso(),
    profile: args.profileName,
    dataset: args.datasetName,
    mode: args.mode,
    row_index: args.rowIndex,
    row_id: args.rowId,
    prompt_sha256: args.promptHash,
    attempt: args.attempt,
    backend_id: args.backend?.backendId ?? null,
    model_id: args.backend?.modelId ?? null,
    provider: args.backend?.inference.provider ?? null,
    api_base_url: args.backend?.inference.apiBaseUrl ?? null,
    model: args.backend?.inference.model ?? null,
    timeout_seconds: args.timeoutSeconds,
    duration_seconds: roundSeconds(Date.now() - args.attemptStarted),
    parser: args.parser,
    text_source: args.textSource,
    thinking_detected: args.thinkingDetected,
    extracted_text: args.extractedText,
    parsed_ok: args.parsed.ok,
    parsed_label: args.parsed.label,
    token_usage: args.tokenUsage,
    error: args.parsed.error,
    response: args.responseDump
  });
}

function appendGenerationException(
  ctx: RunContext,
  args: {
    profileName: string;
    datasetName: string;
    mode: string;
    rowIndex: number | string;
    rowId: unknown;
    promptHash: string;
    attempt: number;
    attemptStarted: number;
    backend: GenerationPoolBackend | undefined;
    timeoutSeconds: number;
    parser: string;
    error: string;
    responseDump: Record<string, unknown> | null;
  }
): void {
  appendJsonl(ctx.rawGenerationPath, {
    timestamp: utcNowIso(),
    profile: args.profileName,
    dataset: args.datasetName,
    mode: args.mode,
    row_index: args.rowIndex,
    row_id: args.rowId,
    prompt_sha256: args.promptHash,
    attempt: args.attempt,
    backend_id: args.backend?.backendId ?? null,
    model_id: args.backend?.modelId ?? null,
    provider: args.backend?.inference.provider ?? null,
    api_base_url: args.backend?.inference.apiBaseUrl ?? null,
    model: args.backend?.inference.model ?? null,
    timeout_seconds: args.timeoutSeconds,
    duration_seconds: roundSeconds(Date.now() - args.attemptStarted),
    parser: args.parser,
    parsed_ok: false,
    parsed_label: null,
    error: args.error,
    response: args.responseDump
  });
}
