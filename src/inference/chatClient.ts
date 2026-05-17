import {
  API_MODE_LMSTUDIO_NATIVE_CHAT,
  API_MODE_OPENAI_CHAT_COMPLETIONS,
  BINARY_SCORE_JSON_SCHEMA_PARSE,
  BINARY_SCORE_PARSE,
  JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL,
  JUDGE_OUTPUT_TEXT_BINARY_LABEL,
  PROVIDER_MINIMAX,
  PROVIDER_OPENROUTER,
  TEXT_PARSE,
  YTA_NTA_PARSE,
  type ExtractedText,
  type InferenceConfig,
  type JudgeConfig,
  type ParsedOutput,
  type TokenUsage
} from "../contracts/autobench.js";
import { tokenUsageForChat } from "./tokenUsage.js";

export class ProviderHttpError extends Error {
  public constructor(
    message: string,
    public readonly provider: string,
    public readonly model: string,
    public readonly status: number,
    public readonly statusText: string,
    public readonly headers: Record<string, string>,
    public readonly body: string,
    public readonly retryAfterSeconds: number | null
  ) {
    super(message);
    this.name = "ProviderHttpError";
  }
}

export class ProviderLimitError extends ProviderHttpError {
  public constructor(error: ProviderHttpError) {
    super(
      error.message,
      error.provider,
      error.model,
      error.status,
      error.statusText,
      error.headers,
      error.body,
      error.retryAfterSeconds
    );
    this.name = "ProviderLimitError";
  }
}

export function isProviderLimitError(error: unknown): error is ProviderLimitError {
  return error instanceof ProviderLimitError;
}

export function stripThinkingBlocks(text: string): { cleaned: string; found: boolean } {
  const patterns = [
    /<think>[\s\S]*?<\/think>/gi,
    /<thinking>[\s\S]*?<\/thinking>/gi,
    /<reasoning>[\s\S]*?<\/reasoning>/gi,
    /\[thinking\][\s\S]*?\[\/thinking\]/gi
  ];

  let cleaned = text;
  let found = false;
  for (const pattern of patterns) {
    if (pattern.test(cleaned)) {
      found = true;
      cleaned = cleaned.replace(pattern, "");
    }
  }

  return { cleaned: cleaned.trim(), found };
}

export function collectNamedStrings(value: unknown, prefix = "", depth = 0): Array<[string, string]> {
  if (depth > 6) {
    return [];
  }
  if (typeof value === "string") {
    return [[prefix, value]];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => collectNamedStrings(item, `${prefix}[${index}]`, depth + 1));
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, item]) => {
      const childPath = prefix ? `${prefix}.${key}` : key;
      return collectNamedStrings(item, childPath, depth + 1);
    });
  }
  return [];
}

export function extractFinalMarker(text: string): string {
  const markers = ["final answer", "final", "answer", "resposta final", "resposta", "verdict", "veredito"];
  for (const marker of markers) {
    const regex = new RegExp(`${escapeRegExp(marker)}\\s*[:：]\\s*(.+)$`, "is");
    const match = regex.exec(text);
    if (match?.[1]) {
      return match[1].trim();
    }
  }
  return "";
}

export function extractAssistantText(messageDump: Record<string, unknown>): ExtractedText {
  const content = messageDump.content;
  if (typeof content === "string" && content.trim()) {
    const stripped = stripThinkingBlocks(content);
    return {
      text: stripped.cleaned,
      source: "content",
      thinkingDetected: stripped.found
    };
  }

  let thinkingDetected = false;
  for (const [name, value] of collectNamedStrings(messageDump)) {
    if (/(reason|think|thought)/i.test(name)) {
      thinkingDetected = true;
    }
    const stripped = stripThinkingBlocks(value);
    thinkingDetected = thinkingDetected || stripped.found;
    const final = extractFinalMarker(stripped.cleaned);
    if (final) {
      return {
        text: final,
        source: name,
        thinkingDetected
      };
    }
    if (looksLikeShortFinal(stripped.cleaned)) {
      return {
        text: stripped.cleaned,
        source: name,
        thinkingDetected
      };
    }
  }

  return {
    text: "",
    source: "none",
    thinkingDetected
  };
}

export function stripMarkdownNoise(text: string): string {
  const stripped = stripThinkingBlocks(text).cleaned;
  return stripped
    .replace(/^[\s>*_`#\-\(\[]+/, "")
    .replace(/[*_`#\]\)]+$/, "")
    .replace(/^(?:final answer|final|answer|resposta final|resposta|verdict|veredito)\s*[:：]\s*/i, "")
    .trim();
}

export function parseYtaNtaLabel(text: string): string | null {
  const cleaned = stripMarkdownNoise(text);
  const match = /^(YTA|NTA)\b/i.exec(cleaned);
  return match ? match[1]!.toUpperCase() : null;
}

export function parseBinaryLabel(text: string): string | null {
  const cleaned = stripMarkdownNoise(text);
  const match = /^([01])\b/.exec(cleaned);
  return match ? match[1]! : null;
}

export function parseOutput(text: string, parser: string): ParsedOutput {
  if (parser === TEXT_PARSE) {
    const stripped = stripThinkingBlocks(text).cleaned;
    if (!stripped) {
      return { ok: false, text: stripped, label: null, parser, error: "blank text" };
    }
    return { ok: true, text: stripped, label: null, parser, error: null };
  }

  if (parser === YTA_NTA_PARSE) {
    const label = parseYtaNtaLabel(text);
    if (!label) {
      return { ok: false, text, label: null, parser, error: "missing leading YTA/NTA verdict" };
    }
    return { ok: true, text, label, parser, error: null };
  }

  if (parser === BINARY_SCORE_PARSE) {
    const label = parseBinaryLabel(text);
    if (!label) {
      return { ok: false, text, label: null, parser, error: "missing leading 0/1 judge label" };
    }
    return { ok: true, text, label, parser, error: null };
  }

  throw new Error(`Unknown parser: ${parser}`);
}

export function parseJsonSchemaBinaryLabel(text: string): ParsedOutput {
  const cleaned = text.trim();
  if (!cleaned) {
    return {
      ok: false,
      text: cleaned,
      label: null,
      parser: BINARY_SCORE_JSON_SCHEMA_PARSE,
      error: "blank judge json response"
    };
  }

  let payload: unknown;
  try {
    payload = JSON.parse(cleaned);
  } catch (error) {
    return {
      ok: false,
      text: cleaned,
      label: null,
      parser: BINARY_SCORE_JSON_SCHEMA_PARSE,
      error: `invalid judge json: ${(error as Error).message}`
    };
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return {
      ok: false,
      text: cleaned,
      label: null,
      parser: BINARY_SCORE_JSON_SCHEMA_PARSE,
      error: "judge json response is not an object"
    };
  }

  const label = (payload as Record<string, unknown>).label;
  if (label !== "0" && label !== "1" && label !== 0 && label !== 1) {
    return {
      ok: false,
      text: cleaned,
      label: null,
      parser: BINARY_SCORE_JSON_SCHEMA_PARSE,
      error: "judge json label must be '0' or '1'"
    };
  }

  return {
    ok: true,
    text: cleaned,
    label: String(label),
    parser: BINARY_SCORE_JSON_SCHEMA_PARSE,
    error: null
  };
}

export function parseJudgeOutput(text: string, outputMode: string): ParsedOutput {
  if (outputMode === JUDGE_OUTPUT_TEXT_BINARY_LABEL) {
    return parseOutput(text, BINARY_SCORE_PARSE);
  }
  if (outputMode === JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL) {
    return parseJsonSchemaBinaryLabel(text);
  }
  throw new Error(`Unknown judge.output_mode: ${outputMode}`);
}

export function buildJudgeResponseFormat(judge: JudgeConfig): Record<string, unknown> | null {
  if (judge.outputMode === JUDGE_OUTPUT_TEXT_BINARY_LABEL) {
    return null;
  }
  if (judge.outputMode === JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL) {
    return {
      type: "json_schema",
      json_schema: {
        name: "judge_label",
        schema: {
          type: "object",
          properties: {
            label: {
              type: "string",
              enum: ["0", "1"]
            }
          },
          required: ["label"],
          additionalProperties: false
        }
      }
    };
  }
  throw new Error(`Unknown judge.output_mode: ${judge.outputMode}`);
}

export function buildExtraBody(inference: InferenceConfig): Record<string, unknown> {
  if (inference.provider === PROVIDER_MINIMAX || inference.provider === PROVIDER_OPENROUTER) {
    return {};
  }
  const extraBody: Record<string, unknown> = {
    context_length: inference.contextLength
  };
  if (inference.thinkingEnabled) {
    extraBody.reasoning = {
      effort: inference.reasoningEffort
    };
  }
  return extraBody;
}

export async function callChat(args: {
  inference: InferenceConfig;
  prompt: string;
  responseFormat?: Record<string, unknown> | null;
  timeoutSeconds?: number;
}): Promise<{ responseDump: Record<string, unknown>; extracted: ExtractedText; tokenUsage: TokenUsage }> {
  const { inference, prompt, responseFormat = null, timeoutSeconds = 600 } = args;
  if (inference.apiMode === API_MODE_LMSTUDIO_NATIVE_CHAT) {
    return callLmStudioNativeChat({ inference, prompt, responseFormat, timeoutSeconds });
  }
  if (inference.apiMode !== API_MODE_OPENAI_CHAT_COMPLETIONS) {
    throw new Error(`Unsupported inference api_mode: ${inference.apiMode}`);
  }
  return callOpenAiChatCompletions({ inference, prompt, responseFormat, timeoutSeconds });
}

async function callOpenAiChatCompletions(args: {
  inference: InferenceConfig;
  prompt: string;
  responseFormat?: Record<string, unknown> | null;
  timeoutSeconds?: number;
}): Promise<{ responseDump: Record<string, unknown>; extracted: ExtractedText; tokenUsage: TokenUsage }> {
  const { inference, prompt, responseFormat = null, timeoutSeconds = 600 } = args;
  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), timeoutSeconds * 1000);
  const normalizedBase = inference.apiBaseUrl.replace(/\/$/, "");
  const body = buildOpenAiChatBody(inference, prompt, responseFormat);

  try {
    const response = await fetch(`${normalizedBase}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${inference.apiKey}`
      },
      body: JSON.stringify(body),
      signal: controller.signal
    });

    if (!response.ok) {
      throw await buildProviderHttpError(response, inference);
    }

    const responseDump = (await response.json()) as Record<string, unknown>;
    const choices = Array.isArray(responseDump.choices) ? responseDump.choices : [];
    const firstChoice = (choices[0] ?? {}) as Record<string, unknown>;
    const message = (firstChoice.message ?? {}) as Record<string, unknown>;
    responseDump._selected_message = message;
    const extracted = extractAssistantText(message);
    return {
      responseDump,
      extracted,
      tokenUsage: tokenUsageForChat({ inference, prompt, outputText: extracted.text, responseDump })
    };
  } finally {
    clearTimeout(timeoutHandle);
  }
}

async function callLmStudioNativeChat(args: {
  inference: InferenceConfig;
  prompt: string;
  responseFormat?: Record<string, unknown> | null;
  timeoutSeconds?: number;
}): Promise<{ responseDump: Record<string, unknown>; extracted: ExtractedText; tokenUsage: TokenUsage }> {
  const { inference, prompt, responseFormat = null, timeoutSeconds = 600 } = args;
  if (responseFormat) {
    throw new Error("LM Studio native chat mode does not support response_format in this runner");
  }

  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), timeoutSeconds * 1000);
  const normalizedBase = inference.apiBaseUrl.replace(/\/$/, "").replace(/\/v1$/, "");
  const body = buildLmStudioNativeBody(inference, prompt);

  try {
    const response = await fetch(`${normalizedBase}/api/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${inference.apiKey}`
      },
      body: JSON.stringify(body),
      signal: controller.signal
    });

    const rawText = await response.text();
    if (!response.ok) {
      const providerError = providerHttpErrorFromRaw(response, rawText, inference, "lmstudio native chat failed");
      throw limitAwareError(providerError);
    }

    const responseDump = parseResponseObject(rawText);
    const content = extractLmStudioNativeMessageText(responseDump);
    const selectedMessage = { content };
    responseDump.model = inference.model;
    responseDump._selected_message = selectedMessage;
    responseDump.choices = [
      {
        finish_reason: "stop",
        message: selectedMessage
      }
    ];
    const extracted = extractAssistantText(selectedMessage);
    return {
      responseDump,
      extracted,
      tokenUsage: tokenUsageForChat({ inference, prompt, outputText: extracted.text, responseDump })
    };
  } finally {
    clearTimeout(timeoutHandle);
  }
}

export function buildLmStudioNativeBody(inference: InferenceConfig, prompt: string): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: inference.model,
    input: prompt,
    system_prompt: inference.systemPrompt,
    temperature: inference.temperature,
    top_p: inference.topP,
    max_output_tokens: inference.maxTokens,
    context_length: inference.contextLength
  };
  if (inference.includeReasoningParameter) {
    body.reasoning = inference.thinkingEnabled ? "on" : "off";
  }
  return body;
}

export function isProviderLimitResponse(status: number, body: string): boolean {
  if (status === 429) {
    return true;
  }
  return /(quota|rate[\s_-]*limit|too many requests|insufficient[_\s-]*quota|token[\s_-]*plan|exhausted|limit reached)/i.test(body);
}

export function buildOpenAiChatBody(
  inference: InferenceConfig,
  prompt: string,
  responseFormat: Record<string, unknown> | null
): Record<string, unknown> {
  const messages: Array<Record<string, string>> = [
    { role: "user", content: prompt }
  ];
  if (inference.systemPrompt) {
    messages.unshift({ role: "system", content: inference.systemPrompt });
  }

  const body: Record<string, unknown> = {
    model: inference.model,
    messages,
    temperature: inference.temperature,
    top_p: inference.topP,
    max_tokens: inference.maxTokens
  };
  const extraBody = buildExtraBody(inference);
  if (Object.keys(extraBody).length > 0) {
    body.extra_body = extraBody;
  }
  if (responseFormat) {
    body.response_format = responseFormat;
  }
  return body;
}

function parseResponseObject(rawText: string): Record<string, unknown> {
  const parsed = JSON.parse(rawText) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Expected object response from chat endpoint: ${rawText.slice(0, 200)}`);
  }
  return parsed as Record<string, unknown>;
}

async function buildProviderHttpError(response: Response, inference: InferenceConfig): Promise<ProviderHttpError> {
  const rawText = await response.text();
  const providerError = providerHttpErrorFromRaw(response, rawText, inference, "chat completion failed");
  return limitAwareError(providerError);
}

function providerHttpErrorFromRaw(response: Response, rawText: string, inference: InferenceConfig, prefix: string): ProviderHttpError {
  const retryAfterSeconds = parseRetryAfterSeconds(response.headers.get("retry-after"));
  const headers = headersToRecord(response.headers);
  const message = `${prefix}: ${response.status} ${response.statusText}: ${rawText.slice(0, 500)}`;
  return new ProviderHttpError(
    message,
    inference.provider,
    inference.model,
    response.status,
    response.statusText,
    headers,
    rawText.slice(0, 4000),
    retryAfterSeconds
  );
}

function limitAwareError(error: ProviderHttpError): ProviderHttpError {
  return isProviderLimitResponse(error.status, error.body) ? new ProviderLimitError(error) : error;
}

function headersToRecord(headers: Headers): Record<string, string> {
  const filtered: Record<string, string> = {};
  headers.forEach((value, key) => {
    if (/retry|rate|limit|quota|request/i.test(key)) {
      filtered[key] = value;
    }
  });
  return filtered;
}

function parseRetryAfterSeconds(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric >= 0) {
    return Math.ceil(numeric);
  }
  const dateMs = Date.parse(value);
  if (Number.isNaN(dateMs)) {
    return null;
  }
  return Math.max(0, Math.ceil((dateMs - Date.now()) / 1000));
}

function extractLmStudioNativeMessageText(responseDump: Record<string, unknown>): string {
  const output = responseDump.output;
  if (!Array.isArray(output)) {
    return "";
  }
  return output
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }
      const entry = item as Record<string, unknown>;
      if (entry.type !== "message") {
        return "";
      }
      return typeof entry.content === "string" ? entry.content : "";
    })
    .filter((value) => value.trim().length > 0)
    .join("\n")
    .trim();
}

export function formatGenerationPrompt(dataset: { promptPrefix: string; promptColumn: string; promptSuffix: string }, row: Record<string, unknown>, constrained: boolean): string {
  const prompt = `${dataset.promptPrefix}${String(row[dataset.promptColumn] ?? "")}${dataset.promptSuffix}`;
  return constrained ? `${prompt}\nOutput only YTA or NTA.` : prompt;
}

export function extractReasoningTokens(responseDump: Record<string, unknown> | null): number | null {
  if (!responseDump) {
    return null;
  }
  const usage = asRecord(responseDump.usage);
  const completionDetails = asRecord(usage?.completion_tokens_details);
  const usageReasoning = toIntegerOrNull(completionDetails?.reasoning_tokens);
  if (usageReasoning !== null) {
    return usageReasoning;
  }
  const stats = asRecord(responseDump.stats);
  return toIntegerOrNull(stats?.reasoning_output_tokens);
}

function looksLikeShortFinal(text: string): boolean {
  const cleaned = text.trim();
  if (cleaned.length > 24) {
    return false;
  }
  return parseYtaNtaLabel(cleaned) !== null || parseBinaryLabel(cleaned) !== null;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function toIntegerOrNull(value: unknown): number | null {
  const numeric = Number(value);
  if (!Number.isInteger(numeric)) {
    return null;
  }
  return numeric;
}
