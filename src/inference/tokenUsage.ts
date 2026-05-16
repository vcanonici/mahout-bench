import fs from "node:fs";

import type { InferenceConfig, TokenUsage, TokenUsageSummary } from "../contracts/autobench.js";
import { readTextFile } from "../io/filesystem.js";

export const EMPTY_TOKEN_USAGE_SUMMARY: TokenUsageSummary = {
  calls: 0,
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  reasoningTokens: 0,
  providerUsageCalls: 0,
  estimatedCalls: 0
};

/**
 * Extracts provider token usage when present, otherwise estimates from prompt/output text.
 */
export function tokenUsageForChat(args: {
  inference: InferenceConfig;
  prompt: string;
  outputText: string;
  responseDump: Record<string, unknown>;
}): TokenUsage {
  return extractProviderTokenUsage(args.responseDump) ?? estimateTokenUsage(args.inference, args.prompt, args.outputText);
}

/**
 * Estimates usage with a deterministic chars/4 heuristic and no tokenizer dependency.
 */
export function estimateTokenUsage(inference: InferenceConfig, prompt: string, outputText: string): TokenUsage {
  const inputTokens = estimateTextTokens(`${inference.systemPrompt}${prompt}`);
  const outputTokens = estimateTextTokens(outputText);
  return {
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens,
    reasoningTokens: null,
    source: "estimated"
  };
}

export function emptyTokenUsageSummary(): TokenUsageSummary {
  return { ...EMPTY_TOKEN_USAGE_SUMMARY };
}

export function addTokenUsage(summary: TokenUsageSummary, usage: TokenUsage | null | undefined): TokenUsageSummary {
  if (!usage) {
    return summary;
  }
  summary.calls += 1;
  summary.inputTokens += usage.inputTokens;
  summary.outputTokens += usage.outputTokens;
  summary.totalTokens += usage.totalTokens;
  summary.reasoningTokens += usage.reasoningTokens ?? 0;
  if (usage.source === "provider_usage") {
    summary.providerUsageCalls += 1;
  } else {
    summary.estimatedCalls += 1;
  }
  return summary;
}

export function combineTokenUsageSummaries(...summaries: TokenUsageSummary[]): TokenUsageSummary {
  const combined = emptyTokenUsageSummary();
  for (const summary of summaries) {
    combined.calls += summary.calls;
    combined.inputTokens += summary.inputTokens;
    combined.outputTokens += summary.outputTokens;
    combined.totalTokens += summary.totalTokens;
    combined.reasoningTokens += summary.reasoningTokens;
    combined.providerUsageCalls += summary.providerUsageCalls;
    combined.estimatedCalls += summary.estimatedCalls;
  }
  return combined;
}

export function aggregateTokenUsageFromJsonl(filePath: string): TokenUsageSummary {
  const summary = emptyTokenUsageSummary();
  if (!fs.existsSync(filePath)) {
    return summary;
  }
  for (const line of readTextFile(filePath).split("\n")) {
    if (!line.trim()) {
      continue;
    }
    const record = JSON.parse(line) as Record<string, unknown>;
    addTokenUsage(summary, parseTokenUsage(record.token_usage));
  }
  return summary;
}

export function parseTokenUsage(value: unknown): TokenUsage | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const inputTokens = toNonNegativeInteger(record.inputTokens);
  const outputTokens = toNonNegativeInteger(record.outputTokens);
  const totalTokens = toNonNegativeInteger(record.totalTokens);
  const reasoningTokens = record.reasoningTokens === null || record.reasoningTokens === undefined
    ? null
    : toNonNegativeInteger(record.reasoningTokens);
  const source = record.source === "provider_usage" ? "provider_usage" : record.source === "estimated" ? "estimated" : null;
  if (inputTokens === null || outputTokens === null || totalTokens === null || reasoningTokens === undefined || source === null) {
    return null;
  }
  return { inputTokens, outputTokens, totalTokens, reasoningTokens, source };
}

function extractProviderTokenUsage(responseDump: Record<string, unknown>): TokenUsage | null {
  const usage = asRecord(responseDump.usage);
  if (!usage) {
    return extractStatsTokenUsage(responseDump);
  }
  const inputTokens = firstInteger(usage.prompt_tokens, usage.input_tokens);
  const outputTokens = firstInteger(usage.completion_tokens, usage.output_tokens);
  const totalTokens = firstInteger(usage.total_tokens);
  if (inputTokens === null && outputTokens === null && totalTokens === null) {
    return extractStatsTokenUsage(responseDump);
  }
  const completionDetails = asRecord(usage.completion_tokens_details);
  const reasoningTokens = firstInteger(completionDetails?.reasoning_tokens, usage.reasoning_tokens);
  const normalizedInput = inputTokens ?? Math.max(0, (totalTokens ?? 0) - (outputTokens ?? 0));
  const normalizedOutput = outputTokens ?? Math.max(0, (totalTokens ?? 0) - normalizedInput);
  return {
    inputTokens: normalizedInput,
    outputTokens: normalizedOutput,
    totalTokens: totalTokens ?? normalizedInput + normalizedOutput,
    reasoningTokens,
    source: "provider_usage"
  };
}

function extractStatsTokenUsage(responseDump: Record<string, unknown>): TokenUsage | null {
  const stats = asRecord(responseDump.stats);
  if (!stats) {
    return null;
  }
  const inputTokens = firstInteger(stats.prompt_tokens, stats.input_tokens, stats.input_tokens_count);
  const outputTokens = firstInteger(stats.completion_tokens, stats.output_tokens, stats.predicted_tokens_count);
  const totalTokens = firstInteger(stats.total_tokens);
  if (inputTokens === null && outputTokens === null && totalTokens === null) {
    return null;
  }
  const normalizedInput = inputTokens ?? Math.max(0, (totalTokens ?? 0) - (outputTokens ?? 0));
  const normalizedOutput = outputTokens ?? Math.max(0, (totalTokens ?? 0) - normalizedInput);
  return {
    inputTokens: normalizedInput,
    outputTokens: normalizedOutput,
    totalTokens: totalTokens ?? normalizedInput + normalizedOutput,
    reasoningTokens: firstInteger(stats.reasoning_output_tokens, stats.reasoning_tokens),
    source: "provider_usage"
  };
}

function estimateTextTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4));
}

function firstInteger(...values: unknown[]): number | null {
  for (const value of values) {
    const numeric = toNonNegativeInteger(value);
    if (numeric !== null) {
      return numeric;
    }
  }
  return null;
}

function toNonNegativeInteger(value: unknown): number | null {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric < 0) {
    return null;
  }
  return numeric;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}
