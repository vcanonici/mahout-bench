import { appendJsonl, utcNowIso } from "../io/filesystem.js";
import { interactiveMenuWithObserver, type TerminalObserver } from "../runtime/terminalObserver.js";
import { BenchmarkAbort } from "./benchmarkAbort.js";
import { logEvent, sleep } from "./runContext.js";
import type { ProviderLimitError } from "../inference/chatClient.js";
import type { RunContext } from "../contracts/autobench.js";

const DEFAULT_MANUAL_WAIT_SECONDS = 5 * 60 * 60;

export async function handleProviderLimit(args: {
  ctx: RunContext;
  observer: TerminalObserver;
  error: ProviderLimitError;
  unit: Record<string, unknown>;
}): Promise<void> {
  const { ctx, observer, error, unit } = args;
  appendJsonl(ctx.providerEventsPath, {
    timestamp: utcNowIso(),
    event: "provider_limit",
    provider: error.provider,
    model: error.model,
    status: error.status,
    status_text: error.statusText,
    retry_after_seconds: error.retryAfterSeconds,
    headers: error.headers,
    body: error.body,
    unit
  });
  logEvent(ctx, "provider_limit", observer, {
    provider: error.provider,
    model: error.model,
    status: error.status,
    retry_after_seconds: error.retryAfterSeconds,
    unit: JSON.stringify(unit)
  });

  const choices: Record<string, string> = {};
  if (error.retryAfterSeconds !== null) {
    choices.r = `esperar Retry-After (${error.retryAfterSeconds}s) e tentar de novo`;
  }
  choices.e = "esperar 5h manualmente e tentar de novo";
  choices.p = "parar agora com checkpoint para retomar depois";
  choices.a = "abortar agora com checkpoint";

  const choice = await interactiveMenuWithObserver(
    observer,
    `Provider sinalizou limite/quota para ${error.provider}/${error.model}. Progresso salvo.`,
    choices
  );
  if (choice === "p" || choice === "a") {
    throw new BenchmarkAbort(`Provider limit stop requested for ${error.provider}/${error.model}`);
  }
  const waitSeconds = choice === "r" && error.retryAfterSeconds !== null
    ? error.retryAfterSeconds
    : DEFAULT_MANUAL_WAIT_SECONDS;
  logEvent(ctx, "provider_limit_wait_started", observer, {
    provider: error.provider,
    model: error.model,
    wait_seconds: waitSeconds
  });
  await sleep(waitSeconds * 1000);
  logEvent(ctx, "provider_limit_wait_finished", observer, {
    provider: error.provider,
    model: error.model
  });
}
