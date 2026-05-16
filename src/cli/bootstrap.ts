#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { fileURLToPath } from "node:url";

import {
  API_MODE_LMSTUDIO_NATIVE_CHAT,
  API_MODE_OPENAI_CHAT_COMPLETIONS,
  PROVIDER_LMSTUDIO,
  PROVIDER_MINIMAX,
  PROVIDER_OLLAMA,
  PROVIDER_OPENROUTER,
  type ProviderCatalog,
  type ProviderCatalogEntry
} from "../contracts/autobench.js";
import { ensureDir, readJsonFile, utcNowIso, writeJson } from "../io/filesystem.js";
import { formatDiscoverySummary, refreshAvailableModelsBestEffort } from "../config/modelDiscovery.js";
import { defaultPackageRoot, resolveDataRoot } from "../runtime/paths.js";
import { setup } from "./setup.js";

const BOOTSTRAP_MARKER_RELATIVE_PATH = path.join("config", "bootstrap.json");
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

interface BootstrapArgs {
  dataRoot: string;
  shouldSkipSetup: boolean;
  shouldForceSetup: boolean;
  shouldUseDefaults: boolean;
  shouldForceSecrets: boolean;
  shouldSkipDiscovery: boolean;
}

export interface BootstrapMarker {
  version: 1;
  completedAt: string;
  dataRoot: string;
  providerIds: string[];
  localProviderIds: string[];
  secretFiles: string[];
}

export interface BootstrapProviderInput {
  id: string;
  label: string;
  provider: string;
  apiMode: string;
  apiBaseUrl: string;
  apiKey: string;
  apiKeyFile: string;
  discoveryUrls: string[];
}

/**
 * Runs the first-use bootstrap assistant for datasets, secrets, and local providers.
 */
export async function bootstrap(argv = process.argv.slice(2), existingReadline: readline.Interface | null = null): Promise<number> {
  if (argv.includes("--help") || argv.includes("-h")) {
    printHelp();
    return 0;
  }
  const args = parseArgs(argv);
  if (!args.shouldSkipSetup) {
    const setupArgs = ["--data-root", args.dataRoot];
    if (args.shouldForceSetup) {
      setupArgs.push("--force");
    }
    const setupCode = await setup(setupArgs);
    if (setupCode !== 0) {
      return setupCode;
    }
  }

  const providers = args.shouldUseDefaults
    ? defaultBootstrapProviders()
    : await collectBootstrapProviders(args.dataRoot, args.shouldForceSecrets, existingReadline);
  validateBootstrapProviders(providers);
  writeBootstrapProviderCatalog(args.dataRoot, providers);
  if (!args.shouldSkipDiscovery) {
    const discovery = await refreshAvailableModelsBestEffort(defaultPackageRoot());
    process.stdout.write(`${formatDiscoverySummary(discovery)}\n`);
  }
  writeBootstrapMarker(args.dataRoot, providers);
  process.stdout.write(`Bootstrap complete: ${args.dataRoot}\n`);
  process.stdout.write("Local runtimes are not installed automatically. Start LM Studio or Ollama, then run discovery from the TUI if a backend is offline.\n");
  return 0;
}

/**
 * Runs bootstrap automatically only for interactive first-use command paths.
 */
export async function maybeRunBootstrap(argv: string[]): Promise<void> {
  if (!shouldAutoBootstrap(argv)) {
    return;
  }
  process.stdout.write("Mahout Bench needs first-run configuration before continuing.\n");
  const code = await bootstrap(["--skip-setup"]);
  if (code !== 0) {
    throw new Error(`Bootstrap failed with exit code ${code}`);
  }
}

export function shouldAutoBootstrap(
  argv: string[],
  isInteractive = Boolean(process.stdin.isTTY && process.stdout.isTTY),
  dataRoot = resolveDataRoot()
): boolean {
  if (!isInteractive || isBootstrapComplete(dataRoot)) {
    return false;
  }
  const command = argv.find((entry) => !entry.startsWith("-")) ?? "";
  if (["bootstrap", "setup", "status"].includes(command)) {
    return false;
  }
  const flags = new Set(argv.filter((entry) => entry.startsWith("-")));
  if (flags.has("--no-bootstrap") || flags.has("--help") || flags.has("-h")) {
    return false;
  }
  if (flags.has("--self-test") || flags.has("--dry-smoke")) {
    return false;
  }
  return true;
}

export function isBootstrapComplete(dataRoot = resolveDataRoot()): boolean {
  return fs.existsSync(bootstrapMarkerPath(dataRoot));
}

export function bootstrapMarkerPath(dataRoot = resolveDataRoot()): string {
  return path.join(dataRoot, BOOTSTRAP_MARKER_RELATIVE_PATH);
}

export function defaultBootstrapProviders(): ProviderCatalogEntry[] {
  return [
    buildProvider({
      id: "bootstrap-lmstudio-local-openai-v1",
      label: "LM Studio local OpenAI-compatible API",
      provider: PROVIDER_LMSTUDIO,
      apiMode: API_MODE_OPENAI_CHAT_COMPLETIONS,
      apiBaseUrl: "http://127.0.0.1:1234/v1",
      apiKey: "lm-studio",
      apiKeyFile: "",
      discoveryUrls: ["http://127.0.0.1:1234/v1/models"]
    })
  ];
}

export function validateBootstrapProviders(providers: ProviderCatalogEntry[]): void {
  if (providers.length === 0) {
    throw new Error("Bootstrap must configure at least one provider");
  }
  const ids = new Set<string>();
  for (const provider of providers) {
    if (!provider.id || ids.has(provider.id)) {
      throw new Error(`Missing or duplicate provider id: ${provider.id}`);
    }
    ids.add(provider.id);
    if (!provider.apiBaseUrl || !provider.label || !provider.provider || !provider.apiMode) {
      throw new Error(`Provider ${provider.id} is missing required fields`);
    }
  }
  const localProviders = providers.filter(isLocalRuntimeProvider);
  if (localProviders.length === 0) {
    throw new Error("Bootstrap requires at least one local LM Studio or Ollama backend");
  }
}

export function isLocalRuntimeProvider(provider: ProviderCatalogEntry): boolean {
  if (provider.provider !== PROVIDER_LMSTUDIO && provider.provider !== PROVIDER_OLLAMA) {
    return false;
  }
  try {
    const parsed = new URL(provider.apiBaseUrl);
    return LOCAL_HOSTS.has(parsed.hostname);
  } catch {
    return false;
  }
}

export function writeBootstrapProviderCatalog(dataRoot: string, providers: ProviderCatalogEntry[]): void {
  validateBootstrapProviders(providers);
  writeJson(path.join(dataRoot, "config", "providers.local.json"), { providers } satisfies ProviderCatalog);
}

function writeBootstrapMarker(dataRoot: string, providers: ProviderCatalogEntry[]): void {
  const marker: BootstrapMarker = {
    version: 1,
    completedAt: utcNowIso(),
    dataRoot,
    providerIds: providers.map((provider) => provider.id),
    localProviderIds: providers.filter(isLocalRuntimeProvider).map((provider) => provider.id),
    secretFiles: providers.map((provider) => provider.apiKeyFile).filter(Boolean)
  };
  writeJson(bootstrapMarkerPath(dataRoot), marker);
}

async function collectBootstrapProviders(
  dataRoot: string,
  shouldForceSecrets: boolean,
  existingReadline: readline.Interface | null
): Promise<ProviderCatalogEntry[]> {
  const rl = existingReadline ?? readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    const providers: ProviderCatalogEntry[] = [];
    const openRouterKey = await optionalAnswer(rl, "OpenRouter API key (blank to skip)");
    const minimaxKey = await optionalAnswer(rl, "MiniMax API key (blank to skip)");
    if (openRouterKey) {
      writeBootstrapSecret(dataRoot, "openrouter_api_key.txt", openRouterKey, shouldForceSecrets);
      providers.push(openRouterProvider());
    }
    if (minimaxKey) {
      writeBootstrapSecret(dataRoot, "minimax_api_key.txt", minimaxKey, shouldForceSecrets);
      providers.push(minimaxProvider());
    }

    while (!providers.some(isLocalRuntimeProvider)) {
      providers.push(await askRuntimeProvider(rl, true));
    }
    while (await confirm(rl, "Add another backend?")) {
      providers.push(await askRuntimeProvider(rl, false));
    }
    return providers;
  } finally {
    if (!existingReadline) {
      rl.close();
    }
  }
}

async function askRuntimeProvider(rl: readline.Interface, mustBeLocal: boolean): Promise<ProviderCatalogEntry> {
  const kind = await choose(rl, mustBeLocal ? "Required local backend" : "Backend kind", [
    "lmstudio-openai",
    "lmstudio-native",
    "ollama-openai"
  ]);
  const defaults = providerDefaults(kind);
  const apiBaseUrl = await optionalAnswer(rl, `API base URL [${defaults.apiBaseUrl}]`) || defaults.apiBaseUrl;
  const label = await optionalAnswer(rl, `Label [${defaults.label}]`) || defaults.label;
  const id = await optionalAnswer(rl, `Provider id [${defaults.id}]`) || defaults.id;
  const discoveryUrl = await optionalAnswer(rl, `Discovery URL [${defaults.discoveryUrls[0]}]`) || defaults.discoveryUrls[0]!;
  const provider = buildProvider({ ...defaults, id: slugifyProviderId(id), label, apiBaseUrl, discoveryUrls: [discoveryUrl] });
  if (mustBeLocal && !isLocalRuntimeProvider(provider)) {
    process.stdout.write("The first runtime backend must point to localhost or 127.0.0.1.\n");
    return askRuntimeProvider(rl, mustBeLocal);
  }
  return provider;
}

function providerDefaults(kind: string): ProviderCatalogEntry {
  if (kind === "lmstudio-native") {
    return buildProvider({
      id: "bootstrap-lmstudio-local-native-v1",
      label: "LM Studio local Native API",
      provider: PROVIDER_LMSTUDIO,
      apiMode: API_MODE_LMSTUDIO_NATIVE_CHAT,
      apiBaseUrl: "http://127.0.0.1:1234",
      apiKey: "lm-studio",
      apiKeyFile: "",
      discoveryUrls: ["http://127.0.0.1:1234/api/v1/models"]
    });
  }
  if (kind === "ollama-openai") {
    return buildProvider({
      id: "bootstrap-ollama-local-openai-v1",
      label: "Ollama local OpenAI-compatible API",
      provider: PROVIDER_OLLAMA,
      apiMode: API_MODE_OPENAI_CHAT_COMPLETIONS,
      apiBaseUrl: "http://127.0.0.1:11434/v1",
      apiKey: "ollama",
      apiKeyFile: "",
      discoveryUrls: ["http://127.0.0.1:11434/v1/models"]
    });
  }
  return defaultBootstrapProviders()[0]!;
}

function buildProvider(input: BootstrapProviderInput): ProviderCatalogEntry {
  return {
    id: input.id,
    label: input.label,
    provider: input.provider,
    apiMode: input.apiMode,
    apiBaseUrl: input.apiBaseUrl,
    apiKey: input.apiKey,
    apiKeyFile: input.apiKeyFile,
    discoveryUrls: input.discoveryUrls
  };
}

function openRouterProvider(): ProviderCatalogEntry {
  return buildProvider({
    id: "openrouter",
    label: "OpenRouter API",
    provider: PROVIDER_OPENROUTER,
    apiMode: API_MODE_OPENAI_CHAT_COMPLETIONS,
    apiBaseUrl: "https://openrouter.ai/api/v1",
    apiKey: "",
    apiKeyFile: "secrets/openrouter_api_key.txt",
    discoveryUrls: ["https://openrouter.ai/api/v1/models"]
  });
}

function minimaxProvider(): ProviderCatalogEntry {
  return buildProvider({
    id: "minimax",
    label: "MiniMax API",
    provider: PROVIDER_MINIMAX,
    apiMode: API_MODE_OPENAI_CHAT_COMPLETIONS,
    apiBaseUrl: "https://api.minimax.io/v1",
    apiKey: "",
    apiKeyFile: "secrets/minimax_api_key.txt",
    discoveryUrls: []
  });
}

export function writeBootstrapSecret(dataRoot: string, fileName: string, value: string, shouldForceSecrets: boolean): void {
  const targetPath = path.join(dataRoot, "secrets", fileName);
  ensureDir(path.dirname(targetPath));
  if (fs.existsSync(targetPath) && !shouldForceSecrets) {
    return;
  }
  fs.writeFileSync(targetPath, `${value.trim()}\n`, { encoding: "utf8", mode: 0o600 });
}

function parseArgs(argv: string[]): BootstrapArgs {
  const args: BootstrapArgs = {
    dataRoot: resolveDataRoot(),
    shouldSkipSetup: false,
    shouldForceSetup: false,
    shouldUseDefaults: false,
    shouldForceSecrets: false,
    shouldSkipDiscovery: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index]!;
    if (current === "--data-root") {
      args.dataRoot = path.resolve(requireValue(argv, ++index, current));
    } else if (current === "--skip-setup") {
      args.shouldSkipSetup = true;
    } else if (current === "--force-setup") {
      args.shouldForceSetup = true;
    } else if (current === "--defaults") {
      args.shouldUseDefaults = true;
    } else if (current === "--force-secrets") {
      args.shouldForceSecrets = true;
    } else if (current === "--skip-discovery") {
      args.shouldSkipDiscovery = true;
    } else {
      throw new Error(`Unknown bootstrap argument: ${current}`);
    }
  }
  return args;
}

async function choose(rl: readline.Interface, prompt: string, choices: string[]): Promise<string> {
  while (true) {
    process.stdout.write(`\n${prompt}\n`);
    choices.forEach((choice, index) => process.stdout.write(`${index + 1}. ${choice}\n`));
    const answer = (await rl.question(">: ")).trim();
    const index = Number(answer) - 1;
    if (Number.isInteger(index) && choices[index]) {
      return choices[index]!;
    }
    const direct = choices.find((choice) => choice.toLowerCase() === answer.toLowerCase());
    if (direct) {
      return direct;
    }
    process.stdout.write("Invalid option.\n");
  }
}

async function confirm(rl: readline.Interface, prompt: string): Promise<boolean> {
  const answer = (await rl.question(`${prompt} [y/N]: `)).trim().toLowerCase();
  return answer === "y" || answer === "yes";
}

async function optionalAnswer(rl: readline.Interface, prompt: string): Promise<string> {
  return (await rl.question(`${prompt}: `)).trim();
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function slugifyProviderId(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .toLowerCase();
}

function printHelp(): void {
  process.stdout.write("mahout-bench bootstrap\n\n");
  process.stdout.write("Downloads data when needed and configures user-owned providers under MAHOUT_BENCH_HOME or ./.mahout-bench.\n\n");
  process.stdout.write("--data-root <path>   Override the data/config root.\n");
  process.stdout.write("--skip-setup         Do not download or verify the data bundle.\n");
  process.stdout.write("--force-setup        Redownload the data bundle during bootstrap.\n");
  process.stdout.write("--defaults           Non-interactive local LM Studio defaults.\n");
  process.stdout.write("--force-secrets      Overwrite existing generated secret files.\n");
  process.stdout.write("--skip-discovery     Do not refresh local model discovery after writing providers.\n");
}

if (path.resolve(process.argv[1] ?? "") === path.resolve(fileURLToPath(import.meta.url))) {
  bootstrap().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}
