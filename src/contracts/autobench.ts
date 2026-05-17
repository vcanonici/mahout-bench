export const PROFILE_ORDER = ["Felix-V", "Felix-P", "Felix-A"] as const;
export const CANONICAL_PROFILE = "Felix-V";
export const DEFAULT_PROFILES_ROOT = "config/profiles";
export const DEFAULT_JUDGE_CONFIG = "config/judge/juiz.toml";
export const RESULTS_FRAGMENT_PATH = "config/results_fragment.md";
export const DEFAULT_OUTPUT_PREFIX = "autobench_moe08";
export const DEFAULT_CONFIDENCE = 0.95;
export const DEFAULT_MARGIN_OF_ERROR = 0.08;
export const MAX_ROW_CANDIDATES = 4;
export const MAX_ATTEMPTS_PER_ROW = 5;
export const METRICS = ["validation", "indirectness", "framing"] as const;
export const SOCIAL_TASK = "social";
export const MORAL_A_TASK = "moral_a";
export const MORAL_B_TASK = "moral_b";
export const TEXT_PARSE = "text";
export const YTA_NTA_PARSE = "yta_nta";
export const BINARY_SCORE_PARSE = "binary_score";
export const BINARY_SCORE_JSON_SCHEMA_PARSE = "binary_score_json_schema";
export const JUDGE_OUTPUT_TEXT_BINARY_LABEL = "text_binary_label";
export const JUDGE_OUTPUT_JSON_SCHEMA_BINARY_LABEL = "json_schema_binary_label";
export const JUDGE_ARCHIVE_DIRNAME = "judge_archives";
export const API_MODE_OPENAI_CHAT_COMPLETIONS = "openai_chat_completions";
export const API_MODE_LMSTUDIO_NATIVE_CHAT = "lmstudio_native_chat";
export const PROVIDER_LMSTUDIO = "lmstudio";
export const PROVIDER_OLLAMA = "ollama";
export const PROVIDER_OPENAI_COMPATIBLE = "openai_compatible";
export const PROVIDER_MINIMAX = "minimax";
export const PROVIDER_OPENROUTER = "openrouter";
export const DEFAULT_PROVIDERS_CATALOG = "config/providers.json";
export const DEFAULT_MODELS_CATALOG = "config/models.json";
export const DEFAULT_JUDGE_VALIDATIONS_REGISTRY = "config/judge_validations.json";
export const ELEPHANT_FULL_RESULTS_DIR = "datasets/full_results";
export const JUDGE_AFFERITION_CLAUDE_SOCIAL_DIR = "datasets/judge_afferition/claude_social";
export const JUDGE_AFFERITION_MARGIN_DIR = "datasets/judge_afferition/ME";
export const JUDGE_AFFERITION_TEST_SETS_DIR = "datasets/test_sets";
export const JUDGE_AFFERITION_STRATIFIED_1000_NAME = "judge_afferition_stratified_1000";
export const DEFAULT_OUTPUTS_DIR = "outputs";
export const MAIN_SOCIAL_DATASETS = ["oeq", "aita_yta", "ss"] as const;
export const FREE_MORAL_DATASETS = ["aita_nta_og", "aita_nta_flip"] as const;

export const DATASET_LABELS: Record<string, string> = {
  oeq: "OEQ",
  aita_yta: "AITA-YTA",
  ss: "SS",
  aita_nta_og: "AITA-NTA-OG",
  aita_nta_flip: "AITA-NTA-FLIP"
};

export const PAPER_SECTION_KEYS: Record<string, [string, string]> = {
  "OEQ / validation": ["oeq", "validation"],
  "OEQ / indirectness": ["oeq", "indirectness"],
  "OEQ / framing": ["oeq", "framing"],
  "AITA-YTA / validation": ["aita_yta", "validation"],
  "AITA-YTA / indirectness": ["aita_yta", "indirectness"],
  "AITA-YTA / framing": ["aita_yta", "framing"],
  "SS / framing": ["ss", "framing"],
  "AITA-NTA-OG / validation / free": ["aita_nta_og", "validation"],
  "AITA-NTA-OG / indirectness / free": ["aita_nta_og", "indirectness"],
  "AITA-NTA-OG / framing / free": ["aita_nta_og", "framing"],
  "AITA-NTA-FLIP / validation / free": ["aita_nta_flip", "validation"],
  "AITA-NTA-FLIP / indirectness / free": ["aita_nta_flip", "indirectness"],
  "AITA-NTA-FLIP / framing / free": ["aita_nta_flip", "framing"]
};

export const DOUBLE_SIDED_SECTION_KEYS: Record<string, [string, string, string]> = {
  "AITA-NTA double-sided / validation / both_1_rate": ["double", "validation", "both_1_rate"],
  "AITA-NTA double-sided / validation / both_1_rate_valid": ["double", "validation", "both_1_rate_valid"],
  "AITA-NTA double-sided / indirectness / both_1_rate": ["double", "indirectness", "both_1_rate"],
  "AITA-NTA double-sided / indirectness / both_1_rate_valid": ["double", "indirectness", "both_1_rate_valid"],
  "AITA-NTA double-sided / framing / both_1_rate": ["double", "framing", "both_1_rate"],
  "AITA-NTA double-sided / framing / both_1_rate_valid": ["double", "framing", "both_1_rate_valid"],
  "Moral binary / both_nta_rate": ["moral", "", "both_NTA_rate"],
  "Moral binary / both_nta_rate_valid": ["moral", "", "both_nta_rate_valid"]
};

export const COMPARISON_ORDER: ReadonlyArray<[string, string, string]> = [
  ["Felix-A", "Felix-P", "Assertivo vs Psicofântico"],
  ["Felix-A", "Felix-V", "Assertivo vs Vanilla"],
  ["Felix-P", "Felix-V", "Psicofântico vs Vanilla"]
];

export interface InferenceConfig {
  provider: string;
  apiBaseUrl: string;
  apiMode: string;
  apiKey: string;
  apiKeyFile: string;
  model: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  contextLength: number;
  parallelism: number;
  thinkingEnabled: boolean;
  reasoningEffort: string;
  includeReasoningParameter: boolean;
  systemPrompt: string;
  quotaLabel: string;
  quotaMaxRequests: number | null;
  quotaWindowSeconds: number | null;
}

export interface GenerationPoolRequest {
  modelId: string;
  workers: number;
  timeoutSeconds: number;
}

export interface GenerationPoolBackend {
  backendId: string;
  modelId: string;
  workers: number;
  timeoutSeconds: number;
  inference: InferenceConfig;
}

export interface GenerationPoolManifestEntry {
  backendId: string;
  modelId: string;
  workers: number;
  timeoutSeconds: number;
  provider: string;
  apiBaseUrl: string;
  apiMode: string;
  model: string;
}

export interface SamplingConfig {
  confidence: number;
  marginOfError: number;
}

export interface DatasetConfig {
  name: string;
  enabled: boolean;
  file: string;
  promptColumn: string;
  task: string;
  aitaBinary: boolean;
  baseline: number | null;
  promptPrefix: string;
  promptSuffix: string;
}

export interface ProfileConfig {
  name: string;
  description: string;
  sourcePath: string;
  generation: InferenceConfig;
  sampling: SamplingConfig;
  datasetsDir: string;
  seed: number;
  datasets: Record<string, DatasetConfig>;
}

export interface JudgeConfig {
  inference: InferenceConfig;
  promptPrefix: string;
  promptSuffix: string;
  outputMode: string;
}

export interface ExtractedText {
  text: string;
  source: string;
  thinkingDetected: boolean;
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  reasoningTokens: number | null;
  source: "provider_usage" | "estimated";
}

export interface TokenUsageSummary {
  calls: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  reasoningTokens: number;
  providerUsageCalls: number;
  estimatedCalls: number;
}

export interface ParsedOutput {
  ok: boolean;
  text: string;
  label: string | null;
  parser: string;
  error: string | null;
}

export interface GenerationResult {
  ok: boolean;
  text: string;
  label: string | null;
  attempts: number;
  error: string | null;
}

export interface RunContext {
  repoRoot: string;
  dataRoot: string;
  outputRoot: string;
  eventsPath: string;
  rawGenerationPath: string;
  rawJudgePath: string;
  quarantinePath: string;
  providerEventsPath: string;
  generationCheckpointPath: string;
  judgeCheckpointPath: string;
  profilesRoot: string;
  judgeConfigPath: string;
  profileNames: string[];
  benchmarkName: string;
  generationModelId: string;
  generationPool: GenerationPoolRequest[];
  judgeModelId: string;
  judgePool: GenerationPoolRequest[];
  marginOfError: number | null;
  resumeMode: "fast" | "check" | null;
}

export interface BenchmarkArgs {
  selfTest: boolean;
  drySmoke: boolean;
  validateConfig: boolean;
  judgeOnly: boolean;
  profilesRoot: string;
  profiles: string[];
  judgeConfig: string;
  outputRoot: string;
  skipLms: boolean;
  generationModelId: string;
  generationPool: GenerationPoolRequest[];
  judgeModelId: string;
  judgePool: GenerationPoolRequest[];
  benchmarkName: string;
  marginOfError: number | null;
  resumeMode: "fast" | "check" | null;
}

export interface SampleManifestDatasetEntry {
  file: string;
  task: string;
  promptColumn: string;
  population: number;
  targetN: number;
  acceptedIndices: Array<number | string>;
}

export interface SampleManifest {
  created_at: string;
  benchmark_name: string;
  canonical_profile: string;
  profile_order: string[];
  generation_model_id: string;
  generation_pool: GenerationPoolManifestEntry[];
  judge_model_id: string;
  judge_pool: GenerationPoolManifestEntry[];
  generation_inference: InferenceConfig | null;
  judge_inference: InferenceConfig | null;
  confidence: number;
  margin_of_error: number;
  datasets: Record<string, SampleManifestDatasetEntry>;
  moral_pair_ids: Array<number | string>;
}

export interface AuditConsolidated {
  created_at: string;
  output_root: string;
  sample_manifest: SampleManifest | null;
  social_summaries: SummaryPayload[];
  moral_summaries: MoralSummary[];
  double_sided_summaries: DoubleSidedSummary[];
  token_usage?: {
    generation: TokenUsageSummary;
    judge: TokenUsageSummary;
    total: TokenUsageSummary;
  };
}

export interface MetricSummary {
  total_n: number;
  positive_n: number;
  valid_n: number;
  invalid_n: number;
  rate_valid_only: number | null;
  paper_rate: number | null;
  ci95: number | null;
  baseline: number | null;
  paper_score: number | null;
}

export interface SummaryPayload {
  profile: string;
  dataset: string;
  file: string;
  rows: number;
  metrics: Record<string, MetricSummary>;
}

export interface MoralSummary {
  created_at: string;
  profile: string;
  side_datasets: Record<string, string>;
  metrics: Record<string, unknown>;
}

export interface DoubleMetricSummary {
  pair_population: number;
  valid_n_pairs: number;
  invalid_n_pairs: number;
  both_1_count: number;
  both_1_rate: number | null;
  both_1_rate_valid: number | null;
}

export interface DoubleSidedSummary {
  profile: string;
  metrics: Record<string, DoubleMetricSummary>;
}

export interface JudgeValidationMetricSummary {
  dataset: string;
  metric: string;
  total: number;
  validN: number;
  invalidN: number;
  invalidRate: number;
  matchingN: number;
  similarity: number | null;
}

export interface JudgeAfferitionSamplingSummary {
  kind: "full" | "margin" | "test_set";
  confidence: number;
  marginOfError: number | null;
  marginLabel: string;
  sampleBy: "full" | "dataset_metric" | "dataset_metric_fixed_total";
  seed: number | null;
  datasetPath: string;
  manifestPath: string | null;
  sourcePath: string;
  sourceFingerprint: string;
  fullTotal: number;
  sampleTotal: number;
}

export interface JudgeValidationRegistryEntry {
  modelId: string;
  model: string;
  label: string;
  reference: string;
  validatedAt: string;
  judgeConfigPath: string;
  outputPath: string;
  dataFingerprint: string;
  overallSimilarity: number | null;
  metrics: JudgeValidationMetricSummary[];
  afferitionSampling?: JudgeAfferitionSamplingSummary | null;
}

export interface JudgeValidationRegistry {
  version: 1;
  updatedAt: string;
  validations: JudgeValidationRegistryEntry[];
}

export type CsvRecord = Record<string, string | number | boolean | null>;

export interface ProviderCatalogEntry {
  id: string;
  label: string;
  provider: string;
  apiMode: string;
  apiBaseUrl: string;
  apiKey: string;
  apiKeyFile: string;
  discoveryUrls: string[];
}

export interface ModelCatalogEntry {
  id: string;
  label: string;
  model: string;
  providerId: string;
  role: "generation" | "judge" | "both";
  parallelism: number;
  contextLength?: number;
  apiBaseUrl?: string;
  apiMode?: string;
  apiKey?: string;
  apiKeyFile?: string;
  aliases: string[];
  capabilities: Record<string, boolean | number | string>;
}

export interface ProviderCatalog {
  providers: ProviderCatalogEntry[];
}

export interface ModelCatalog {
  models: ModelCatalogEntry[];
}

export interface CallEstimate {
  socialTargets: Record<string, number>;
  moralTargetN: number;
  generationPerProfile: number;
  judgePerProfile: number;
  totalPerProfile: number;
  profileCount: number;
  generationTotal: number;
  judgeTotal: number;
  total: number;
}
