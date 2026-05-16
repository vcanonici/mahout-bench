import seedrandom from "seedrandom";

import {
  DEFAULT_CONFIDENCE,
  DEFAULT_MARGIN_OF_ERROR,
  SOCIAL_TASK,
  type CallEstimate,
  type CsvRecord,
  type DatasetConfig,
  type ProfileConfig,
  type RunContext,
  type SamplingConfig,
  type GenerationPoolBackend,
  type GenerationPoolManifestEntry,
  type SampleManifest
} from "../contracts/autobench.js";
import { stableSeedOffset, utcNowIso } from "../io/filesystem.js";

export function sampleTargetN(population: number, sampling: SamplingConfig = defaultSampling()): number {
  if (population <= 0) {
    return 0;
  }
  if (sampling.marginOfError <= 0) {
    return population;
  }
  const zValue = inverseNormalCdf(1 - (1 - sampling.confidence) / 2);
  const n0 = (zValue * zValue * 0.25) / (sampling.marginOfError * sampling.marginOfError);
  const corrected = n0 / (1 + (n0 - 1) / population);
  return Math.min(population, Math.ceil(corrected));
}

export function shuffleIndices(rows: CsvRecord[], seed: number): number[] {
  const indices = rows.map((_row, index) => index);
  shuffleInPlace(indices, seed);
  return indices;
}

export function shuffleIds(ids: string[], seed: number): string[] {
  const cloned = [...ids];
  shuffleInPlace(cloned, seed);
  return cloned;
}

export function commonMoralIds(aRows: CsvRecord[], bRows: CsvRecord[]): string[] {
  const bIds = new Set(bRows.map((row) => String(row.id ?? "")));
  return aRows
    .map((row) => String(row.id ?? ""))
    .filter((value) => bIds.has(String(value)))
    .sort((left, right) => String(left).localeCompare(String(right)));
}

export function buildSampleManifest(args: {
  ctx: RunContext;
  referenceProfile: ProfileConfig;
  profileOrder: string[];
  generationPool: GenerationPoolBackend[];
  judgePool: GenerationPoolBackend[];
  judgeInference: SampleManifest["judge_inference"];
  socialIndices: Record<string, number[]>;
  datasetPopulations: Record<string, number>;
  moralIds: string[];
}): SampleManifest {
  const { ctx, referenceProfile, socialIndices, datasetPopulations, moralIds } = args;
  const datasets: SampleManifest["datasets"] = {};
  for (const dataset of Object.values(referenceProfile.datasets).filter((item) => item.enabled)) {
    const population = datasetPopulations[dataset.name] ?? 0;
    datasets[dataset.name] = {
      file: dataset.file,
      task: dataset.task,
      promptColumn: dataset.promptColumn,
      population,
      targetN: sampleTargetN(population, referenceProfile.sampling),
      acceptedIndices: dataset.task === SOCIAL_TASK ? socialIndices[dataset.name] ?? [] : []
    };
  }
  return {
    created_at: utcNowIso(),
    benchmark_name: ctx.benchmarkName,
    canonical_profile: referenceProfile.name,
    profile_order: args.profileOrder,
    generation_model_id: ctx.generationModelId || referenceProfile.generation.model,
    generation_pool: generationPoolManifest(args.generationPool),
    judge_model_id: ctx.judgeModelId,
    judge_pool: generationPoolManifest(args.judgePool),
    generation_inference: referenceProfile.generation,
    judge_inference: args.judgeInference,
    confidence: referenceProfile.sampling.confidence,
    margin_of_error: referenceProfile.sampling.marginOfError,
    datasets,
    moral_pair_ids: moralIds
  };
}

export function generationPoolManifest(backends: GenerationPoolBackend[]): GenerationPoolManifestEntry[] {
  return backends.map((backend) => ({
    backendId: backend.backendId,
    modelId: backend.modelId,
    workers: backend.workers,
    timeoutSeconds: backend.timeoutSeconds,
    provider: backend.inference.provider,
    apiBaseUrl: backend.inference.apiBaseUrl,
    apiMode: backend.inference.apiMode,
    model: backend.inference.model
  }));
}

export function socialTargetMap(reference: ProfileConfig, socialDatasets: DatasetConfig[], populations: Record<string, number>): Record<string, number> {
  return Object.fromEntries(
    socialDatasets.map((dataset) => [dataset.name, sampleTargetN(populations[dataset.name] ?? 0, reference.sampling)])
  );
}

export function plannedOverallUnits(profileCount: number, socialTargets: Record<string, number>, moralTargetN: number): number {
  const socialGenerationUnits = Object.values(socialTargets).reduce((total, value) => total + value, 0);
  const judgeSocialUnits = Object.entries(socialTargets).reduce((total, [datasetName, targetN]) => {
    const metricCount = datasetName === "ss" ? 1 : 3;
    return total + targetN * metricCount;
  }, 0);
  const generationUnits = profileCount * (socialGenerationUnits + moralTargetN * 4);
  const judgeMoralUnits = profileCount * moralTargetN * 3 * 2;
  const judgePreflightUnits = 3;
  return generationUnits + judgePreflightUnits + profileCount * judgeSocialUnits + judgeMoralUnits;
}

export function estimateCalls(args: {
  profileCount: number;
  socialDatasets: DatasetConfig[];
  populations: Record<string, number>;
  moralPopulation: number;
  sampling: SamplingConfig;
}): CallEstimate {
  const socialTargets = socialTargetMap(
    {
      sampling: args.sampling
    } as ProfileConfig,
    args.socialDatasets,
    args.populations
  );
  const moralTargetN = sampleTargetN(args.moralPopulation, args.sampling);
  const socialGenerationUnits = Object.values(socialTargets).reduce((total, value) => total + value, 0);
  const judgeSocialUnits = Object.entries(socialTargets).reduce((total, [datasetName, targetN]) => {
    const metricCount = datasetName === "ss" ? 1 : 3;
    return total + targetN * metricCount;
  }, 0);
  const judgePreflightUnits = 3;
  const generationPerProfile = socialGenerationUnits + moralTargetN * 4;
  const judgePerProfile = judgeSocialUnits + moralTargetN * 2 * 3;
  const generationTotal = generationPerProfile * args.profileCount;
  const judgeTotal = judgePerProfile * args.profileCount + judgePreflightUnits;
  return {
    socialTargets,
    moralTargetN,
    generationPerProfile,
    judgePerProfile,
    totalPerProfile: generationPerProfile + judgePerProfile,
    profileCount: args.profileCount,
    generationTotal,
    judgeTotal,
    total: generationTotal + judgeTotal
  };
}

function shuffleInPlace<T>(items: T[], seed: number): void {
  const rng = seedrandom(String(seed));
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    const current = items[index]!;
    items[index] = items[swapIndex]!;
    items[swapIndex] = current;
  }
}

function defaultSampling(): SamplingConfig {
  return {
    confidence: DEFAULT_CONFIDENCE,
    marginOfError: DEFAULT_MARGIN_OF_ERROR
  };
}

function inverseNormalCdf(probability: number): number {
  if (!(probability > 0 && probability < 1)) {
    throw new Error(`Probability must be between 0 and 1: ${probability}`);
  }

  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416
  ];
  const low = 0.02425;
  const high = 1 - low;

  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }

  if (probability > high) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }

  const q = probability - 0.5;
  const r = q * q;
  return (((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q /
    (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
}
