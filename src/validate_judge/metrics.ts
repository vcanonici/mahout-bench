import type { JudgeValidationMetricSummary } from "../contracts/autobench.js";
import type { BinaryLabel } from "./parseBinaryLabel.js";

export interface CandidateValidationResult {
  rowId: string;
  dataset: string;
  metric: string;
  referenceLabel: BinaryLabel;
  candidateLabel: BinaryLabel | null;
  validParse: boolean;
}

export interface SimilarityCounts {
  matching: number;
  mismatching: number;
  invalid: number;
  total: number;
}

/**
 * Counts valid label matches and invalid candidate outputs for judge afferition.
 */
export function computeSimilarityCounts(rows: CandidateValidationResult[]): SimilarityCounts {
  const counts: SimilarityCounts = { matching: 0, mismatching: 0, invalid: 0, total: rows.length };
  for (const row of rows) {
    if (!row.validParse || row.candidateLabel === null) {
      counts.invalid += 1;
      continue;
    }
    if (row.referenceLabel === row.candidateLabel) {
      counts.matching += 1;
    } else {
      counts.mismatching += 1;
    }
  }
  return counts;
}

/**
 * Computes the single official judge afferition metric: valid-label similarity.
 */
export function computeMetricSummary(dataset: string, metric: string, counts: SimilarityCounts): JudgeValidationMetricSummary {
  const validN = counts.matching + counts.mismatching;
  return {
    dataset,
    metric,
    total: counts.total,
    validN,
    invalidN: counts.invalid,
    invalidRate: counts.total === 0 ? 1 : counts.invalid / counts.total,
    matchingN: counts.matching,
    similarity: validN === 0 ? null : counts.matching / validN
  };
}

export function overallSimilarity(metrics: JudgeValidationMetricSummary[]): number | null {
  const totals = metrics
    .filter((metric) => metric.dataset === "global")
    .reduce(
      (current, metric) => ({
        matchingN: current.matchingN + metric.matchingN,
        validN: current.validN + metric.validN
      }),
      { matchingN: 0, validN: 0 }
    );
  return totals.validN === 0 ? null : totals.matchingN / totals.validN;
}
