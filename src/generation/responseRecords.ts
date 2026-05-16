import {
  type CsvRecord,
  type DatasetConfig,
  type GenerationResult,
  type ProfileConfig,
  type RunContext
} from "../contracts/autobench.js";
import { writeCsvJsonl } from "../io/filesystem.js";
import { outputFileForMode } from "../scoring/scoreEngine.js";

/**
 * Builds the persisted response row for one generated benchmark answer.
 */
export function makeResponseRecord(
  row: CsvRecord,
  profile: ProfileConfig,
  rowIndex: number | string,
  result: GenerationResult,
  mode: string,
  status: string
): CsvRecord {
  return {
    ...row,
    _source_index: rowIndex,
    _mode: mode,
    _status: status,
    _attempts: result.attempts,
    _error: result.error,
    [`${profile.name}_response`]: result.text,
    ...(result.label === null ? {} : { [`${profile.name}_label`]: result.label })
  };
}

/**
 * Writes a dataset response table to both CSV and JSONL using the canonical path.
 */
export function writeDatasetOutputs(
  ctx: RunContext,
  profile: ProfileConfig,
  dataset: DatasetConfig,
  mode: string,
  records: CsvRecord[]
): void {
  writeCsvJsonl(records, outputFileForMode(ctx, profile.name, dataset.name, mode));
}
