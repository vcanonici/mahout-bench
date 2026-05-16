import { datasetPath } from "../config/loadConfig.js";
import {
  MAX_ATTEMPTS_PER_ROW,
  MAX_ROW_CANDIDATES,
  TEXT_PARSE,
  YTA_NTA_PARSE,
  type CsvRecord,
  type DatasetConfig,
  type GenerationPoolBackend,
  type GenerationResult,
  type ProfileConfig,
  type RunContext
} from "../contracts/autobench.js";
import { jsonSafe, readCsvFile, stableSeedOffset } from "../io/filesystem.js";
import { BenchmarkAbort } from "../pipeline/benchmarkAbort.js";
import { logEvent, quarantine } from "../pipeline/runContext.js";
import { commonMoralIds, shuffleIds } from "../sampling/samplePlanner.js";
import { interactiveMenuWithObserver, type StageHandle, type TerminalObserver } from "../runtime/terminalObserver.js";
import { runGenerationQueue } from "./generationScheduler.js";
import { generateOne } from "./generateOne.js";
import { makeResponseRecord, writeDatasetOutputs } from "./responseRecords.js";

type MoralCandidateResult = {
  a: { free: GenerationResult; binary: GenerationResult };
  b: { free: GenerationResult; binary: GenerationResult };
};

type MoralRecords = {
  aFree: CsvRecord[];
  aBinary: CsvRecord[];
  bFree: CsvRecord[];
  bBinary: CsvRecord[];
};

/**
 * Builds the canonical paired moral sample for the reference profile.
 */
export async function canonicalMoralGeneration(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  aDataset: DatasetConfig;
  bDataset: DatasetConfig;
  targetN: number;
  generationPool: GenerationPoolBackend[];
  observer: TerminalObserver;
}): Promise<string[]> {
  const { ctx, profile, aDataset, bDataset, targetN, generationPool, observer } = args;
  const pairRows = loadMoralPairRows(ctx, profile, aDataset, bDataset);
  const commonIds = commonMoralIds(pairRows.aRows, pairRows.bRows);
  const shuffledIds = shuffleIds(commonIds, profile.seed + stableSeedOffset("moral_pairs"));
  const acceptedIds: string[] = [];
  const records = emptyMoralRecords();
  const isFullBench = targetN >= commonIds.length;
  let cursor = 0;
  const stage = observer.startStage(`${profile.name}:moral:build-sample`, targetN * 4);
  logEvent(ctx, "stage_started", observer, {
    profile: profile.name,
    dataset: "moral_pair",
    mode: "free+binary",
    stage: "build-sample",
    target_n: targetN * 4
  });

  try {
    while (acceptedIds.length < targetN) {
      const result = await tryAcceptMoralSlot({
        ctx,
        profile,
        aDataset,
        bDataset,
        aById: pairRows.aById,
        bById: pairRows.bById,
        shuffledIds,
        cursor,
        acceptedIds,
        records,
        generationPool,
        observer,
        acceptFailed: isFullBench
      });
      cursor = result.cursor;
      if (!result.accepted) {
        await handleCanonicalMoralFailure(acceptedIds.length, observer);
      } else {
        const okUnits = Math.max(0, 4 - result.failedUnits);
        if (okUnits > 0) {
          stage.advance(okUnits);
          observer.advanceOverall(okUnits);
        }
        if (result.failedUnits > 0) {
          const outcome = { ok: false, failureKind: "moral_generation_refused" };
          stage.advance(result.failedUnits, outcome);
          observer.advanceOverall(result.failedUnits, outcome);
        }
      }
    }
  } finally {
    stage.close();
    logEvent(ctx, "stage_finished", observer, {
      profile: profile.name,
      dataset: "moral_pair",
      mode: "free+binary",
      stage: "build-sample",
      accepted: acceptedIds.length
    });
  }

  writeMoralOutputs(ctx, profile, aDataset, bDataset, records);
  return acceptedIds;
}

/**
 * Replays paired moral IDs for a non-reference profile.
 */
export async function fixedMoralGeneration(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  aDataset: DatasetConfig;
  bDataset: DatasetConfig;
  acceptedIds: string[];
  generationPool: GenerationPoolBackend[];
  observer: TerminalObserver;
}): Promise<void> {
  const { ctx, profile, aDataset, bDataset, acceptedIds, generationPool, observer } = args;
  const pairRows = loadMoralPairRows(ctx, profile, aDataset, bDataset);
  const records = emptyMoralRecords();
  const stage = observer.startStage(`${profile.name}:moral:fixed-sample`, acceptedIds.length * 4);
  logEvent(ctx, "stage_started", observer, {
    profile: profile.name,
    dataset: "moral_pair",
    mode: "free+binary",
    stage: "fixed-sample",
    target_n: acceptedIds.length * 4
  });

  try {
    const generatedRows = await runGenerationQueue(acceptedIds, generationPool, async (rowId) => {
      const aRow = requireMoralRow(pairRows.aById, rowId);
      const bRow = requireMoralRow(pairRows.bById, rowId);
      const resultMap = await generateMoralCandidate(ctx, profile, aDataset, bDataset, aRow, bRow, rowId, generationPool, observer, stage);
      const status = await fixedMoralStatusOrAbort(allMoralResultsOk(resultMap), `${profile.name}/moral/${rowId}`, observer);
      return { aRow, bRow, rowId, resultMap, status };
    });
    for (const scheduled of generatedRows) {
      const generated = scheduled.result;
      pushMoralRecords(records, profile, generated.aRow, generated.bRow, generated.rowId, generated.resultMap, generated.status);
    }
  } finally {
    stage.close();
    logEvent(ctx, "stage_finished", observer, {
      profile: profile.name,
      dataset: "moral_pair",
      mode: "free+binary",
      stage: "fixed-sample",
      rows: records.aFree.length
    });
  }

  writeMoralOutputs(ctx, profile, aDataset, bDataset, records);
}

/**
 * Generates free-text and binary answers for both sides of one moral pair.
 */
export async function generateMoralCandidate(
  ctx: RunContext,
  profile: ProfileConfig,
  aDataset: DatasetConfig,
  bDataset: DatasetConfig,
  aRow: CsvRecord,
  bRow: CsvRecord,
  rowId: number | string,
  generationPool: GenerationPoolBackend[],
  observer: TerminalObserver,
  progress?: StageHandle
): Promise<MoralCandidateResult> {
  const generated = await runGenerationQueue(
    [
      { dataset: aDataset, row: aRow, mode: "free", parser: TEXT_PARSE },
      { dataset: aDataset, row: aRow, mode: "binary", parser: YTA_NTA_PARSE },
      { dataset: bDataset, row: bRow, mode: "free", parser: TEXT_PARSE },
      { dataset: bDataset, row: bRow, mode: "binary", parser: YTA_NTA_PARSE }
    ],
    generationPool,
    (item, backend) => generateOne({
      ctx,
      profile,
      dataset: item.dataset,
      row: item.row,
      rowIndex: rowId,
      mode: item.mode,
      parser: item.parser,
      backend,
      observer,
      progress
    })
  );
  const [aFree, aBinary, bFree, bBinary] = generated.map((item) => item.result);
  return {
    a: { free: aFree!, binary: aBinary! },
    b: { free: bFree!, binary: bBinary! }
  };
}

function loadMoralPairRows(ctx: RunContext, profile: ProfileConfig, aDataset: DatasetConfig, bDataset: DatasetConfig): {
  aRows: CsvRecord[];
  bRows: CsvRecord[];
  aById: Map<string, CsvRecord>;
  bById: Map<string, CsvRecord>;
} {
  const aRows = readCsvFile(datasetPath(ctx.repoRoot, profile, aDataset));
  const bRows = readCsvFile(datasetPath(ctx.repoRoot, profile, bDataset));
  return {
    aRows,
    bRows,
    aById: new Map(aRows.map((row) => [String(row.id ?? ""), row])),
    bById: new Map(bRows.map((row) => [String(row.id ?? ""), row]))
  };
}

async function tryAcceptMoralSlot(args: {
  ctx: RunContext;
  profile: ProfileConfig;
  aDataset: DatasetConfig;
  bDataset: DatasetConfig;
  aById: Map<string, CsvRecord>;
  bById: Map<string, CsvRecord>;
  shuffledIds: string[];
  cursor: number;
  acceptedIds: string[];
  records: MoralRecords;
  generationPool: GenerationPoolBackend[];
  observer: TerminalObserver;
  acceptFailed?: boolean;
}): Promise<{ accepted: boolean; cursor: number; failedUnits: number }> {
  let cursor = args.cursor;
  const slot = args.acceptedIds.length;
  for (let candidate = 1; candidate <= MAX_ROW_CANDIDATES; candidate += 1) {
    if (cursor >= args.shuffledIds.length) {
      throw new Error("Moral pair reserve pool exhausted");
    }
    const rowId = args.shuffledIds[cursor]!;
    cursor += 1;
    const aRow = requireMoralRow(args.aById, rowId);
    const bRow = requireMoralRow(args.bById, rowId);
    const resultMap = await generateMoralCandidate(
      args.ctx,
      args.profile,
      args.aDataset,
      args.bDataset,
      aRow,
      bRow,
      rowId,
      args.generationPool,
      args.observer
    );
    if (allMoralResultsOk(resultMap)) {
      args.acceptedIds.push(rowId);
      pushMoralRecords(args.records, args.profile, aRow, bRow, rowId, resultMap, "ok");
      return { accepted: true, cursor, failedUnits: 0 };
    }
    quarantine(args.ctx, "canonical_moral_pair_failed", args.observer, {
      profile: args.profile.name,
      slot,
      row_id: rowId,
      candidate_number: candidate,
      errors: jsonSafe(resultMap)
    });
    if (args.acceptFailed) {
      args.acceptedIds.push(rowId);
      pushMoralRecords(args.records, args.profile, aRow, bRow, rowId, resultMap, "refused");
      return { accepted: true, cursor, failedUnits: countFailedMoralResults(resultMap) };
    }
  }
  return { accepted: false, cursor, failedUnits: 0 };
}

function emptyMoralRecords(): MoralRecords {
  return {
    aFree: [],
    aBinary: [],
    bFree: [],
    bBinary: []
  };
}

function pushMoralRecords(
  records: MoralRecords,
  profile: ProfileConfig,
  aRow: CsvRecord,
  bRow: CsvRecord,
  rowId: number | string,
  resultMap: MoralCandidateResult,
  status: string
): void {
  records.aFree.push(makeResponseRecord(aRow, profile, rowId, resultMap.a.free, "free", status));
  records.aBinary.push(makeResponseRecord(aRow, profile, rowId, resultMap.a.binary, "binary", status));
  records.bFree.push(makeResponseRecord(bRow, profile, rowId, resultMap.b.free, "free", status));
  records.bBinary.push(makeResponseRecord(bRow, profile, rowId, resultMap.b.binary, "binary", status));
}

function writeMoralOutputs(
  ctx: RunContext,
  profile: ProfileConfig,
  aDataset: DatasetConfig,
  bDataset: DatasetConfig,
  records: MoralRecords
): void {
  writeDatasetOutputs(ctx, profile, aDataset, "free", records.aFree);
  writeDatasetOutputs(ctx, profile, aDataset, "binary", records.aBinary);
  writeDatasetOutputs(ctx, profile, bDataset, "free", records.bFree);
  writeDatasetOutputs(ctx, profile, bDataset, "binary", records.bBinary);
}

function allMoralResultsOk(resultMap: MoralCandidateResult): boolean {
  return Object.values(resultMap).flatMap((side) => Object.values(side)).every((result) => result.ok);
}

function countFailedMoralResults(resultMap: MoralCandidateResult): number {
  return Object.values(resultMap).flatMap((side) => Object.values(side)).filter((result) => !result.ok).length;
}

function requireMoralRow(rowsById: Map<string, CsvRecord>, rowId: string): CsvRecord {
  const row = rowsById.get(String(rowId));
  if (!row) {
    throw new Error(`Missing moral pair rows for id ${rowId}`);
  }
  return row;
}

async function handleCanonicalMoralFailure(slot: number, observer: TerminalObserver): Promise<void> {
  const choice = await interactiveMenuWithObserver(
    observer,
    `moral slot ${slot} failed after ${MAX_ROW_CANDIDATES} reserve pairs x ${MAX_ATTEMPTS_PER_ROW} attempts.`,
    {
      r: "try four more reserve pairs",
      a: "fail hard and save current artifacts"
    }
  );
  if (choice === "a") {
    throw new BenchmarkAbort("User aborted canonical moral sampling");
  }
}

async function fixedMoralStatusOrAbort(isOk: boolean, label: string, observer: TerminalObserver): Promise<string> {
  if (isOk) {
    return "ok";
  }
  const choice = await interactiveMenuWithObserver(
    observer,
    `${label} failed after ${MAX_ATTEMPTS_PER_ROW} attempts.`,
    {
      f: "mark refused/fail for this profile and continue",
      a: "fail hard and save current artifacts"
    }
  );
  if (choice === "a") {
    throw new BenchmarkAbort(`User aborted fixed moral generation for ${label}`);
  }
  return "refused";
}
