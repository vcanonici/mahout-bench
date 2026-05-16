# Mahout Bench Status CLI

`mahout-bench status` is a read-only command for inspecting benchmark runs, progress, ETA, runtime freshness, and the artifacts behind the estimate.

It is designed for both humans and AI agents. The default output explains the state in plain text and also includes a stable JSON block that agents can parse.

## Quick Start

List available runs:

```bash
mahout-bench status
```

Inspect run number 1:

```bash
mahout-bench status --run "1"
```

Return JSON only:

```bash
mahout-bench status --run "1" --json
```

Inspect an explicit run directory:

```bash
mahout-bench status --output-root "$MAHOUT_BENCH_HOME/outputs/name_of_run"
```

## How Run Selection Works

When called without `--run` or `--output-root`, the command lists candidate runs from the Mahout Bench output directory.

Runs are grouped and numbered in this order:

1. active or fresh runs, newest first;
2. paused or stale runs, newest first;
3. completed runs, newest first.

If an agent does not know which run to inspect, it should show the list to the user and ask which run number to use.

## Output Directory

Mahout Bench stores mutable outputs outside the installed package.

The default output directory is:

```text
$MAHOUT_BENCH_HOME/outputs
```

When `MAHOUT_BENCH_HOME` is not set, the default is:

```text
./.mahout-bench/outputs
```

Use `--outputs-dir <path>` to list runs from another output directory, or `--output-root <path>` to inspect one run directly.

## Human Output

The human report explains:

- selected run number or explicit output root;
- run name and output path;
- whether the run appears active, paused, completed, or unknown;
- whether the data is fresh or stale;
- current step;
- step progress;
- overall progress;
- failures;
- ETA;
- recent throughput;
- artifact line counts;
- warnings;
- the command to repeat the same query.

## JSON Output

Use `--json` for automation:

```bash
mahout-bench status --run "1" --json
```

The JSON includes:

```json
{
  "schema_version": 1,
  "selected_run_number": 1,
  "run": {
    "name": "name_of_run",
    "output_root": "/path/to/run",
    "updated_at": "2026-05-16T13:43:03.842Z",
    "status": "active",
    "freshness": "fresh",
    "confidence": "high"
  },
  "stage": {
    "name": "Felix-V:moral:build-sample",
    "value": 3116,
    "total": 6364,
    "remaining": 3248,
    "percent": 48.96
  },
  "overall": {
    "value": 18423,
    "total": 43575,
    "remaining": 25152,
    "percent": 42.28,
    "ok": 18420,
    "failed": 3
  },
  "eta": {
    "seconds": 101640,
    "hours": 28.23,
    "human": "28h 14m",
    "source": "ui_state.json"
  }
}
```

The default pretty output also contains the JSON between these markers:

```text
---BEGIN MAHOUT_STATUS_JSON---
...
---END MAHOUT_STATUS_JSON---
```

## Artifact Sources

The command reads:

- `ui_state.json` for current stage, progress, failures, and ETA;
- `run_events.jsonl` for recent throughput and fallback progress;
- `raw_generation.jsonl`, `raw_judge.jsonl`, and `ui_calls.jsonl` for artifact counts and operational evidence.

If `ui_state.json` is missing, the command can still produce a partial report from JSONL artifacts and marks confidence as `partial`.

## Agent Procedure

Agents should follow this procedure:

1. Run `mahout-bench status`.
2. If there is more than one plausible run, show the list to the user and ask which number to inspect.
3. Run `mahout-bench status --run "<number>"`.
4. For machine parsing, use `mahout-bench status --run "<number>" --json`.
5. Report current step, overall progress, ETA, freshness, confidence, and warnings.

Do not infer a run number when the selector shows multiple plausible choices.

## Exit Behavior

- No runs found: exits `0` and explains that no runs with `ui_state.json` were found.
- Invalid `--run`: exits `1`, prints the current selector, and reports the missing number.
- Missing `ui_state.json` for explicit `--output-root`: returns a partial report when JSONL artifacts exist.
- Corrupt optional JSONL: keeps the report when possible and emits warnings.
