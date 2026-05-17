# mahout-bench

[![npm version](https://img.shields.io/npm/v/mahout-bench)](https://www.npmjs.com/package/mahout-bench)
[![MIT license](https://img.shields.io/npm/l/mahout-bench)](LICENSE)

A CLI benchmark for measuring and mitigating sycophancy in large language models.

Mahout Bench moves from "is this model sycophantic?" to "what can I change to reduce that behavior?" It evaluates LLMs across configurable system prompts, inference hyperparameters, judges, providers, and sampling margins — so the same model can be compared under different mitigation strategies before deciding whether the model itself needs replacing.

The benchmark uses a margin-of-error sampling model to reduce generation and judge calls while preserving controlled comparisons. It supports full and reduced-call runs, automated CLI execution, an interactive TUI, resumable artifacts, multi-provider backends, and judge **aferition** workflows that compare candidate judges against GPT-4o reference labels.

**Aferition** (from Latin *aferetio*): the process of evaluating a judge's accuracy by measuring its agreement with a reference standard. In Mahout Bench, it refers to workflows that assess whether a candidate judge can reliably replace GPT-4o as the labeling authority.

## Install

```bash
npm install -g mahout-bench
mahout-bench bootstrap
mahout-bench setup
mahout-bench run --dry-smoke
mahout-bench status
```

By default, mutable data lives in `./.mahout-bench`. Set `MAHOUT_BENCH_HOME` to use a shared or absolute data root:

```bash
export MAHOUT_BENCH_HOME="$HOME/.mahout-bench"
mahout-bench setup
```

## Quick start

```bash
npm install -g mahout-bench
mahout-bench bootstrap
mahout-bench setup
mahout-bench run --dry-smoke
mahout-bench status
```

For non-interactive diagnostics:

```bash
mahout-bench --no-bootstrap run --self-test
mahout-bench bootstrap --help
mahout-bench --no-bootstrap status
```

## Designed for long runs

The judge aferition workflow is designed for operational use: it can test alternative judges, record agreement against reference labels, compare margin configurations (full, 10pp, 8pp, 5pp, and others), and reuse aferition datasets so judge selection does not require repeating the most expensive calls. This makes Mahout Bench useful both for benchmark replication and for mitigation experiments over prompts, parameters, and model/provider choices.

## Commands

```bash
mahout-bench bootstrap
mahout-bench setup
mahout-bench run --self-test
mahout-bench run --dry-smoke
mahout-bench run --validate-config
mahout-bench status
mahout-bench status --run "1"
mahout-bench status --run "1" --json
mahout-bench tui
```

Bench resume supports `--resume-mode fast|check`. The TUI asks for `fast resume` or `checked resume` before selecting the run. Fast resume reconstructs completed generation from existing `responses*.jsonl` files and continues without replaying every checkpoint hit; checked resume also writes `resume_check_report.json` in the run directory.

`setup` downloads `mahout-bench-data-v0.0.5.zip` and its manifest from the `vcanonici/mahout-bench` GitHub Release, verifies SHA256 and size, extracts into the data root, and checks required dataset paths. Mahout Bench is the distributor/source of the setup bundle; ELEPHANT remains the upstream research/data origin and citation.

`bootstrap` is the recommended first command after npm install. It prepares the data root, asks for optional OpenRouter and MiniMax API keys, requires at least one local LM Studio or Ollama-compatible backend, lets you add any number of additional local-network or remote backends, and writes user-owned config under `MAHOUT_BENCH_HOME` or `./.mahout-bench`. It never writes real secrets into the npm package. First interactive runs also offer bootstrap when no bootstrap marker exists, and the TUI includes `bootstrap/configure providers`.

Profiles in `config/profiles` are intentionally small: they define profile identity, generation hyperparameters, and `system_prompt`. The TUI chooses model, provider, pools, precision, judge, and output root at runtime. Use `config/profile.example.toml` as the copyable template for new profiles. Profiles that omit `[datasets]` use the default benchmark contract in `datasets/full_results` for `OEQ.csv`, `AITA-YTA.csv`, `SS.csv`, `AITA-NTA-OG.csv`, and `AITA-NTA-FLIP.csv`.

New generation traces include `system_prompt_sha256` and `system_prompt_chars`, so runs can audit which system prompt was active without duplicating full prompt text in every `raw_generation.jsonl` row.

## Status and ETA

```bash
mahout-bench status
mahout-bench status --run "1"
mahout-bench status --run "1" --json
mahout-bench status --output-root "$MAHOUT_BENCH_HOME/outputs/name_of_run"
```

The default output is intentionally verbose: it prints a human explanation plus a stable JSON block between `---BEGIN MAHOUT_STATUS_JSON---` and `---END MAHOUT_STATUS_JSON---`. Use `--json` for automation.

Full human/agent documentation is available in [`docs/status.md`](docs/status.md).

## Public package boundary

The npm package intentionally excludes:

- `AgentDATA/`
- private root `AGENTS.md`
- `.env` / `.ENV`
- real secrets
- run outputs
- dataset archives and extracted datasets
- private DSI/Ollama tunnel configuration

Provider configuration is public and generic: LM Studio local endpoints, OpenRouter, and MiniMax. Put real API keys outside the package and point config at your own secret files or environment-managed copies.

LM Studio is treated as a passive provider. Mahout Bench does not load, unload, start, or tune LM Studio models; keep models loaded and configured in LM Studio GUI or your own operational tooling. The runner only sends HTTP inference requests with the selected model, context window, and generation hyperparameters.

The npm package includes a public `AGENTS.md` for AI coding agents. It documents the package architecture, data-root contract, validation commands, style rules, and packaging boundary without carrying private repository instructions.



## How to Cite Mahout Bench

Citation metadata is available in `CITATION.cff`.

```bibtex
@software{canonici_mahout_bench_2026,
  author = {Vinicius Garcia Canonici and Luis Miguel da Rocha de Matos and Ana Paula de Carvalho Soares},
  title = {Mahout Bench: From Measuring to Mitigating Sycophancy in Large Language Models},
  version = {1.0.1},
  year = {2026},
  url = {https://github.com/vcanonici/mahout-bench},
  note = {Public TypeScript runner for measuring and mitigating sycophancy in large language models}
}
```

Authors:

- Vinicius Garcia Canonici, ORCID `0009-0006-8269-9004`, Departamento de Sistemas de Informacao (DSI), Universidade do Minho; CIPsi, Escola de Psicologia, Universidade do Minho
- Luis Miguel da Rocha de Matos, Departamento de Sistemas de Informacao (DSI), Universidade do Minho
- Ana Paula de Carvalho Soares, Departamento de Psicologia Basica, Escola de Psicologia, Universidade do Minho

## Philosophy

Mahout Bench is released under MIT to maximize reuse, modification, forking, benchmarking, and integration into research and production workflows.

See [`docs/philosophy.md`](docs/philosophy.md) for the full statement.

## Data Citation

Mahout Bench uses/adapts data and procedure from ELEPHANT / Social Sycophancy:

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. "ELEPHANT: Measuring and understanding social sycophancy in LLMs" / "Social Sycophancy: A Broader Understanding of LLM Sycophancy."

Links:

- https://arxiv.org/abs/2505.13995
- https://openreview.net/forum?id=igbRHKEiAs
- https://github.com/myracheng/elephant

The upstream `myracheng/elephant` repository declares `CC0-1.0` for its released material. Mahout Bench code is MIT licensed; the separate setup data bundle records Mahout Bench distribution metadata plus upstream attribution and license metadata in its manifest.
