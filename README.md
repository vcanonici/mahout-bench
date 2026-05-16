# mahout-bench

Sycophantic Behavior Benchmark automated TUI for LLMs

Mahout Bench is a practical benchmark system for moving from "is this model sycophantic?" to "what can I change to reduce that behavior?" It evaluates large language models across configurable system prompts, inference hyperparameters, judges, providers, and sampling margins, so the same model can be compared under different mitigation strategies before deciding whether the model itself needs to be replaced.

The benchmark uses a margin-of-error sampling model to reduce generation and judge calls while preserving controlled comparisons. It supports full and reduced-call runs, automated CLI execution, an interactive TUI, resumable artifacts, multiprovider backends, LM Studio integration, and judge aferition workflows that compare candidate judges against GPT-4o reference labels.

The judge workflow is designed for operational use: it can test alternative judges, record agreement against reference labels, compare margin configurations such as full, 10pp, 8pp, 5pp and any other, and reuse aferition datasets so judge selection does not require repeating the most expensive calls. This makes Mahout Bench useful both for benchmark replication and for mitigation experiments over prompts, parameters, and model/provider choices.

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

`setup` downloads `mahout-bench-data-v0.0.5.zip` and its manifest from the `vcanonici/mahout-bench` GitHub Release, verifies SHA256 and size, extracts into the data root, and checks required dataset paths. Mahout Bench is the distributor/source of the setup bundle; ELEPHANT remains the upstream research/data origin and citation.

`bootstrap` is the recommended first command after npm install. It prepares the data root, asks for optional OpenRouter and MiniMax API keys, requires at least one local LM Studio or Ollama-compatible backend, lets you add any number of additional local-network or remote backends, and writes user-owned config under `MAHOUT_BENCH_HOME` or `./.mahout-bench`. It never writes real secrets into the npm package. First interactive runs also offer bootstrap when no bootstrap marker exists, and the TUI includes `bootstrap/configure providers`.

For non-interactive diagnostics:

```bash
mahout-bench --no-bootstrap run --self-test
mahout-bench bootstrap --help
mahout-bench --no-bootstrap status
```

## Status And ETA

`mahout-bench status` lists benchmark runs from the user-owned output directory and helps humans or agents choose a run before requesting details:

```bash
mahout-bench status
mahout-bench status --run "1"
mahout-bench status --run "1" --json
mahout-bench status --output-root "$MAHOUT_BENCH_HOME/outputs/name_of_run"
```

The default output is intentionally verbose: it prints a human explanation plus a stable JSON block between `---BEGIN MAHOUT_STATUS_JSON---` and `---END MAHOUT_STATUS_JSON---`. Use `--json` for automation.

Full human/agent documentation is available in [`docs/status.md`](docs/status.md).

## Public Package Boundary

The npm package intentionally excludes:

- `AgentDATA/`
- private root `AGENTS.md`
- `.env` / `.ENV`
- real secrets
- run outputs
- dataset archives and extracted datasets
- private DSI/Ollama tunnel configuration

Provider configuration is public and generic: LM Studio local endpoints, OpenRouter, and MiniMax. Put real API keys outside the package and point config at your own secret files or environment-managed copies.

The npm package includes a public `AGENTS.md` for AI coding agents. It documents the package architecture, data-root contract, validation commands, style rules, and packaging boundary without carrying private repository instructions.



## How to Cite Mahout Bench

Citation metadata is available in `CITATION.cff`.

```bibtex
@software{canonici_mahout_bench_2026,
  author = {Vinicius Garcia Canonici and Luis Miguel da Rocha de Matos and Ana Paula de Carvalho Soares},
  title = {Mahout Bench: From Measuring to Mitigating Sycophancy in Large Language Models},
  version = {0.0.7},
  year = {2026},
  url = {https://github.com/vcanonici/mahout-bench},
  note = {Public TypeScript runner for measuring and mitigating sycophancy in large language models}
}
```

Authors:

- Vinicius Garcia Canonici, ORCID `0009-0006-8269-9004`, Departamento de Sistemas de Informacao (DSI), Universidade do Minho; CIPsi, Escola de Psicologia, Universidade do Minho
- Luis Miguel da Rocha de Matos, Departamento de Sistemas de Informacao (DSI), Universidade do Minho
- Ana Paula de Carvalho Soares, Departamento de Psicologia Basica, Escola de Psicologia, Universidade do Minho

## THE VISION

This project is released under the MIT License to maximize reuse, modification,
forking, benchmarking, and integration into research and production workflows.

Use it, fork it, break it, improve it.

Science moves faster when tools are not locked behind permission walls.
Knowledge should compound, not be hoarded. Every benchmark, every script,
every optimization, and every failed experiment is another brick in the road
from the stone age to the scientific age.

This project follows a simple rule: if it helps people test, understand,
replicate, or improve AI systems, it should be easy to take apart and rebuild.

No gatekeeping. No artificial scarcity. No permission walls.

Build on it. Challenge it. Replace it. Make it better.

That is, **10 billion percent**, the point.

Science is not magic.
Science is the long, stubborn process of turning curiosity into tools.

And that is exhilarating.

## Data Citation

Mahout Bench uses/adapts data and procedure from ELEPHANT / Social Sycophancy:

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. "ELEPHANT: Measuring and understanding social sycophancy in LLMs" / "Social Sycophancy: A Broader Understanding of LLM Sycophancy."

Links:

- https://arxiv.org/abs/2505.13995
- https://openreview.net/forum?id=igbRHKEiAs
- https://github.com/myracheng/elephant

The upstream `myracheng/elephant` repository declares `CC0-1.0` for its released material. Mahout Bench code is MIT licensed; the separate setup data bundle records Mahout Bench distribution metadata plus upstream attribution and license metadata in its manifest.
