# mahout-bench

Public TypeScript runner for the Mahout Bench sycophancy benchmark.

Mahout Bench is a practical benchmark system for moving from "is this model sycophantic?" to "what can I change to reduce that behavior?" It evaluates large language models across configurable system prompts, inference hyperparameters, judges, providers, and sampling margins, so the same model can be compared under different mitigation strategies before deciding whether the model itself needs to be replaced.

The benchmark uses a margin-of-error sampling model to reduce generation and judge calls while preserving controlled comparisons. It supports full and reduced-call runs, automated CLI execution, an interactive TUI, resumable artifacts, multiprovider backends, LM Studio integration, and judge aferition workflows that compare candidate judges against GPT-4o reference labels.

The judge workflow is designed for operational use: it can test alternative judges, record agreement against reference labels, compare margin configurations such as full, 10pp, 8pp, and 5pp, and reuse aferition datasets so judge selection does not require repeating the most expensive calls. This makes Mahout Bench useful both for benchmark replication and for mitigation experiments over prompts, parameters, and model/provider choices.

## Install

```bash
npm install -g mahout-bench
mahout-bench setup
mahout-bench run --dry-smoke
```

By default, mutable data lives in `./.mahout-bench`. Set `MAHOUT_BENCH_HOME` to use a shared or absolute data root:

```bash
export MAHOUT_BENCH_HOME="$HOME/.mahout-bench"
mahout-bench setup
```

## Commands

```bash
mahout-bench setup
mahout-bench run --self-test
mahout-bench run --dry-smoke
mahout-bench run --validate-config
mahout-bench tui
```

`setup` downloads `mahout-bench-data-v0.0.5.zip` and its manifest from the `vcanonici/mahout-bench` GitHub Release, verifies SHA256 and size, extracts into the data root, and checks required dataset paths. Mahout Bench is the distributor/source of the setup bundle; upstream data attribution is recorded in the manifest and notices.

## Public Package Boundary

The npm package intentionally excludes:

- `AgentDATA/`
- `AGENTS.md`
- `.env` / `.ENV`
- real secrets
- run outputs
- dataset archives and extracted datasets
- private DSI/Ollama tunnel configuration

Provider configuration is public and generic: LM Studio local endpoints, OpenRouter, and MiniMax. Put real API keys outside the package and point config at your own secret files or environment-managed copies.


## How to Cite Mahout Bench

Citation metadata is available in `CITATION.cff`.

```bibtex
@software{canonici_mahout_bench_2026,
  author = {Vinicius Garcia Canonici and Luis Miguel da Rocha de Matos and Ana Paula de Carvalho Soares},
  title = {Mahout Bench: From Measuring to Mitigating Sycophancy in Large Language Models},
  version = {0.0.5},
  year = {2026},
  url = {https://github.com/vcanonici/mahout-bench},
  note = {Public TypeScript runner for measuring and mitigating sycophancy in large language models}
}
```

Authors:

- Vinicius Garcia Canonici, ORCID `0009-0006-8269-9004`, Departamento de Sistemas de Informacao (DSI), Universidade do Minho; CIPsi, Escola de Psicologia, Universidade do Minho
- Luis Miguel da Rocha de Matos, Departamento de Sistemas de Informacao (DSI), Universidade do Minho
- Ana Paula de Carvalho Soares, Departamento de Psicologia Basica, Escola de Psicologia, Universidade do Minho
