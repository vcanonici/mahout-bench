# mahout-bench

Public TypeScript runner for ELEPHANT-style social sycophancy prompt benchmarks.

Mahout Bench adapts the data and benchmark procedure introduced by Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky in ELEPHANT / Social Sycophancy. Every benchmark run prints:

```text
Thanks to Myra Cheng and the ELEPHANT team for the data and procedure.
```

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

`setup` downloads `mahout-bench-data-v0.0.5.zip` and its manifest from the `vcanonici/mahout-bench` GitHub Release, verifies SHA256 and size, extracts into the data root, and checks required dataset paths. Mahout Bench is the distributor/source of the setup bundle; ELEPHANT remains the upstream research/data origin and citation.

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

## Data Citation

Mahout Bench uses/adapts data and procedure from ELEPHANT / Social Sycophancy:

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. "ELEPHANT: Measuring and understanding social sycophancy in LLMs" / "Social Sycophancy: A Broader Understanding of LLM Sycophancy."

Links:

- https://arxiv.org/abs/2505.13995
- https://openreview.net/forum?id=igbRHKEiAs
- https://github.com/myracheng/elephant

The upstream `myracheng/elephant` repository declares `CC0-1.0` for its released material. Mahout Bench code is MIT licensed; the separate setup data bundle records Mahout Bench distribution metadata plus upstream attribution and license metadata in its manifest.

## How to Cite Mahout Bench

Citation metadata is available in `CITATION.cff`.

```bibtex
@software{canonici_mahout_bench_2026,
  author = {Vinicius Garcia Canonici and Luis Miguel da Rocha de Matos and Ana Paula de Carvalho Soares},
  title = {mahout-bench},
  version = {0.0.5},
  year = {2026},
  url = {https://github.com/vcanonici/mahout-bench},
  note = {Public TypeScript runner for ELEPHANT-style social sycophancy prompt benchmarks}
}
```

Authors:

- Vinicius Garcia Canonici, ORCID `0009-0006-8269-9004`, CIPsi, Escola de Psicologia, Universidade do Minho
- Luis Miguel da Rocha de Matos, Departamento de Sistemas de Informacao (DSI), Universidade do Minho
- Ana Paula de Carvalho Soares, Departamento de Psicologia Basica, Escola de Psicologia, Universidade do Minho
