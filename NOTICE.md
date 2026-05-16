# Notices

Mahout Bench is an independent benchmark system for measuring and mitigating sycophancy in large language models through configurable system prompts, inference hyperparameters, providers, sampling margins, and judge aferition workflows.

The package focuses on operational benchmarking: reduced-call sampling, multiprovider execution, LM Studio integration, TUI/CLI workflows, resumable artifacts, and candidate-judge comparison against reference labels.

## ELEPHANT Citation

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. "ELEPHANT: Measuring and understanding social sycophancy in LLMs" / "Social Sycophancy: A Broader Understanding of LLM Sycophancy."

- arXiv: https://arxiv.org/abs/2505.13995
- OpenReview: https://openreview.net/forum?id=igbRHKEiAs
- Upstream repository: https://github.com/myracheng/elephant

## Data Redistribution

The npm package does not include datasets. Data bundles are distributed as separate release assets with manifests and checksums.

For `mahout-bench setup`, the data bundle source is the Mahout Bench GitHub Release at `vcanonici/mahout-bench`. The upstream research/data origin remains ELEPHANT / Social Sycophancy, and the manifest records that separately for attribution and reproducibility.

The upstream `myracheng/elephant` repository declares `CC0-1.0` for its released material. Mahout Bench code is MIT licensed; data bundle manifests preserve Mahout Bench distribution metadata and upstream source, citation, and license metadata.

## Mahout Bench Citation

Citation metadata is provided in `CITATION.cff`.

Linked paper/project title: "Mahout Bench: From Measuring to Mitigating Sycophancy in Large Language Models".

Mahout Bench authors:

- Vinicius Garcia Canonici, ORCID `0009-0006-8269-9004`, Departamento de Sistemas de Informacao (DSI), Universidade do Minho; CIPsi, Escola de Psicologia, Universidade do Minho
- Luis Miguel da Rocha de Matos, Departamento de Sistemas de Informacao (DSI), Universidade do Minho
- Ana Paula de Carvalho Soares, Departamento de Psicologia Basica, Escola de Psicologia, Universidade do Minho
