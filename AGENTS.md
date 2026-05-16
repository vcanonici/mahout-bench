# AGENTS.md

## Purpose

Mahout Bench is a public TypeScript CLI for running sycophancy benchmark workflows. Agents working here should keep the package reproducible, publishable to npm, and safe to install without private data or secrets.

## Repository Shape

- `src/cli/`: public commands such as `bootstrap`, `setup`, `run`, and `tui`.
- `src/config/`: TOML/JSON catalog loading, model discovery, and provider resolution.
- `src/pipeline/`: benchmark orchestration, run context, checkpoints, and validation.
- `src/generation/`, `src/judging/`, `src/scoring/`, `src/reporting/`: benchmark execution stages.
- `src/runtime/`: runtime paths, LM Studio lifecycle, concurrency, and terminal observer.
- `config/`: public default providers, models, profiles, judges, and report fragments.
- `docs/`: public Markdown documentation intended for humans, static documentation sites, and AI agents.
- `tests/`: Vitest coverage. Tests that require the external data bundle must guard on data presence.

Mutable user data must live under `MAHOUT_BENCH_HOME` or `./.mahout-bench`, never in the installed package folder.

## Runtime Contracts

- Package root is code/config owned by npm.
- Data root is user-owned and may contain `datasets/`, `outputs/`, `config/*.local.json`, `secrets/`, and bootstrap state.
- Public catalogs are `config/providers.json` and `config/models.json`.
- User catalogs are `dataRoot/config/providers.local.json` and `dataRoot/config/models.local.json`.
- Secret paths may be relative. Prefer resolving relative secret files from data root before package root.
- Do not write real API keys, outputs, datasets, or local tunnel configuration into the npm package.

## Commands

Use these commands from the package root:

```bash
pnpm typecheck
pnpm test
pnpm build
npm pack --dry-run --json
```

Useful CLI smoke checks:

```bash
node build/cli/index.js --help
node build/cli/index.js bootstrap --help
node build/cli/index.js setup --help
node build/cli/index.js run --self-test
node build/cli/index.js run --dry-smoke
```

`mahout-bench bootstrap` configures user-owned providers and secrets. It must not prompt from npm `postinstall`.

## Coding Rules

- Use TypeScript with strict types. Avoid `any` unless a boundary truly has unknown external shape.
- Prefer `camelCase` for functions, variables, and file names unless an existing contract uses another style.
- Public/exported functions should have concise documentation comments.
- Keep functions small and specific; split orchestration from parsing, validation, and filesystem writes.
- Validate external boundaries: CLI args, JSON, TOML, CSV, provider catalogs, model catalogs, and file paths.
- Errors must include the failing stage or file path when useful.
- Preserve deterministic ordering where catalogs, manifests, samples, or reports depend on it.

## Testing Rules

- Add focused tests for new parsing, validation, catalog merge, and CLI behavior.
- Do not require network or installed datasets for the default test suite.
- Tests that use the public data bundle must skip or return early when the required files are absent.
- Do not update snapshots or expected values just to silence failures.

## Packaging Rules

The npm tarball should include only:

- `build/**`
- `config/**`
- `docs/**`
- `AGENTS.md`
- `README.md`
- `LICENSE`
- `NOTICE.md`
- `CITATION.cff`
- `package.json`

It must exclude datasets, outputs, `.env`, real secrets, private `AgentDATA`, release archives, and local runtime config.

## Git Hygiene

- Inspect `git status` before staging and committing.
- Stage only files changed for the current task; never use `git add .` or `git add -A`.
- Do not revert unrelated user changes.
- Commit relevant completed work with `type(scope): imperative summary`.
