# Contributing and Agent Setup

## Agent instruction baseline

- `AGENTS.md` is canonical for all repository-specific instructions.
- `CLAUDE.md` and `GEMINI.md` are symlink aliases of `AGENTS.md`.
- `skills/` is the canonical skills registry.
- `.claude/skills` and `.gemini/skills` are the provider-facing aliases.

## Repository workflow

- Pull request start:
  - Confirm instruction parity: `npm run agents:verify`
  - Read the latest files in `docs/` relevant to your change area.
  - If touching CLI/runtime behavior, re-run the command parity checks in `src/tooling/command-api.js` context before edits.
- Implementation:
  - Prefer config/schema-first changes over ad-hoc flags.
  - Keep debug output in `src/debug/index.js`; avoid one-off `console.*` in runtime code.
  - For inference/runtime edits, call out kernel-path/config interactions explicitly in the PR body.

## Contributor parity checks

Run parity verification before and after agent instruction updates:

```bash
npm run agents:verify
```

## Default CI contract

`npm run ci:check` is the same deterministic contract used by GitHub Actions. It combines generated-artifact checks, kernel registry validation, source typechecking, diffusion/training contract gates, and the curated CPU test suite.

`npm run check:green` remains the fast read-only contract subset:

```
agents:verify              # AGENTS.md / CLAUDE.md / GEMINI.md / skills parity
public:boundaries:check    # src/index*, src/generation, etc. forbidden-import rules
api:docs:check             # docs/api/reference/* reflects real public exports
imports:check:browser      # browser import graph has no Node-only leaks
pending:check              # *.pending.test.js files have owned policy entries
exports:parity:check       # sibling .js / .d.ts export name sets agree
```

GPU, browser, and model-download validation is explicit rather than part of automatic CI. Run the `Manual Runtime Validation` workflow for `node-kernels`, `browser-kernels`, `opfs-text`, or `opfs-embedding` when a change touches those surfaces.

## Repo touch policy reminder

The repo is code-first and instruction-first. Prefer linking to canonical docs over duplicating policy descriptions in product-facing docs such as `README.md`.

## Writing architecture and benchmark claims

- Use explicit engine naming: `Doppler` and `Transformers.js (v4)`.
- Prefer plain language over niche runtime jargon.
- For each comparison claim, include:
  - one architectural difference,
  - one measurable effect,
  - one artifact reference (`benchmarks/vendors/results/...`) and the reproduce command.
- Avoid absolute claims like "always faster." Use scoped phrasing tied to workload, mode, and cache/load settings.

## Change categories

- Core runtime (`src/inference`, `src/gpu`, `src/loader`):
  - Update type contracts (`.d.ts`) in the same commit.
  - Prefer deterministic behavior changes with explicit config gates.
- Conversion (`src/converter`, `tools/convert-*`):
  - Add or update converter regressions under `tests/converter`.
  - Preserve manifest semantics and shard naming expectations.
- Tooling/CLI (`tools`, `src/tooling`, `src/index*.js`):
  - Ensure command contracts stay parity-safe between browser and Node.
  - Update `src/tooling/command-api.js` and `docs/style/command-interface-design-guide.md` references when command contracts change.

## Pre-merge checklist

- [ ] `npm run ci:check`
- [ ] Run at least one manual runtime lane (`test:gpu` or browser harness) when the change touches GPU execution
- [ ] Any behavior change has a regression test under `tests/` and artifact trail
- [ ] Public API changes reflected in the correct entry surface (`src/index.js`, `src/index-browser.js`, or subpath exports)
