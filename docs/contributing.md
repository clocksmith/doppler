# Contributing and Agent Setup

## Agent instruction baseline

- `AGENTS.md` is canonical for all repository-specific instructions.
- `CLAUDE.md` and `GEMINI.md` are symlink aliases of `AGENTS.md`.
- `skills/` is the canonical skills registry.
- `.claude/skills` and `.gemini/skills` are the provider-facing aliases.

## Contributor parity checks

Run parity verification before and after agent instruction updates:

```bash
npm run agents:verify
```

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
