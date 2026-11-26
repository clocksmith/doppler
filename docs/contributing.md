# Contributing

- Workflow and repository rules: [AGENTS.md](../AGENTS.md). Verified changes are
  committed and pushed straight to `main`; there are no pull requests or
  pre-merge reviews. `CLAUDE.md` and `GEMINI.md` are symlinks to `AGENTS.md`,
  and `skills/` is the canonical skill registry.
- Which checks to run, and what CI runs: [testing runbook](testing-runbook.md)
  (the tier table in [AGENTS.md](../AGENTS.md#style-guides) is the short form).
- Run `npm run agents:verify` only when the conditions in
  [AGENTS.md](../AGENTS.md#agent-instruction--skills-parity-required) apply
  (symlinks, `skills/` entries, or listed AGENTS.md headings change).
- Benchmark and comparison claims: [benchmark methodology](benchmark-methodology.md).
- Public priorities and where to pick work: [GOALS.md](../GOALS.md#current-priorities)
  and [GitHub issues](https://github.com/clocksmith/doppler/issues).
