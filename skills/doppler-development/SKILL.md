---
name: doppler-development
description: Implement and validate Doppler runtime, command, configuration, model, loader, conversion, inference, kernel, cache, adapter, package, and browser changes. Use for general Doppler code work that is not already owned by doppler-debug, doppler-perf, doppler-bench, doppler-convert, or doppler-kernel-reviewer.
---

# Doppler Development

## Route To A Specialist

Use the matching skill first:

- Runtime failure or parity: `doppler-debug`
- Throughput diagnosis or tuning: `doppler-perf`
- Comparable evidence: `doppler-bench`
- Model conversion: `doppler-convert`
- WGSL review: `doppler-kernel-reviewer`

Use this skill for shared APIs, contracts, orchestration, packaging, and changes
that cross those owners.

## Load The Contract

Read `AGENTS.md` and the invariant sections in the general, JavaScript, and
config style guides. Read the matching developer guide for extension work.

## Ownership Map

- Command parity: `src/tooling/`, `src/cli/`
- Config and checked registries: `src/config/`, `src/rules/`
- Artifacts and loading: `src/loader/`, `src/storage/`, `models/`
- Execution: `src/inference/`, `src/generation/`, `src/gpu/`
- Public API and package: `src/client/`, exports, package closure
- Experimental lanes: `src/experimental/`

## Implement And Validate

1. Preserve manifest-first identity and explicit runtime configuration.
2. Keep browser, Node, and CLI semantics aligned through the shared command API.
3. Fail on unsupported capabilities; never add a silent runtime fallback.
4. Add `.d.ts` changes with public JavaScript changes.
5. Regenerate registries, digests, matrices, and package closure through their
   owning scripts.

Run the focused test, then select from:

- `npm run test:unit`
- `npm run typecheck:source-runtime`
- `npm run contracts:check`
- `npm run package:closure:check`
- `npm run ci:check`
- `npm run check:green`

GPU, browser, benchmark, and hosted-artifact claims require their own runs and
receipts.
