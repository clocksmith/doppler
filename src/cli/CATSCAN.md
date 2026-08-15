# CATSCAN: Command-Line Surface

Component: `doppler.runtime-source.cli`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Present the shared Doppler command contract as a clear, scriptable Node command-line interface.

## Authority

- Owns CLI parsing, presentation, process exit behavior, server startup, and command-line model resolution.
- Does not own command semantics or invent runtime fields outside the shared command request.

## Scope

- Public `doppler` executable entrypoints and CLI-only presentation adapters.

## Contracts

- Input: [Shared tooling command contract](../tooling/CATSCAN.md) and CLI policy assets.
- Output: Normalized command execution, JSON or human output, and explicit failures.

## Invariants

- CLI shorthand normalizes into the same command contract used by other surfaces.
- Unsupported surfaces fail closed.
- Exit status reflects command outcome.

## Acceptance

- Command-surface and CLI integration tests pass.
- Evidence: [command API integration test](../../tests/integration/command-api.test.js).

## Non-goals

- A CLI-specific runtime, model resolver truth, or benchmark methodology.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
