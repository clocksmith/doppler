# Add a Command-Surface Command

## Goal

Add a new top-level command to Doppler's browser and Node tooling surface.

## When To Use This Guide

- The feature belongs next to `convert`, `debug`, `bench`, `verify`, `lora`, or `distill`.
- The command is part of the public command contract rather than an ad hoc script.

## Blast Radius

- Cross-surface JS + docs

## Required Touch Points

- `src/tooling/command-api.js` and `.d.ts`
- `src/tooling/browser-command-runner.js`
- `src/tooling/node-command-runner.js`
- `src/cli/doppler-cli.js`
- `src/tooling-exports.shared.js` and `.d.ts` or `src/tooling-exports.js` and `.d.ts`
- `docs/api/tooling.md`
- Tests for normalization, surface support, and result envelope shape

## Recommended Order

1. Add normalization and validation in `src/tooling/command-api.js`.
2. Implement browser and Node handling, or make unsupported surfaces fail closed through `ensureCommandSupportedOnSurface()`.
3. Wire the command into `src/cli/doppler-cli.js`.
4. Export any new helpers from the tooling barrels and declaration files.
5. Update tooling docs and add tests for both surfaces.

## Verification

- `npm run test:unit`
- Exercise the command on Node and browser relay surfaces, or confirm the unsupported surface fails closed with the documented error

## Common Misses

- Adding the command to `command-api.js` but not both runners.
- Letting one surface silently substitute different behavior.
- Returning a non-standard success or error envelope.
- Forgetting the tooling export barrels or `.d.ts` updates.
- Treating command semantics as CLI-only instead of shared browser/Node contract.

## Related Guides

- [09-sampling-strategy.md](09-sampling-strategy.md)
- [composite-pipeline-family.md](composite-pipeline-family.md)

## Canonical References

- `src/tooling/command-api.js`
- `src/tooling/browser-command-runner.js`
- `src/tooling/node-command-runner.js`
- `src/cli/doppler-cli.js`
- [../style/command-interface-design-guide.md](../style/command-interface-design-guide.md)
- [../api/tooling.md](../api/tooling.md)
