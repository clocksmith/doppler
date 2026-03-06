# Tooling API

## Purpose

Browser-safe and Node tooling surface for command runners, diagnostics, registry helpers, storage access, and shared command contracts.

## Import Path

```js
import {
  runBrowserCommand,
  normalizeBrowserCommand,
} from '@simulatte/doppler/tooling';
```

## Audience

Tool builders, demo/harness code, and consumers that need CLI/browser command parity or browser-safe storage/diagnostic helpers.

## Stability

Public, but tooling-oriented rather than app-facing.

## Primary Exports

- shared command contract helpers
- browser command runner helpers
- Node command runner helpers
- storage registry and shard-manager helpers
- runtime/config inspection helpers

The generated export inventory is the authoritative symbol list for this surface because the tooling subpath is broad.

## Core Behaviors

- shared browser/Node command contract
- browser-safe exports and Node-only exports gathered under one subpath
- intended for harnesses, diagnostics, demos, registry/storage workflows, and command runners

## Symbol Groups

### Command contract and runners

- `normalizeBrowserCommand(...)`
- `runBrowserCommand(...)`
- `normalizeNodeCommand(...)`
- `runNodeCommand(...)`
- `normalizeNodeBrowserCommand(...)`
- `runBrowserCommandInNode(...)`

### Config and preset helpers

- `listPresets(...)`
- `detectPreset(...)`
- `resolvePreset(...)`
- `getRuntimeConfig()`
- `setRuntimeConfig(...)`

### Storage and manifest helpers

- shard-manager exports
- storage registry exports
- manifest parsing helpers

### Device and diagnostics helpers

- `initDevice(...)`
- `getDevice()`
- `isWebGPUAvailable()`
- browser suite helpers

## Minimal Example

```js
import { normalizeBrowserCommand, runBrowserCommand } from '@simulatte/doppler/tooling';

const command = normalizeBrowserCommand({
  command: 'verify',
  request: { suite: 'inference' },
});

const result = await runBrowserCommand(command);
console.log(result.ok);
```

## Advanced Example

```js
import {
  listRegisteredModels,
  loadRuntimePreset,
  normalizeBrowserCommand,
  runBrowserCommand,
} from '@simulatte/doppler/tooling';

const preset = await loadRuntimePreset('modes/debug');
console.log(await listRegisteredModels());

const result = await runBrowserCommand(normalizeBrowserCommand({
  command: 'verify',
  request: {
    suite: 'inference',
    runtimeConfig: preset,
  },
}));

console.log(result.ok);
```

## Code Pointers

- tooling export surface: [src/tooling-exports.d.ts](../../src/tooling-exports.d.ts)
- shared tooling exports: [src/tooling-exports.shared.d.ts](../../src/tooling-exports.shared.d.ts)
- command contract: [src/tooling/command-api.js](../../src/tooling/command-api.js)

## Related Surfaces

- [Generated export inventory](reference/exports.md)
