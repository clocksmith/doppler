# Experimental Tooling API

## Purpose

Experimental browser-conversion and P2P helper surface kept separate from the tier1 tooling core.

## Import Path

```js
import { convertModel, pickModelFiles } from 'doppler-gpu/tooling-experimental';
```

## Audience

Advanced tooling, demo, and research workflows that explicitly need browser conversion helpers or P2P/distribution utilities.

## Stability

Exported, but experimental. This subpath is outside the tier1 quickstart and demo-default proof contract.
See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- browser conversion helpers such as `convertModel(...)`, `createRemoteModelSources(...)`, and `isConversionSupported(...)`
- file-picker helpers such as `pickModelDirectory()` and `pickModelFiles()`
- P2P/distribution helpers such as `createBrowserWebRTCDataPlaneTransport(...)`, `normalizeP2PControlPlaneConfig(...)`, and `buildP2PDashboardSnapshot(...)`

## Minimal Example

```js
import { isConversionSupported, pickModelFiles } from 'doppler-gpu/tooling-experimental';

if (isConversionSupported()) {
  const files = await pickModelFiles({ multiple: true });
  console.log(files.length);
}
```

## Code Pointers

- experimental tooling export surface: [src/tooling-experimental-exports.js](../../src/tooling-experimental-exports.js)
- shared experimental tooling exports: [src/tooling-experimental-exports.shared.js](../../src/tooling-experimental-exports.shared.js)
- browser conversion helpers: [src/experimental/browser/browser-converter.js](../../src/experimental/browser/browser-converter.js)
- P2P helpers: [src/experimental/distribution/p2p-control-plane.js](../../src/experimental/distribution/p2p-control-plane.js)

## Related Surfaces

- [Tooling API](tooling.md)
- [Advanced Export Map](advanced-root-exports.md)
- [Generated export inventory](reference/exports.md)
