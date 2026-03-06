# Provider API

## Purpose

Legacy/demo-oriented singleton provider surface for managed model loading and chat/generation helpers.

## Import Path

```js
import {
  initDoppler,
  loadModel,
  generate,
  DopplerProvider,
} from '@simulatte/doppler/provider';
```

## Audience

Existing integrations, demo-oriented code, and consumers that want a singleton provider model instead of explicit `doppler.load()` instances.

## Stability

Public, but advanced and legacy-leaning compared with the root facade.

## Primary Exports

- `initDoppler()`
- `loadModel(modelId, modelUrl?, onProgress?, localPath?)`
- `generate(...)`
- `dopplerChat(...)`
- `prefillKV(...)`
- `generateWithPrefixKV(...)`
- `DopplerProvider`

## Core Behaviors

- singleton/provider-style ownership
- explicit `init()` / `loadModel()` lifecycle
- model management and generation helpers share one provider state
- capability state is exposed through `DopplerCapabilities` and `getCapabilities()`

## Symbol Notes

### Lifecycle and model management

- `initDoppler()`
- `loadModel(...)`
- `unloadModel()`
- `destroyDoppler()`
- `getCurrentModelId()`
- `getAvailableModels()`

### Generation and chat

- `generate(...)`
- `dopplerChat(...)`
- `prefillKV(...)`
- `generateWithPrefixKV(...)`
- `formatGemmaChat(...)`
- `formatLlama3Chat(...)`
- `formatGptOssChat(...)`
- `formatChatMessages(...)`
- `buildChatPrompt(...)`

### Adapter management

- `loadLoRAAdapter(...)`
- `activateLoRAFromTrainingOutput(...)`
- `unloadLoRAAdapter()`
- `getActiveLoRA()`

### Provider object

- `DopplerProvider`
- `DOPPLER_PROVIDER_VERSION`
- `DopplerCapabilities`

## Minimal Example

```js
import { initDoppler, loadModel, generate } from '@simulatte/doppler/provider';

await initDoppler();
await loadModel('gemma-3-270m-it-wq4k-ef16-hf16');

const reply = await generate('Hello');
console.log(reply);
```

## Advanced Example

```js
import {
  initDoppler,
  loadModel,
  dopplerChat,
  loadLoRAAdapter,
  unloadLoRAAdapter,
} from '@simulatte/doppler/provider';

await initDoppler();
await loadModel('gemma-3-270m-it-wq4k-ef16-hf16');
await loadLoRAAdapter('oneshift-twoshift-redshift-blueshift');

const reply = await dopplerChat([
  { role: 'user', content: 'Write a short line about local inference.' },
]);

console.log(reply.content);
await unloadLoRAAdapter();
```

## Code Pointers

- provider export surface: [src/client/doppler-provider.js](../../src/client/doppler-provider.js)
- provider types: [src/client/doppler-provider.d.ts](../../src/client/doppler-provider.d.ts)
- provider contract types: [src/client/doppler-provider/types.d.ts](../../src/client/doppler-provider/types.d.ts)

## Related Surfaces

- [Root API](root.md)
- [Generated export inventory](reference/exports.md)
