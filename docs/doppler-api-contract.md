# Doppler Public API Contract

Status: transitional design/contract note. The canonical public API guides now live under [docs/api/](api/index.md).
This file captures contract intent and historical design notes for the root `doppler` surface.

## Design Goals

1. Two lines to first token (load, generate).
2. Streaming is the default. Non-streaming is the convenience wrapper.
3. No hidden singleton state. Every model is an explicit instance.
4. One import covers simple and advanced use cases.
5. Fail fast on every ambiguous input.
6. Config-first: runtime behavior flows from presets and validated config, not ad-hoc flags.

## Exports

```
@simulatte/doppler         -> doppler (function + namespace)
@simulatte/doppler/provider -> DopplerProvider (singleton, legacy/demo)
```

The root export is `doppler`. Everything hangs off it.

---

## Model Resolution

`doppler.load()` and all convenience functions accept a `model` parameter. The input type is discriminated — no heuristic parsing.

```ts
type ModelInput =
  | string                        // registry ID (must exist in bundled registry)
  | { url: string }               // fetch manifest from URL
  | { path: string }              // read manifest from filesystem (Node) or OPFS (browser)
  | { manifest: RDRRManifest }    // use manifest object directly
```

### Rules

- A bare string is **always** a registry ID. It is never interpreted as a URL or path.
- `{ url }` requires network access. Fails with status + URL on network error.
- `{ path }` is environment-gated:
  - **Node**: reads from filesystem. Fails if file not found.
  - **Browser**: reads from OPFS. Fails if entry not found.
  - Passing a filesystem path in browser (or vice versa) fails fast with an environment mismatch error.
- `{ manifest }` validates `modelType` is present. Fails if missing with re-conversion instructions.
- Unknown registry IDs fail with an actionable error listing available IDs.
- No silent fallback between resolution strategies.

### Registry

- **Bundled JSON**: ships with the package, always available, no network required.
- **Optional remote refresh**: `doppler.refreshRegistry()` fetches updated registry. Never called implicitly during `load()`.
- `load()` never makes implicit network requests for registry resolution.

---

## Tier 2: `doppler.load()` (README Hero)

```ts
function doppler.load(model: ModelInput, options?: LoadOptions): Promise<DopplerModel>
```

### LoadOptions

```ts
interface LoadOptions {
  onProgress?: (event: ProgressEvent) => void;  // { phase, percent, bytesLoaded, bytesTotal }
  signal?: AbortSignal;                          // abort load (rejects with AbortError)
  runtimePreset?: string;                        // named preset from src/config/presets/runtime/
  runtimeConfig?: Partial<RuntimeConfig>;         // validated overrides merged on top of preset
}
```

- No `device` option. Doppler is WebGPU-only. If WebGPU is unavailable, `load()` throws. No silent CPU fallback.
- `runtimePreset` and `runtimeConfig` are validated against the runtime config schema. Unknown fields fail fast.
- `contexts` is removed. Pipeline contexts are derived from `runtimeConfig` and the manifest. If you need raw context control, use the Tier 3 internal API directly.

### DopplerModel

```ts
interface DopplerModel {
  // --- Generation ---
  generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string>;
  generateText(prompt: string, options?: GenerateOptions): Promise<string>;

  // --- Chat ---
  chat(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string>;
  chatText(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;

  // --- LoRA ---
  loadLoRA(adapter: string | LoRAManifest): Promise<void>;
  unloadLoRA(): Promise<void>;
  readonly activeLoRA: string | null;

  // --- Lifecycle ---
  unload(): Promise<void>;
  readonly loaded: boolean;
  readonly modelId: string;

  // --- Introspection ---
  readonly manifest: RDRRManifest;
  readonly deviceInfo: DeviceInfo;

  // --- Advanced (KV cache reuse) ---
  readonly advanced: {
    prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
    generateWithPrefixKV(
      prefix: KVCacheSnapshot,
      prompt: string,
      options?: GenerateOptions,
    ): AsyncGenerator<string>;
  };
}
```

### GenerateOptions

```ts
interface GenerateOptions {
  maxTokens?: number;       // default: from runtime config
  temperature?: number;     // default: from runtime config
  topK?: number;            // default: from runtime config
  topP?: number;            // default: from runtime config
  stopSequences?: string[]; // default: []
  seed?: number;            // default: undefined (non-deterministic)
  signal?: AbortSignal;     // abort generation (yields stop, does not throw)
  onToken?: (token: string) => void;  // optional callback (streaming still primary)
}
```

### ChatMessage / ChatResponse

```ts
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}
```

### Example (README Hero)

```js
import { doppler } from '@simulatte/doppler';

const model = await doppler.load('gemma3-270m');

for await (const token of model.generate('Hello, world')) {
  process.stdout.write(token);
}
```

### With options

```js
const controller = new AbortController();

const model = await doppler.load('gemma3-270m', {
  onProgress: ({ phase, percent }) => console.log(`${phase}: ${percent}%`),
  signal: controller.signal,
});

// Non-streaming
const text = await model.generateText('Explain WebGPU in one sentence');

// Chat
const reply = await model.chatText([
  { role: 'user', content: 'Write a haiku about browsers' },
]);
console.log(reply.content);

// LoRA
await model.loadLoRA('my-adapter');
for await (const token of model.generate('Hello with adapter')) {
  process.stdout.write(token);
}

// Multi-model
const small = await doppler.load('gemma3-270m');
const large = await doppler.load('google-embeddinggemma-300m-wq4k-ef16');

// Cleanup
await model.unload();
```

---

## Tier 1: `doppler()` (Convenience Sugar)

```ts
function doppler(prompt: string, options: DopplerCallOptions): AsyncGenerator<string>
```

### DopplerCallOptions

```ts
interface DopplerCallOptions extends GenerateOptions {
  model: ModelInput;                              // required — no implicit default model
  onProgress?: (event: ProgressEvent) => void;    // emitted only on first load for this cache key
}
```

Tier 1 does **not** accept load-affecting options (`runtimePreset`, `runtimeConfig`). These alter pipeline behavior and would make cache-key semantics ambiguous. If you need load-affecting options, use `doppler.load()` (Tier 2).

### Behavior

1. Resolve model from `options.model`.
2. Compute cache key from resolved model ID string.
3. If an in-flight load for this key exists, await it (single-flight dedupe).
4. If cached and `.loaded` is true, use existing instance.
5. If cached but `.loaded` is false, evict and reload.
6. If not cached, call `doppler.load(model)`, cache the result.
7. Call `.generate(prompt, options)` on the instance.
8. Return the AsyncGenerator.

### Cache Rules

- Cache key is the resolved model ID string (after registry/path/URL/manifest resolution).
- Cache is **per-process** (module-level Map, not global).
- `doppler.load()` instances are NOT in the convenience cache (explicit ownership).
- `doppler.evict(modelId)` removes a cached instance and calls `.unload()`.
- `doppler.evictAll()` clears all cached instances.
- **Single-flight**: concurrent `doppler()` calls for the same model ID share one in-flight `load()`. The first caller triggers the load; subsequent callers await the same Promise. If the load fails, all waiters reject and the key is not cached.

### Fail-fast Rules

- `options.model` is required. Omitting it throws immediately (no default model).
- Load-affecting options (`runtimePreset`, `runtimeConfig`) in Tier 1 throw immediately with a message directing the caller to use `doppler.load()`.
- First call pays load latency. This is by design. No background prefetch.

### Convenience Variants

```ts
function doppler.text(prompt: string, options: DopplerCallOptions): Promise<string>
function doppler.chat(messages: ChatMessage[], options: DopplerChatOptions): AsyncGenerator<string>
function doppler.chatText(messages: ChatMessage[], options: DopplerChatOptions): Promise<ChatResponse>
```

All follow the same cache-key, single-flight, and resolve rules as `doppler()`.

### Example

```js
import { doppler } from '@simulatte/doppler';

// Streaming (caches model on first call)
for await (const token of doppler('Hello', { model: 'gemma3-270m' })) {
  process.stdout.write(token);
}

// Non-streaming
const text = await doppler.text('Summarize WebGPU', { model: 'gemma3-270m' });

// Chat
const reply = await doppler.chatText([
  { role: 'user', content: 'Explain OPFS' },
], { model: 'gemma3-270m' });

// Evict when done
await doppler.evict('gemma3-270m');
```

---

## Utility Statics

```ts
doppler.listModels(): Promise<string[]>            // available registry IDs
doppler.refreshRegistry(): Promise<void>           // fetch updated registry (explicit, never implicit)
doppler.version: string                            // package version
doppler.evict(modelId: string): Promise<void>      // evict from convenience cache + unload
doppler.evictAll(): Promise<void>                  // evict all from convenience cache + unload all
```

---

## Fail-Fast Contract

| Scenario | Behavior |
|---|---|
| Unknown registry ID | Throw with list of available IDs |
| WebGPU unavailable | Throw, never silent fallback |
| Manifest missing `modelType` | Throw with re-conversion instructions |
| `generate()` after `unload()` | Throw "model unloaded" |
| LoRA adapter incompatible | Throw with shape mismatch details |
| `doppler()` without `model` option | Throw "model is required" |
| Load-affecting options in Tier 1 | Throw "use doppler.load() for runtimePreset/runtimeConfig" |
| `{ path }` in wrong environment | Throw "filesystem paths are Node-only" / "OPFS paths are browser-only" |
| Network error during `{ url }` fetch | Throw with URL and HTTP status |
| GPU device lost during generation | Throw with device loss context, mark model as unloaded |
| Concurrent load for same model (Tier 1) | Single-flight: share one Promise, no double-load |

---

## What This Wraps

`doppler.load()` composes existing internals:

```
doppler.load(model, opts)
  -> resolveModelInput(model)        // discriminated: string=registry, {url}, {path}, {manifest}
  -> initGPU()                       // lazy, once per process (existing initDoppler)
  -> getDopplerLoader(loaderConfig)  // existing src/loader/doppler-loader.js (takes loader config, not raw manifest)
  -> createPipeline(manifest, ctx)   // existing src/inference/pipelines/text.js
  -> return new DopplerModel(pipeline, manifest, runtimeConfig)
```

No new engine work. This is a surface layer over existing primitives.

---

## What This Does NOT Cover

- Diffusion pipelines (separate surface, `@simulatte/doppler/diffusion`)
- Training/backward pass
- P2P distribution
- Speculative decoding (expose on `model.advanced` later if adoption warrants it)
- Model conversion (stays in tooling surface)
