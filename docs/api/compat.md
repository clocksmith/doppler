# Compatibility API

## Purpose

`doppler-gpu/compat` preserves the former manifest-loading application facade
while supported production models migrate to signed Packs. It exports `dr`,
`doppler`, `load`, `open`, `generate`, `openPack`, and provider construction.

```js
import { dr } from 'doppler-gpu/compat';

const session = await dr.open('qwen3-0.8b');
const result = await session.generate('Describe WebGPU briefly');
await session.close();
```

This route is explicit compatibility, not the default production authority.
New integrations should import `doppler-gpu` and execute signed Packs. The
compatibility `openPack()` still requires explicit signer trust and rejects
behavior-changing `modelLoadOptions`.

## Code pointers

- [Compatibility entrypoint](../../src/index.js)
- [Pack Runtime API](root.md)

