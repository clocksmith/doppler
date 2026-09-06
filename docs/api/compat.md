# Compatibility API

## Purpose

`doppler-gpu/compat` preserves the former manifest-loading application facade
while supported production models migrate to signed Capsules. It exports `dr`,
`doppler`, `load`, `open`, `generate`, `openCapsule`, and provider construction.

```js
import { dr } from 'doppler-gpu/compat';

const session = await dr.open('qwen3-0.8b');
const result = await session.generate('Describe WebGPU briefly');
await session.close();
```

This route is explicit compatibility, not the default production authority.
New integrations should import `doppler-gpu` and execute signed Capsules. The
compatibility `openCapsule()` still requires explicit signer trust and rejects
behavior-changing `modelLoadOptions`.

## Code pointers

- [Compatibility entrypoint](../../src/index.js)
- [Capsule Runtime API](root.md)

