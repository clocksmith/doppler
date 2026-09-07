import assert from 'node:assert/strict';
for(const spec of ["doppler-gpu","doppler-gpu/runtime","doppler-gpu/host","doppler-gpu/serve","doppler-gpu/capsule","doppler-gpu/compat","doppler-gpu/provider","doppler-gpu/tooling","doppler-gpu/tooling/storage","doppler-gpu/tooling/device","doppler-gpu/tooling/manifest","doppler-gpu/tooling/runtime","doppler-gpu/tooling/evidence","doppler-gpu/structured","doppler-gpu/client/model-manager","doppler-gpu/electron","doppler-gpu/models/qwen3","doppler-gpu/models/gemma3","doppler-gpu/models/gemma4","doppler-gpu/models/diffusiongemma","doppler-gpu/models/embeddinggemma","doppler-gpu/tooling-experimental","doppler-gpu/loaders","doppler-gpu/orchestration","doppler-gpu/generation","doppler-gpu/training","doppler-gpu/diffusion","doppler-gpu/energy"]) await import(spec);
const runtime=await import('doppler-gpu');
assert.equal(runtime.DOPPLER_VERSION,'0.6.0');
assert.equal(typeof runtime.openCapsule,'function');
assert.equal('openPack' in runtime,false);
await import('./node_modules/doppler-gpu/src/tooling-exports.browser.js');
console.log('Registry import smoke passed (28 exports and browser tooling; version 0.6.0).');
