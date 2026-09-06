import assert from 'node:assert/strict';
import { loadLoRAFromManifest } from '../../src/experimental/adapters/lora-loader.js';
import { applyLoRA } from '../../src/inference/pipelines/text/lora-apply.js';
import { resolveLoRAFormatLayout } from '../../src/config/lora-layouts.js';

const manifest = { id: 'layout-check', name: 'Layout check', baseModel: 'base', rank: 2, alpha: 4,
  targetModules: ['q_proj'], weightsLayout: 'peft', tensors: [
    { name: 'layers.0.q_proj.lora_A', shape: [2, 3], dtype: 'f32', data: [1, 2, 3, 4, 5, 6] },
    { name: 'layers.0.q_proj.lora_B', shape: [4, 2], dtype: 'f32', data: [1, 2, 3, 4, 5, 6, 7, 8] },
  ] };
const adapter = await loadLoRAFromManifest(manifest);
const weights = adapter.layers.get(0).q_proj;
assert.equal(weights.weightsLayout, resolveLoRAFormatLayout('peft_safetensors').name);
assert.deepEqual(weights.aShape, [2, 3]);
assert.deepEqual(weights.bShape, [4, 2]);
assert.deepEqual(Array.from(weights.a), manifest.tensors[0].data, 'byte order is preserved for WGSL');

// Equal byte counts cannot disguise a different projection shape. Fail before GPU allocation.
let allocated = false;
await assert.rejects(applyLoRA({ dtype: 'f32' }, { dtype: 'f32' },
  { ...weights, aShape: [3, 2] }, { M: 1, N: 4, K: 3 }, () => { allocated = true; }), /declared shape/);
assert.equal(allocated, false);
await assert.rejects(loadLoRAFromManifest({ ...manifest, weightsLayout: 'input-major' }), /shape conflicts/);
await assert.rejects(loadLoRAFromManifest(manifest, { weightsLayout: 'input-major' }), /conflicts/);
await assert.rejects(loadLoRAFromManifest({ ...manifest, weightsLayout: 'guess' }), /Unsupported/);
assert.throws(() => resolveLoRAFormatLayout('unknown'), /Unsupported/);

// Square matrices still need layout identity, even when shape alone cannot distinguish them.
const square = { ...manifest, tensors: manifest.tensors.map(tensor => ({ ...tensor, shape: [2, 2], data: [1, 2, 3, 4] })) };
const peft = await loadLoRAFromManifest(square);
const native = await loadLoRAFromManifest({ ...square, weightsLayout: 'input-major' });
const legacy = await loadLoRAFromManifest({ ...square, weightsLayout: undefined });
assert.notEqual(peft.identity.digest, native.identity.digest);
assert.equal(legacy.identity.digest, native.identity.digest);

console.log('lora-runtime-matmul-layout-contract.test: ok');
