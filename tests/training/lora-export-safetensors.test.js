import assert from 'node:assert/strict';

import {
  exportLoRAAdapter,
  serializeLoRASafetensors,
} from '../../src/experimental/training/export.js';
import { validateManifest } from '../../src/experimental/adapters/adapter-manifest.js';
import { loadLoRAFromManifest } from '../../src/experimental/adapters/lora-loader.js';

const tensors = [
  {
    name: 'layers.0.q_proj.lora_a',
    shape: [3, 2],
    tensor: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
  },
  {
    name: 'layers.0.q_proj.lora_b',
    shape: [2, 4],
    tensor: new Float32Array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
  },
];

const raw = serializeLoRASafetensors(tensors.map((tensor) => ({
  name: tensor.name,
  shape: tensor.shape,
  data: tensor.tensor,
})));
assert.equal(raw instanceof ArrayBuffer, true);
assert.equal(new DataView(raw).getBigUint64(0, true) > 0n, true);

const exported = await exportLoRAAdapter({
  id: 'columbo_export_test',
  name: 'Columbo export test',
  baseModel: 'gemma4-e2b-it',
  rank: 2,
  alpha: 4,
  targetModules: ['q_proj'],
  tensors,
  weightsFormat: 'safetensors',
  weightsPath: 'adapters.safetensors',
});

assert.equal(exported.weights instanceof ArrayBuffer, true);
assert.equal(exported.weightsPath, 'adapters.safetensors');
assert.equal(exported.weightsSha256, exported.manifest.checksum);
assert.match(exported.weightsSha256, /^[a-f0-9]{64}$/);
assert.equal(exported.manifest.weightsFormat, 'safetensors');
assert.equal(exported.manifest.weightsPath, 'adapters.safetensors');
assert.equal(exported.manifest.weightsSize, exported.weights.byteLength);
assert.equal(exported.manifest.tensors, undefined);
assert.equal(validateManifest(exported.manifest).valid, true);

const loaded = await loadLoRAFromManifest(exported.manifest, {
  readFile: async (filePath) => {
    assert.equal(filePath, 'adapters.safetensors');
    return exported.weights;
  },
});

assert.equal(loaded.baseModel, 'gemma4-e2b-it');
assert.equal(loaded.rank, 2);
assert.equal(loaded.alpha, 4);
assert.deepEqual(loaded.targetModules, ['q_proj']);
const layer = loaded.layers.get(0);
assert.equal(layer.q_proj.a.length, 6);
assert.equal(layer.q_proj.b.length, 8);
assert.equal(layer.q_proj.scale, 2);

const padded = await exportLoRAAdapter({
  id: 'columbo_padded_export_test',
  name: 'Columbo padded export test',
  baseModel: 'gemma4-e2b-it',
  rank: 2,
  alpha: 4,
  targetModules: ['q_proj'],
  tensors: [{
    name: 'layers.0.q_proj.lora_a',
    shape: [3, 2],
    tensor: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 99, 100]),
  }, {
    name: 'layers.0.q_proj.lora_b',
    shape: [2, 4],
    tensor: new Float32Array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 99, 100]),
  }],
  weightsFormat: 'safetensors',
  weightsPath: 'padded.adapters.safetensors',
});
const paddedLoaded = await loadLoRAFromManifest(padded.manifest, {
  readFile: async () => padded.weights,
});
assert.equal(paddedLoaded.layers.get(0).q_proj.a.length, 6);
assert.equal(paddedLoaded.layers.get(0).q_proj.b.length, 8);

console.log('lora-export-safetensors.test: ok');
