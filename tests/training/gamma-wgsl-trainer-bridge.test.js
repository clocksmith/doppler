import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { serializeLoRASafetensors } from '../../src/experimental/training/export.js';
import {
  buildGammaProcessEnv,
  readGammaAdapterTensors,
} from '../../tools/trainers/gamma-wgsl-trainer.js';

const gammaEnv = buildGammaProcessEnv({
  PYTHONPATH: '/base/site',
  GAMMA_WGSL_PYTHONPATH: '/overlay/site',
  KEEP_ME: 'yes',
});
assert.equal(gammaEnv.PYTHONPATH, '/overlay/site');
assert.equal(gammaEnv.KEEP_ME, 'yes');

const root = await mkdtemp(join(tmpdir(), 'doppler-gamma-wgsl-'));
const adapter = join(root, 'adapter');
await mkdir(adapter);
const weights = serializeLoRASafetensors([
  {
    name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight',
    shape: [2, 3],
    data: new Float32Array([1, 2, 3, 4, 5, 6]),
  },
  {
    name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight',
    shape: [4, 2],
    data: new Float32Array([7, 8, 9, 10, 11, 12, 13, 14]),
  },
  {
    name: 'base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_A.default.weight',
    shape: [2, 3],
    data: new Float32Array([15, 16, 17, 18, 19, 20]),
  },
  {
    name: 'base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_B.default.weight',
    shape: [4, 2],
    data: new Float32Array([21, 22, 23, 24, 25, 26, 27, 28]),
  },
]);
await writeFile(join(adapter, 'adapter_model.safetensors'), new Uint8Array(weights));

const tensors = await readGammaAdapterTensors(adapter);
assert.equal(tensors.length, 4);
assert.deepEqual(tensors.map((tensor) => tensor.name), [
  'layers.0.q_proj.lora_a',
  'layers.0.q_proj.lora_b',
  'layers.1.gate_proj.lora_a',
  'layers.1.gate_proj.lora_b',
]);
assert.deepEqual(tensors[0].shape, [3, 2]);
assert.deepEqual(Array.from(tensors[0].tensor), [1, 4, 2, 5, 3, 6]);
assert.deepEqual(tensors[1].shape, [2, 4]);
assert.deepEqual(Array.from(tensors[1].tensor), [7, 9, 11, 13, 8, 10, 12, 14]);

console.log('gamma-wgsl-trainer-bridge.test: ok');
