import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { serializeLoRASafetensors } from '../../src/experimental/training/export.js';
import {
  GammaHandoffRequiredError,
  buildGammaRequest,
  readGammaAdapterTensors,
  runGammaWgslRequest,
} from '../../tools/trainers/gamma-wgsl-trainer.js';

const sftRequest = buildGammaRequest({
  workload: {
    id: 'wgsl-v12-fixture',
    seed: 11,
    baseModelId: 'fixture-model',
    pipeline: {
      baseModelRef: 'Qwen/Qwen3.5-9B',
      baseModelRevision: 'fixture-revision',
      maxLength: 2048,
      rowOrder: 'seed-hash-sorted-v1',
    },
  },
  adapter: {
    rank: 32,
    alpha: 64,
    dropout: 0.05,
    targetModules: ['q_proj'],
  },
  dataset: { absolutePath: '/tmp/wgsl-v12.jsonl' },
  training: {
    steps: 1200,
    accumSteps: 8,
    precision: { activations: 'bf16' },
    optimizer: { lr: 0.0001, weightDecay: 0.01 },
    gradientClipping: { maxNorm: 1 },
  },
}, '/tmp/wgsl-v12-output');
assert.equal(sftRequest.training.rowOrder, 'seed-hash-sorted-v1');

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

// Doppler does not run Gamma's trainer: the first call writes the request and
// asks for an external run; a matching response is then validated and returned.
const handoffRoot = join(root, 'handoff');
const request = { protocol: 'gamma_wgsl_trainer_json_v1', action: 'probe', outputRoot: handoffRoot, value: 1 };
await assert.rejects(runGammaWgslRequest(request), (error) => {
  assert.ok(error instanceof GammaHandoffRequiredError);
  assert.match(error.message, /outside Doppler/);
  return true;
});
const requestPath = join(handoffRoot, 'probe-request.json');
const responsePath = join(handoffRoot, 'probe-response.json');
assert.deepEqual(JSON.parse(await readFile(requestPath, 'utf8')), request);
await writeFile(responsePath, JSON.stringify({ protocol: 'gamma_wgsl_trainer_json_v1', ok: true, action: 'probe', result: { x: 2 } }));
const exchanged = await runGammaWgslRequest(request);
assert.equal(exchanged.response.result.x, 2);
assert.equal(exchanged.paths.responsePath, responsePath);
await writeFile(responsePath, JSON.stringify({ protocol: 'gamma_wgsl_trainer_json_v1', ok: true, action: 'other' }));
await assert.rejects(runGammaWgslRequest(request), /gamma_wgsl_trainer_invalid_response/);
// A changed request discards the earlier response and asks for a new run.
await writeFile(responsePath, JSON.stringify({ protocol: 'gamma_wgsl_trainer_json_v1', ok: true, action: 'probe' }));
await assert.rejects(runGammaWgslRequest({ ...request, value: 2 }), GammaHandoffRequiredError);

console.log('gamma-wgsl-trainer-bridge.test: ok');
