import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { verifyQ40Resources, validateQ40Inventory, verifyQ40Numerical } from './verify-q4_0.js';
const bytes = new Uint8Array([0, 1, 2, 3]);
const inventory = { schema: 'doppler.gguf-native-weight-inventory.v1', completed: true,
 sourceSha256: 'a'.repeat(64), chunkBytes: 4, excludedTensors: [{ name: 'blk.40.nextn.eh_proj.weight', reason: 'next-token-prediction layer outside text decoder' }], tensors: [{ name: 'weight', dtype: 'F32', shape: [1], bytes: 4,
 chunks: [{ offset: 0, bytes: 4, sha256: createHash('sha256').update(bytes).digest('hex') }] }] };
const resources = new Map([['weight', { dtype: 'F32', shape: [1], bytes: 4, location: 'gpu-bound', async readRange() { return bytes; } }]]);
assert.equal((await verifyQ40Resources(inventory, resources)).pass, true);
await assert.rejects(verifyQ40Resources(inventory, new Map()), /Every consumed/);
await assert.rejects(verifyQ40Resources(inventory, new Map([['weight', { ...resources.get('weight'), dtype: 'F16' }]])), /representation changed/);
await assert.rejects(verifyQ40Resources(inventory, new Map([['weight', { ...resources.get('weight'), async readRange() { return new Uint8Array(4); } }]])), /materialization changed/);
assert.throws(() => validateQ40Inventory({ ...inventory, tensors: [...inventory.tensors, ...inventory.tensors] }), /Incomplete/);
assert.deepEqual((await verifyQ40Resources(inventory, resources)).excludedTensors, inventory.excludedTensors);
for (const excludedTensors of [undefined, [{ name: 'blk.40.x' }], [{ name: 'weight', reason: 'overlaps consumed' }]]) {
  assert.throws(() => validateQ40Inventory({ ...inventory, excludedTensors }), /excluded source tensor/);
}
const contract = { vocabSize: 2, shared: { maxNewTokens: 1 }, gates: { logitPositions: [0], logitAbsoluteTolerance: .001 } };
const prompts = [{ id: 'x', promptIds: [3] }], rows = [{ id: 'x', promptIds: [3], tokenIds: [1], logits: [{ step: 0, values: [0, 1] }] }];
assert.equal(verifyQ40Numerical(contract, prompts, rows, rows).pass, true);
assert.throws(() => verifyQ40Numerical(contract, prompts, rows, [{ ...rows[0], logits: [{ step: 0, values: [0, 1.1] }] }]), /parity failed/);
console.log('q4_0.test: ok');
