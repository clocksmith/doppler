import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createTensorRoleClosureReceipt } from '../../src/converter/tensor-role-closure.js';

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

const policy = await readJson('src/config/forge/tensor-role-closure/glimmer-30b-text.json');
const [modelIRReceipt, headers, checkedInReceipt] = await Promise.all([
  readJson(policy.modelIRReceipt),
  readJson(policy.headerEvidence),
  readJson(policy.output),
]);
const receipt = createTensorRoleClosureReceipt({ modelIR: modelIRReceipt.modelIR, headers, policy });
assert.deepEqual(receipt, checkedInReceipt, 'tensor-role closure receipt must be deterministic');
assert.equal(receipt.complete, true);
assert.equal(receipt.expectedTensorCount, 627);
assert.equal(receipt.observedTensorCount, 627);
assert.equal(receipt.outOfScopeTensorCount, 809);
assert.deepEqual(receipt.missingTensors, []);
assert.deepEqual(receipt.unexpectedTensors, []);
assert.equal(receipt.bindings.filter((binding) => binding.scope === 'layer').length, 12);
assert.ok(receipt.bindings.filter((binding) => binding.scope === 'layer').every((binding) => binding.matchedTensors === 52));

const missing = structuredClone(headers);
delete missing.tensors['model.language_model.layers.51.self_attn.q_proj.weight'];
assert.throws(
  () => createTensorRoleClosureReceipt({ modelIR: modelIRReceipt.modelIR, headers: missing, policy }),
  /missing tensor/
);

const wrongShape = structuredClone(headers);
wrongShape.tensors['model.language_model.layers.0.self_attn.k_proj.weight'].shape = [128, 6656];
assert.throws(
  () => createTensorRoleClosureReceipt({ modelIR: modelIRReceipt.modelIR, headers: wrongShape, policy }),
  /shape mismatch/
);

const unexpected = structuredClone(headers);
unexpected.tensors['model.language_model.layers.0.self_attn.bias'] = {
  dtype: 'BF16',
  shape: [4096],
  sourceFile: 'model-00001-of-00002.safetensors',
};
assert.throws(
  () => createTensorRoleClosureReceipt({ modelIR: modelIRReceipt.modelIR, headers: unexpected, policy }),
  /1 unexpected tensor/
);

const moduleSource = await fs.readFile('src/converter/tensor-role-closure.js', 'utf8');
assert.doesNotMatch(moduleSource, /glimmer|qwen/i, 'generic tensor closure must not contain model-family names');

console.log('✔ tensor-role-closure.test.js passed');
