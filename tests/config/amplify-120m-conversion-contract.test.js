import assert from 'node:assert/strict';
import fs from 'node:fs';

import { applySourceTensorRules } from '../../src/converter/source-tensor-rules.js';
import { resolveConversionPlan } from '../../src/converter/conversion-plan.js';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

const configPath = 'src/config/conversion/amplify/amplify-120m-f16-af32.json';
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

let offset = 0;
function tensor(name, shape) {
  const size = shape.reduce((product, dimension) => product * dimension, 1) * 4;
  const descriptor = { name, dtype: 'F32', shape, offset, size };
  offset += size;
  return descriptor;
}

const sourceTensors = [
  tensor('decoder.bias', [27]),
  tensor('decoder.weight', [27, 640]),
  tensor('encoder.weight', [27, 640]),
  tensor('layer_norm_2.weight', [640]),
];
for (let layer = 0; layer < 24; layer += 1) {
  sourceTensors.push(
    tensor(`transformer_encoder.${layer}.attention_norm.weight`, [640]),
    tensor(`transformer_encoder.${layer}.ffn.w12.weight`, [3424, 640]),
    tensor(`transformer_encoder.${layer}.ffn.w3.weight`, [640, 1712]),
    tensor(`transformer_encoder.${layer}.ffn_norm.weight`, [640]),
    tensor(`transformer_encoder.${layer}.k.weight`, [640, 640]),
    tensor(`transformer_encoder.${layer}.q.weight`, [640, 640]),
    tensor(`transformer_encoder.${layer}.v.weight`, [640, 640]),
    tensor(`transformer_encoder.${layer}.wo.weight`, [640, 640]),
  );
}

assert.equal(sourceTensors.length, 196);
const transformed = applySourceTensorRules(sourceTensors, config.sourceTensors);
assert.equal(transformed.length, 220);
assert.deepEqual(
  transformed.filter((entry) => entry.name.includes('model.layers.0.mlp.')).map(
    ({ name, shape }) => ({ name, shape })
  ),
  [
    { name: 'model.layers.0.mlp.gate_proj.weight', shape: [1712, 640] },
    { name: 'model.layers.0.mlp.up_proj.weight', shape: [1712, 640] },
    { name: 'model.layers.0.mlp.down_proj.weight', shape: [640, 1712] },
  ]
);
assert.deepEqual(
  transformed.find((entry) => entry.name === 'lm_head.bias'),
  {
    ...sourceTensors[0],
    name: 'lm_head.bias',
    role: 'other',
    group: 'head',
  }
);

validateRequiredInferenceFields(config.inference, config.output.modelBaseId);
const plan = resolveConversionPlan({
  rawConfig: {
    model_type: 'AMPLIFY',
    hidden_size: 640,
    intermediate_size: 2560,
    num_hidden_layers: 24,
    num_attention_heads: 10,
    vocab_size: 27,
    max_length: 2048,
  },
  tensors: transformed,
  tensorNames: transformed.map((entry) => entry.name),
  converterConfig: config,
  modelId: config.output.modelBaseId,
});

assert.equal(plan.modelType, 'embedding');
assert.equal(plan.manifestInference.schema, 'doppler.execution/v1');
assert.equal(plan.manifestInference.supportsSequence, true);
assert.deepEqual(plan.manifestInference.sequence, config.inference.sequence);

// Catalog lifecycle is mutable status owned by models/catalog.json. Assert the
// claim-to-evidence invariant instead of pinning the current status values: a
// verified/passing claim must be backed by committed passing receipts for this
// exact model, and a hosted artifact must name an immutable revision.
const catalog = JSON.parse(fs.readFileSync('models/catalog.json', 'utf8'));
const catalogEntry = catalog.models.find((entry) => entry.modelId === config.output.modelBaseId);
if (catalogEntry) {
  if (catalogEntry.lifecycle?.availability?.hf === true) {
    assert.match(catalogEntry.hf?.revision ?? '', /^[0-9a-f]{40}$/u,
      'a hosted catalog artifact must pin a full commit revision');
  }
  const claimsVerified = catalogEntry.lifecycle?.status?.tested === 'verified'
    || catalogEntry.lifecycle?.tested?.result === 'pass';
  if (claimsVerified) {
    const citedReceipts = [...new Set(
      (catalogEntry.lifecycle?.tested?.notes ?? '').match(/docs\/status\/[\w.-]+\.json/gu) ?? []
    )];
    assert.ok(citedReceipts.length > 0, 'a verified catalog claim must cite its qualification receipt');
    for (const receiptPath of citedReceipts) {
      assert.ok(fs.existsSync(receiptPath), `cited receipt must be committed: ${receiptPath}`);
      const receipt = JSON.parse(fs.readFileSync(receiptPath, 'utf8'));
      assert.equal(receipt.passed, true, `cited receipt must pass: ${receiptPath}`);
      assert.equal(receipt.model?.modelId, catalogEntry.modelId,
        `cited receipt must qualify this exact model: ${receiptPath}`);
    }
  }
}

console.log('amplify-120m-conversion-contract.test: ok');
