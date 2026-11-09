import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateTensorConfigConsistency } from '../../src/formats/rdrr/tensor-config-validator.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getKernelPathActivationSpec } from '../../src/config/kernel-path-loader.js';

const config = JSON.parse(await readFile(new URL('../../src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json', import.meta.url), 'utf8'));
const inference = { schema: 'doppler.execution/v1', ...config.inference, session: config.session, execution: config.execution };
const manifest = {
  inference,
  tensors: Object.fromEntries(Array.from({ length: 16 }, (_, layer) => [`model.layers.${layer}.ffn_norm.weight`, { shape: [2048] }])),
};
assert.deepEqual(validateTensorConfigConsistency(manifest).errors, [], 'LFM2 feed-forward norms must survive manifest admission');
const compiled = compileExecutionV1({
  manifestInference: inference, modelId: config.output.modelBaseId, numLayers: 16, headDim: 64,
  capabilities: { hasSubgroups: true, hasF16: true, hasSubgroupsF16: true, maxWorkgroupStorageSize: 32768 },
  platform: { id: 'lfm2-contract', vendor: 'test', architecture: 'test' },
  kernelPathPolicy: { mode: 'capability-aware', sourceScope: ['manifest', 'model', 'config'], onIncompatible: 'remap' },
});
for (const phase of ['prefill', 'decode']) {
  for (const layer of [0, 2, 15]) {
    assert.ok(getKernelPathActivationSpec('silu', phase, layer, compiled.runtimeInferencePatch?.kernelPath));
  }
}
console.log('lfm2-conversion-contract.test: ok');
