import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { getKernelPathActivationSpec } from '../../src/config/kernel-path-loader.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';

const step = { op: 'activation', kernel: 'gelu.wgsl', entry: 'main', constants: { GELU_ERF: true, HAS_GATE: false } };
const path = steps => ({ decode: { steps }, prefill: { steps } });
assert.deepEqual(getKernelPathActivationSpec('gelu', 'prefill', 0, path([step])),
  { variant: 'gelu', constants: step.constants });
assert.equal(getKernelPathActivationSpec('gelu', undefined, undefined, null), null);
assert.throws(() => getKernelPathActivationSpec('gelu', undefined, 0, path([step])), /explicit phase/);
assert.throws(() => getKernelPathActivationSpec('gelu', 'prefill', 0, path([])), /exactly one/);
assert.throws(() => getKernelPathActivationSpec('gelu', 'prefill', 0, path([step, step])), /exactly one/);
assert.throws(() => getKernelPathActivationSpec('gelu', 'prefill', 0, path([{ ...step, entry: 'missing' }])), /exact registered/);
assert.throws(() => getKernelPathActivationSpec('gelu', 'prefill', 0, path([{ ...step, kernel: 'silu.wgsl' }])), /exact registered/);
const recipe = JSON.parse(await fs.readFile(new URL('../../src/config/conversion/esm/esm2-t12-35m-ur50d-f32-af32.json', import.meta.url)));
const compiled = compileExecutionV1({ modelId: recipe.output.modelBaseId, numLayers: 12, headDim: 24,
  weightDtype: 'f32', manifestInference: { ...recipe.inference, schema: 'doppler.execution/v1', execution: recipe.execution, session: recipe.session } });
for (const phase of ['prefill', 'decode']) {
  assert.equal(getKernelPathActivationSpec('gelu', phase, 0, compiled.runtimeInferencePatch.kernelPath).constants.GELU_ERF, true);
}
console.log('activation-kernel-path.test: ok');
