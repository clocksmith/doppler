import assert from 'node:assert/strict';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import {
  acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool, isBufferActive,
} from '../../src/memory/buffer-pool.js';
import { resolveLayerPipeline } from '../../src/inference/pipelines/text/layer-plan.js';
import { processLayerPlanGPU } from '../../src/inference/pipelines/text/layer-plan-gpu.js';
import { buildLayerPipelineFromExecution } from '../../src/inference/pipelines/text/execution-runtime-builders.js';

const normStep = { op: 'layernorm', src: 'state', dst: 'state', weight: 'post_attn' };
assert.throws(() => resolveLayerPipeline({ steps: [{ ...normStep, residual: 'state' }] }, null, 1), /explicit residual_add/);
assert.throws(() => resolveLayerPipeline({ steps: [{ ...normStep, weight: 'unknown' }] }, null, 1), /unknown affine/);
const lowered = buildLayerPipelineFromExecution(expandExecutionV1({
  kernels: { norm: { kernel: 'layernorm.wgsl', entry: 'main',
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['layernorm.wgsl#main']}` } },
  prefill: [['layernorm', 'norm', 'post_attn']],
}), { strict: true });
assert.equal(lowered.steps[0].op, 'layernorm');
assert.equal(lowered.steps[0].weight, 'post_attn');

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`layer-plan-layernorm.test: skipped physical checks (${gpu.reason})`);
  process.exit(0);
}
const device = getDevice();
const hiddenSize = 4;
const numTokens = 2;
const epsilon = 1e-5;
const values = new Float32Array([1, 2, 4, 9, -2, 5, 7, 3]);
const gamma = new Float32Array([1, 2, -1, 0.5]);
const beta = new Float32Array([0.5, -0.25, 1, 2]);
const input = acquireBuffer(values.byteLength, undefined, 'plan_norm_test_input');
uploadData(input, values);
const borrowedBias = acquireBuffer(beta.byteLength, undefined, 'plan_norm_test_bias');
uploadData(borrowedBias, beta);
const biasWeight = createWeightBuffer(borrowedBias, 'f32', 'row', [hiddenSize], 'plan_norm_bias');

function context(steps, recorder) {
  return {
    config: { hiddenSize, numHeads: 1, rmsNormEps: epsilon },
    weightConfig: { rmsNormWeightOffset: false }, debugFlags: {},
    activationDtype: 'f32', finitenessGuardEnabled: false,
    pipelinePlan: resolveLayerPipeline({ steps }, null, 1), recorder,
  };
}

function reference() {
  const result = [];
  for (let row = 0; row < numTokens; row++) {
    const residual = Array.from(values.slice(row * hiddenSize, (row + 1) * hiddenSize), value => value + value);
    const mean = residual.reduce((a, b) => a + b, 0) / hiddenSize;
    const variance = residual.reduce((sum, value) => sum + (value - mean) ** 2, 0) / hiddenSize;
    result.push(...residual.map((value, column) => (value - mean) / Math.sqrt(variance + epsilon) * gamma[column] + beta[column]));
  }
  return result;
}
const steps = [
  { op: 'save', src: 'state', name: 'residual' },
  { op: 'residual_add', a: 'state', b: 'residual', dst: 'state' },
  normStep,
];
try {
  for (const recorded of [false, true]) {
    const recorder = recorded ? new CommandRecorder(device, 'layernorm_plan_test') : null;
    const output = await processLayerPlanGPU(0, input, numTokens, true, values.length,
      context(steps, recorder), { postAttnNorm: gamma, postAttentionNormBias: biasWeight }, {});
    if (recorder) await recorder.submitAndWait();
    const actual = new Float32Array(await readBuffer(output, values.byteLength));
    reference().forEach((expected, i) => assert.ok(Math.abs(actual[i] - expected) < 1e-5, `row element ${i}: ${actual[i]} != ${expected}`));
    releaseBuffer(output);
    assert(isBufferActive(input) && isBufferActive(borrowedBias));

    const activeBefore = getBufferPool().getStats().activeBuffers;
    const failingRecorder = recorded ? new CommandRecorder(device, 'layernorm_plan_failure') : null;
    await assert.rejects(processLayerPlanGPU(0, input, numTokens, true, values.length,
      context([normStep, { ...normStep, weight: 'post_ffn' }], failingRecorder),
      { postAttnNorm: gamma, postAttentionNormBias: biasWeight, postFeedforwardNorm: gamma }, {}), /requires affine weight and bias/);
    if (failingRecorder) failingRecorder.abort();
    assert.equal(getBufferPool().getStats().activeBuffers, activeBefore, 'failed second operation must release the current state');
    assert(isBufferActive(input) && isBufferActive(borrowedBias));
  }

  const activeBefore = getBufferPool().getStats().activeBuffers;
  const originalWrite = device.queue.writeBuffer;
  device.queue.writeBuffer = function(buffer, ...args) {
    if (args[1] === beta) throw new Error('injected bias upload failure');
    return originalWrite.call(this, buffer, ...args);
  };
  try {
    await assert.rejects(processLayerPlanGPU(0, input, numTokens, true, values.length,
      context([normStep], null), { postAttnNorm: gamma, postAttentionNormBias: beta }, {}), /injected bias upload failure/);
  } finally {
    device.queue.writeBuffer = originalWrite;
  }
  assert.equal(getBufferPool().getStats().activeBuffers, activeBefore, 'partial affine upload must release its weight');
  await assert.rejects(processLayerPlanGPU(0, input, numTokens, true, values.length,
    context([{ ...normStep, outputDtype: 'f16' }], null),
    { postAttnNorm: gamma, postAttentionNormBias: biasWeight }, {}), /output dtype mismatch/);
  assert.equal(getBufferPool().getStats().activeBuffers, activeBefore);
} finally {
  releaseBuffer(input);
  releaseBuffer(borrowedBias);
}
console.log('layer-plan-layernorm.test: physical immediate/recorded parity and failure cleanup passed');
