import assert from 'node:assert/strict';

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const {
  acquireBuffer,
  uploadData,
  readBuffer,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { processLayer } = await import('../../src/inference/pipelines/text/layer.js');
const { resolveLayerPipeline } = await import('../../src/inference/pipelines/text/layer-plan.js');
const { initConvLayerState } = await import('../../src/inference/pipelines/text/ops.js');

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`lfm2-layer-plan-conv-state.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

function uploadWeight(values, shape, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return createWeightBuffer(buffer, 'f32', 'row', shape, label);
}

const hiddenSize = 2;
const inputValues = new Float32Array([1, 1]);
const inputBuffer = acquireBuffer(inputValues.byteLength, undefined, 'lfm2_plan_conv_input');
uploadData(inputBuffer, inputValues);

const convInProj = uploadWeight(new Float32Array([
  1, 0,
  0, 1,
  2, 0,
  0, 2,
  3, 0,
  0, 3,
]), [hiddenSize * 3, hiddenSize], 'lfm2_plan_conv_in_proj');
const convOutProj = uploadWeight(new Float32Array([
  1, 0,
  0, 1,
]), [hiddenSize, hiddenSize], 'lfm2_plan_conv_out_proj');
const convKernel = uploadWeight(new Float32Array([
  0, 0, 1,
  0, 0, 1,
]), [hiddenSize, 1, 3], 'lfm2_plan_conv_kernel');

const convState = {};
const convLayerStates = new Map();
const pipelinePlan = resolveLayerPipeline({
  steps: [
    {
      op: 'conv',
      phase: 'both',
      src: 'state',
      dst: 'state',
    },
  ],
}, null, 1);

let outputBuffer = null;
try {
  await initConvLayerState(
    convState,
    convKernel,
    convInProj,
    hiddenSize,
    'L0.plan_conv',
    0
  );
  convLayerStates.set(0, convState);

  outputBuffer = await processLayer(0, inputBuffer, 1, true, {
    config: {
      hiddenSize,
      numHeads: 1,
      numKVHeads: 1,
      headDim: hiddenSize,
      rmsNormEps: 1e-5,
      layerTypes: ['conv'],
      slidingWindow: null,
      attnLogitSoftcapping: null,
      queryPreAttnScalar: hiddenSize,
      queryKeyNorm: false,
      attentionOutputGate: false,
      causalAttention: true,
      rmsNormWeightOffset: 0,
      ropeRotaryDim: hiddenSize,
      ropeInterleaved: false,
      swigluLimit: null,
      useMoE: false,
    },
    useGPU: true,
    weights: new Map([[
      'layer_0',
      {
        convInProj,
        convKernel,
        convOutProj,
      },
    ]]),
    weightConfig: {
      rmsNormWeightOffset: 0,
    },
    debugFlags: {},
    kvCache: null,
    ropeFreqsCos: null,
    ropeFreqsSin: null,
    recorder: null,
    pipelinePlan,
    currentSeqLen: 0,
    activationDtype: 'f32',
    kernelPath: null,
    convLayerStates,
    linearAttentionRuntime: null,
    runtimeComputeConfig: null,
    finitenessGuardEnabled: false,
    debug: false,
    debugLayers: null,
    debugProbes: null,
    stats: {},
    currentTokenIds: null,
    lora: null,
    decodeBuffers: null,
  });

  const actual = new Float32Array(await readBuffer(outputBuffer, hiddenSize * Float32Array.BYTES_PER_ELEMENT));
  assert.equal(actual.length, hiddenSize);
  for (const value of actual) {
    assert.ok(
      Math.abs(value - 6) < 1e-5,
      `execution-v0 conv path must use conv state; expected 6, got ${value}`
    );
  }
} finally {
  if (outputBuffer) {
    releaseBuffer(outputBuffer);
  }
  if (convState.convWeightGPU) {
    releaseBuffer(convState.convWeightGPU);
  }
  if (convState.convStateGPU) {
    releaseBuffer(convState.convStateGPU);
  }
  if (convState.inProjF32GPU) {
    releaseBuffer(convState.inProjF32GPU);
  }
  releaseBuffer(inputBuffer);
  releaseBuffer(convInProj.buffer);
  releaseBuffer(convOutProj.buffer);
  releaseBuffer(convKernel.buffer);
}

console.log('lfm2-layer-plan-conv-state.test: ok');
