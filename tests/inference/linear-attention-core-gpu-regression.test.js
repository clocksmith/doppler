import assert from 'node:assert/strict';

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { runLinearAttentionCoreGPU } = await import('../../src/gpu/kernels/linear-attention-core.js');
const { float16ToFloat32, float32ToFloat16 } = await import('../../src/converter/quantizer.js');

function silu(x) {
  return x / (1 + Math.exp(-x));
}

function softplus(x) {
  if (x > 20) return x;
  if (x < -20) return Math.exp(x);
  return Math.log(1 + Math.exp(x));
}

function assertArraysClose(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label} length mismatch`);
  for (let i = 0; i < actual.length; i += 1) {
    const delta = Math.abs(actual[i] - expected[i]);
    assert.ok(
      delta <= tolerance,
      `${label}[${i}] mismatch: actual=${actual[i]} expected=${expected[i]} delta=${delta} tolerance=${tolerance}`
    );
  }
}

function buildReferenceOutput({ qkv, z, a, b, layerState, numTokens, qkL2NormEps }) {
  const convState = new Float32Array(layerState.convState);
  const recurrentState = new Float32Array(layerState.recurrentState);
  const convOut = new Float32Array(numTokens * layerState.convDim);
  const output = new Float32Array(numTokens * layerState.valueDim);

  for (let channel = 0; channel < layerState.convDim; channel += 1) {
    const stateBase = channel * layerState.convKernelSize;
    for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx += 1) {
      const newest = qkv[tokenIdx * layerState.convDim + channel];
      for (let k = 0; k + 1 < layerState.convKernelSize; k += 1) {
        convState[stateBase + k] = convState[stateBase + k + 1];
      }
      convState[stateBase + layerState.convKernelSize - 1] = newest;

      let mixed = 0;
      for (let k = 0; k < layerState.convKernelSize; k += 1) {
        mixed += convState[stateBase + k] * layerState.convWeight[k + stateBase];
      }
      convOut[tokenIdx * layerState.convDim + channel] = silu(mixed);
    }
  }

  for (let head = 0; head < layerState.numVHeads; head += 1) {
    const headScale = 1 / Math.sqrt(layerState.headKDim);
    const recurrentHeadBase = head * layerState.headKDim * layerState.headVDim;
    const srcHead = Math.floor(head / Math.max(layerState.qRep, 1));
    const qBase = srcHead * layerState.headKDim;
    const kBase = layerState.qSize + srcHead * layerState.headKDim;
    const vBase = layerState.qSize + layerState.kSize + head * layerState.headVDim;

    for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx += 1) {
      const convRowBase = tokenIdx * layerState.convDim;
      const zRowBase = tokenIdx * layerState.valueDim + head * layerState.headVDim;
      const abRowBase = tokenIdx * layerState.numVHeads + head;
      const outRowBase = tokenIdx * layerState.valueDim + head * layerState.headVDim;

      let qNormSq = 0;
      let kNormSq = 0;
      for (let d = 0; d < layerState.headKDim; d += 1) {
        const qValue = convOut[convRowBase + qBase + d];
        const kValue = convOut[convRowBase + kBase + d];
        qNormSq += qValue * qValue;
        kNormSq += kValue * kValue;
      }

      const qNormScale = headScale / Math.sqrt(qNormSq + qkL2NormEps);
      const kNormScale = 1 / Math.sqrt(kNormSq + qkL2NormEps);
      const beta = 1 / (1 + Math.exp(-b[abRowBase]));
      const g = layerState.aNegExp[head] * softplus(a[abRowBase] + layerState.dtBias[head]);
      const gExp = Math.exp(g);

      for (let i = 0; i < layerState.headKDim * layerState.headVDim; i += 1) {
        recurrentState[recurrentHeadBase + i] *= gExp;
      }

      for (let vd = 0; vd < layerState.headVDim; vd += 1) {
        let kvMem = 0;
        for (let kd = 0; kd < layerState.headKDim; kd += 1) {
          const kNormed = convOut[convRowBase + kBase + kd] * kNormScale;
          const stateIdx = recurrentHeadBase + kd * layerState.headVDim + vd;
          kvMem += recurrentState[stateIdx] * kNormed;
        }
        const delta = (convOut[convRowBase + vBase + vd] - kvMem) * beta;
        for (let kd = 0; kd < layerState.headKDim; kd += 1) {
          const kNormed = convOut[convRowBase + kBase + kd] * kNormScale;
          const stateIdx = recurrentHeadBase + kd * layerState.headVDim + vd;
          recurrentState[stateIdx] += kNormed * delta;
        }
      }

      let meanSq = 0;
      for (let vd = 0; vd < layerState.headVDim; vd += 1) {
        let outValue = 0;
        for (let kd = 0; kd < layerState.headKDim; kd += 1) {
          const qNormed = convOut[convRowBase + qBase + kd] * qNormScale;
          const stateIdx = recurrentHeadBase + kd * layerState.headVDim + vd;
          outValue += recurrentState[stateIdx] * qNormed;
        }
        output[outRowBase + vd] = outValue;
        meanSq += outValue * outValue;
      }

      const invRms = 1 / Math.sqrt(meanSq / layerState.headVDim + layerState.rmsNormEps);
      for (let vd = 0; vd < layerState.headVDim; vd += 1) {
        const gate = silu(z[zRowBase + vd]);
        const normIndex = layerState.normMode === 'per_head'
          ? head * layerState.headVDim + vd
          : vd;
        output[outRowBase + vd] = output[outRowBase + vd] * invRms * layerState.normWeight[normIndex] * gate;
      }
    }
  }

  return { output, recurrentState };
}

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

function createF16View(values) {
  const encoded = new Uint16Array(values.length);
  const rounded = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    encoded[i] = float32ToFloat16(values[i]);
    rounded[i] = float16ToFloat32(encoded[i]);
  }
  return { encoded, rounded };
}

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`linear-attention-core-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

const numTokens = 3;
const qkL2NormEps = 1e-6;
const qkv = new Float32Array([
  0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.7,
  -0.3, 0.6, -0.5, 0.2, 0.8, 0.4, -0.2, 0.9,
  0.7, 0.1, 0.2, -0.4, -0.6, 0.3, 0.5, -0.8,
]);
const z = new Float32Array([
  0.3, -0.2, 0.5, 0.1,
  -0.4, 0.6, 0.2, -0.1,
  0.7, 0.2, -0.3, 0.4,
]);
const a = new Float32Array([0.1, -0.2, 0.3, 0.4, -0.5, 0.2]);
const b = new Float32Array([0.2, -0.1, -0.3, 0.7, 0.4, -0.2]);
const layerState = {
  layerIdx: 0,
  convKernelSize: 1,
  convDim: 8,
  keyDim: 2,
  valueDim: 4,
  numKHeads: 1,
  numVHeads: 2,
  headKDim: 2,
  headVDim: 2,
  qSize: 2,
  kSize: 2,
  vSize: 4,
  qRep: 2,
  normMode: 'shared',
  rmsNormEps: 1e-6,
  convWeight: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]),
  dtBias: new Float32Array([0.2, -0.1]),
  aNegExp: new Float32Array([-0.6, -0.4]),
  normWeight: new Float32Array([1.1, 0.9]),
  convState: new Float32Array(8),
  recurrentState: new Float32Array(8),
};

async function runCase(inputDtype) {
  const qkvInput = inputDtype === 'f16' ? createF16View(qkv) : null;
  const zInput = inputDtype === 'f16' ? createF16View(z) : null;
  const aInput = inputDtype === 'f16' ? createF16View(a) : null;
  const bInput = inputDtype === 'f16' ? createF16View(b) : null;
  const reference = buildReferenceOutput({
    qkv: qkvInput?.rounded ?? qkv,
    z: zInput?.rounded ?? z,
    a: aInput?.rounded ?? a,
    b: bInput?.rounded ?? b,
    layerState,
    numTokens,
    qkL2NormEps,
  });
  const gpuLayerState = {
    ...layerState,
    convWeightGPU: createGpuBuffer(layerState.convWeight, `linear_conv_weight_${inputDtype}`),
    dtBiasGPU: createGpuBuffer(layerState.dtBias, `linear_dt_bias_${inputDtype}`),
    aNegExpGPU: createGpuBuffer(layerState.aNegExp, `linear_a_neg_exp_${inputDtype}`),
    normWeightGPU: createGpuBuffer(layerState.normWeight, `linear_norm_weight_${inputDtype}`),
    convStateGPU: createGpuBuffer(layerState.convState, `linear_conv_state_${inputDtype}`),
    recurrentStateGPU: createGpuBuffer(layerState.recurrentState, `linear_recurrent_state_${inputDtype}`),
  };
  const qkvBuffer = createGpuBuffer(qkvInput?.encoded ?? qkv, `linear_qkv_${inputDtype}`);
  const zBuffer = createGpuBuffer(zInput?.encoded ?? z, `linear_z_${inputDtype}`);
  const aBuffer = createGpuBuffer(aInput?.encoded ?? a, `linear_a_${inputDtype}`);
  const bBuffer = createGpuBuffer(bInput?.encoded ?? b, `linear_b_${inputDtype}`);

  let outputTensor = null;
  try {
    outputTensor = await runLinearAttentionCoreGPU(
      createTensor(qkvBuffer, inputDtype, [numTokens, layerState.convDim], `linear_qkv_${inputDtype}`),
      createTensor(zBuffer, inputDtype, [numTokens, layerState.valueDim], `linear_z_${inputDtype}`),
      createTensor(aBuffer, inputDtype, [numTokens, layerState.numVHeads], `linear_a_${inputDtype}`),
      createTensor(bBuffer, inputDtype, [numTokens, layerState.numVHeads], `linear_b_${inputDtype}`),
      gpuLayerState,
      { numTokens, qkL2NormEps }
    );

    const gpuOutput = new Float32Array(await readBuffer(outputTensor.buffer, reference.output.byteLength));
    const gpuState = new Float32Array(
      await readBuffer(gpuLayerState.recurrentStateGPU, reference.recurrentState.byteLength)
    );
    const tolerance = inputDtype === 'f16' ? 7e-4 : 1e-5;
    assertArraysClose(gpuOutput, reference.output, tolerance, `${inputDtype} output`);
    assertArraysClose(gpuState, reference.recurrentState, tolerance, `${inputDtype} recurrent_state`);
  } finally {
    if (outputTensor?.buffer) {
      releaseBuffer(outputTensor.buffer);
    }
    for (const buffer of [
      qkvBuffer,
      zBuffer,
      aBuffer,
      bBuffer,
      gpuLayerState.convWeightGPU,
      gpuLayerState.dtBiasGPU,
      gpuLayerState.aNegExpGPU,
      gpuLayerState.normWeightGPU,
      gpuLayerState.convStateGPU,
      gpuLayerState.recurrentStateGPU,
    ]) {
      releaseBuffer(buffer);
    }
  }
}

await runCase('f32');
await runCase('f16');

console.log('linear-attention-core-gpu-regression.test: ok');
