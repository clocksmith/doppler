import { getDevice } from '../../../gpu/device.js';
import { createTensor, dtypeBytes } from '../../../gpu/tensor.js';
import { getBuffer } from '../../../gpu/weight-buffer.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import {
  runConv2D,
  runTranspose,
  runGather,
  runLayerNorm,
  runRMSNorm,
  runMatmul,
  runAttention,
  runGeLU,
  runSiLURowSplit,
  runResidualAdd,
  runBiasAdd,
  runModulate,
  runPixelShuffle,
  recordConv2D,
  recordTranspose,
  recordGather,
  recordLayerNorm,
  recordRMSNorm,
  recordMatmul,
  recordAttention,
  recordGeLU,
  recordSiLURowSplit,
  recordResidualAdd,
  recordBiasAdd,
  recordModulate,
  recordPixelShuffle,
} from '../../../gpu/kernels/index.js';
import {
  expectDiffusionWeight,
  inferDiffusionMatmulDtypeFromBuffer,
  normalizeDiffusionLocationDtype,
  normalizeDiffusionMatmulLocationDtype,
  resolveDiffusionActivationDtype,
} from './weight-contract.js';
import {
  resolveSD3BiasDtype,
  resolveSD3MatmulDtype,
} from './sd3/plan.js';
import { executeSD3Transformer } from './sd3/execution.js';


function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label);
}

function createKernelOps(recorder) {
  if (!recorder) {
    return {
      conv2d: runConv2D,
      transpose: runTranspose,
      gather: runGather,
      layerNorm: runLayerNorm,
      rmsNorm: runRMSNorm,
      attention: runAttention,
      gelu: runGeLU,
      siluRowSplit: runSiLURowSplit,
      residualAdd: runResidualAdd,
      biasAdd: runBiasAdd,
      modulate: runModulate,
      pixelShuffle: runPixelShuffle,
    };
  }
  return {
    conv2d: (...args) => recordConv2D(recorder, ...args),
    transpose: (...args) => recordTranspose(recorder, ...args),
    gather: (...args) => recordGather(recorder, ...args),
    layerNorm: (...args) => recordLayerNorm(recorder, ...args),
    rmsNorm: (...args) => recordRMSNorm(recorder, ...args),
    attention: (...args) => recordAttention(recorder, ...args),
    gelu: (...args) => recordGeLU(recorder, ...args),
    siluRowSplit: (...args) => recordSiLURowSplit(recorder, ...args),
    residualAdd: (...args) => recordResidualAdd(recorder, ...args),
    biasAdd: (...args) => recordBiasAdd(recorder, ...args),
    modulate: (...args) => recordModulate(recorder, ...args),
    pixelShuffle: (...args) => recordPixelShuffle(recorder, ...args),
  };
}

function createVectorBuffer(device, data, label) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}


async function runMatmulResolved(input, weight, resolver, name, M, N, K, options = {}) {
  const { recorder = null, ...rest } = options;
  const resolved = resolveSD3MatmulDtype(weight?.dtype, resolver && name ? resolver.dtype(name) : null);
  const bDtype = inferDiffusionMatmulDtypeFromBuffer(weight, N, K, resolved);
  const nextOptions = bDtype ? { ...rest, bDtype } : rest;
  if (recorder) {
    return recordMatmul(recorder, input, weight, M, N, K, nextOptions);
  }
  return runMatmul(input, weight, M, N, K, nextOptions);
}

function createBiasTensorWithDtype(weight, size, label, resolver, name) {
  if (!weight) return null;
  const dtype = resolveSD3BiasDtype(weight?.dtype, resolver && name ? resolver.dtype(name) : null);
  return createTensor(getBuffer(weight), dtype, [size], label);
}


async function splitQKV(qkv, numTokens, hiddenSize, label, recorder) {
  const device = getDevice();
  const bytesPerElement = dtypeBytes(qkv.dtype);
  const sliceBytes = numTokens * hiddenSize * bytesPerElement;
  const qBuf = acquireBuffer(sliceBytes, undefined, `${label}_q`);
  const kBuf = acquireBuffer(sliceBytes, undefined, `${label}_k`);
  const vBuf = acquireBuffer(sliceBytes, undefined, `${label}_v`);

  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
  encoder.copyBufferToBuffer(qkv.buffer, 0, qBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes, kBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes * 2, vBuf, 0, sliceBytes);
  if (!recorder) {
    device.queue.submit([encoder.finish()]);
  }

  return {
    q: createTensor(qBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_q`),
    k: createTensor(kBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_k`),
    v: createTensor(vBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_v`),
  };
}

async function runFusedQKV(input, weight, biasTensor, numTokens, hiddenSize, outputDtype, label, matmul, weightName, ops, release, recorder) {
  const qkv = await matmul(input, weight, weightName, numTokens, hiddenSize * 3, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });

  let qkvTensor = qkv;
  if (biasTensor) {
    qkvTensor = await ops.biasAdd(qkv, biasTensor, numTokens, hiddenSize * 3);
  }

  const split = await splitQKV(qkvTensor, numTokens, hiddenSize, label, recorder);
  release(qkvTensor.buffer);
  return split;
}

async function runQKV(input, weights, bias, numTokens, hiddenSize, label, matmul, weightNames, ops, release, recorder) {
  const outputDtype = input.dtype;
  if (weights.qkv) {
    return runFusedQKV(
      input,
      weights.qkv,
      bias?.qkv ?? null,
      numTokens,
      hiddenSize,
      outputDtype,
      label,
      matmul,
      weightNames?.qkv ?? null,
      ops,
      release,
      recorder
    );
  }

  const qWeight = expectDiffusionWeight(weights.q, `${label}.q`);
  const kWeight = expectDiffusionWeight(weights.k, `${label}.k`);
  const vWeight = expectDiffusionWeight(weights.v, `${label}.v`);

  let q = await matmul(input, qWeight, weightNames?.q ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });
  let k = await matmul(input, kWeight, weightNames?.k ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });
  let v = await matmul(input, vWeight, weightNames?.v ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });

  if (bias?.q) q = await ops.biasAdd(q, bias.q, numTokens, hiddenSize);
  if (bias?.k) k = await ops.biasAdd(k, bias.k, numTokens, hiddenSize);
  if (bias?.v) v = await ops.biasAdd(v, bias.v, numTokens, hiddenSize);

  return { q, k, v };
}

async function applyQKNorm(tensor, weight, numTokens, numHeads, headDim, eps, ops) {
  const flattened = createTensor(tensor.buffer, tensor.dtype, [numTokens * numHeads, headDim], 'qk_norm_in');
  const normed = await ops.rmsNorm(flattened, getBuffer(weight), eps, {
    batchSize: numTokens * numHeads,
    hiddenSize: headDim,
  });
  return reshapeTensor(normed, [numTokens, numHeads, headDim], 'qk_norm_out');
}

async function concatKV(a, b, numTokensA, numTokensB, hiddenSize, recorder) {
  const device = getDevice();
  const bytesPerElement = a.dtype === 'f16' ? 2 : 4;
  const outputSize = (numTokensA + numTokensB) * hiddenSize * bytesPerElement;
  const output = acquireBuffer(outputSize, undefined, 'kv_concat');
  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
  encoder.copyBufferToBuffer(a.buffer, 0, output, 0, numTokensA * hiddenSize * bytesPerElement);
  encoder.copyBufferToBuffer(b.buffer, 0, output, numTokensA * hiddenSize * bytesPerElement, numTokensB * hiddenSize * bytesPerElement);
  if (!recorder) {
    device.queue.submit([encoder.finish()]);
  }
  return createTensor(output, a.dtype, [numTokensA + numTokensB, hiddenSize], 'kv_concat');
}


async function buildModulation(timeText, weight, bias, hiddenSize, segments, runtime, matmul, weightName, ops) {
  const device = getDevice();
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const outDim = hiddenSize * segments;
  const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
  const bufferSize = (outDim + hiddenSize) * bytesPerElement;
  const outputBuffer = acquireBuffer(bufferSize, undefined, 'sd3_modulate');

  const mod = await matmul(timeText, weight, weightName, 1, outDim, hiddenSize, {
    outputDtype: activationDtype,
    transposeB: 'auto',
    outputBuffer,
  });

  if (bias) {
    await ops.biasAdd(mod, bias, 1, outDim);
  }

  const zeroOffset = outDim * bytesPerElement;
  device.queue.writeBuffer(outputBuffer, zeroOffset, new Uint8Array(hiddenSize * bytesPerElement));

  return {
    tensor: createTensor(outputBuffer, activationDtype, [1, outDim], 'sd3_mod'),
    zeroOffset: outDim,
  };
}

async function applyAdaLayerNorm(input, weight, bias, eps, mod, offsets, runtime, ops, release, options = {}) {
  const { numTokens, hiddenSize } = options;
  const normed = await ops.layerNorm(input, weight, bias, eps, { batchSize: numTokens, hiddenSize });
  const modulated = await ops.modulate(normed, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.scale,
    shiftOffset: offsets.shift,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: true,
  });
  release(normed.buffer);
  return modulated;
}

async function applyGate(output, mod, offsets, ops, release, options = {}) {
  const { numTokens, hiddenSize, zeroOffset } = options;
  const gated = await ops.modulate(output, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.gate,
    shiftOffset: zeroOffset,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: false,
  });
  release(output.buffer);
  return gated;
}

async function runFFN(input, weights, bias, numTokens, hiddenSize, runtime, matmul, weightNames, ops, release) {
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const upDim = weights.up.shape[0];
  const downInput = weights.down.shape[1];
  let up = await matmul(input, weights.up, weightNames?.up ?? null, numTokens, upDim, hiddenSize, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.up) up = await ops.biasAdd(up, bias.up, numTokens, upDim);

  let act = null;
  let intermediate = upDim;
  if (Number.isFinite(downInput) && upDim === downInput * 2) {
    act = await ops.siluRowSplit(up, {
      numTokens,
      dim: downInput,
      activation: 'gelu',
      swigluLimit: null,
    });
    intermediate = downInput;
  } else {
    act = await ops.gelu(up, { size: numTokens * upDim });
  }
  release(up.buffer);

  let down = await matmul(act, weights.down, weightNames?.down ?? null, numTokens, hiddenSize, intermediate, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.down) down = await ops.biasAdd(down, bias.down, numTokens, hiddenSize);
  release(act.buffer);
  return down;
}


export async function runSD3Transformer(
  latents,
  context,
  timeText,
  weightsEntry,
  modelConfig,
  runtime,
  options = {}
) {
  return executeSD3Transformer(
    latents,
    context,
    timeText,
    weightsEntry,
    modelConfig,
    runtime,
    options,
    {
      applyAdaLayerNorm,
      applyGate,
      applyQKNorm,
      buildModulation,
      concatKV,
      createBiasTensorWithDtype,
      createKernelOps,
      createVectorBuffer,
      runFFN,
      runMatmulResolved,
      runQKV,
    }
  );
}
