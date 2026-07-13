import {
  runAttention,
  runMatmul,
  runQwenAttentionSplitQGate,
  runRMSNorm,
  runResidualAdd,
  runRoPE,
  runSiLU,
} from '../../gpu/kernels/index.js';
import {
  runQwenAttentionSplitQGateBackward,
  runQwenGqaAttentionBackward,
  runRmsNormBackward,
  runRoPEBackward,
  runSigmoidGatedBackward,
} from '../../gpu/kernels/backward/index.js';
import { runMatmulBackwardDx } from '../../gpu/kernels/backward/utils.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function resolveDimensions(options) {
  const seqLen = positiveInteger(options?.seqLen, 'seqLen');
  const hiddenSize = positiveInteger(options?.hiddenSize, 'hiddenSize');
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const numKVHeads = positiveInteger(options?.numKVHeads, 'numKVHeads');
  const headDim = positiveInteger(options?.headDim, 'headDim');
  const rotaryDim = positiveInteger(options?.rotaryDim, 'rotaryDim');
  const pairSpanDim = positiveInteger(options?.pairSpanDim, 'pairSpanDim');
  const rmsEps = Number(options?.rmsEps);
  if (numHeads % numKVHeads !== 0 || rotaryDim > headDim
    || pairSpanDim < rotaryDim || pairSpanDim > headDim
    || !Number.isFinite(rmsEps) || rmsEps <= 0) {
    throw new Error('invalid Qwen full-attention training geometry.');
  }
  return {
    seqLen,
    hiddenSize,
    numHeads,
    numKVHeads,
    headDim,
    rotaryDim,
    pairSpanDim,
    rmsEps,
    interleaved: options?.interleaved === true,
    startPos: Math.max(0, Math.floor(Number(options?.startPos ?? 0))),
    querySize: numHeads * headDim,
    kvSize: numKVHeads * headDim,
    scale: 1 / Math.sqrt(headDim),
  };
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseTensorMap(tensors) {
  if (!tensors) return;
  for (const tensor of Object.values(tensors)) releaseTensor(tensor);
}

export async function runQwenFullAttentionTrainingModuleForward(inputs, options = {}) {
  const dims = resolveDimensions(options);
  let qProjection = null;
  let split = null;
  let keyProjection = null;
  let valueProjection = null;
  let queryRope = null;
  let keyRope = null;
  let attention = null;
  let gated = null;
  let output = null;
  let completed = false;
  try {
    qProjection = await runMatmul(
      inputs.hidden,
      inputs.qWeight,
      dims.seqLen,
      dims.querySize * 2,
      dims.hiddenSize,
      { transposeB: true, outputDtype: 'f32' }
    );
    split = await runQwenAttentionSplitQGate(qProjection, {
      numTokens: dims.seqLen,
      numHeads: dims.numHeads,
      headDim: dims.headDim,
    });
    keyProjection = await runMatmul(
      inputs.hidden,
      inputs.kWeight,
      dims.seqLen,
      dims.kvSize,
      dims.hiddenSize,
      { transposeB: true, outputDtype: 'f32' }
    );
    valueProjection = await runMatmul(
      inputs.hidden,
      inputs.vWeight,
      dims.seqLen,
      dims.kvSize,
      dims.hiddenSize,
      { transposeB: true, outputDtype: 'f32' }
    );
    queryRope = await runRMSNorm(split.query, inputs.qNormWeight, dims.rmsEps, {
      batchSize: dims.seqLen * dims.numHeads,
      hiddenSize: dims.headDim,
      rmsNormWeightOffset: true,
    });
    keyRope = await runRMSNorm(keyProjection, inputs.kNormWeight, dims.rmsEps, {
      batchSize: dims.seqLen * dims.numKVHeads,
      hiddenSize: dims.headDim,
      rmsNormWeightOffset: true,
    });
    const ropeOptions = {
      headDim: dims.headDim,
      rotaryDim: dims.rotaryDim,
      pairSpanDim: dims.pairSpanDim,
      interleaved: dims.interleaved,
      startPos: dims.startPos,
    };
    await runRoPE(queryRope, inputs.cos, inputs.sin, dims.seqLen, {
      ...ropeOptions,
      numHeads: dims.numHeads,
    });
    await runRoPE(keyRope, inputs.cos, inputs.sin, dims.seqLen, {
      ...ropeOptions,
      numHeads: dims.numKVHeads,
    });
    attention = await runAttention(
      queryRope,
      keyRope,
      valueProjection,
      null,
      dims.numHeads,
      dims.headDim,
      {
        seqLen: dims.seqLen,
        kvLen: dims.seqLen,
        numKVHeads: dims.numKVHeads,
        causal: true,
        startPos: dims.startPos,
        scale: dims.scale,
      }
    );
    gated = await runSiLU(attention, {
      size: dims.seqLen * dims.querySize,
      gate: split.gate,
      gateActivation: 'sigmoid',
      inputActivation: 'identity',
      swigluLimit: null,
    });
    output = await runMatmul(
      gated,
      inputs.oWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.querySize,
      { transposeB: true, outputDtype: 'f32' }
    );
    completed = true;
    return {
      output,
      cache: {
        dims,
        rawQuery: split.query,
        gate: split.gate,
        rawKey: keyProjection,
        value: valueProjection,
        queryRope,
        keyRope,
        attention,
      },
    };
  } finally {
    releaseTensor(qProjection);
    releaseTensor(gated);
    if (!completed) {
      releaseTensor(output);
      releaseTensorMap(split);
      releaseTensor(keyProjection);
      releaseTensor(valueProjection);
      releaseTensor(queryRope);
      releaseTensor(keyRope);
      releaseTensor(attention);
    }
  }
}

export async function runQwenFullAttentionTrainingModuleBackward(
  inputs,
  gradOutput,
  cache,
  options = {}
) {
  const dims = resolveDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`Qwen full-attention cache mismatch for ${key}.`);
    }
  }
  let gradGated = null;
  let gateGradients = null;
  let attentionGradients = null;
  let gradQueryNorm = null;
  let gradKeyNorm = null;
  let gradQuery = null;
  let gradKey = null;
  let gradQProjection = null;
  const contributions = [];
  let sumQueryKey = null;
  let hidden = null;
  let completed = false;
  try {
    gradGated = await runMatmulBackwardDx(
      gradOutput,
      inputs.oWeight,
      dims.seqLen,
      dims.querySize,
      dims.hiddenSize,
      { transposeB: true }
    );
    gateGradients = await runSigmoidGatedBackward(
      cache.attention,
      cache.gate,
      gradGated,
      { count: dims.seqLen * dims.querySize }
    );
    attentionGradients = await runQwenGqaAttentionBackward(
      cache.queryRope,
      cache.keyRope,
      cache.value,
      gateGradients.input,
      {
        seqLen: dims.seqLen,
        numHeads: dims.numHeads,
        numKVHeads: dims.numKVHeads,
        headDim: dims.headDim,
        scale: dims.scale,
        causal: true,
      }
    );
    const ropeOptions = {
      seqLen: dims.seqLen,
      headDim: dims.headDim,
      rotaryDim: dims.rotaryDim,
      pairSpanDim: dims.pairSpanDim,
      interleaved: dims.interleaved,
      startPos: dims.startPos,
    };
    gradQueryNorm = await runRoPEBackward(
      attentionGradients.query,
      inputs.cos,
      inputs.sin,
      { ...ropeOptions, numHeads: dims.numHeads }
    );
    gradKeyNorm = await runRoPEBackward(
      attentionGradients.key,
      inputs.cos,
      inputs.sin,
      { ...ropeOptions, numHeads: dims.numKVHeads }
    );
    gradQuery = await runRmsNormBackward(
      cache.rawQuery,
      inputs.qNormWeight,
      gradQueryNorm,
      {
        numTokens: dims.seqLen * dims.numHeads,
        hiddenSize: dims.headDim,
        eps: dims.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    gradKey = await runRmsNormBackward(
      cache.rawKey,
      inputs.kNormWeight,
      gradKeyNorm,
      {
        numTokens: dims.seqLen * dims.numKVHeads,
        hiddenSize: dims.headDim,
        eps: dims.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    gradQProjection = await runQwenAttentionSplitQGateBackward(
      gradQuery,
      gateGradients.gate,
      { numTokens: dims.seqLen, numHeads: dims.numHeads, headDim: dims.headDim }
    );
    for (const [gradient, weight, outputSize] of [
      [gradQProjection, inputs.qWeight, dims.querySize * 2],
      [gradKey, inputs.kWeight, dims.kvSize],
      [attentionGradients.value, inputs.vWeight, dims.kvSize],
    ]) {
      contributions.push(await runMatmulBackwardDx(
        gradient,
        weight,
        dims.seqLen,
        dims.hiddenSize,
        outputSize,
        { transposeB: true }
      ));
    }
    const hiddenElements = dims.seqLen * dims.hiddenSize;
    sumQueryKey = await runResidualAdd(contributions[0], contributions[1], hiddenElements);
    hidden = await runResidualAdd(sumQueryKey, contributions[2], hiddenElements);
    completed = true;
    return { hidden };
  } finally {
    releaseTensor(gradGated);
    releaseTensorMap(gateGradients);
    releaseTensorMap(attentionGradients);
    releaseTensor(gradQueryNorm);
    releaseTensor(gradKeyNorm);
    releaseTensor(gradQuery);
    releaseTensor(gradKey);
    releaseTensor(gradQProjection);
    for (const tensor of contributions) releaseTensor(tensor);
    releaseTensor(sumQueryKey);
    if (!completed) releaseTensor(hidden);
  }
}

export function releaseQwenFullAttentionTrainingModuleCache(cache) {
  releaseTensor(cache?.rawQuery);
  releaseTensor(cache?.gate);
  releaseTensor(cache?.rawKey);
  releaseTensor(cache?.value);
  releaseTensor(cache?.queryRope);
  releaseTensor(cache?.keyRope);
  releaseTensor(cache?.attention);
}
