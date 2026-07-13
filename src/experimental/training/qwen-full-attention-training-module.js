import {
  runAttention,
  runMatmul,
  runQwenAttentionSplitQGate,
  runRMSNorm,
  runResidualAdd,
  runRoPE,
  runScale,
  runSiLU,
} from '../../gpu/kernels/index.js';
import {
  runMatmulBackward,
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

function requireMatrixShape(tensor, rows, columns, label) {
  if (!tensor || tensor.shape?.length !== 2
    || tensor.shape[0] !== rows || tensor.shape[1] !== columns) {
    throw new Error(`${label} must have shape [${rows}, ${columns}].`);
  }
}

function resolveAdapter(adapter, inputSize, outputSize, label) {
  if (!adapter) return null;
  const rank = positiveInteger(adapter.rank, `${label} LoRA rank`);
  const alpha = Number(adapter.alpha);
  if (!Number.isFinite(alpha)) {
    throw new Error(`${label} LoRA alpha must be finite.`);
  }
  requireMatrixShape(adapter.A, inputSize, rank, `${label} LoRA A`);
  requireMatrixShape(adapter.B, rank, outputSize, `${label} LoRA B`);
  return { A: adapter.A, B: adapter.B, rank, scale: alpha / rank };
}

async function runProjectionForward(
  input,
  weight,
  rows,
  inputSize,
  outputSize,
  adapter,
  label
) {
  const resolvedAdapter = resolveAdapter(adapter, inputSize, outputSize, label);
  let base = null;
  let down = null;
  let up = null;
  let scaled = null;
  let output = null;
  let completed = false;
  try {
    base = await runMatmul(input, weight, rows, outputSize, inputSize, {
      transposeB: true,
      outputDtype: 'f32',
    });
    if (!resolvedAdapter) {
      completed = true;
      return { output: base, down: null };
    }
    down = await runMatmul(input, resolvedAdapter.A, rows, resolvedAdapter.rank, inputSize, {
      transposeB: false,
      outputDtype: 'f32',
    });
    up = await runMatmul(down, resolvedAdapter.B, rows, outputSize, resolvedAdapter.rank, {
      transposeB: false,
      outputDtype: 'f32',
    });
    scaled = await runScale(up, resolvedAdapter.scale, { count: rows * outputSize });
    output = await runResidualAdd(base, scaled, rows * outputSize);
    completed = true;
    return { output, down };
  } finally {
    if (resolvedAdapter) {
      releaseTensor(base);
      releaseTensor(up);
      releaseTensor(scaled);
    }
    if (!completed) {
      releaseTensor(output);
      releaseTensor(down);
    }
  }
}

async function runProjectionBackward(
  input,
  weight,
  gradOutput,
  rows,
  inputSize,
  outputSize,
  adapter,
  down,
  label
) {
  const resolvedAdapter = resolveAdapter(adapter, inputSize, outputSize, label);
  let baseInput = null;
  let scaledGradient = null;
  let downGradients = null;
  let inputGradients = null;
  let combinedInput = null;
  let completed = false;
  try {
    baseInput = await runMatmulBackwardDx(
      gradOutput,
      weight,
      rows,
      inputSize,
      outputSize,
      { transposeB: true }
    );
    if (!resolvedAdapter) {
      completed = true;
      return { input: baseInput, A: null, B: null };
    }
    if (!down) throw new Error(`${label} LoRA backward requires its forward cache.`);
    scaledGradient = await runScale(gradOutput, resolvedAdapter.scale, {
      count: rows * outputSize,
    });
    downGradients = await runMatmulBackward(
      down,
      resolvedAdapter.B,
      scaledGradient,
      { M: rows, N: outputSize, K: resolvedAdapter.rank, transposeB: false }
    );
    inputGradients = await runMatmulBackward(
      input,
      resolvedAdapter.A,
      downGradients.gradInput,
      { M: rows, N: resolvedAdapter.rank, K: inputSize, transposeB: false }
    );
    combinedInput = await runResidualAdd(
      baseInput,
      inputGradients.gradInput,
      rows * inputSize
    );
    completed = true;
    return {
      input: combinedInput,
      A: inputGradients.gradWeight,
      B: downGradients.gradWeight,
    };
  } finally {
    releaseTensor(scaledGradient);
    if (resolvedAdapter) {
      releaseTensor(baseInput);
      releaseTensor(downGradients?.gradInput);
      releaseTensor(inputGradients?.gradInput);
    }
    if (!completed) {
      releaseTensor(combinedInput);
      releaseTensor(downGradients?.gradWeight);
      releaseTensor(inputGradients?.gradWeight);
    }
  }
}

function releaseAdapterGradients(gradients) {
  if (!gradients) return;
  releaseTensor(gradients.A);
  releaseTensor(gradients.B);
}

export async function runQwenFullAttentionTrainingModuleForward(inputs, options = {}) {
  const dims = resolveDimensions(options);
  let qProjectionResult = null;
  let qProjection = null;
  let split = null;
  let keyProjectionResult = null;
  let keyProjection = null;
  let valueProjectionResult = null;
  let valueProjection = null;
  let queryRope = null;
  let keyRope = null;
  let attention = null;
  let gated = null;
  let outputResult = null;
  let output = null;
  let completed = false;
  try {
    qProjectionResult = await runProjectionForward(
      inputs.hidden,
      inputs.qWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.querySize * 2,
      inputs.lora?.q,
      'q_proj'
    );
    qProjection = qProjectionResult.output;
    split = await runQwenAttentionSplitQGate(qProjection, {
      numTokens: dims.seqLen,
      numHeads: dims.numHeads,
      headDim: dims.headDim,
    });
    keyProjectionResult = await runProjectionForward(
      inputs.hidden,
      inputs.kWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.kvSize,
      inputs.lora?.k,
      'k_proj'
    );
    keyProjection = keyProjectionResult.output;
    valueProjectionResult = await runProjectionForward(
      inputs.hidden,
      inputs.vWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.kvSize,
      inputs.lora?.v,
      'v_proj'
    );
    valueProjection = valueProjectionResult.output;
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
    outputResult = await runProjectionForward(
      gated,
      inputs.oWeight,
      dims.seqLen,
      dims.querySize,
      dims.hiddenSize,
      inputs.lora?.o,
      'o_proj'
    );
    output = outputResult.output;
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
        gated,
        projectionDowns: {
          q: qProjectionResult.down,
          k: keyProjectionResult.down,
          v: valueProjectionResult.down,
          o: outputResult.down,
        },
      },
    };
  } finally {
    releaseTensor(qProjection);
    if (!completed) {
      releaseTensor(output);
      releaseTensorMap(split);
      releaseTensor(keyProjection);
      releaseTensor(valueProjection);
      releaseTensor(queryRope);
      releaseTensor(keyRope);
      releaseTensor(attention);
      releaseTensor(gated);
      releaseTensor(qProjectionResult?.down);
      releaseTensor(keyProjectionResult?.down);
      releaseTensor(valueProjectionResult?.down);
      releaseTensor(outputResult?.down);
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
  let outputProjectionGradients = null;
  let qProjectionGradients = null;
  let kProjectionGradients = null;
  let vProjectionGradients = null;
  const contributions = [];
  let sumQueryKey = null;
  let hidden = null;
  let completed = false;
  try {
    outputProjectionGradients = await runProjectionBackward(
      cache.gated,
      inputs.oWeight,
      gradOutput,
      dims.seqLen,
      dims.querySize,
      dims.hiddenSize,
      inputs.lora?.o,
      cache.projectionDowns?.o,
      'o_proj'
    );
    gradGated = outputProjectionGradients.input;
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
    qProjectionGradients = await runProjectionBackward(
      inputs.hidden,
      inputs.qWeight,
      gradQProjection,
      dims.seqLen,
      dims.hiddenSize,
      dims.querySize * 2,
      inputs.lora?.q,
      cache.projectionDowns?.q,
      'q_proj'
    );
    contributions.push(qProjectionGradients.input);
    kProjectionGradients = await runProjectionBackward(
      inputs.hidden,
      inputs.kWeight,
      gradKey,
      dims.seqLen,
      dims.hiddenSize,
      dims.kvSize,
      inputs.lora?.k,
      cache.projectionDowns?.k,
      'k_proj'
    );
    contributions.push(kProjectionGradients.input);
    vProjectionGradients = await runProjectionBackward(
      inputs.hidden,
      inputs.vWeight,
      attentionGradients.value,
      dims.seqLen,
      dims.hiddenSize,
      dims.kvSize,
      inputs.lora?.v,
      cache.projectionDowns?.v,
      'v_proj'
    );
    contributions.push(vProjectionGradients.input);
    const hiddenElements = dims.seqLen * dims.hiddenSize;
    sumQueryKey = await runResidualAdd(contributions[0], contributions[1], hiddenElements);
    hidden = await runResidualAdd(sumQueryKey, contributions[2], hiddenElements);
    completed = true;
    return {
      hidden,
      lora: {
        q: { A: qProjectionGradients.A, B: qProjectionGradients.B },
        k: { A: kProjectionGradients.A, B: kProjectionGradients.B },
        v: { A: vProjectionGradients.A, B: vProjectionGradients.B },
        o: { A: outputProjectionGradients.A, B: outputProjectionGradients.B },
      },
    };
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
    if (!completed) {
      releaseTensor(hidden);
      releaseAdapterGradients(outputProjectionGradients);
      releaseAdapterGradients(qProjectionGradients);
      releaseAdapterGradients(kProjectionGradients);
      releaseAdapterGradients(vProjectionGradients);
    }
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
  releaseTensor(cache?.gated);
  releaseTensorMap(cache?.projectionDowns);
}
