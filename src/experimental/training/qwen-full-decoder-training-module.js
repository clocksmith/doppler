import {
  runRMSNorm,
  runResidualAdd,
  runSiLU,
} from '../../gpu/kernels/index.js';
import {
  runRmsNormBackward,
  runSiluGatedBackward,
} from '../../gpu/kernels/backward/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  releaseQwenFullAttentionTrainingModuleCache,
  runFrozenLoraProjectionBackward,
  runFrozenLoraProjectionForward,
  runQwenFullAttentionTrainingModuleBackward,
  runQwenFullAttentionTrainingModuleForward,
} from './qwen-full-attention-training-module.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function resolveDimensions(options) {
  return {
    seqLen: positiveInteger(options?.seqLen, 'seqLen'),
    hiddenSize: positiveInteger(options?.hiddenSize, 'hiddenSize'),
    intermediateSize: positiveInteger(options?.intermediateSize, 'intermediateSize'),
  };
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseTensorMap(tensors) {
  if (!tensors) return;
  for (const tensor of Object.values(tensors)) releaseTensor(tensor);
}

function releaseProjectionGradients(gradients) {
  if (!gradients) return;
  releaseTensor(gradients.A);
  releaseTensor(gradients.B);
}

function releaseAttentionAdapterGradients(gradients) {
  if (!gradients?.lora) return;
  for (const projection of Object.values(gradients.lora)) {
    releaseProjectionGradients(projection);
  }
}

export async function runQwenFullDecoderLayerForward(inputs, options = {}) {
  const dims = resolveDimensions(options);
  let inputNorm = null;
  let attention = null;
  let normalizedAttention = null;
  let postAttention = null;
  let gate = null;
  let up = null;
  let activated = null;
  let down = null;
  let output = null;
  let completed = false;
  try {
    inputNorm = await runRMSNorm(inputs.hidden, inputs.inputNormWeight, options.rmsEps, {
      batchSize: dims.seqLen,
      hiddenSize: dims.hiddenSize,
      rmsNormWeightOffset: true,
    });
    const attentionInputs = { ...inputs.attention, hidden: inputNorm };
    attention = await runQwenFullAttentionTrainingModuleForward(attentionInputs, options);
    normalizedAttention = await runRMSNorm(
      attention.output,
      inputs.postAttentionNormWeight,
      options.rmsEps,
      {
        batchSize: dims.seqLen,
        hiddenSize: dims.hiddenSize,
        rmsNormWeightOffset: true,
      }
    );
    postAttention = await runResidualAdd(
      inputs.hidden,
      normalizedAttention,
      dims.seqLen * dims.hiddenSize
    );
    gate = await runFrozenLoraProjectionForward(
      postAttention,
      inputs.mlp.gateWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.mlp.lora?.gate,
      'gate_proj'
    );
    up = await runFrozenLoraProjectionForward(
      postAttention,
      inputs.mlp.upWeight,
      dims.seqLen,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.mlp.lora?.up,
      'up_proj'
    );
    activated = await runSiLU(up.output, {
      size: dims.seqLen * dims.intermediateSize,
      gate: gate.output,
      inputActivation: 'identity',
      swigluLimit: null,
    });
    down = await runFrozenLoraProjectionForward(
      activated,
      inputs.mlp.downWeight,
      dims.seqLen,
      dims.intermediateSize,
      dims.hiddenSize,
      inputs.mlp.lora?.down,
      'down_proj'
    );
    output = await runResidualAdd(
      postAttention,
      down.output,
      dims.seqLen * dims.hiddenSize
    );
    completed = true;
    return {
      output,
      cache: {
        dims,
        inputNorm,
        attentionInputs,
        attentionOutput: attention.output,
        attentionCache: attention.cache,
        postAttention,
        gate: gate.output,
        up: up.output,
        activated,
        projectionDowns: {
          gate: gate.down,
          up: up.down,
          down: down.down,
        },
      },
    };
  } finally {
    releaseTensor(normalizedAttention);
    releaseTensor(down?.output);
    if (!completed) {
      releaseTensor(output);
      releaseTensor(inputNorm);
      releaseTensor(attention?.output);
      if (attention?.cache) releaseQwenFullAttentionTrainingModuleCache(attention.cache);
      releaseTensor(postAttention);
      releaseTensor(gate?.output);
      releaseTensor(up?.output);
      releaseTensor(activated);
      releaseTensor(gate?.down);
      releaseTensor(up?.down);
      releaseTensor(down?.down);
    }
  }
}

export async function runQwenFullDecoderLayerBackward(inputs, gradOutput, cache, options = {}) {
  const dims = resolveDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`Qwen full-decoder cache mismatch for ${key}.`);
    }
  }
  let downGradients = null;
  let activationGradients = null;
  let gateGradients = null;
  let upGradients = null;
  let mlpInputGradient = null;
  let postAttentionGradient = null;
  let attentionOutputGradient = null;
  let attentionGradients = null;
  let inputNormGradient = null;
  let hidden = null;
  let completed = false;
  try {
    downGradients = await runFrozenLoraProjectionBackward(
      cache.activated,
      inputs.mlp.downWeight,
      gradOutput,
      dims.seqLen,
      dims.intermediateSize,
      dims.hiddenSize,
      inputs.mlp.lora?.down,
      cache.projectionDowns?.down,
      'down_proj'
    );
    activationGradients = await runSiluGatedBackward(
      cache.gate,
      cache.up,
      downGradients.input,
      { count: dims.seqLen * dims.intermediateSize, swigluLimit: null }
    );
    gateGradients = await runFrozenLoraProjectionBackward(
      cache.postAttention,
      inputs.mlp.gateWeight,
      activationGradients.gate,
      dims.seqLen,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.mlp.lora?.gate,
      cache.projectionDowns?.gate,
      'gate_proj'
    );
    upGradients = await runFrozenLoraProjectionBackward(
      cache.postAttention,
      inputs.mlp.upWeight,
      activationGradients.up,
      dims.seqLen,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.mlp.lora?.up,
      cache.projectionDowns?.up,
      'up_proj'
    );
    mlpInputGradient = await runResidualAdd(
      gateGradients.input,
      upGradients.input,
      dims.seqLen * dims.hiddenSize
    );
    postAttentionGradient = await runResidualAdd(
      gradOutput,
      mlpInputGradient,
      dims.seqLen * dims.hiddenSize
    );
    attentionOutputGradient = await runRmsNormBackward(
      cache.attentionOutput,
      inputs.postAttentionNormWeight,
      postAttentionGradient,
      {
        numTokens: dims.seqLen,
        hiddenSize: dims.hiddenSize,
        eps: options.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    attentionGradients = await runQwenFullAttentionTrainingModuleBackward(
      cache.attentionInputs,
      attentionOutputGradient,
      cache.attentionCache,
      options
    );
    inputNormGradient = await runRmsNormBackward(
      inputs.hidden,
      inputs.inputNormWeight,
      attentionGradients.hidden,
      {
        numTokens: dims.seqLen,
        hiddenSize: dims.hiddenSize,
        eps: options.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    hidden = await runResidualAdd(
      postAttentionGradient,
      inputNormGradient,
      dims.seqLen * dims.hiddenSize
    );
    completed = true;
    return {
      hidden,
      lora: {
        q: attentionGradients.lora.q,
        k: attentionGradients.lora.k,
        v: attentionGradients.lora.v,
        o: attentionGradients.lora.o,
        gate: { A: gateGradients.A, B: gateGradients.B },
        up: { A: upGradients.A, B: upGradients.B },
        down: { A: downGradients.A, B: downGradients.B },
      },
    };
  } finally {
    releaseTensor(downGradients?.input);
    releaseTensorMap(activationGradients);
    releaseTensor(gateGradients?.input);
    releaseTensor(upGradients?.input);
    releaseTensor(mlpInputGradient);
    releaseTensor(postAttentionGradient);
    releaseTensor(attentionOutputGradient);
    releaseTensor(attentionGradients?.hidden);
    releaseTensor(inputNormGradient);
    if (!completed) {
      releaseTensor(hidden);
      releaseProjectionGradients(downGradients);
      releaseProjectionGradients(gateGradients);
      releaseProjectionGradients(upGradients);
      releaseAttentionAdapterGradients(attentionGradients);
    }
  }
}

export function releaseQwenFullDecoderLayerCache(cache) {
  releaseTensor(cache?.inputNorm);
  releaseTensor(cache?.attentionOutput);
  releaseQwenFullAttentionTrainingModuleCache(cache?.attentionCache);
  releaseTensor(cache?.postAttention);
  releaseTensor(cache?.gate);
  releaseTensor(cache?.up);
  releaseTensor(cache?.activated);
  releaseTensorMap(cache?.projectionDowns);
}
