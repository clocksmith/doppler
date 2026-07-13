import {
  runRMSNorm,
  runResidualAdd,
} from '../../gpu/kernels/index.js';
import { runRmsNormBackward } from '../../gpu/kernels/backward/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  releaseQwenFullAttentionTrainingModuleCache,
  runQwenFullAttentionTrainingModuleBackward,
  runQwenFullAttentionTrainingModuleForward,
} from './qwen-full-attention-training-module.js';
import {
  releaseQwenDecoderMlpCache,
  runQwenDecoderMlpBackward,
  runQwenDecoderMlpForward,
} from './qwen-decoder-mlp-training-module.js';

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

function releaseProjectionGradients(gradients) {
  if (!gradients) return;
  releaseTensor(gradients.A);
  releaseTensor(gradients.B);
}

function releaseMlpAdapterGradients(gradients) {
  if (!gradients?.lora) return;
  for (const projection of Object.values(gradients.lora)) {
    releaseProjectionGradients(projection);
  }
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
  let mlp = null;
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
    mlp = await runQwenDecoderMlpForward(
      { ...inputs.mlp, hidden: postAttention },
      {
        numTokens: dims.seqLen,
        hiddenSize: dims.hiddenSize,
        intermediateSize: dims.intermediateSize,
      }
    );
    output = await runResidualAdd(
      postAttention,
      mlp.output,
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
        mlpCache: mlp.cache,
      },
    };
  } finally {
    releaseTensor(normalizedAttention);
    releaseTensor(mlp?.output);
    if (!completed) {
      releaseTensor(output);
      releaseTensor(inputNorm);
      releaseTensor(attention?.output);
      if (attention?.cache) releaseQwenFullAttentionTrainingModuleCache(attention.cache);
      releaseTensor(postAttention);
      if (mlp?.cache) releaseQwenDecoderMlpCache(mlp.cache);
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
  let mlpGradients = null;
  let postAttentionGradient = null;
  let attentionOutputGradient = null;
  let attentionGradients = null;
  let inputNormGradient = null;
  let hidden = null;
  let completed = false;
  try {
    mlpGradients = await runQwenDecoderMlpBackward(
      { ...inputs.mlp, hidden: cache.postAttention },
      gradOutput,
      cache.mlpCache,
      {
        numTokens: dims.seqLen,
        hiddenSize: dims.hiddenSize,
        intermediateSize: dims.intermediateSize,
      }
    );
    postAttentionGradient = await runResidualAdd(
      gradOutput,
      mlpGradients.hidden,
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
        gate: mlpGradients.lora.gate,
        up: mlpGradients.lora.up,
        down: mlpGradients.lora.down,
      },
    };
  } finally {
    releaseTensor(mlpGradients?.hidden);
    releaseTensor(postAttentionGradient);
    releaseTensor(attentionOutputGradient);
    releaseTensor(attentionGradients?.hidden);
    releaseTensor(inputNormGradient);
    if (!completed) {
      releaseTensor(hidden);
      releaseMlpAdapterGradients(mlpGradients);
      releaseAttentionAdapterGradients(attentionGradients);
    }
  }
}

export function releaseQwenFullDecoderLayerCache(cache) {
  releaseTensor(cache?.inputNorm);
  releaseTensor(cache?.attentionOutput);
  releaseQwenFullAttentionTrainingModuleCache(cache?.attentionCache);
  releaseTensor(cache?.postAttention);
  releaseQwenDecoderMlpCache(cache?.mlpCache);
}
