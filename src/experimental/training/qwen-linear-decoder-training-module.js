import { runRMSNorm, runResidualAdd } from '../../gpu/kernels/index.js';
import { runRmsNormBackward } from '../../gpu/kernels/backward/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  releaseQwenDecoderMlpCache,
  runQwenDecoderMlpBackward,
  runQwenDecoderMlpForward,
} from './qwen-decoder-mlp-training-module.js';
import {
  releaseQwenLinearAttentionTrainingModuleCache,
  runQwenLinearAttentionTrainingModuleBackward,
  runQwenLinearAttentionTrainingModuleForward,
} from './qwen-linear-attention-training-core.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function resolveDimensions(options) {
  return {
    numTokens: positiveInteger(options?.numTokens, 'numTokens'),
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

export async function runQwenLinearDecoderLayerForward(inputs, options = {}) {
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
      batchSize: dims.numTokens,
      hiddenSize: dims.hiddenSize,
      rmsNormWeightOffset: true,
    });
    const attentionInputs = { ...inputs.attention, hidden: inputNorm };
    attention = await runQwenLinearAttentionTrainingModuleForward(attentionInputs, options);
    normalizedAttention = await runRMSNorm(
      attention.output,
      inputs.postAttentionNormWeight,
      options.rmsEps,
      {
        batchSize: dims.numTokens,
        hiddenSize: dims.hiddenSize,
        rmsNormWeightOffset: true,
      }
    );
    postAttention = await runResidualAdd(
      inputs.hidden,
      normalizedAttention,
      dims.numTokens * dims.hiddenSize
    );
    mlp = await runQwenDecoderMlpForward(
      { ...inputs.mlp, hidden: postAttention },
      dims
    );
    output = await runResidualAdd(
      postAttention,
      mlp.output,
      dims.numTokens * dims.hiddenSize
    );
    completed = true;
    return {
      output,
      finalState: attention.finalState,
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
      releaseTensor(attention?.finalState);
      if (attention?.cache) releaseQwenLinearAttentionTrainingModuleCache(attention.cache);
      releaseTensor(postAttention);
      if (mlp?.cache) releaseQwenDecoderMlpCache(mlp.cache);
    }
  }
}

export async function runQwenLinearDecoderLayerBackward(inputs, gradOutput, cache, options = {}) {
  const dims = resolveDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`Qwen linear-decoder cache mismatch for ${key}.`);
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
      dims
    );
    postAttentionGradient = await runResidualAdd(
      gradOutput,
      mlpGradients.hidden,
      dims.numTokens * dims.hiddenSize
    );
    attentionOutputGradient = await runRmsNormBackward(
      cache.attentionOutput,
      inputs.postAttentionNormWeight,
      postAttentionGradient,
      {
        numTokens: dims.numTokens,
        hiddenSize: dims.hiddenSize,
        eps: options.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    attentionGradients = await runQwenLinearAttentionTrainingModuleBackward(
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
        numTokens: dims.numTokens,
        hiddenSize: dims.hiddenSize,
        eps: options.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    hidden = await runResidualAdd(
      postAttentionGradient,
      inputNormGradient,
      dims.numTokens * dims.hiddenSize
    );
    completed = true;
    return {
      hidden,
      initialState: attentionGradients.initialState,
      lora: mlpGradients.lora,
    };
  } finally {
    releaseTensor(mlpGradients?.hidden);
    releaseTensor(postAttentionGradient);
    releaseTensor(attentionOutputGradient);
    releaseTensor(attentionGradients?.hidden);
    releaseTensor(inputNormGradient);
    if (!completed) {
      releaseTensor(hidden);
      releaseTensor(attentionGradients?.initialState);
      releaseMlpAdapterGradients(mlpGradients);
    }
  }
}

export function releaseQwenLinearDecoderLayerCache(cache) {
  releaseTensor(cache?.inputNorm);
  releaseTensor(cache?.attentionOutput);
  releaseQwenLinearAttentionTrainingModuleCache(cache?.attentionCache);
  releaseTensor(cache?.postAttention);
  releaseQwenDecoderMlpCache(cache?.mlpCache);
}
