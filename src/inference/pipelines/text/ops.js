

import {
  runRMSNorm, runResidualAdd, runMatmul, runSiLU, runGeLU,
  recordRMSNorm, recordResidualAdd, recordMatmul, recordSiLU, recordGeLU,
  runSiLURowSplit, recordSiLURowSplit,
  runMatmulRMSNormFused, recordMatmulRMSNormFused,
  runConv2D, recordConv2D,
} from '../../../gpu/kernel-selector.js';
import {
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
} from '../../../gpu/kernels/cast.js';
import { createTensor } from '../../../gpu/tensor.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import {
  runLayerAttentionGPU,
  recordLayerAttentionGPU,
} from './attention.js';


export function isDecodeBuffer(decodeBuffers, buffer) {
  return !!decodeBuffers?.ownsBuffer(buffer);
}


export function releaseOrTrack(recorder, buffer, decodeBuffers) {
  if (isDecodeBuffer(decodeBuffers, buffer)) {
    return;
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}


export async function doRMSNorm(input, weight, eps, options, recorder) {
  const result = recorder
    ? await recordRMSNorm(recorder, input, weight, eps, options)
    : await runRMSNorm(input, weight, eps, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'rmsnorm';
    await traceStep('rmsnorm', label, layer, result.buffer, [options.batchSize, options.hiddenSize]);
  }

  return result;
}


export async function doResidualAdd(a, b, size, recorder, traceOptions) {
  const options = traceOptions?.outputBuffer ? { outputBuffer: traceOptions.outputBuffer } : {};
  const result = recorder
    ? await recordResidualAdd(recorder, a, b, size, options)
    : await runResidualAdd(a, b, size, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && traceOptions) {
    await traceStep('residual_add', traceOptions.label ?? 'residual', traceOptions.layerIdx ?? -1, result.buffer, [size]);
  }

  return result;
}


export async function doMatmul(A, B, M, N, K, options = {}, recorder) {
  const result = recorder
    ? await recordMatmul(recorder, A, B, M, N, K, options)
    : await runMatmul(A, B, M, N, K, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'matmul';
    await traceStep('matmul', label, layer, result.buffer, [M, N]);
  }

  return result;
}


export async function doSiLU(input, options = {}, recorder) {
  const result = recorder
    ? await recordSiLU(recorder, input, options)
    : await runSiLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('silu', options.label ?? 'silu', options.layerIdx ?? -1, result.buffer, [options.size]);
  }

  return result;
}


export async function doGeLU(input, options = {}, recorder) {
  const result = recorder
    ? await recordGeLU(recorder, input, options)
    : await runGeLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('gelu', options.label ?? 'gelu', options.layerIdx ?? -1, result.buffer, [options.size]);
  }

  return result;
}


export async function doSiLURowSplit(input, options, recorder) {
  const result = recorder
    ? await recordSiLURowSplit(recorder, input, options)
    : await runSiLURowSplit(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('silu_row_split', options.label ?? 'ffn_activation', options.layerIdx ?? -1, result.buffer, [options.numTokens, options.dim]);
  }

  return result;
}


export async function doMatmulRMSNormFused(input, weight, normWeight, options, recorder) {
  // The fused kernel takes Tensor input but residual is still GPUBuffer
  const fusedOptions = {
    N: options.N,
    K: options.K,
    eps: options.eps,
    residual: options.residual?.buffer ?? null,
    outputBuffer: options.outputBuffer,
    transposeB: options.transposeB,
    rmsNormWeightOffset: options.rmsNormWeightOffset,
    label: options.label ?? null,
  };
  const resultTensor = recorder
    ? await recordMatmulRMSNormFused(recorder, input, weight, normWeight, fusedOptions)
    : await runMatmulRMSNormFused(input, weight, normWeight, fusedOptions);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('fused_matmul_rmsnorm', options.label ?? 'fused_matmul_rmsnorm', options.layerIdx ?? -1, resultTensor.buffer, [1, options.N]);
  }

  return resultTensor;
}

export async function doConv(
  inputTensor,
  convInProj,
  convKernel,
  convOutProj,
  options = {},
  recorder
) {
  const numTokens = Number(options.numTokens);
  const hiddenSize = Number(options.hiddenSize);
  const layerIdx = Number.isFinite(options.layerIdx) ? options.layerIdx : -1;
  const label = options.label ?? 'conv';
  const kernelPath = options.kernelPath ?? null;

  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('doConv requires numTokens > 0.');
  }
  if (!Number.isFinite(hiddenSize) || hiddenSize <= 0) {
    throw new Error('doConv requires hiddenSize > 0.');
  }

  // Use the first 2x hidden projection channels as a gated conv-state projection.
  const inProj = await doMatmul(
    inputTensor,
    convInProj,
    numTokens,
    hiddenSize * 2,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `${label}.in_proj`,
      layerIdx,
      kernelPath,
      role: 'conv_in_proj',
    },
    recorder
  );
  const activated = await doSiLURowSplit(inProj, {
    numTokens,
    dim: hiddenSize,
    activation: 'silu',
    swigluLimit: options.swigluLimit ?? null,
    label: `${label}.activation`,
    layerIdx,
  }, recorder);

  if (recorder) {
    recorder.trackTemporaryBuffer(inProj.buffer);
  } else {
    releaseBuffer(inProj.buffer);
  }

  // Optional generic conv2d stage when explicit shape metadata is provided.
  // LFM2 depthwise conv kernels use model-specific packing, so this path is best-effort only.
  let convInput = activated;
  if (convKernel && options.conv2d && options.conv2d.enabled === true) {
    const convTensorInput = createTensor(activated.buffer, activated.dtype, [
      options.conv2d.inChannels,
      options.conv2d.height,
      options.conv2d.width,
    ], `${label}.conv_input`);
    const convOptions = {
      inChannels: options.conv2d.inChannels,
      outChannels: options.conv2d.outChannels,
      height: options.conv2d.height,
      width: options.conv2d.width,
      kernelH: options.conv2d.kernelH,
      kernelW: options.conv2d.kernelW,
      stride: options.conv2d.stride ?? 1,
      pad: options.conv2d.pad ?? 0,
    };
    const convResult = recorder
      ? await recordConv2D(recorder, convTensorInput, convKernel, null, convOptions)
      : await runConv2D(convTensorInput, convKernel, null, convOptions);
    convInput = createTensor(
      convResult.buffer,
      convResult.dtype,
      [numTokens, hiddenSize],
      `${label}.conv_output`
    );
    if (recorder) {
      recorder.trackTemporaryBuffer(activated.buffer);
    } else {
      releaseBuffer(activated.buffer);
    }
  }

  const outProj = await doMatmul(
    convInput,
    convOutProj,
    numTokens,
    hiddenSize,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `${label}.out_proj`,
      layerIdx,
      kernelPath,
      role: 'conv_out_proj',
    },
    recorder
  );

  if (convInput.buffer !== activated.buffer) {
    if (recorder) {
      recorder.trackTemporaryBuffer(convInput.buffer);
    } else {
      releaseBuffer(convInput.buffer);
    }
  } else if (recorder) {
    recorder.trackTemporaryBuffer(activated.buffer);
  } else {
    releaseBuffer(activated.buffer);
  }

  if (kernelTrace.enabled && !recorder) {
    await traceStep('conv', label, layerIdx, outProj.buffer, [numTokens, hiddenSize]);
  }
  return outProj;
}

export async function doCast(input, toDtype, recorder) {
  if (toDtype !== 'f16' && toDtype !== 'f32') {
    throw new Error(`Unsupported cast target dtype "${toDtype}"`);
  }
  if (input.dtype === toDtype) {
    return input;
  }
  if (input.dtype === 'f16' && toDtype === 'f32') {
    return recorder
      ? recordCastF16ToF32(recorder, input)
      : castF16ToF32(input);
  }
  if (input.dtype === 'f32' && toDtype === 'f16') {
    return recorder
      ? recordCastF32ToF16(recorder, input)
      : castF32ToF16(input);
  }
  throw new Error(`Unsupported cast path ${input.dtype} -> ${toDtype}`);
}


export async function doAttention(
  inputTensor,
  layerWeights,
  config,
  state,
  debug,
  debugFlags,
  getWeightBufferFn,
  getNormWeightBufferFn,
  debugCheckBuffer,
  recorder,
  lora
) {
  const isBDPA = state?.kvCache?.layout === 'bdpa_paged';
  if (recorder && isBDPA) {
    throw new Error('BDPA attention does not support command recorder mode. Disable command batching for BDPA.');
  }

  if (recorder) {
    return recordLayerAttentionGPU(
      recorder,
      inputTensor,
      layerWeights,
      config,
      state,
      debug,
      debugFlags,
      getWeightBufferFn,
      getNormWeightBufferFn,
      debugCheckBuffer,
      lora
    );
  }
  return runLayerAttentionGPU(
    inputTensor,
    layerWeights,
    config,
    state,
    debug,
    debugFlags,
    getWeightBufferFn,
    getNormWeightBufferFn,
    debugCheckBuffer,
    lora
  );
}
