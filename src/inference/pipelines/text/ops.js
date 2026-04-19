

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
import { releaseBuffer, readBuffer, acquireBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { isGpuBufferInstance } from '../../../gpu/weight-buffer.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import {
  runLayerAttentionGPU,
  recordLayerAttentionGPU,
} from './attention/index.js';
import { runLinearAttentionLayer } from './linear-attention.js';
import { runGatedShortConvGPU } from '../../../gpu/kernels/gated-short-conv.js';


function isDecodeBuffer(decodeBuffers, buffer) {
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
  const options = {};
  if (traceOptions?.outputBuffer) {
    options.outputBuffer = traceOptions.outputBuffer;
  }
  if (traceOptions?.executionPolicies) {
    options.executionPolicies = traceOptions.executionPolicies;
  }
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

  // LFM2 gated short convolution (GPU-native):
  // in_proj → 3×hidden → GPU kernel: split(B,C,x) + B*x + causal conv1d + C*conv_out → out_proj
  let inProj = null;
  let convOut = null;
  let outProj = null;
  try {
    const convState = options.convState;
    const hasConvState = Boolean(convState?.convWeightGPU && convState?.convStateGPU);
    const projN = hasConvState ? hiddenSize * 3 : hiddenSize * 2;

    // Project input
    inProj = await doMatmul(
      inputTensor,
      convInProj,
      numTokens,
      projN,
      hiddenSize,
      {
        transposeB: 'auto',
        label: `${label}.in_proj`,
        layerIdx,
        kernelPath,
        role: 'conv_in_proj',
        executionPolicies: options.executionPolicies ?? null,
      },
      recorder
    );

    if (hasConvState) {
      // GPU gated short conv kernel: B*x → conv1d → C*conv_out (all on GPU)
      convOut = await runGatedShortConvGPU(inProj, convState, {
        numTokens,
        layerIdx,
        recorder,
      });
    } else {
      // SwiGLU gated activation fallback: silu(first_half) * second_half
      convOut = await doSiLURowSplit(inProj, {
        numTokens,
        dim: hiddenSize,
        activation: 'silu',
        swigluLimit: options.swigluLimit ?? null,
        label: `${label}.activation`,
        layerIdx,
      }, recorder);
    }

    releaseOrTrack(recorder, inProj.buffer);
    inProj = null;

    // Output projection
    outProj = await doMatmul(
      convOut,
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
        executionPolicies: options.executionPolicies ?? null,
      },
      recorder
    );

    releaseOrTrack(recorder, convOut.buffer);
    convOut = null;

    if (kernelTrace.enabled && !recorder) {
      await traceStep('conv', label, layerIdx, outProj.buffer, [numTokens, hiddenSize]);
    }
    return outProj;
  } catch (error) {
    if (outProj) releaseOrTrack(recorder, outProj.buffer);
    if (convOut) releaseOrTrack(recorder, convOut.buffer);
    if (inProj) releaseOrTrack(recorder, inProj.buffer);
    throw error;
  }
}

export async function initConvLayerState(convState, convKernel, convInProj, hiddenSize, label, layerIdx) {
  const { isWeightBuffer } = await import('../../../gpu/weight-buffer.js');
  const isWB = typeof isWeightBuffer === 'function' && isWeightBuffer(convKernel);
  const kernelBuf = isWB ? convKernel.buffer : (isGpuBufferInstance(convKernel) ? convKernel : convKernel.buffer ?? convKernel);
  const kernelDtype = isWB ? String(convKernel.dtype ?? '').toLowerCase() : null;

  // Determine kernel size from weight shape
  let kernelSize = 3;
  if (isWB && Array.isArray(convKernel.shape)) {
    kernelSize = Number(convKernel.shape[convKernel.shape.length - 1]) || 3;
  }

  // Dequantize conv kernel weights to F32
  const totalElements = hiddenSize * kernelSize;
  const { QK_K, Q4K_BLOCK_BYTES } = await import('../../../config/schema/index.js');
  const { dequantizeQ4KM } = await import('../../../converter/quantizer.js');
  const { getDevice } = await import('../../../gpu/device.js');
  const device = getDevice();

  const isQ4K = kernelDtype === 'q4k' || kernelDtype === 'q4_k_m' || kernelDtype === 'q4_k';
  let weightF32;

  if (isQ4K) {
    const numBlocks = Math.ceil(totalElements / QK_K);
    const q4kBytes = numBlocks * Q4K_BLOCK_BYTES;
    // GPU readBuffer returns zeros for some Q4K weight buffers, so prefer
    // CPU-side rawBytes from the WeightBuffer when available.
    const hasRawBytes = isWB && convKernel.rawBytes;
    if (hasRawBytes) {
      weightF32 = dequantizeQ4KM(new Uint8Array(convKernel.rawBytes), numBlocks, [totalElements]);
    } else {
      if (device) await device.queue.onSubmittedWorkDone();
      const raw = await readBuffer(kernelBuf, q4kBytes);
      weightF32 = dequantizeQ4KM(new Uint8Array(raw), numBlocks, [totalElements]);
    }
  } else if (kernelDtype === 'f16' || kernelDtype === 'bf16') {
    if (device) await device.queue.onSubmittedWorkDone();
    const raw = await readBuffer(kernelBuf, totalElements * 2);
    const { decodeReadback } = await import('./debug-utils/index.js');
    weightF32 = decodeReadback(raw, 'f16');
  } else {
    if (device) await device.queue.onSubmittedWorkDone();
    const raw = await readBuffer(kernelBuf, totalElements * 4);
    weightF32 = new Float32Array(raw);
  }

  // Validate dequantized weights are non-degenerate
  let maxAbs = 0;
  for (let i = 0; i < weightF32.length; i++) {
    const abs = Math.abs(weightF32[i]);
    if (abs > maxAbs) maxAbs = abs;
  }
  if (maxAbs === 0) {
    const { log } = await import('../../../debug/index.js');
    log.error('Pipeline', `${label} conv kernel weights are all zeros after dequantization (dtype=${kernelDtype}, elements=${totalElements}). Conv layers will produce degenerate output.`);
  }

  // Upload dequantized weights to GPU
  const weightGPU = acquireBuffer(weightF32.byteLength, undefined, `${label}.conv_weight_f32`);
  uploadData(weightGPU, weightF32);

  // Create zeroed conv state buffer
  const stateSize = hiddenSize * (kernelSize - 1) * Float32Array.BYTES_PER_ELEMENT;
  const stateGPU = acquireBuffer(stateSize, undefined, `${label}.conv_state`);
  uploadData(stateGPU, new Float32Array(hiddenSize * (kernelSize - 1)));

  convState.convWeightGPU = weightGPU;
  convState.convStateGPU = stateGPU;
  convState.hiddenSize = hiddenSize;
  convState.kernelSize = kernelSize;

  // Pre-dequantize in_proj weight to F32 via CPU dequantization of the raw Q4K buffer.
  // GPU readBuffer returns zeros for some Q4K weight buffers, so we dequantize from the
  // WeightBuffer's raw bytes instead.
  if (isWB && isWeightBuffer(convInProj)) {
    const inProjDtype = String(convInProj.dtype ?? '').toLowerCase();
    const isInProjQ4K = inProjDtype === 'q4k' || inProjDtype === 'q4_k_m' || inProjDtype === 'q4_k';
    if (isInProjQ4K && convInProj.rawBytes) {
      const inProjElements = hiddenSize * 3 * hiddenSize;
      const inProjBlocks = Math.ceil(inProjElements / QK_K);
      const inProjF32 = dequantizeQ4KM(new Uint8Array(convInProj.rawBytes), inProjBlocks, [inProjElements]);
      const inProjGPU = acquireBuffer(inProjF32.byteLength, undefined, `${label}.in_proj_f32`);
      uploadData(inProjGPU, inProjF32);
      convState.inProjF32GPU = inProjGPU;
    }
  }
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
  const normalizedLayerType = String(config?.layerType ?? '').trim().toLowerCase();
  const isLinearLayer = normalizedLayerType === 'linear_attention'
    || normalizedLayerType === 'linear'
    || normalizedLayerType === 'gated_delta'
    || normalizedLayerType === 'gated_delta_net';
  if (isLinearLayer) {
    return {
      output: await runLinearAttentionLayer(inputTensor, layerWeights, {
        layerIdx: config.layerIdx,
        numTokens: config.numTokens,
        hiddenSize: config.hiddenSize,
        config,
        currentSeqLen: config.currentSeqLen,
        activationDtype: config.activationDtype,
        kernelPath: config.kernelPath ?? null,
        linearRuntime: state?.linearRuntime ?? null,
        getWeightBuffer: getWeightBufferFn,
        getNormWeightBuffer: getNormWeightBufferFn,
        recorder: recorder ?? null,
      }),
      residualFused: false,
    };
  }

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
