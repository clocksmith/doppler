/**
 * Logits computation - final layer norm and LM head projection.
 *
 * This module handles the final steps of inference:
 * - Apply final RMS norm to hidden states
 * - Project to vocabulary size via LM head
 * - Handle tied embeddings (transposeB for HuggingFace format)
 * - CPU fallback for non-GPU execution
 *
 * @module inference/pipeline/logits
 */

import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../gpu/kernel-selector.js';
import { recordMatmul } from '../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../gpu/kernels/rmsnorm.js';
import { createTensor, type Tensor } from '../../gpu/tensor.js';
import { castF32ToF16, castF16ToF32, recordCastF16ToF32 } from '../../gpu/kernels/cast.js';
import { type WeightBuffer, type CpuWeightBuffer, createWeightBuffer, isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { type LargeWeightConfigSchema, DEFAULT_LARGE_WEIGHT_CONFIG, type ProbeConfigSchema } from '../../config/schema/index.js';
import { runProbes } from './probes.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for logits computation.
 */
export interface LogitsConfig {
  hiddenSize: number;
  vocabSize: number;
  rmsNormEps: number;
  useTiedEmbeddings: boolean;
  embeddingVocabSize: number | null;
  finalLogitSoftcapping: number | null;  // Gemma 2: 30.0 - applies tanh(x/cap)*cap
  largeWeights?: LargeWeightConfigSchema;
  /** Dtype for hidden state activations */
  activationDtype?: 'f16' | 'f32';
}

/**
 * Weights required for logits computation.
 */
export interface LogitsWeights {
  finalNorm: GPUBuffer | Float32Array;
  lmHead: GPUBuffer | Float32Array | WeightBuffer | CpuWeightBuffer;
}

/**
 * Debug flags for logits computation.
 */
export interface LogitsDebugFlags {
  finalNormDebugDone?: boolean;
  afterFinalNormDebugDone?: boolean;
}

// ============================================================================
// CPU Fallback Functions
// ============================================================================

/**
 * CPU RMSNorm implementation.
 *
 * Computes: output[i] = (x[i] / rms) * weight[i]
 * where rms = sqrt(mean(x^2) + eps)
 *
 * @param x - Input tensor
 * @param weight - Norm weights
 * @param eps - Epsilon for numerical stability
 * @returns Normalized tensor
 */
export function rmsNormCPU(
  x: Float32Array,
  weight: Float32Array,
  eps: number = 1e-5
): Float32Array {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += x[i] * x[i];
  }
  const rms = Math.sqrt(sumSq / n + eps);

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = (x[i] / rms) * weight[i % weight.length];
  }
  return result;
}

function f16ToF32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

function f16BufferToF32(data: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(data);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    out[i] = f16ToF32(u16[i]);
  }
  return out;
}

/**
 * CPU matmul implementation (fallback for non-GPU).
 *
 * Computes: output = input @ weight^T
 * Input: [M, K], Weight: [N, K] (row) or [K, N] (column), Output: [M, N]
 *
 * @param input - Input tensor [M, K]
 * @param weight - Weight tensor [N, K] (row) or [K, N] (column)
 * @param M - Batch size (num tokens)
 * @param N - Output size (vocab size)
 * @param K - Hidden size
 * @returns Output tensor [M, N]
 */
export function matmulCPU(
  input: Float32Array,
  weight: Float32Array,
  M: number,
  N: number,
  K: number,
  layout: 'row' | 'column' = 'row',
  weightStride?: number | null
): Float32Array {
  const result = new Float32Array(M * N);
  const stride = weightStride ?? (layout === 'row' ? K : N);

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        // Row layout: weight is [N, K] (vocab x hidden).
        // Column layout: weight is [K, N] (hidden x vocab).
        const weightIndex = layout === 'row'
          ? n * stride + k
          : k * stride + n;
        sum += input[m * K + k] * weight[weightIndex];
      }
      result[m * N + n] = sum;
    }
  }
  return result;
}

/**
 * Apply softcapping to logits (Gemma 2 style).
 *
 * Computes: logits = tanh(logits / cap) * cap
 *
 * This bounds logits to [-cap, cap] with smooth transitions,
 * preventing extreme values from dominating softmax.
 *
 * @param logits - Logits tensor to modify in-place
 * @param cap - Softcap value (Gemma 2 default: 30.0)
 */
export function applySoftcapping(logits: Float32Array, cap: number): void {
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.tanh(logits[i] / cap) * cap;
  }
}

function resolveCpuWeightDims(lmHead: CpuWeightBuffer): { vocabSize: number; hiddenSize: number } {
  if (lmHead.shape.length !== 2) {
    throw new Error(`[Logits] CPU LM head shape must be 2D, got [${lmHead.shape.join(', ')}]`);
  }
  if (lmHead.layout === 'column') {
    return { hiddenSize: lmHead.shape[0], vocabSize: lmHead.shape[1] };
  }
  return { vocabSize: lmHead.shape[0], hiddenSize: lmHead.shape[1] };
}

function resolveLmHeadChunkRows(
  device: GPUDevice,
  numTokens: number,
  hiddenSize: number,
  config?: LargeWeightConfigSchema
): number {
  const resolved = config ?? DEFAULT_LARGE_WEIGHT_CONFIG;
  const safety = Math.min(Math.max(resolved.safetyRatio ?? 0.9, 0.1), 1);
  const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxBytes = Math.floor(maxBinding * safety);

  const maxRowsByWeight = Math.floor(maxBytes / (hiddenSize * 4));
  const maxRowsByOutput = Math.floor(maxBytes / (numTokens * 4));
  const maxRows = Math.min(maxRowsByWeight, maxRowsByOutput);

  if (!Number.isFinite(maxRows) || maxRows <= 0) {
    throw new Error(
      `[Logits] LM head chunk size underflow (maxBytes=${maxBytes}, hiddenSize=${hiddenSize}, numTokens=${numTokens}).`
    );
  }

  const override = resolved.lmHeadChunkRows ?? null;
  if (override && override > 0) {
    return Math.min(override, maxRows);
  }
  return maxRows;
}

function extractLmHeadChunk(
  data: Float32Array,
  layout: 'row' | 'column',
  hiddenSize: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number
): Float32Array {
  if (layout === 'row') {
    const start = rowOffset * hiddenSize;
    return data.subarray(start, start + rowCount * hiddenSize);
  }

  const chunk = new Float32Array(hiddenSize * rowCount);
  for (let k = 0; k < hiddenSize; k++) {
    const srcOffset = k * vocabSize + rowOffset;
    const dstOffset = k * rowCount;
    chunk.set(data.subarray(srcOffset, srcOffset + rowCount), dstOffset);
  }
  return chunk;
}

function writeChunkLogits(
  target: Float32Array,
  chunk: Float32Array,
  numTokens: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number
): void {
  for (let t = 0; t < numTokens; t++) {
    const srcOffset = t * rowCount;
    const dstOffset = t * vocabSize + rowOffset;
    target.set(chunk.subarray(srcOffset, srcOffset + rowCount), dstOffset);
  }
}

async function computeChunkedLogitsGPU(
  normedTensor: Tensor,
  lmHead: CpuWeightBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  weightVocabSize: number,
  debugProbes?: ProbeConfigSchema[] | null,
  largeWeightConfig?: LargeWeightConfigSchema
): Promise<Float32Array> {
  const device = getDevice();
  if (!device) {
    throw new Error('[Logits] GPU device not available for chunked LM head.');
  }

  const chunkRows = resolveLmHeadChunkRows(device, numTokens, hiddenSize, largeWeightConfig);
  const caps = getKernelCapabilities();
  const preferF16 = (largeWeightConfig?.preferF16 ?? true) && lmHead.dtype === 'f16' && caps.hasF16;
  const logits = new Float32Array(numTokens * vocabSize);

  if (isTraceEnabled('logits')) {
    trace.logits(`LM_HEAD_CHUNKED: vocab=${vocabSize}, chunkRows=${chunkRows}, layout=${lmHead.layout}, f16=${preferF16}`);
  }

  for (let rowOffset = 0; rowOffset < vocabSize; rowOffset += chunkRows) {
    const rowCount = Math.min(chunkRows, vocabSize - rowOffset);
    const chunkData = extractLmHeadChunk(
      lmHead.data,
      lmHead.layout,
      hiddenSize,
      weightVocabSize,
      rowOffset,
      rowCount
    );

    const f32Buffer = acquireBuffer(chunkData.byteLength, undefined, 'lm_head_chunk_f32');
    device.queue.writeBuffer(
      f32Buffer,
      0,
      chunkData.buffer as ArrayBuffer,
      chunkData.byteOffset,
      chunkData.byteLength
    );

    const chunkShape = lmHead.layout === 'column'
      ? [hiddenSize, rowCount]
      : [rowCount, hiddenSize];

    let weightBuffer = createWeightBuffer(f32Buffer, 'f32', lmHead.layout, chunkShape, 'lm_head_chunk_f32');

    if (preferF16) {
      const f32Tensor = createTensor(f32Buffer, 'f32', chunkShape, 'lm_head_chunk_f32');
      const f16Tensor = await castF32ToF16(f32Tensor);
      releaseBuffer(f32Buffer);
      weightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', lmHead.layout, chunkShape, 'lm_head_chunk_f16');
    }

    const logitsTensor = await runMatmul(normedTensor, weightBuffer, numTokens, rowCount, hiddenSize, {
      transposeB: 'auto',
      role: 'lm_head',
    });

    if (debugProbes?.length) {
      await runProbes('logits', logitsTensor.buffer, {
        numTokens,
        hiddenSize: rowCount,
        probes: debugProbes,
      });
    }

    const chunkLogitsData = await readBuffer(logitsTensor.buffer, numTokens * rowCount * 4);
    const chunkLogits = new Float32Array(chunkLogitsData);
    writeChunkLogits(logits, chunkLogits, numTokens, vocabSize, rowOffset, rowCount);

    releaseBuffer(logitsTensor.buffer);
    releaseBuffer(weightBuffer.buffer);
  }

  return logits;
}

async function finalizeLogits(
  rawLogits: Float32Array,
  numTokens: number,
  matmulVocabSize: number,
  vocabSize: number,
  config: LogitsConfig,
  debugProbes?: ProbeConfigSchema[] | null
): Promise<Float32Array> {
  let logits = rawLogits;

  if (matmulVocabSize < vocabSize) {
    const paddedLogits = new Float32Array(numTokens * vocabSize);
    for (let t = 0; t < numTokens; t++) {
      const srcOffset = t * matmulVocabSize;
      const dstOffset = t * vocabSize;
      for (let i = 0; i < matmulVocabSize; i++) {
        paddedLogits[dstOffset + i] = rawLogits[srcOffset + i];
      }
      for (let i = matmulVocabSize; i < vocabSize; i++) {
        paddedLogits[dstOffset + i] = -Infinity;
      }
    }
    logits = paddedLogits;
  }

  if (config.finalLogitSoftcapping) {
    applySoftcapping(logits, config.finalLogitSoftcapping);
  }

  await runProbes('logits_final', logits, {
    numTokens,
    hiddenSize: vocabSize,
    probes: debugProbes,
  });

  return logits;
}

// ============================================================================
// Main Logits Computation
// ============================================================================

/**
 * Compute logits from hidden states.
 *
 * This function:
 * 1. Applies final RMS normalization
 * 2. Projects to vocabulary via LM head matrix multiplication
 * 3. Handles tied embeddings (uses transposeB for HF format)
 * 4. Falls back to CPU if GPU unavailable
 *
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens (required for GPU buffer input)
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @param useGPU - Whether to use GPU
 * @param debugFlags - Debug flags to prevent repeated logging
 * @param getNormWeightBuffer - Helper to get norm weight buffer (from pipeline)
 * @param debugCheckBuffer - Helper for debug buffer checking (from pipeline)
 * @returns Logits tensor [numTokens, vocabSize]
 */
export async function computeLogits(
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  useGPU: boolean,
  debugFlags: LogitsDebugFlags = {},
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  debugProbes?: ProbeConfigSchema[] | null
): Promise<Float32Array> {
  if (isTraceEnabled('logits')) {
    trace.logits(`LOGITS_ENTRY: numTokens=${numTokens}, useGPU=${useGPU}`);
  }
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    largeWeights,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded, returning zeros');
    return new Float32Array(vocabSize);
  }

  const requestedVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;
  let matmulVocabSize = requestedVocabSize;
  let cpuWeightVocabSize: number | null = null;
  let cpuWeightLayout: 'row' | 'column' | null = null;

  if (isCpuWeightBuffer(lmHead)) {
    const dims = resolveCpuWeightDims(lmHead);
    cpuWeightVocabSize = dims.vocabSize;
    cpuWeightLayout = lmHead.layout;
    if (dims.hiddenSize !== hiddenSize) {
      log.warn('Logits', `LM head hiddenSize mismatch: weight=${dims.hiddenSize}, expected=${hiddenSize}`);
    }
    if (matmulVocabSize > dims.vocabSize) {
      log.warn('Logits', `LM head vocabSize smaller than requested: weight=${dims.vocabSize}, requested=${matmulVocabSize}. Clamping.`);
      matmulVocabSize = dims.vocabSize;
    }
  }

  // Check if input is GPU buffer
  const inputIsGPU = hiddenStates instanceof GPUBuffer;

  // CPU fallback path
  if (isTraceEnabled('logits')) {
    trace.logits(`LOGITS_PATH: device=${!!device}, useGPU=${useGPU}, taking ${(!device || !useGPU) ? 'CPU' : 'GPU'} path`);
  }
  if (!device || !useGPU) {
    let cpuHiddenStates: Float32Array;
    if (inputIsGPU) {
      const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
      const data = await readBuffer(hiddenStates, numTokens * hiddenSize * bytesPerElement);
      cpuHiddenStates = activationDtype === 'f16'
        ? f16BufferToF32(data)
        : new Float32Array(data);
    } else {
      cpuHiddenStates = hiddenStates as Float32Array;
    }
    const normed = rmsNormCPU(cpuHiddenStates, finalNorm as Float32Array, rmsNormEps);
    const rawLogits = isCpuWeightBuffer(lmHead)
      ? matmulCPU(
          normed,
          lmHead.data,
          numTokens,
          matmulVocabSize,
          hiddenSize,
          cpuWeightLayout ?? 'row',
          cpuWeightLayout === 'column' ? cpuWeightVocabSize : null
        )
      : matmulCPU(normed, lmHead as Float32Array, numTokens, matmulVocabSize, hiddenSize);
    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // GPU path
  // 1. Get or create input buffer
  let inputBuffer: GPUBuffer;
  let inputBufferOwned = false;
  if (inputIsGPU) {
    inputBuffer = hiddenStates as GPUBuffer;
  } else {
    inputBuffer = acquireBuffer((hiddenStates as Float32Array).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0, hiddenStates as unknown as BufferSource);
    inputBufferOwned = true;
  }
  await runProbes('pre_final_norm', inputBuffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
  });

  const inputDtype: 'f16' | 'f32' = inputIsGPU ? activationDtype : 'f32';

  // 2. Apply final RMSNorm
  let normWeightBuffer: GPUBuffer;
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer((finalNorm as Float32Array).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0, finalNorm as unknown as BufferSource);
  }

  // Debug: Check hidden state before final norm
  if (!debugFlags.finalNormDebugDone && debugCheckBuffer) {
    debugFlags.finalNormDebugDone = true;
    await debugCheckBuffer(inputBuffer, 'Before final norm', numTokens);
    await debugCheckBuffer(normWeightBuffer, 'Final norm weights', 1, 100);
  }

  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normInputTensor = inputDtype === 'f16' ? await castF16ToF32(inputTensor) : inputTensor;
  const normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });
  if (normInputTensor !== inputTensor) {
    releaseBuffer(normInputTensor.buffer);
  }
  await runProbes('final_norm', normedTensor.buffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
  });

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, normedTensor.buffer, [numTokens, hiddenSize]);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(normedTensor.buffer, 'After final norm', numTokens);
  }

  if (isCpuWeightBuffer(lmHead)) {
    const rawLogits = await computeChunkedLogitsGPU(
      normedTensor,
      lmHead,
      numTokens,
      hiddenSize,
      matmulVocabSize,
      cpuWeightVocabSize ?? matmulVocabSize,
      debugProbes,
      largeWeights
    );

    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(normedTensor.buffer);
    if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);

    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // 3. Project to vocab via LM head
  let lmHeadBuffer: GPUBuffer | WeightBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0, lmHead as unknown as BufferSource);
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  // Debug: Log buffer info for lm_head matmul
  const lmHeadGPU = isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer;
  const lmHeadDtype = getWeightDtype(lmHeadBuffer);  // dtype from WeightBuffer metadata
  const normedDtype = normedTensor.dtype;
  if (isTraceEnabled('logits')) {
    trace.logits(`LM_HEAD_MATMUL: M=${numTokens}, N=${matmulVocabSize}, K=${hiddenSize}, lmHeadDtype=${lmHeadDtype}, normedDtype=${normedDtype}, size=${lmHeadGPU.size}, bufLabel=${lmHeadGPU.label}`);
  }

  // HuggingFace models store lm_head as [vocabSize, hiddenSize], so transposeB=true
  const logitsTensor = await runMatmul(normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });
  await runProbes('logits', logitsTensor.buffer, {
    numTokens,
    hiddenSize: matmulVocabSize,
    probes: debugProbes,
  });

  // Trace lm_head output
  if (kernelTrace.enabled) {
    await traceStep('matmul', 'lm_head', -1, logitsTensor.buffer, [numTokens, matmulVocabSize]);
  }

  // 4. Read back logits
  const logitsData = await readBuffer(logitsTensor.buffer, numTokens * matmulVocabSize * 4);

  // Cleanup
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedTensor.buffer);
  releaseBuffer(logitsTensor.buffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);

  const rawLogits = new Float32Array(logitsData);
  return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
}

/**
 * Compute logits and return GPU buffer directly (deferred readback).
 *
 * This variant avoids the ~1MB readback per token, enabling GPU-side sampling.
 * Use with runGPUSample or runArgmax to sample directly on GPU.
 *
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @param debugFlags - Debug flags
 * @param getNormWeightBuffer - Helper to get norm weight buffer
 * @returns GPU buffer containing logits [numTokens, vocabSize]
 */
export async function computeLogitsGPU(
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  debugFlags: LogitsDebugFlags = {},
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
): Promise<{ logitsBuffer: GPUBuffer; vocabSize: number } | null> {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!device) {
    return null;
  }

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded');
    return null;
  }
  if (isCpuWeightBuffer(lmHead)) {
    return null;
  }

  // Get or create input buffer
  let inputBuffer: GPUBuffer;
  let inputBufferOwned = false;
  if (hiddenStates instanceof GPUBuffer) {
    inputBuffer = hiddenStates;
  } else {
    inputBuffer = acquireBuffer((hiddenStates as Float32Array).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0, hiddenStates as unknown as BufferSource);
    inputBufferOwned = true;
  }

  // Apply final RMSNorm
  let normWeightBuffer: GPUBuffer;
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer((finalNorm as Float32Array).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0, finalNorm as unknown as BufferSource);
  }

  const inputDtype: 'f16' | 'f32' = hiddenStates instanceof GPUBuffer ? activationDtype : 'f32';
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normInputTensor = inputDtype === 'f16' ? await castF16ToF32(inputTensor) : inputTensor;
  const normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });
  if (normInputTensor !== inputTensor) {
    releaseBuffer(normInputTensor.buffer);
  }

  // Project to vocab via LM head
  let lmHeadBuffer: GPUBuffer | WeightBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0, lmHead as unknown as BufferSource);
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;

  const logitsTensor = await runMatmul(normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });

  // Cleanup intermediate buffers (but keep logitsBuffer)
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedTensor.buffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize };
}

/**
 * Record logits computation (batched, no submit).
 *
 * This variant uses the CommandRecorder to batch logits computation with
 * preceding layer operations, avoiding a GPU sync point.
 *
 * @param recorder - CommandRecorder for batched operations
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @returns GPU buffer containing logits [numTokens, vocabSize] and vocab size
 */
export async function recordLogitsGPU(
  recorder: CommandRecorder,
  hiddenStates: GPUBuffer,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
): Promise<{ logitsBuffer: GPUBuffer; vocabSize: number }> {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize ? embeddingVocabSize : vocabSize;

  if (!finalNorm || !lmHead) {
    throw new Error('[recordLogitsGPU] Final norm or LM head not loaded');
  }
  if (isCpuWeightBuffer(lmHead)) {
    throw new Error('[recordLogitsGPU] CPU-resident LM head not supported in recorded path');
  }

  // Get norm weight buffer
  let normWeightBuffer: GPUBuffer;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer((finalNorm as Float32Array).byteLength, undefined, 'final_norm_w');
    recorder.device.queue.writeBuffer(normWeightBuffer, 0, finalNorm as unknown as BufferSource);
  }

  const inputDtype: 'f16' | 'f32' = activationDtype;
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(hiddenStates, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normInputTensor = inputDtype === 'f16' ? await recordCastF16ToF32(recorder, inputTensor) : inputTensor;

  // Record RMSNorm (no submit)
  const normedTensor = await recordRMSNorm(recorder, normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });
  if (normInputTensor !== inputTensor) {
    recorder.trackTemporaryBuffer(normInputTensor.buffer);
  }

  // Get LM head buffer
  let lmHeadBuffer: GPUBuffer | WeightBuffer;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(rawBuffer, 0, lmHead as unknown as BufferSource);
    lmHeadBuffer = rawBuffer;
  }

  // Record matmul (no submit)
  const logitsTensor = await recordMatmul(recorder, normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });

  // Track intermediate buffer for cleanup after submit
  recorder.trackTemporaryBuffer(normedTensor.buffer);

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize };
}

/**
 * Extract logits for only the last position.
 *
 * Used after prefill to get logits for sampling the first generated token.
 *
 * @param logits - Full logits tensor [numTokens, vocabSize]
 * @param numTokens - Number of tokens
 * @param vocabSize - Vocabulary size
 * @returns Logits for last position [vocabSize]
 */
export function extractLastPositionLogits(
  logits: Float32Array,
  numTokens: number,
  vocabSize: number
): Float32Array {
  const lastPosLogits = new Float32Array(vocabSize);
  const lastPosOffset = (numTokens - 1) * vocabSize;

  for (let i = 0; i < vocabSize; i++) {
    lastPosLogits[i] = logits[lastPosOffset + i];
  }

  return lastPosLogits;
}
