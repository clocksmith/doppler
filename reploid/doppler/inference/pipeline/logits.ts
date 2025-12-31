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

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../gpu/kernel-selector.js';
import { recordMatmul } from '../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../gpu/kernels/rmsnorm.js';
import { getBufferDtype } from '../../gpu/buffer-dtypes.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import { log, trace } from '../../debug/index.js';

// ============================================================================
// Debug Configuration
// ============================================================================

/**
 * Enable detailed debug logging with GPU readbacks.
 * Set window.DOPPLER_DEBUG_LOGITS = true to enable.
 * WARNING: Adds 3-4 extra GPU submits per token when enabled!
 */
const DEBUG_LOGITS_FLAGS = typeof window !== 'undefined'
  ? (window as unknown as { DOPPLER_DEBUG_LOGITS?: boolean })
  : null;
const LOGITS_DEBUG = Boolean(DEBUG_LOGITS_FLAGS?.DOPPLER_DEBUG_LOGITS);
const ENABLE_DEBUG_READBACKS = LOGITS_DEBUG;

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
}

/**
 * Weights required for logits computation.
 */
export interface LogitsWeights {
  finalNorm: GPUBuffer | Float32Array;
  lmHead: GPUBuffer | Float32Array;
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

/**
 * CPU matmul implementation (fallback for non-GPU).
 *
 * Computes: output = input @ weight^T
 * Input: [M, K], Weight: [N, K] (transposed), Output: [M, N]
 *
 * @param input - Input tensor [M, K]
 * @param weight - Weight tensor [N, K]
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
  K: number
): Float32Array {
  const result = new Float32Array(M * N);

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        // Weight is [N, K] (vocab x hidden) - row-major
        sum += input[m * K + k] * weight[n * K + k];
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
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>
): Promise<Float32Array> {
  if (LOGITS_DEBUG) {
    trace.logits(`LOGITS_ENTRY: numTokens=${numTokens}, useGPU=${useGPU}`);
  }
  const { hiddenSize, vocabSize, rmsNormEps, useTiedEmbeddings, embeddingVocabSize } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded, returning zeros');
    return new Float32Array(vocabSize);
  }

  // Check if input is GPU buffer
  const inputIsGPU = hiddenStates instanceof GPUBuffer;

  // CPU fallback path
  if (LOGITS_DEBUG) {
    trace.logits(`LOGITS_PATH: device=${!!device}, useGPU=${useGPU}, taking ${(!device || !useGPU) ? 'CPU' : 'GPU'} path`);
  }
  if (!device || !useGPU) {
    let cpuHiddenStates: Float32Array;
    if (inputIsGPU) {
      const data = await readBuffer(hiddenStates, numTokens * hiddenSize * 4);
      cpuHiddenStates = new Float32Array(data);
    } else {
      cpuHiddenStates = hiddenStates as Float32Array;
    }
    const normed = rmsNormCPU(cpuHiddenStates, finalNorm as Float32Array, rmsNormEps);
    const logits = matmulCPU(normed, lmHead as Float32Array, numTokens, vocabSize, hiddenSize);
    // Apply Gemma 2 softcapping if configured
    if (config.finalLogitSoftcapping) {
      applySoftcapping(logits, config.finalLogitSoftcapping);
    }
    return logits;
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

  // Debug: Always log hidden state BEFORE final norm for last token
  if (ENABLE_DEBUG_READBACKS && allowReadback('logits.debug.pre-norm')) {
    const lastTokenOffset = (numTokens - 1) * hiddenSize * 4;
    // Read full last-token vector for accurate maxAbs comparisons vs HF.
    const sampleSize = hiddenSize * 4;
    const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(inputBuffer, lastTokenOffset, staging, 0, sampleSize);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
    trace.logits(`BEFORE_FINAL_NORM[pos=${numTokens - 1}]: maxAbs=${maxAbs.toFixed(4)}, first5=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);

    // Also read FULL final norm weights
    const wsSize = normWeightBuffer.size;
    const wstaging = device.createBuffer({ size: wsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const wenc = device.createCommandEncoder();
    wenc.copyBufferToBuffer(normWeightBuffer, 0, wstaging, 0, wsSize);
    device.queue.submit([wenc.finish()]);
    await wstaging.mapAsync(GPUMapMode.READ);
    const wdata = new Float32Array(wstaging.getMappedRange().slice(0));
    wstaging.unmap();
    wstaging.destroy();
    const wmaxAbs = Math.max(...Array.from(wdata).map(x => Math.abs(x)));
    trace.logits(`FULL_FINAL_NORM_WEIGHTS: maxAbs=${wmaxAbs.toFixed(4)}, size=${wdata.length}, first5=[${Array.from(wdata).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
  }

  const normedBuffer = await runRMSNorm(inputBuffer, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, normedBuffer, [numTokens, hiddenSize]);
  }

  // Debug: check post-norm values at LAST token position (used for logits)
  if (ENABLE_DEBUG_READBACKS && allowReadback('logits.debug.post-norm')) {
    // Hidden state for LAST token is at offset (numTokens-1) * hiddenSize
    const lastTokenOffset = (numTokens - 1) * hiddenSize * 4;  // F32
    // Read full last-token vector for accurate maxAbs comparisons vs HF.
    const sampleSize = hiddenSize * 4;
    const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(normedBuffer, lastTokenOffset, staging, 0, sampleSize);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
    const nonZero = Array.from(data).filter(x => x !== 0).length;
    trace.logits(`LAST_TOKEN_HIDDEN[pos=${numTokens - 1}]: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(normedBuffer, 'After final norm', numTokens);
  }

  // 3. Project to vocab via LM head
  let lmHeadBuffer: GPUBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else {
    lmHeadBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(lmHeadBuffer, 0, lmHead as unknown as BufferSource);
    lmHeadBufferOwned = true;
  }

  // For tied embeddings, use actual embedding matrix vocab size
  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;

  // Debug: Log buffer info for lm_head matmul
  const lmHeadDtype = getBufferDtype(lmHeadBuffer);
  const normedDtype = getBufferDtype(normedBuffer);
  if (LOGITS_DEBUG) {
    trace.logits(`LM_HEAD_MATMUL: M=${numTokens}, N=${matmulVocabSize}, K=${hiddenSize}, lmHeadDtype=${lmHeadDtype}, normedDtype=${normedDtype}, size=${lmHeadBuffer.size}, bufLabel=${lmHeadBuffer.label}`);
  }

  // Debug: Sample lm_head weights at start of buffer to verify values look sane
  // GGUF layout: [hiddenSize, vocabSize] - row k contains all vocab weights for hidden dim k
  if (ENABLE_DEBUG_READBACKS && allowReadback('logits.debug.lm-head')) {
    const bytesPerElem = lmHeadDtype === 'f16' ? 2 : 4;
    // Sample from the start of the buffer (first hidden dimension's weights)
    const sampleSize = Math.min(64, lmHeadBuffer.size);
    const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(lmHeadBuffer, 0, staging, 0, sampleSize);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const rawData = staging.getMappedRange().slice(0);

    // Convert F16 to F32 for display
    let values: number[];
    if (lmHeadDtype === 'f16') {
      const u16 = new Uint16Array(rawData);
      const f32 = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) {
        const h = u16[i];
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7C00) >> 10;
        const mant = h & 0x03FF;
        let f: number;
        if (exp === 0) {
          f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024);
        } else if (exp === 31) {
          f = mant === 0 ? Infinity : NaN;
        } else {
          f = Math.pow(2, exp - 15) * (1 + mant / 1024);
        }
        f32[i] = sign ? -f : f;
      }
      values = Array.from(f32);
    } else {
      values = Array.from(new Float32Array(rawData));
    }

    staging.unmap();
    staging.destroy();

    const maxAbs = Math.max(...values.map(x => Math.abs(x)));
    const nonZero = values.filter(x => x !== 0).length;
    trace.logits(`LM_HEAD_SAMPLE: dtype=${lmHeadDtype}, bufSize=${lmHeadBuffer.size}, maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${values.length}, first8=[${values.slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
  }

  // HuggingFace models store lm_head as [vocabSize, hiddenSize], so transposeB=true
  const logitsBuffer = await runMatmul(normedBuffer, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
  });

  // Trace lm_head output
  if (kernelTrace.enabled) {
    await traceStep('matmul', 'lm_head', -1, logitsBuffer, [numTokens, matmulVocabSize]);
  }

  // DEBUG: Manual dot product verification for token "blue" (3730)
  // This checks if the matmul is computing the correct values
  if (ENABLE_DEBUG_READBACKS && allowReadback('logits.debug.manual-dot')) {
    const blueTokenId = 3730;
    const numSamples = 128; // Sample 128 dimensions for reasonable accuracy

    // Read 128 dimensions of last-token hidden state (after norm)
    const hSize = numSamples * 4;
    const hStaging = device.createBuffer({ size: hSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const hOffset = (numTokens - 1) * hiddenSize * 4;
    const hEnc = device.createCommandEncoder();
    hEnc.copyBufferToBuffer(normedBuffer, hOffset, hStaging, 0, hSize);
    device.queue.submit([hEnc.finish()]);
    await hStaging.mapAsync(GPUMapMode.READ);
    const hData = new Float32Array(hStaging.getMappedRange().slice(0));
    hStaging.unmap();
    hStaging.destroy();

    // For GGUF [K, N] layout, to read the embedding for token `blueTokenId`:
    // Element [k, blueTokenId] is at offset: k * vocabSize + blueTokenId
    // These are strided, not contiguous, so we need to sample them individually
    // For efficiency, just sample a few dimensions
    const bytesPerElem = lmHeadDtype === 'f16' ? 2 : 4;
    const embValues: number[] = [];

    // Read ALL hidden dimension values for embedding of token blueTokenId
    // For GGUF [K, N] layout: element [k, n] is at offset k * N + n
    // These are strided, not contiguous - read in chunks for efficiency
    const stride = matmulVocabSize * bytesPerElem;
    const chunkStart = blueTokenId * bytesPerElem;
    // Align the read offset to 4-byte boundary
    const alignedOffset = Math.floor(chunkStart / 4) * 4;
    const valueOffsetInChunk = (chunkStart - alignedOffset) / bytesPerElem;
    const readSize = 4; // Minimum 4 bytes for alignment

    // Read numSamples dimensions (128 for reasonable accuracy)
    for (let k = 0; k < numSamples; k++) {
      const offset = k * stride + alignedOffset;
      if (offset + readSize > lmHeadBuffer.size) break;

      const emStaging = device.createBuffer({ size: readSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const emEnc = device.createCommandEncoder();
      emEnc.copyBufferToBuffer(lmHeadBuffer, offset, emStaging, 0, readSize);
      device.queue.submit([emEnc.finish()]);
      await emStaging.mapAsync(GPUMapMode.READ);
      const rawData = emStaging.getMappedRange().slice(0);

      if (lmHeadDtype === 'f16') {
        const u16Array = new Uint16Array(rawData);
        const h = u16Array[valueOffsetInChunk] || u16Array[0];
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7C00) >> 10;
        const mant = h & 0x03FF;
        let f: number;
        if (exp === 0) { f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024); }
        else if (exp === 31) { f = mant === 0 ? Infinity : NaN; }
        else { f = Math.pow(2, exp - 15) * (1 + mant / 1024); }
        embValues.push(sign ? -f : f);
      } else {
        embValues.push(new Float32Array(rawData)[0]);
      }

      emStaging.unmap();
      emStaging.destroy();
    }

    // Compute FULL dot product
    let fullDot = 0;
    for (let i = 0; i < embValues.length; i++) {
      fullDot += hData[i] * embValues[i];
    }

    trace.logits(`MANUAL_DOT_CHECK[blue=${blueTokenId}]: nDims=${embValues.length}, hidden[0..3]=[${Array.from(hData.slice(0, 4)).map(x => x.toFixed(4)).join(', ')}], emb[0..3]=[${Array.from(embValues.slice(0, 4)).map(x => x.toFixed(4)).join(', ')}]`);
    trace.logits(`MANUAL_DOT_CHECK[blue=${blueTokenId}]: fullDot(${embValues.length}dims)=${fullDot.toFixed(4)}`);
    const partialDot = fullDot; // For compatibility with below code

    // Also read the GPU-computed logit for "blue" directly from logitsBuffer
    // Logit for last token position (6) and token id 3730
    const logitOffset = ((numTokens - 1) * matmulVocabSize + blueTokenId) * 4;  // F32
    const logitStaging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const logitEnc = device.createCommandEncoder();
    logitEnc.copyBufferToBuffer(logitsBuffer, logitOffset, logitStaging, 0, 4);
    device.queue.submit([logitEnc.finish()]);
    await logitStaging.mapAsync(GPUMapMode.READ);
    const gpuLogit = new Float32Array(logitStaging.getMappedRange().slice(0))[0];
    logitStaging.unmap();
    logitStaging.destroy();

    // Compare partial dot product (128 dims) with extrapolated full
    const extrapolatedFull = partialDot * (hiddenSize / numSamples);
    trace.logits(`LOGIT_COMPARE[blue=${blueTokenId}]: gpuLogit=${gpuLogit.toFixed(4)}, cpuPartial(${numSamples}dims)=${partialDot.toFixed(4)}, extrapolated=${extrapolatedFull.toFixed(4)}, ratio=${(gpuLogit / extrapolatedFull).toFixed(4)}`);

    // Also check token 36889 (_scripts) which is winning
    const scriptsTokenId = 36889;
    const scriptsOffset = ((numTokens - 1) * matmulVocabSize + scriptsTokenId) * 4;
    const scriptsStaging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const scriptsEnc = device.createCommandEncoder();
    scriptsEnc.copyBufferToBuffer(logitsBuffer, scriptsOffset, scriptsStaging, 0, 4);
    device.queue.submit([scriptsEnc.finish()]);
    await scriptsStaging.mapAsync(GPUMapMode.READ);
    const scriptsLogit = new Float32Array(scriptsStaging.getMappedRange().slice(0))[0];
    scriptsStaging.unmap();
    scriptsStaging.destroy();

    trace.logits(`LOGIT_CHECK[_scripts=${scriptsTokenId}]: gpuLogit=${scriptsLogit.toFixed(4)}, vs blue=${gpuLogit.toFixed(4)}, diff=${(scriptsLogit - gpuLogit).toFixed(4)}`);

    // Compare embedding values for _scripts vs blue
    // Read 128 embedding dimensions for _scripts (same as blue)
    const scriptsEmbValues: number[] = [];
    const scriptsChunkStart = scriptsTokenId * bytesPerElem;
    const scriptsAlignedOffset = Math.floor(scriptsChunkStart / 4) * 4;
    const scriptsValueOffset = (scriptsChunkStart - scriptsAlignedOffset) / bytesPerElem;

    for (let k = 0; k < numSamples; k++) {
      const offset = k * stride + scriptsAlignedOffset;
      if (offset + readSize > lmHeadBuffer.size) break;

      const emStaging = device.createBuffer({ size: readSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const emEnc = device.createCommandEncoder();
      emEnc.copyBufferToBuffer(lmHeadBuffer, offset, emStaging, 0, readSize);
      device.queue.submit([emEnc.finish()]);
      await emStaging.mapAsync(GPUMapMode.READ);
      const rawData = emStaging.getMappedRange().slice(0);

      if (lmHeadDtype === 'f16') {
        const u16Array = new Uint16Array(rawData);
        const h = u16Array[scriptsValueOffset] || u16Array[0];
        const sign = (h & 0x8000) >> 15;
        const exp = (h & 0x7C00) >> 10;
        const mant = h & 0x03FF;
        let f: number;
        if (exp === 0) { f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024); }
        else if (exp === 31) { f = mant === 0 ? Infinity : NaN; }
        else { f = Math.pow(2, exp - 15) * (1 + mant / 1024); }
        scriptsEmbValues.push(sign ? -f : f);
      } else {
        scriptsEmbValues.push(new Float32Array(rawData)[0]);
      }

      emStaging.unmap();
      emStaging.destroy();
    }

    // Compute dot products for comparison
    let blueDot = 0, scriptsDot = 0;
    for (let i = 0; i < scriptsEmbValues.length && i < embValues.length; i++) {
      blueDot += hData[i] * embValues[i];
      scriptsDot += hData[i] * scriptsEmbValues[i];
    }

    // Extrapolate to full dimension
    const blueExtrapolated = blueDot * (hiddenSize / numSamples);
    const scriptsExtrapolated = scriptsDot * (hiddenSize / numSamples);

    trace.logits(`EMB_COMPARE: blue[0..3]=[${embValues.slice(0, 4).map(x => x.toFixed(4)).join(', ')}], _scripts[0..3]=[${scriptsEmbValues.slice(0, 4).map(x => x.toFixed(4)).join(', ')}]`);
    trace.logits(`DOT_COMPARE[${numSamples}dims]: blueDot=${blueDot.toFixed(4)}, scriptsDot=${scriptsDot.toFixed(4)}`);
    trace.logits(`DOT_EXTRAPOLATED: blueEst=${blueExtrapolated.toFixed(2)}, scriptsEst=${scriptsExtrapolated.toFixed(2)}, actualBlue=${gpuLogit.toFixed(2)}, actualScripts=${scriptsLogit.toFixed(2)}`);

    // Check if maybe the GPU is using the WRONG token position (row 0 instead of row 6)
    // Read hidden state at position 0 and compute dot product with _scripts
    const h0Size = numSamples * 4;
    const h0Staging = device.createBuffer({ size: h0Size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const h0Enc = device.createCommandEncoder();
    h0Enc.copyBufferToBuffer(normedBuffer, 0, h0Staging, 0, h0Size); // Position 0
    device.queue.submit([h0Enc.finish()]);
    await h0Staging.mapAsync(GPUMapMode.READ);
    const h0Data = new Float32Array(h0Staging.getMappedRange().slice(0));
    h0Staging.unmap();
    h0Staging.destroy();

    let scriptsDotPos0 = 0;
    for (let i = 0; i < Math.min(scriptsEmbValues.length, h0Data.length); i++) {
      scriptsDotPos0 += h0Data[i] * scriptsEmbValues[i];
    }
    const scriptsExtrapolatedPos0 = scriptsDotPos0 * (hiddenSize / numSamples);

    trace.logits(`POS0_CHECK: hidden0[0..3]=[${Array.from(h0Data.slice(0, 4)).map(x => x.toFixed(4)).join(', ')}]`);
    trace.logits(`SCRIPTS_POS0_DOT: dot=${scriptsDotPos0.toFixed(4)}, extrapolated=${scriptsExtrapolatedPos0.toFixed(2)}, actualScripts=${scriptsLogit.toFixed(2)}`);

    // Also check what logit the GPU computed at position 0 for _scripts
    const scriptsPos0Offset = (0 * matmulVocabSize + scriptsTokenId) * 4;  // Row 0
    const sp0Staging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const sp0Enc = device.createCommandEncoder();
    sp0Enc.copyBufferToBuffer(logitsBuffer, scriptsPos0Offset, sp0Staging, 0, 4);
    device.queue.submit([sp0Enc.finish()]);
    await sp0Staging.mapAsync(GPUMapMode.READ);
    const scriptsLogitPos0 = new Float32Array(sp0Staging.getMappedRange().slice(0))[0];
    sp0Staging.unmap();
    sp0Staging.destroy();

    trace.logits(`GPU_LOGIT_BY_POS[_scripts]: pos0=${scriptsLogitPos0.toFixed(2)}, pos6=${scriptsLogit.toFixed(2)}, diff=${(scriptsLogit - scriptsLogitPos0).toFixed(2)}`);

    // Verify by using gather kernel to read _scripts embedding and compare with manual read
    // Import runGather to read the embedding for token 36889
    const { runGather } = await import('../../gpu/kernel-selector.js');
    const tokenIdBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(tokenIdBuffer, 0, new Uint32Array([scriptsTokenId]));

    const gatherOutput = await runGather(tokenIdBuffer, lmHeadBuffer, 1, hiddenSize, matmulVocabSize);
    tokenIdBuffer.destroy();

    // Read first 8 values from gather output
    const gStaging = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const gEnc = device.createCommandEncoder();
    gEnc.copyBufferToBuffer(gatherOutput, 0, gStaging, 0, 32);
    device.queue.submit([gEnc.finish()]);
    await gStaging.mapAsync(GPUMapMode.READ);
    const gatherVals = new Float32Array(gStaging.getMappedRange().slice(0));
    gStaging.unmap();
    gStaging.destroy();

    trace.logits(`GATHER_VS_MANUAL[_scripts]: gather[0..3]=[${Array.from(gatherVals).slice(0, 4).map(x => x.toFixed(4)).join(', ')}], manual[0..3]=[${scriptsEmbValues.slice(0, 4).map(x => x.toFixed(4)).join(', ')}]`);

    // Read full embedding via gather and compute full dot product
    const fullGStaging = device.createBuffer({ size: hiddenSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const fullGEnc = device.createCommandEncoder();
    fullGEnc.copyBufferToBuffer(gatherOutput, 0, fullGStaging, 0, hiddenSize * 4);
    device.queue.submit([fullGEnc.finish()]);
    await fullGStaging.mapAsync(GPUMapMode.READ);
    const fullGatherVals = new Float32Array(fullGStaging.getMappedRange().slice(0));
    fullGStaging.unmap();
    fullGStaging.destroy();
    releaseBuffer(gatherOutput);

    // Also read full hidden state for pos 6
    const fullHStaging = device.createBuffer({ size: hiddenSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const fullHEnc = device.createCommandEncoder();
    const fullHOffset = (numTokens - 1) * hiddenSize * 4;
    fullHEnc.copyBufferToBuffer(normedBuffer, fullHOffset, fullHStaging, 0, hiddenSize * 4);
    device.queue.submit([fullHEnc.finish()]);
    await fullHStaging.mapAsync(GPUMapMode.READ);
    const fullHiddenVals = new Float32Array(fullHStaging.getMappedRange().slice(0));
    fullHStaging.unmap();
    fullHStaging.destroy();

    // Compute full dot product on CPU
    let cpuFullDot = 0;
    for (let i = 0; i < hiddenSize; i++) {
      cpuFullDot += fullHiddenVals[i] * fullGatherVals[i];
    }

    trace.logits(`FULL_DOT_PRODUCT[_scripts]: cpuFullDot=${cpuFullDot.toFixed(4)}, gpuLogit=${scriptsLogit.toFixed(4)}, diff=${(scriptsLogit - cpuFullDot).toFixed(4)}`);
  }

  // 4. Read back logits
  const logitsData = await readBuffer(logitsBuffer, numTokens * matmulVocabSize * 4);

  // Cleanup
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedBuffer);
  releaseBuffer(logitsBuffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(lmHeadBuffer);

  // Pad with -Infinity if matmulVocabSize < vocabSize
  if (matmulVocabSize < vocabSize) {
    const paddedLogits = new Float32Array(numTokens * vocabSize);
    const rawLogits = new Float32Array(logitsData);
    for (let t = 0; t < numTokens; t++) {
      const srcOffset = t * matmulVocabSize;
      const dstOffset = t * vocabSize;
      // Copy actual logits
      for (let i = 0; i < matmulVocabSize; i++) {
        paddedLogits[dstOffset + i] = rawLogits[srcOffset + i];
      }
      // Pad extra slots with -Infinity
      for (let i = matmulVocabSize; i < vocabSize; i++) {
        paddedLogits[dstOffset + i] = -Infinity;
      }
    }
    // Apply Gemma 2 softcapping if configured
    if (config.finalLogitSoftcapping) {
      applySoftcapping(paddedLogits, config.finalLogitSoftcapping);
    }
    return paddedLogits;
  }

  const logits = new Float32Array(logitsData);
  // Apply Gemma 2 softcapping if configured
  if (config.finalLogitSoftcapping) {
    applySoftcapping(logits, config.finalLogitSoftcapping);
  }
  return logits;
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
  const { hiddenSize, vocabSize, rmsNormEps, useTiedEmbeddings, embeddingVocabSize } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!device) {
    return null;
  }

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded');
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

  const normedBuffer = await runRMSNorm(inputBuffer, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });

  // Project to vocab via LM head
  let lmHeadBuffer: GPUBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else {
    lmHeadBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(lmHeadBuffer, 0, lmHead as unknown as BufferSource);
    lmHeadBufferOwned = true;
  }

  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;

  const logitsBuffer = await runMatmul(normedBuffer, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
  });

  // Cleanup intermediate buffers (but keep logitsBuffer)
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedBuffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(lmHeadBuffer);

  return { logitsBuffer, vocabSize: matmulVocabSize };
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
  const { hiddenSize, vocabSize, rmsNormEps, useTiedEmbeddings, embeddingVocabSize } = config;
  const { finalNorm, lmHead } = weights;
  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize ? embeddingVocabSize : vocabSize;

  if (!finalNorm || !lmHead) {
    throw new Error('[recordLogitsGPU] Final norm or LM head not loaded');
  }

  // Get norm weight buffer
  let normWeightBuffer: GPUBuffer;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer((finalNorm as Float32Array).byteLength, undefined, 'final_norm_w');
    recorder.device.queue.writeBuffer(normWeightBuffer, 0, finalNorm as unknown as BufferSource);
  }

  // Record RMSNorm (no submit)
  const normedBuffer = await recordRMSNorm(recorder, hiddenStates, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });

  // Get LM head buffer
  let lmHeadBuffer: GPUBuffer;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else {
    lmHeadBuffer = acquireBuffer((lmHead as Float32Array).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(lmHeadBuffer, 0, lmHead as unknown as BufferSource);
  }

  // Record matmul (no submit)
  const logitsBuffer = await recordMatmul(recorder, normedBuffer, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
  });

  // Track intermediate buffer for cleanup after submit
  recorder.trackTemporaryBuffer(normedBuffer);

  return { logitsBuffer, vocabSize: matmulVocabSize };
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
