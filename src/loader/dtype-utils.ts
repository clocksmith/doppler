/**
 * Dtype Utilities
 *
 * Data type conversion utilities for tensor loading.
 *
 * @module loader/dtype-utils
 */

import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../gpu/buffer-pool.js';
import { log, trace as debugTrace } from '../debug/index.js';
import type { TensorLocation } from './loader-types.js';

/**
 * Convert F16 (half precision) to F32 (single precision)
 */
export function f16ToF32(h: number): number {
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

/**
 * Convert BF16 buffer to F32 on GPU
 */
export async function convertBF16ToF32GPU(
  srcBuffer: GPUBuffer,
  numElements: number,
  name: string
): Promise<GPUBuffer> {
  debugTrace.loader(`[BF16→F32] Importing cast.js...`);
  const castModule = await import('../gpu/kernels/cast.js');
  debugTrace.loader(`[BF16→F32] castModule keys:`, Object.keys(castModule));
  const { runBF16ToF32 } = castModule;
  debugTrace.loader(`[BF16→F32] runBF16ToF32 type: ${typeof runBF16ToF32}`);
  const resultTensor = await runBF16ToF32(srcBuffer, [numElements], name);
  debugTrace.loader(`[BF16→F32] runBF16ToF32 returned, result.size=${resultTensor.buffer?.size}`);

  // Debug: Verify conversion produced non-zero values
  if (name.includes('embed') && name.includes('embed_tokens')) {
    try {
      debugTrace.loader(`[BF16→F32] Checking embed buffer for non-zeros...`);
      const device = getDevice();
      const sampleSize = Math.min(1024, resultTensor.buffer.size);
      debugTrace.loader(`[BF16→F32] Creating staging buffer size=${sampleSize}`);
      const stagingBuffer = device!.createBuffer({
        size: sampleSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      debugTrace.loader(`[BF16→F32] Copying to staging buffer...`);
      const encoder = device!.createCommandEncoder();
      encoder.copyBufferToBuffer(resultTensor.buffer, 0, stagingBuffer, 0, sampleSize);
      device!.queue.submit([encoder.finish()]);
      debugTrace.loader(`[BF16→F32] Mapping staging buffer...`);
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      debugTrace.loader(`[BF16→F32] Reading data...`);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();
      const nonZero = Array.from(data).filter(x => x !== 0);
      const nanCount = data.filter(x => !Number.isFinite(x)).length;
      debugTrace.loader(`[BF16→F32] nonZero=${nonZero.length}/${data.length}, nan=${nanCount}, sample=[${nonZero.slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
    } catch (err) {
      log.error('Loader', 'BF16→F32 embed buffer check error:', (err as Error).message);
    }
  }

  return resultTensor.buffer;
}

/**
 * Decide whether a quantized tensor should be dequantized directly to f16.
 * Returns true for matmul weights (projections, FFN, lm_head, embeddings).
 */
export function shouldDequantizeToF16(name: string): boolean {
  const lower = name.toLowerCase();
  const matmulSuffixes = [
    // HuggingFace/SafeTensors naming
    'q_proj.weight',
    'k_proj.weight',
    'v_proj.weight',
    'o_proj.weight',
    'attention.wq.weight',
    'attention.wk.weight',
    'attention.wv.weight',
    'attention.wo.weight',
    'gate_proj.weight',
    'up_proj.weight',
    'down_proj.weight',
    'w1.weight',
    'w2.weight',
    'w3.weight',
    'lm_head.weight',
    'output.weight',
    // Weight-tied embedding/lm_head (Gemma, Llama, etc.)
    'embed_tokens.weight',
    'wte.weight',
    'token_embd.weight',
    // GGUF naming (blk.X.attn_q.weight, blk.X.ffn_gate.weight, etc.)
    'attn_q.weight',
    'attn_k.weight',
    'attn_v.weight',
    'attn_output.weight',
    'ffn_gate.weight',
    'ffn_up.weight',
    'ffn_down.weight',
    'ffn_gate_up.weight',
  ];

  return matmulSuffixes.some(suffix => lower.endsWith(suffix));
}

/**
 * Check if a weight is an embedding weight (needs column layout for LM head matmul).
 * GGUF stores all weights as [N,K] (transposed). For layer weights, we need transposeB=true
 * to compute A@W = A@W.T^T. But for embeddings used as LM head, we need transposeB=false
 * to compute hidden@E.T directly (the embedding IS already transposed in GGUF).
 */
export function isEmbeddingWeight(name: string): boolean {
  const lower = name.toLowerCase();
  // Only match actual embedding and lm_head weights
  // Be careful NOT to match 'attn_output.weight' which ends with 'output.weight'
  const embeddingPatterns = [
    'embed_tokens.weight',
    'wte.weight',
    'token_embd.weight',
    'lm_head.weight',
  ];
  // Check exact suffix matches for embedding patterns
  if (embeddingPatterns.some(suffix => lower.endsWith(suffix))) {
    return true;
  }
  // For 'output.weight', only match if it's the top-level (no 'attn_' prefix)
  // This handles models that use 'output.weight' as the LM head name
  if (lower.endsWith('output.weight') && !lower.includes('attn_')) {
    return true;
  }
  return false;
}

/**
 * Apply layout metadata to a GPU buffer if the tensor has column-major storage.
 * Note: Layout is now tracked via WeightBuffer for matmul weights.
 * This function is kept for API compatibility but is a no-op for non-matmul weights (norms).
 */
export function applyBufferLayout(buffer: GPUBuffer, _location: TensorLocation): GPUBuffer {
  // Note: WeakMap layout tracking removed - layout is stored in WeightBuffer
  // For non-matmul weights (norms), layout doesn't affect kernel selection
  return buffer;
}

/**
 * Apply +1 offset to norm weights for Gemma models.
 *
 * IMPORTANT: actualNumElements must be provided to avoid reading garbage padding
 * from the buffer pool's power-of-2 bucketing.
 *
 * @param bufferDtype - Optional dtype for GPU buffer (defaults to 'f32')
 */
export async function applyNormWeightOffset(
  tensor: GPUBuffer | Float32Array,
  actualNumElements?: number,
  normOffsetDebugLogged = false,
  bufferDtype: 'f16' | 'f32' | 'bf16' = 'f32'
): Promise<{ tensor: GPUBuffer | Float32Array; debugLogged: boolean }> {
  const device = getDevice();
  if (!device) {
    log.warn('Loader', ' No GPU device for norm offset');
    return { tensor, debugLogged: normOffsetDebugLogged };
  }

  let debugLogged = normOffsetDebugLogged;

  if (tensor instanceof GPUBuffer) {
    // Use provided dtype to determine element size (WeakMap tracking removed)
    const isF16 = bufferDtype === 'f16' || bufferDtype === 'bf16';
    const bytesPerElement = isF16 ? 2 : 4;

    // Use actual element count if provided, otherwise infer from buffer size
    const numElements = actualNumElements ?? Math.floor(tensor.size / bytesPerElement);
    const dataSize = numElements * bytesPerElement;

    // Ensure we don't read past the buffer
    const readSize = Math.min(dataSize, tensor.size);

    const stagingBuffer = device.createBuffer({
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor, 0, stagingBuffer, 0, readSize);
    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const rawData = stagingBuffer.getMappedRange().slice(0);
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    // Convert to F32 for offset calculation
    let data: Float32Array;
    if (isF16) {
      const u16Data = new Uint16Array(rawData);
      data = new Float32Array(u16Data.length);
      for (let i = 0; i < u16Data.length; i++) {
        data[i] = f16ToF32(u16Data[i]);
      }
    } else {
      data = new Float32Array(rawData);
    }

    const offsetData = new Float32Array(numElements);
    for (let i = 0; i < numElements; i++) {
      offsetData[i] = 1.0 + data[i];
    }

    // Debug: log first norm weight transformation (once per model load)
    if (!debugLogged) {
      debugLogged = true;
      const beforeMin = Math.min(...Array.from(data.slice(0, Math.min(256, numElements))));
      const beforeMax = Math.max(...Array.from(data.slice(0, Math.min(256, numElements))));
      const afterMin = Math.min(...Array.from(offsetData.slice(0, Math.min(256, numElements))));
      const afterMax = Math.max(...Array.from(offsetData.slice(0, Math.min(256, numElements))));
      debugTrace.loader(`Norm +1 offset: before=[${beforeMin.toFixed(3)}, ${beforeMax.toFixed(3)}] after=[${afterMin.toFixed(3)}, ${afterMax.toFixed(3)}]`);
    }

    releaseBuffer(tensor);
    const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
    device.queue.writeBuffer(newBuffer, 0, offsetData);
    return { tensor: newBuffer, debugLogged };
  }

  if (tensor instanceof Float32Array) {
    const numElements = actualNumElements ?? tensor.length;
    const offsetData = new Float32Array(numElements);
    for (let i = 0; i < numElements; i++) {
      offsetData[i] = 1.0 + tensor[i];
    }
    // Always upload to GPU to prevent double-offset in pipeline
    // Pipeline's getNormWeightBuffer returns GPUBuffer as-is, skipping offset
    const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
    device.queue.writeBuffer(newBuffer, 0, offsetData);
    return { tensor: newBuffer, debugLogged };
  }

  log.warn('Loader', ' Unknown tensor type for norm offset');
  return { tensor, debugLogged };
}

/**
 * Find alternative tensor name (handles different naming conventions).
 * Returns null if no alternative is found.
 */
export function findAlternativeTensorName(
  name: string,
  tensorLocations: Map<string, TensorLocation>
): string | null {
  const prefixes = [
    'language_model.model.',
    'language_model.',
    'model.',
    '',
  ];

  for (const prefix of prefixes) {
    const prefixedName = prefix + name;
    if (prefixedName !== name && tensorLocations.has(prefixedName)) {
      return prefixedName;
    }
  }

  const patterns: [RegExp, string][] = [
    [/^layers\.(\d+)\./, 'model.layers.$1.'],
    [/^model\.layers\.(\d+)\./, 'layers.$1.'],
    [/\.weight$/, ''],
    [/$/, '.weight'],
    [/attention/, 'self_attn'],
    [/self_attn/, 'attention'],
    [/ffn/, 'mlp'],
    [/mlp/, 'ffn'],
  ];

  for (const [pattern, replacement] of patterns) {
    const altName = name.replace(pattern, replacement);
    if (altName !== name && tensorLocations.has(altName)) {
      return altName;
    }
    for (const prefix of prefixes) {
      const prefixedAlt = prefix + altName;
      if (tensorLocations.has(prefixedAlt)) {
        return prefixedAlt;
      }
    }
  }
  return null;
}
