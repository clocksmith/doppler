/**
 * Dtype Utilities
 *
 * Data type conversion utilities for tensor loading.
 *
 * @module loader/dtype-utils
 */

import { getDevice } from '../gpu/device.js';
import { isTraceEnabled, log, trace as debugTrace } from '../debug/index.js';

/**
 * Convert F16 (half precision) to F32 (single precision)
 * @param {number} h
 * @returns {number}
 */
export function f16ToF32(h) {
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
 * @param {GPUBuffer} srcBuffer
 * @param {number} numElements
 * @param {string} name
 * @returns {Promise<GPUBuffer>}
 */
export async function convertBF16ToF32GPU(srcBuffer, numElements, name) {
  debugTrace.loader(`[BF16→F32] Importing cast.js...`);
  const castModule = await import('../gpu/kernels/cast.js');
  debugTrace.loader(`[BF16→F32] castModule keys:`, Object.keys(castModule));
  const { runBF16ToF32 } = castModule;
  debugTrace.loader(`[BF16→F32] runBF16ToF32 type: ${typeof runBF16ToF32}`);
  const resultTensor = await runBF16ToF32(srcBuffer, [numElements], name);
  debugTrace.loader(`[BF16→F32] runBF16ToF32 returned, result.size=${resultTensor.buffer?.size}`);

  // Debug: Verify conversion produced non-zero values
  const shouldCheckEmbed = isTraceEnabled('loader') &&
    name.includes('embed') &&
    name.includes('embed_tokens');
  if (shouldCheckEmbed) {
    try {
      debugTrace.loader(`[BF16→F32] Checking embed buffer for non-zeros...`);
      const device = getDevice();
      const sampleSize = Math.min(1024, resultTensor.buffer.size);
      debugTrace.loader(`[BF16→F32] Creating staging buffer size=${sampleSize}`);
      const stagingBuffer = device.createBuffer({
        size: sampleSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      debugTrace.loader(`[BF16→F32] Copying to staging buffer...`);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(resultTensor.buffer, 0, stagingBuffer, 0, sampleSize);
      device.queue.submit([encoder.finish()]);
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
      log.error('Loader', 'BF16→F32 embed buffer check error:', /** @type {Error} */ (err).message);
    }
  }

  return resultTensor.buffer;
}

/**
 * Decide whether a quantized tensor should be dequantized directly to f16.
 * Returns true for matmul weights (projections, FFN, lm_head, embeddings).
 * @param {string} name
 * @returns {boolean}
 */
export function shouldDequantizeToF16(name) {
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
 * @param {string} name
 * @returns {boolean}
 */
export function isEmbeddingWeight(name) {
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
 * @param {GPUBuffer} buffer
 * @param {import('./loader-types.js').TensorLocation} _location
 * @returns {GPUBuffer}
 */
export function applyBufferLayout(buffer, _location) {
  // Note: WeakMap layout tracking removed - layout is stored in WeightBuffer
  // For non-matmul weights (norms), layout doesn't affect kernel selection
  return buffer;
}

/**
 * Find alternative tensor name (handles different naming conventions).
 * Returns null if no alternative is found.
 * @param {string} name
 * @param {Map<string, import('./loader-types.js').TensorLocation>} tensorLocations
 * @returns {string | null}
 */
export function findAlternativeTensorName(name, tensorLocations) {
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

  /** @type {Array<[RegExp, string]>} */
  const patterns = [
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
