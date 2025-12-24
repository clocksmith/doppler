/**
 * Reference RoPE (Rotary Position Embedding) Implementation
 */

export interface RopeFrequencies {
  cos: Float32Array;
  sin: Float32Array;
}

/**
 * Precompute RoPE frequencies
 * @param dim Head dimension
 * @param maxSeqLen Maximum sequence length
 * @param base Base for frequency computation (default 10000)
 * @returns Precomputed cos/sin [maxSeqLen, dim/2]
 */
export function computeRopeFreqs(dim: number, maxSeqLen: number, base: number = 10000): RopeFrequencies {
  const halfDim = dim / 2;
  const cos = new Float32Array(maxSeqLen * halfDim);
  const sin = new Float32Array(maxSeqLen * halfDim);

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const freq = 1.0 / Math.pow(base, (2 * i) / dim);
      const angle = pos * freq;
      cos[pos * halfDim + i] = Math.cos(angle);
      sin[pos * halfDim + i] = Math.sin(angle);
    }
  }

  return { cos, sin };
}

/**
 * Reference RoPE application
 * Applies rotary position embeddings to Q and K
 *
 * @param x Input [seqLen, numHeads, headDim]
 * @param cos Cos frequencies [seqLen, headDim/2]
 * @param sin Sin frequencies [seqLen, headDim/2]
 * @param seqLen Sequence length
 * @param numHeads Number of heads
 * @param headDim Head dimension
 * @param startPos Starting position (for KV cache)
 * @returns Output with RoPE applied
 */
export function ropeRef(
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  startPos: number = 0
): Float32Array {
  const output = new Float32Array(x.length);
  const halfDim = headDim / 2;

  for (let s = 0; s < seqLen; s++) {
    const pos = s + startPos;

    for (let h = 0; h < numHeads; h++) {
      const offset = s * numHeads * headDim + h * headDim;

      for (let i = 0; i < halfDim; i++) {
        const x0 = x[offset + i];
        const x1 = x[offset + i + halfDim];

        const cosVal = cos[pos * halfDim + i];
        const sinVal = sin[pos * halfDim + i];

        // Apply rotation
        output[offset + i] = x0 * cosVal - x1 * sinVal;
        output[offset + i + halfDim] = x0 * sinVal + x1 * cosVal;
      }
    }
  }

  return output;
}

/**
 * Alternative RoPE layout (interleaved pairs)
 * Some models use [x0, x1, x2, x3, ...] -> rotate pairs (x0,x1), (x2,x3), ...
 */
export function ropeInterleavedRef(
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  startPos: number = 0
): Float32Array {
  const output = new Float32Array(x.length);
  const halfDim = headDim / 2;

  for (let s = 0; s < seqLen; s++) {
    const pos = s + startPos;

    for (let h = 0; h < numHeads; h++) {
      const offset = s * numHeads * headDim + h * headDim;

      for (let i = 0; i < halfDim; i++) {
        const x0 = x[offset + 2 * i];
        const x1 = x[offset + 2 * i + 1];

        const cosVal = cos[pos * halfDim + i];
        const sinVal = sin[pos * halfDim + i];

        output[offset + 2 * i] = x0 * cosVal - x1 * sinVal;
        output[offset + 2 * i + 1] = x0 * sinVal + x1 * cosVal;
      }
    }
  }

  return output;
}

export default ropeRef;
