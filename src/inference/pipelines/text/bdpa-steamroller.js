/**
 * Basis-Decomposed Paged Attention (BDPA) 
 * Steamroller Ingestion Algorithm
 * 
 * Implements the Host-Side (JS) Least Significant Digit (LSD) Radix Argsort
 * to re-order incoming Context tokens by their Token ID, generating the 
 * Vocab Basis Table ($T_{basis}$), Quantized Residuals ($P_{delta}$), and 
 * Execution Index ($I_{flat}$) for the GPU.
 */

/**
 * Perform an out-of-place LSD Radix Argsort over Token IDs.
 * Groups semantic clusters together to maximize GPU L2 Cache hits on $T_{basis}$.
 * 
 * @param {Int32Array|Uint32Array} tokens The input token IDs
 * @returns {Uint32Array} The argsort permutation indices [new_pos] -> original_pos
 */
export function steamrollerRadixArgsort(tokens) {
    const n = tokens.length;
    // Initialize sorting permutation
    let indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    let auxIndices = new Uint32Array(n);

    // 4 passes for 32-bit integers, 8-bits per pass (Radix 256)
    for (let shift = 0; shift < 32; shift += 8) {
        const counts = new Uint32Array(256);

        // Histogram
        for (let i = 0; i < n; i++) {
            const val = tokens[indices[i]];
            const digit = (val >>> shift) & 0xFF;
            counts[digit]++;
        }

        // Prefix sum (offsets)
        const offsets = new Uint32Array(256);
        let currentOffset = 0;
        for (let i = 0; i < 256; i++) {
            offsets[i] = currentOffset;
            currentOffset += counts[i];
        }

        // Scatter
        for (let i = 0; i < n; i++) {
            const idx = indices[i];
            const val = tokens[idx];
            const digit = (val >>> shift) & 0xFF;
            auxIndices[offsets[digit]] = idx;
            offsets[digit]++;
        }

        // Swap buffers
        const temp = indices;
        indices = auxIndices;
        auxIndices = temp;
    }

    // Because there are 4 passes (even number), the final sorted data ends up in indices, so no extra copy is needed.
    return indices;
}

/**
 * Generate the Vocabulary Basis Table ($T_{basis}$) and Execution Index ($I_{flat}$)
 * from the argsorted semantic blocks.
 * 
 * @param {Uint32Array} sortedIndices The argsorted indices [new_pos -> old_pos]
 * @param {Float32Array|Uint16Array} kBuffer The full Float16 or Float32 Key tensor (length = N * kvSize)
 * @param {Float32Array|Uint16Array} vBuffer The full Float16 or Float32 Value tensor (length = N * kvSize)
 * @param {Int32Array|Uint32Array} tokens The input token IDs
 * @param {number} kvSize The size of a single Key/Value vector (numKVHeads * headDim)
 * @returns {Object} { tBasisK, tBasisV, pDeltaK, pDeltaV, iFlat }
 */
export function generateBDPAData(sortedIndices, kBuffer, vBuffer, tokens, kvSize) {
    const n = sortedIndices.length;

    // Track unique tokens for basic Basis Allocation
    const uniqueTokenIDs = new Map();
    let nextBasisIdx = 0;

    // 1. First Pass: Find unique tokens and assign Basis Indexes
    for (let i = 0; i < n; i++) {
        const origPos = sortedIndices[i];
        const tokenID = tokens[origPos];
        if (!uniqueTokenIDs.has(tokenID)) {
            uniqueTokenIDs.set(tokenID, nextBasisIdx++);
        }
    }

    const numBasisVectors = nextBasisIdx;

    // Allocate T_basis using Float32 for accumulation precision
    const tBasisKAccumulator = new Float32Array(numBasisVectors * kvSize);
    const tBasisVAccumulator = new Float32Array(numBasisVectors * kvSize);
    const basisCounts = new Uint32Array(numBasisVectors);

    // 2. Accumulate matching tokens into their Basis centroid
    const isKBufferF16 = kBuffer instanceof Uint16Array;
    // Note: We use the `types.js` f16 bits converter or rely on outer cast since we need f32 math
    // For the sake of this JS Host implementation, we assume kBuffer is f32 from JS layer cache for simplicity

    for (let i = 0; i < n; i++) {
        const origPos = sortedIndices[i];
        const tokenID = tokens[origPos];
        const basisIdx = uniqueTokenIDs.get(tokenID);

        const kvOffset = origPos * kvSize;
        const basisOffset = basisIdx * kvSize;

        for (let j = 0; j < kvSize; j++) {
            tBasisKAccumulator[basisOffset + j] += kBuffer[kvOffset + j];
            tBasisVAccumulator[basisOffset + j] += vBuffer[kvOffset + j];
        }
        basisCounts[basisIdx]++;
    }

    // 3. Average the accumulators to finalize T_basis
    for (let b = 0; b < numBasisVectors; b++) {
        const count = basisCounts[b];
        const basisOffset = b * kvSize;
        for (let j = 0; j < kvSize; j++) {
            tBasisKAccumulator[basisOffset + j] /= count;
            tBasisVAccumulator[basisOffset + j] /= count;
        }
    }

    // 4. Generate Deltas (P_delta) and Execution Index (I_flat)
    // P_delta will store the Int8 quantized residuals
    const pDeltaK = new Int8Array(n * kvSize);
    const pDeltaV = new Int8Array(n * kvSize);

    // I_flat: [BasisPtr, DeltaPagePtr, OriginalPos] 
    const iFlat = new Uint32Array(n * 3);

    for (let i = 0; i < n; i++) {
        const origPos = sortedIndices[i];
        const tokenID = tokens[origPos];
        const basisIdx = uniqueTokenIDs.get(tokenID);

        const kvOffset = origPos * kvSize;
        const basisOffset = basisIdx * kvSize;

        const deltaOffset = i * kvSize; // Packed contiguously sorted

        for (let j = 0; j < kvSize; j++) {
            // Calculate Residual
            const rawK = kBuffer[kvOffset + j];
            const rawV = vBuffer[kvOffset + j];
            const basisK = tBasisKAccumulator[basisOffset + j];
            const basisV = tBasisVAccumulator[basisOffset + j];

            const resK = rawK - basisK;
            const resV = rawV - basisV;

            // Super simple symmetric Int8 Quantization (scale = 1/127 of max across all, simplified here)
            // Real BDPA uses block-wise scaling
            const scaleStr = 127.0;
            pDeltaK[deltaOffset + j] = Math.max(-127, Math.min(127, Math.round(resK * scaleStr)));
            pDeltaV[deltaOffset + j] = Math.max(-127, Math.min(127, Math.round(resV * scaleStr)));
        }

        // Write I_flat
        const iFlatOffset = i * 3;
        iFlat[iFlatOffset + 0] = basisIdx; // BasisPtr
        iFlat[iFlatOffset + 1] = i;        // DeltaPagePtr (1:1 mapping in this dense prototype)
        iFlat[iFlatOffset + 2] = origPos;  // Original Position for RoPE Restore
    }

    return {
        tBasisK: tBasisKAccumulator,
        tBasisV: tBasisVAccumulator,
        pDeltaK,
        pDeltaV,
        iFlat,
        numBasisVectors
    };
}
