


export function moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
  // Count tokens per expert
  const tokenCounts = new Uint32Array(numExperts);

  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      tokenCounts[expertIdx]++;
    }
  }

  // Find max tokens per expert for allocation
  let maxTokensPerExpert = 0;
  for (let e = 0; e < numExperts; e++) {
    maxTokensPerExpert = Math.max(maxTokensPerExpert, tokenCounts[e]);
  }

  // Allocate output buffers
  const gatheredTokens = new Float32Array(numExperts * maxTokensPerExpert * hiddenSize);
  const tokenIndices = new Uint32Array(numExperts * maxTokensPerExpert);

  // Fill with invalid index
  tokenIndices.fill(0xFFFFFFFF);

  // Reset counts for second pass
  const currentCounts = new Uint32Array(numExperts);

  // Gather tokens by expert
  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      const slotIdx = currentCounts[expertIdx];

      // Copy token to expert's slot
      const srcOffset = t * hiddenSize;
      const dstOffset = expertIdx * maxTokensPerExpert * hiddenSize + slotIdx * hiddenSize;

      for (let d = 0; d < hiddenSize; d++) {
        gatheredTokens[dstOffset + d] = tokens[srcOffset + d];
      }

      // Record original token index
      tokenIndices[expertIdx * maxTokensPerExpert + slotIdx] = t;

      currentCounts[expertIdx]++;
    }
  }

  return {
    gatheredTokens,
    tokenCounts,
    tokenIndices,
    maxTokensPerExpert,
  };
}


export function moeComputeAssignmentsRef(expertIndices, numTokens, numExperts, topK) {
  const tokenCounts = new Uint32Array(numExperts);
  const expertOffsets = new Uint32Array(numExperts);

  // Count tokens per expert
  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      tokenCounts[expertIdx]++;
    }
  }

  // Compute prefix sum for offsets
  let offset = 0;
  for (let e = 0; e < numExperts; e++) {
    expertOffsets[e] = offset;
    offset += tokenCounts[e];
  }

  return { tokenCounts, expertOffsets, totalAssignments: offset };
}

export default moeGatherRef;
