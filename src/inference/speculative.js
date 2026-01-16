








export class SpeculativeDecoder {
  
  numDraftTokens;

  
  maxRejectionRetries;

  
  enableTreeDraft;

  
  draftModel = null;

  
  mainModel = null;

  
  stats = {
    totalDrafted: 0,
    totalAccepted: 0,
    totalRejected: 0,
    averageAcceptRate: 0,
  };

  
  constructor(config = {}) {
    this.numDraftTokens = config.numDraftTokens ?? 5;
    this.maxRejectionRetries = config.maxRejectionRetries ?? 3;
    this.enableTreeDraft = config.enableTreeDraft ?? false;
  }

  
  setDraftModel(model) {
    this.draftModel = model;
  }

  
  setMainModel(model) {
    this.mainModel = model;
  }

  
  async generateDraftTokens(inputIds, kvCache, numTokens = this.numDraftTokens) {
    if (!this.draftModel) {
      throw new Error('Draft model not set');
    }

    
    const draftTokens = [];
    
    const draftLogprobs = [];

    // Clone KV cache for draft generation (don't pollute main cache)
    const draftKVCache = kvCache?.clone?.() ?? kvCache;
    let currentIds = [...inputIds];

    for (let i = 0; i < numTokens; i++) {
      // Forward pass through draft model
      const { logits, newKVCache } = await this.draftModel.forward(
        currentIds,
        draftKVCache
      );

      // Sample next token
      const { token, logprob } = this.sampleToken(logits);
      draftTokens.push(token);
      draftLogprobs.push(logprob);

      // Append for next iteration
      currentIds = [...currentIds, token];
    }

    return {
      tokens: draftTokens,
      logprobs: draftLogprobs,
    };
  }

  
  sampleToken(logits, temperature = 1.0) {
    const vocabSize = logits.length;

    // Apply temperature
    const scaledLogits = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      scaledLogits[i] = logits[i] / temperature;
    }

    // Compute softmax (log probabilities)
    const logprobs = this.logSoftmax(scaledLogits);
    const probs = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      probs[i] = Math.exp(logprobs[i]);
    }

    // Sample from distribution
    const r = Math.random();
    let cumSum = 0;
    for (let i = 0; i < vocabSize; i++) {
      cumSum += probs[i];
      if (r < cumSum) {
        return { token: i, logprob: logprobs };
      }
    }

    // Fallback to last token
    return { token: vocabSize - 1, logprob: logprobs };
  }

  
  logSoftmax(logits) {
    const n = logits.length;
    const result = new Float32Array(n);

    // Find max for stability
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      if (logits[i] > max) max = logits[i];
    }

    // Compute log-sum-exp
    let sumExp = 0;
    for (let i = 0; i < n; i++) {
      sumExp += Math.exp(logits[i] - max);
    }
    const logSumExp = max + Math.log(sumExp);

    // Compute log probabilities
    for (let i = 0; i < n; i++) {
      result[i] = logits[i] - logSumExp;
    }

    return result;
  }

  
  async verifyDraftTokens(inputIds, draftTokens, draftLogprobs, kvCache) {
    if (!this.mainModel) {
      throw new Error('Main model not set');
    }

    // Concatenate input + draft tokens for parallel verification
    const fullSequence = [...inputIds, ...draftTokens];

    // Forward pass through main model (processes all positions at once)
    const { logits: mainLogits } = await this.mainModel.forward(
      fullSequence,
      kvCache
    );

    const numDraft = draftTokens.length;
    
    const acceptedTokens = [];
    let acceptedCount = 0;

    // Verify each draft token using rejection sampling
    for (let i = 0; i < numDraft; i++) {
      const draftToken = draftTokens[i];
      const draftLogprob = draftLogprobs[i][draftToken];

      // Get main model's logprob for this position
      const mainLogprob = this.logSoftmax(
        mainLogits.subarray(
          (i * mainLogits.length) / numDraft,
          ((i + 1) * mainLogits.length) / numDraft
        )
      );
      const mainTokenLogprob = mainLogprob[draftToken];

      // Acceptance probability: min(1, p_main / p_draft)
      const acceptProb = Math.min(1, Math.exp(mainTokenLogprob - draftLogprob));

      if (Math.random() < acceptProb) {
        // Accept this token
        acceptedTokens.push(draftToken);
        acceptedCount++;
      } else {
        // Reject - stop here and sample from adjusted distribution
        break;
      }
    }

    // Sample final token from the residual distribution
    const sampledToken = this.sampleFromResidual(
      mainLogits,
      draftLogprobs[acceptedCount] ?? new Float32Array(mainLogits.length),
      acceptedCount < numDraft
    );

    // Update statistics
    this.stats.totalDrafted += numDraft;
    this.stats.totalAccepted += acceptedCount;
    this.stats.totalRejected += numDraft - acceptedCount;
    this.stats.averageAcceptRate =
      this.stats.totalAccepted / this.stats.totalDrafted;

    return {
      acceptedCount,
      acceptedTokens,
      sampledToken,
      allAccepted: acceptedCount === numDraft,
    };
  }

  
  sampleFromResidual(mainLogits, draftLogprobs, wasRejected) {
    if (!wasRejected) {
      // All accepted - sample normally from main model
      return this.sampleToken(mainLogits).token;
    }

    const vocabSize = mainLogits.length;
    const mainProbs = new Float32Array(vocabSize);
    const draftProbs = new Float32Array(vocabSize);

    // Convert to probabilities
    const mainLogprobsArr = this.logSoftmax(mainLogits);
    for (let i = 0; i < vocabSize; i++) {
      mainProbs[i] = Math.exp(mainLogprobsArr[i]);
      draftProbs[i] = Math.exp(draftLogprobs[i] ?? -Infinity);
    }

    // Compute residual: max(0, p_main - p_draft)
    const residual = new Float32Array(vocabSize);
    let residualSum = 0;
    for (let i = 0; i < vocabSize; i++) {
      residual[i] = Math.max(0, mainProbs[i] - draftProbs[i]);
      residualSum += residual[i];
    }

    // Normalize and sample
    if (residualSum > 0) {
      const r = Math.random() * residualSum;
      let cumSum = 0;
      for (let i = 0; i < vocabSize; i++) {
        cumSum += residual[i];
        if (r < cumSum) {
          return i;
        }
      }
    }

    // Fallback: sample from main distribution
    return this.sampleToken(mainLogits).token;
  }

  
  async step(inputIds, mainKVCache, draftKVCache) {
    // Generate draft tokens
    const { tokens: draftTokens, logprobs: draftLogprobs } =
      await this.generateDraftTokens(inputIds, draftKVCache);

    // Verify against main model
    const result = await this.verifyDraftTokens(
      inputIds,
      draftTokens,
      draftLogprobs,
      mainKVCache
    );

    // Combine accepted tokens + sampled token
    const newTokens = [...result.acceptedTokens, result.sampledToken];

    return {
      newTokens,
      mainKVCache,
      acceptRate: result.acceptedCount / draftTokens.length,
    };
  }

  
  getStats() {
    return {
      ...this.stats,
      speedup: this.estimateSpeedup(),
    };
  }

  
  estimateSpeedup() {
    if (this.stats.totalDrafted === 0) return 1.0;

    const acceptRate = this.stats.averageAcceptRate;
    const k = this.numDraftTokens;

    // Account for draft model overhead (assume ~0.1x main model cost)
    const draftOverhead = 0.1 * k;

    // Effective tokens per main model call
    const tokensPerCall = 1 + acceptRate * k;

    return tokensPerCall / (1 + draftOverhead);
  }

  
  resetStats() {
    this.stats = {
      totalDrafted: 0,
      totalAccepted: 0,
      totalRejected: 0,
      averageAcceptRate: 0,
    };
  }
}

export default SpeculativeDecoder;
