/**
 * speculative.ts - Speculative Decoding
 *
 * Implements speculative decoding for faster inference.
 * Uses a draft model to generate candidate tokens, then verifies
 * them in parallel with the main model.
 *
 * Based on: "Fast Inference from Transformers via Speculative Decoding"
 * (Leviathan et al., 2022)
 *
 * @module inference/speculative
 */

/**
 * Draft Model Interface
 * Smaller/faster model used to generate candidate tokens
 */
export interface DraftModel {
  /**
   * Forward pass through the model
   * @param inputIds - Token sequence
   * @param kvCache - KV cache state
   * @returns Logits and updated KV cache
   */
  forward(
    inputIds: number[],
    kvCache?: KVCache
  ): Promise<{ logits: Float32Array; newKVCache?: KVCache }>;
}

/**
 * Main Model Interface
 * Used for verification of draft tokens
 */
export interface MainModel {
  /**
   * Forward pass through the model
   * @param inputIds - Token sequence
   * @param kvCache - KV cache state
   * @returns Logits and updated KV cache
   */
  forward(
    inputIds: number[],
    kvCache?: KVCache
  ): Promise<{ logits: Float32Array; newKVCache?: KVCache }>;
}

/**
 * KV Cache Interface
 * Represents key-value cache state - compatible with KVCache class from kv-cache.ts
 */
export interface KVCache {
  clone?(): KVCache;
  currentSeqLen?: number;
}

/**
 * Speculative Decoding Policy Configuration
 */
export interface SpeculativePolicy {
  /** Number of tokens to draft in each iteration */
  draftTokens: number;
  /** Acceptance threshold for draft tokens (0-1) */
  acceptanceThreshold: number;
  /** Whether to fallback to greedy sampling on rejection */
  fallbackToGreedy: boolean;
}

/**
 * Speculative Decoding Configuration
 */
export interface SpeculativeConfig {
  /** Number of tokens to draft (default: 5) */
  numDraftTokens?: number;
  /** Max retries after rejection (default: 3) */
  maxRejectionRetries?: number;
  /** Use tree-based drafting (experimental) */
  enableTreeDraft?: boolean;
}

/**
 * Verification Result
 * Results from verifying draft tokens against main model
 */
export interface VerificationResult {
  /** Number of accepted draft tokens */
  acceptedCount: number;
  /** The accepted token IDs */
  acceptedTokens: number[];
  /** Token sampled from corrected distribution */
  sampledToken: number;
  /** Whether all draft tokens were accepted */
  allAccepted: boolean;
}

/**
 * Token Sampling Result
 */
interface SampleResult {
  /** Sampled token ID */
  token: number;
  /** Log probabilities for entire vocabulary */
  logprob: Float32Array;
}

/**
 * Draft Generation Result
 */
interface DraftResult {
  /** Generated draft tokens */
  tokens: number[];
  /** Log probabilities for each drafted token */
  logprobs: Float32Array[];
}

/**
 * Step Result
 * Results from one speculative decoding step
 */
export interface StepResult {
  /** New tokens generated (accepted + sampled) */
  newTokens: number[];
  /** Updated main model KV cache */
  mainKVCache?: KVCache;
  /** Acceptance rate for this step */
  acceptRate: number;
}

/**
 * Decoding Statistics
 */
export interface DecodingStats {
  /** Total number of tokens drafted */
  totalDrafted: number;
  /** Total number of tokens accepted */
  totalAccepted: number;
  /** Total number of tokens rejected */
  totalRejected: number;
  /** Average acceptance rate */
  averageAcceptRate: number;
}

/**
 * Statistics with Speedup Estimate
 */
export interface StatsWithSpeedup extends DecodingStats {
  /** Estimated speedup factor */
  speedup: number;
}

/**
 * Speculative Decoder
 * Implements speculative decoding for faster inference
 */
export class SpeculativeDecoder {
  private numDraftTokens: number;
  private maxRejectionRetries: number;
  private enableTreeDraft: boolean;

  // Draft model reference (smaller/faster model)
  protected draftModel: DraftModel | null = null;
  // Main model reference (for verification)
  private mainModel: MainModel | null = null;

  // Statistics
  private stats: DecodingStats = {
    totalDrafted: 0,
    totalAccepted: 0,
    totalRejected: 0,
    averageAcceptRate: 0,
  };

  constructor(config: SpeculativeConfig = {}) {
    this.numDraftTokens = config.numDraftTokens ?? 5;
    this.maxRejectionRetries = config.maxRejectionRetries ?? 3;
    this.enableTreeDraft = config.enableTreeDraft ?? false;
  }

  /**
   * Set the draft model for speculation
   */
  setDraftModel(model: DraftModel): void {
    this.draftModel = model;
  }

  /**
   * Set the main model for verification
   */
  setMainModel(model: MainModel): void {
    this.mainModel = model;
  }

  /**
   * Generate draft tokens using the smaller model
   */
  async generateDraftTokens(
    inputIds: number[],
    kvCache?: KVCache,
    numTokens: number = this.numDraftTokens
  ): Promise<DraftResult> {
    if (!this.draftModel) {
      throw new Error('Draft model not set');
    }

    const draftTokens: number[] = [];
    const draftLogprobs: Float32Array[] = [];

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

  /**
   * Sample a token from logits using temperature sampling
   */
  sampleToken(logits: Float32Array, temperature: number = 1.0): SampleResult {
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

  /**
   * Compute log softmax for numerical stability
   */
  protected logSoftmax(logits: Float32Array): Float32Array {
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

  /**
   * Verify draft tokens against the main model
   * Uses parallel forward pass for efficiency
   */
  async verifyDraftTokens(
    inputIds: number[],
    draftTokens: number[],
    draftLogprobs: Float32Array[],
    kvCache?: KVCache
  ): Promise<VerificationResult> {
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
    const acceptedTokens: number[] = [];
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

  /**
   * Sample from residual distribution after rejection
   */
  private sampleFromResidual(
    mainLogits: Float32Array,
    draftLogprobs: Float32Array,
    wasRejected: boolean
  ): number {
    if (!wasRejected) {
      // All accepted - sample normally from main model
      return this.sampleToken(mainLogits).token;
    }

    const vocabSize = mainLogits.length;
    const mainProbs = new Float32Array(vocabSize);
    const draftProbs = new Float32Array(vocabSize);

    // Convert to probabilities
    const mainLogprobs = this.logSoftmax(mainLogits);
    for (let i = 0; i < vocabSize; i++) {
      mainProbs[i] = Math.exp(mainLogprobs[i]);
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

  /**
   * Run one step of speculative decoding
   */
  async step(
    inputIds: number[],
    mainKVCache?: KVCache,
    draftKVCache?: KVCache
  ): Promise<StepResult> {
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

  /**
   * Get speculative decoding statistics
   */
  getStats(): StatsWithSpeedup {
    return {
      ...this.stats,
      speedup: this.estimateSpeedup(),
    };
  }

  /**
   * Estimate speedup from speculative decoding
   * Theoretical: (1 + α*k) where α is accept rate, k is num draft tokens
   */
  private estimateSpeedup(): number {
    if (this.stats.totalDrafted === 0) return 1.0;

    const α = this.stats.averageAcceptRate;
    const k = this.numDraftTokens;

    // Account for draft model overhead (assume ~0.1x main model cost)
    const draftOverhead = 0.1 * k;

    // Effective tokens per main model call
    const tokensPerCall = 1 + α * k;

    return tokensPerCall / (1 + draftOverhead);
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      totalDrafted: 0,
      totalAccepted: 0,
      totalRejected: 0,
      averageAcceptRate: 0,
    };
  }
}

/**
 * Tree Configuration for Tree-based Speculative Decoding
 */
export interface TreeConfig extends SpeculativeConfig {
  /** Number of branches per node */
  branchFactor?: number;
  /** Maximum depth of the tree */
  maxDepth?: number;
}

/**
 * Tree Node for Draft Tree
 */
interface TreeNode {
  /** Token ID (null for root) */
  token: number | null;
  /** Child nodes */
  children: TreeNode[];
  /** Cumulative log probability */
  logprob: number;
  /** Depth in the tree */
  depth: number;
}

/**
 * Top-K Token Result
 */
interface TopKToken {
  /** Token ID */
  token: number;
  /** Log probability */
  logprob: number;
}

/**
 * Tree-based speculative decoding (experimental)
 * Generates multiple candidate paths and verifies in parallel
 */
export class TreeSpeculativeDecoder extends SpeculativeDecoder {
  private branchFactor: number;
  private maxDepth: number;

  constructor(config: TreeConfig = {}) {
    super(config);
    this.branchFactor = config.branchFactor ?? 2;
    this.maxDepth = config.maxDepth ?? 3;
  }

  /**
   * Generate tree of draft tokens
   */
  async generateDraftTree(inputIds: number[], kvCache?: KVCache): Promise<TreeNode> {
    // Build tree structure with multiple branches
    const root: TreeNode = { token: null, children: [], logprob: 0, depth: 0 };

    const buildTree = async (
      node: TreeNode,
      ids: number[],
      depth: number
    ): Promise<void> => {
      if (depth >= this.maxDepth) return;

      if (!this.draftModel) {
        throw new Error('Draft model not set');
      }

      const { logits } = await this.draftModel.forward(ids, kvCache);
      const logprobs = this.logSoftmax(logits);

      // Get top-k tokens as branches
      const topK = this.getTopK(logprobs, this.branchFactor);

      for (const { token, logprob } of topK) {
        const child: TreeNode = {
          token,
          logprob: node.logprob + logprob,
          children: [],
          depth: depth + 1,
        };
        node.children.push(child);

        // Recursively build subtree
        await buildTree(child, [...ids, token], depth + 1);
      }
    };

    await buildTree(root, inputIds, 0);
    return root;
  }

  /**
   * Get top-k tokens from log probabilities
   */
  private getTopK(logprobs: Float32Array, k: number): TopKToken[] {
    const indexed: TopKToken[] = [];
    for (let i = 0; i < logprobs.length; i++) {
      indexed.push({ token: i, logprob: logprobs[i] });
    }
    indexed.sort((a, b) => b.logprob - a.logprob);
    return indexed.slice(0, k);
  }
}

export default SpeculativeDecoder;
