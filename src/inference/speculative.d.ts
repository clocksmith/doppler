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
 * Speculative Decoding Configuration
 */
export interface SpeculativeConfig {
  /** Number of tokens to draft */
  numDraftTokens: number;
  /** Max retries after rejection */
  maxRejectionRetries: number;
  /** Use tree-based drafting (experimental) */
  enableTreeDraft: boolean;
  /** Temperature for draft sampling */
  temperature: number;
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
export declare class SpeculativeDecoder {
  private numDraftTokens;
  private maxRejectionRetries;
  private enableTreeDraft;
  private temperature;

  // Draft model reference (smaller/faster model)
  protected draftModel: DraftModel | null;
  // Main model reference (for verification)
  private mainModel;

  // Statistics
  private stats;

  constructor(config: SpeculativeConfig);

  /**
   * Set the draft model for speculation
   */
  setDraftModel(model: DraftModel): void;

  /**
   * Set the main model for verification
   */
  setMainModel(model: MainModel): void;

  /**
   * Generate draft tokens using the smaller model
   */
  generateDraftTokens(
    inputIds: number[],
    kvCache?: KVCache,
    numTokens?: number
  ): Promise<DraftResult>;

  /**
   * Sample a token from logits using temperature sampling
   */
  sampleToken(logits: Float32Array, temperature: number): SampleResult;

  /**
   * Compute log softmax for numerical stability
   */
  protected logSoftmax(logits: Float32Array): Float32Array;

  /**
   * Verify draft tokens against the main model
   * Uses parallel forward pass for efficiency
   */
  verifyDraftTokens(
    inputIds: number[],
    draftTokens: number[],
    draftLogprobs: Float32Array[],
    kvCache?: KVCache
  ): Promise<VerificationResult>;

  /**
   * Sample from residual distribution after rejection
   */
  private sampleFromResidual(
    mainLogits: Float32Array,
    draftLogprobs: Float32Array,
    wasRejected: boolean
  ): number;

  /**
   * Run one step of speculative decoding
   */
  step(
    inputIds: number[],
    mainKVCache?: KVCache,
    draftKVCache?: KVCache
  ): Promise<StepResult>;

  /**
   * Get speculative decoding statistics
   */
  getStats(): StatsWithSpeedup;

  /**
   * Estimate speedup from speculative decoding
   * Theoretical: (1 + α*k) where α is accept rate, k is num draft tokens
   */
  private estimateSpeedup(): number;

  /**
   * Reset statistics
   */
  resetStats(): void;
}

export default SpeculativeDecoder;
