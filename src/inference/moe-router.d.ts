/**
 * moe-router.ts - Mixture of Experts Router
 *
 * Implements top-k expert selection for Mixtral-style MoE models.
 * Handles gating network computation and expert selection.
 *
 * @module inference/moe-router
 */

/**
 * MoE Router Configuration (extended)
 */
export interface MoEConfig {
  /** Total number of experts (e.g., 8 for Mixtral) */
  numExperts: number;
  /** Number of experts to select per token (e.g., 2) */
  topK: number;
  /** Hidden dimension size */
  hiddenSize: number;
  /** Whether to renormalize weights after top-k */
  normalizeWeights?: boolean;
}

/**
 * Expert Selection Result for a single token
 */
export interface ExpertSelection {
  /** Selected expert indices */
  indices: number[];
  /** Corresponding weights for each selected expert */
  weights: Float32Array;
  /** Raw router logits (for auxiliary loss) */
  routerLogits: Float32Array;
}

/**
 * GPU context for computations
 */
interface GpuContext {
  device: GPUDevice;
}

/**
 * Load balancing statistics
 */
interface LoadBalanceStats {
  expertCounts: Uint32Array;
  totalTokens: number;
}

/**
 * Expert utilization stats
 */
export interface UtilizationStats {
  experts: Array<{
    index: number;
    count: number;
    percentage: number;
  }>;
  totalTokens: number;
  loadBalanceLoss: number;
}

export declare class MoERouter {
  numExperts: number;
  topK: number;
  hiddenSize: number;
  normalizeWeights: boolean;

  // Router gate weights (linear projection: hidden_size -> num_experts)
  // Will be loaded from model weights
  gateWeight: Float32Array | GPUBuffer | null;
  // Router bias (optional, used by GPT-OSS)
  gateBias: Float32Array | GPUBuffer | null;

  // Track active experts for the current batch
  activeExperts: Set<number>;

  // Auxiliary load balancing stats
  loadBalanceStats: LoadBalanceStats;

  constructor(config: MoEConfig);

  /**
   * Load router gate weights from model
   * @param weights - Gate weight matrix [hidden_size, num_experts]
   * @param bias - Optional gate bias vector [num_experts]
   */
  loadWeights(weights: Float32Array | GPUBuffer, bias?: Float32Array | GPUBuffer | null): void;

  /**
   * Compute router logits from hidden states (CPU fallback)
   * @param hiddenStates - Input tensor [batchSize * seqLen, hiddenSize]
   * @param numTokens - Number of tokens
   * @returns Router logits [numTokens, numExperts]
   */
  computeRouterLogitsCPU(hiddenStates: Float32Array, numTokens: number): Float32Array;

  /**
   * Compute router logits using GPU (when available)
   * @param hiddenStates - Input tensor on GPU [numTokens, hiddenSize]
   * @param numTokens - Number of tokens
   * @param gpuContext - GPU context (optional, uses global device if not provided)
   * @returns Router logits on GPU [numTokens, numExperts]
   */
  computeRouterLogitsGPU(
    hiddenStates: GPUBuffer,
    numTokens: number,
    gpuContext?: GpuContext | null
  ): Promise<GPUBuffer>;

  /**
   * Route tokens using GPU and read back results
   * @param hiddenStates - Hidden states on GPU
   * @param numTokens - Number of tokens
   * @returns Expert selections for each token
   */
  routeGPU(hiddenStates: GPUBuffer, numTokens: number): Promise<ExpertSelection[]>;

  /**
   * Apply softmax to logits
   * @param logits - Input logits
   * @param size - Size of softmax dimension
   * @returns Softmax probabilities
   */
  softmax(logits: Float32Array, size: number): Float32Array;

  /**
   * Select top-k experts for a single token
   * @param logits - Router logits for one token [numExperts]
   * @returns Selected experts with weights
   */
  selectExpertsForToken(logits: Float32Array): ExpertSelection;

  /**
   * Route a batch of tokens to experts
   * @param hiddenStates - Input hidden states [numTokens, hiddenSize]
   * @param numTokens - Number of tokens
   * @returns Expert selections for each token
   */
  route(hiddenStates: Float32Array, numTokens: number): ExpertSelection[];

  /**
   * Get currently active expert indices
   * @returns Array of active expert indices
   */
  getActiveExperts(): number[];

  /**
   * Compute auxiliary load balancing loss
   * Used during training to encourage balanced expert utilization.
   * @returns Load balancing loss value
   */
  computeLoadBalanceLoss(): number;

  /**
   * Reset load balancing statistics
   */
  resetStats(): void;

  /**
   * Get expert utilization statistics
   * @returns Utilization stats per expert
   */
  getUtilizationStats(): UtilizationStats;
}

/**
 * Expert execution plan entry
 */
interface ExpertExecutionPlanEntry {
  tokenIndices: number[];
  weights: Float32Array;
}

/**
 * Create a grouped expert execution plan
 * Groups tokens by their selected experts for efficient batched computation
 *
 * @param selections - Expert selections for all tokens
 * @param numExperts - Total number of experts
 * @returns Map of expert index to token indices and weights
 */
export declare function createExpertExecutionPlan(
  selections: ExpertSelection[],
  numExperts: number
): Map<number, ExpertExecutionPlanEntry>;

/**
 * Combine expert outputs with routing weights
 *
 * @param expertOutputs - Output from each expert [numTokens, hiddenSize]
 * @param selections - Original routing decisions
 * @param numTokens - Number of tokens
 * @param hiddenSize - Hidden dimension
 * @returns Combined output [numTokens, hiddenSize]
 */
export declare function combineExpertOutputs(
  expertOutputs: Map<number, Float32Array>,
  selections: ExpertSelection[],
  numTokens: number,
  hiddenSize: number
): Float32Array;

export default MoERouter;
