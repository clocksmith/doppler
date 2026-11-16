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

interface RouterLogitsOptions {
  kernelPath?: import("../config/schema/execution/kernel-path.schema.js").KernelPathSchema | null;
  phase?: 'prefill' | 'decode';
  layerIdx?: number;
  outputBuffer?: GPUBuffer | null;
  recorder?: import('../gpu/command-recorder.js').CommandRecorder | null;
  inputDtype?: 'f16' | 'f32';
  outputDtype?: 'f16' | 'f32';
}

type RouterVector = Float32Array | GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer;

export declare class MoERouter {
  numExperts: number;
  topK: number;
  hiddenSize: number;
  normalizeWeights: boolean;

  // Router gate weights (linear projection: hidden_size -> num_experts)
  // Will be loaded from model weights
  gateWeight: Float32Array | GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | null;
  // Router bias (optional, used by GPT-OSS)
  gateBias: Float32Array | GPUBuffer | null;
  // Optional DiffusionGemma router input scale
  gateScale: RouterVector | null;
  // Optional DiffusionGemma per-expert output scale
  perExpertScale: RouterVector | null;

  lastLogitsDtype: 'f16' | 'f32';

  constructor(config: MoEConfig);

  /**
   * Load router gate weights from model
   * @param weights - Gate weight matrix [hidden_size, num_experts]
   * @param bias - Optional gate bias vector [num_experts]
   * @param scale - Optional router input scale vector [hidden_size]
   * @param perExpertScale - Optional selected-expert weight scale [num_experts]
   */
  loadWeights(
    weights: Float32Array | GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer,
    bias?: Float32Array | GPUBuffer | null,
    scale?: RouterVector | null,
    perExpertScale?: RouterVector | null
  ): void;

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
    gpuContext?: GpuContext | null,
    options?: RouterLogitsOptions
  ): Promise<GPUBuffer>;

}

export default MoERouter;
