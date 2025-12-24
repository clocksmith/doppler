/**
 * moe-router.ts - Mixture of Experts Router
 *
 * Implements top-k expert selection for Mixtral-style MoE models.
 * Handles gating network computation and expert selection.
 *
 * @module inference/moe-router
 */

import { getDevice } from '../gpu/device.js';
import { runMatmul, runSoftmax } from '../gpu/kernel-selector.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../gpu/buffer-pool.js';
import type { ExpertPlan, RouterConfig } from '../types/inference.js';

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
 * Routing decision for a single token
 */
export interface RoutingDecision {
  tokenIdx: number;
  selectedExperts: number[];
  weights: number[];
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

export class MoERouter {
  numExperts: number;
  topK: number;
  hiddenSize: number;
  normalizeWeights: boolean;

  // Router gate weights (linear projection: hidden_size -> num_experts)
  // Will be loaded from model weights
  gateWeight: Float32Array | GPUBuffer | null = null;
  // Router bias (optional, used by GPT-OSS)
  gateBias: Float32Array | GPUBuffer | null = null;

  // Track active experts for the current batch
  activeExperts: Set<number>;

  // Auxiliary load balancing stats
  loadBalanceStats: LoadBalanceStats;

  // Cached GPU pipeline for router bias add
  private _biasAddPipeline: GPUComputePipeline | null = null;
  private _gateBiasGPU: GPUBuffer | null = null;
  private _gateWeightGPU: GPUBuffer | null = null;

  constructor(config: MoEConfig) {
    this.numExperts = config.numExperts || 8;
    this.topK = config.topK || 2;
    this.hiddenSize = config.hiddenSize || 4096;
    this.normalizeWeights = config.normalizeWeights !== false;

    // Track active experts for the current batch
    this.activeExperts = new Set<number>();

    // Auxiliary load balancing stats
    this.loadBalanceStats = {
      expertCounts: new Uint32Array(this.numExperts),
      totalTokens: 0
    };
  }

  /**
   * Load router gate weights from model
   * @param weights - Gate weight matrix [hidden_size, num_experts]
   * @param bias - Optional gate bias vector [num_experts]
   */
  loadWeights(weights: Float32Array | GPUBuffer, bias: Float32Array | GPUBuffer | null = null): void {
    this.gateWeight = weights;
    this.gateBias = bias;
    // Clear cached GPU uploads when swapping router parameters (e.g., per-layer routers).
    this._gateBiasGPU = null;
    this._gateWeightGPU = null;
  }

  /**
   * Compute router logits from hidden states (CPU fallback)
   * @param hiddenStates - Input tensor [batchSize * seqLen, hiddenSize]
   * @param numTokens - Number of tokens
   * @returns Router logits [numTokens, numExperts]
   */
  computeRouterLogitsCPU(hiddenStates: Float32Array, numTokens: number): Float32Array {
    if (!this.gateWeight) {
      throw new Error('Router gate weights not loaded');
    }

    if (this.gateWeight instanceof GPUBuffer) {
      throw new Error('Gate weights are on GPU, use computeRouterLogitsGPU instead');
    }

    const logits = new Float32Array(numTokens * this.numExperts);

    // Matrix multiply: hidden_states @ gate_weight
    // SafeTensors stores linear weights as [out, in] = [numExperts, hiddenSize].
    for (let t = 0; t < numTokens; t++) {
      for (let e = 0; e < this.numExperts; e++) {
        let sum = 0;
        for (let h = 0; h < this.hiddenSize; h++) {
          sum += hiddenStates[t * this.hiddenSize + h] *
                 this.gateWeight[e * this.hiddenSize + h];
        }
        // Add bias if present (GPT-OSS style)
        if (this.gateBias && this.gateBias instanceof Float32Array) {
          sum += this.gateBias[e];
        }
        logits[t * this.numExperts + e] = sum;
      }
    }

    return logits;
  }

  /**
   * Compute router logits using GPU (when available)
   * @param hiddenStates - Input tensor on GPU [numTokens, hiddenSize]
   * @param numTokens - Number of tokens
   * @param gpuContext - GPU context (optional, uses global device if not provided)
   * @returns Router logits on GPU [numTokens, numExperts]
   */
  async computeRouterLogitsGPU(
    hiddenStates: GPUBuffer,
    numTokens: number,
    gpuContext: GpuContext | null = null
  ): Promise<GPUBuffer> {
    const device = gpuContext?.device || getDevice();
    if (!device) {
      throw new Error('GPU device not available');
    }

    if (!this.gateWeight) {
      throw new Error('Router gate weights not loaded');
    }

    // Ensure gate weight is on GPU.
    // SafeTensors weights are [out, in] = [numExperts, hiddenSize], so we use transposeB.
    let gateWeightBuffer = this.gateWeight;
    if (!(gateWeightBuffer instanceof GPUBuffer)) {
      const uploaded = device.createBuffer({
        label: 'moe_gate_weight',
        size: gateWeightBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uploaded, 0, gateWeightBuffer as GPUAllowSharedBufferSource);
      this._gateWeightGPU = uploaded;
      this.gateWeight = uploaded;
      gateWeightBuffer = uploaded;
    }

    // Matrix multiply: hidden_states [numTokens, hiddenSize] @ gate_weight [hiddenSize, numExperts]
    // Result: [numTokens, numExperts]
    const logitsBuffer = await runMatmul(
      hiddenStates,
      gateWeightBuffer,
      numTokens,           // M
      this.numExperts,     // N
      this.hiddenSize,     // K
      {
        preferF16: false,  // Use F32 for routing precision
        transposeB: true,
      }
    );

    // Add bias on GPU if present (GPT-OSS style)
    if (this.gateBias) {
      const biasBuffer = await this._getGateBiasBuffer(device);
      await this._addBiasInPlace(logitsBuffer, biasBuffer, numTokens, device);
    }

    return logitsBuffer;
  }

  /**
   * Ensure router bias is available as a GPUBuffer.
   * @private
   */
  private async _getGateBiasBuffer(device: GPUDevice): Promise<GPUBuffer> {
    if (this.gateBias instanceof GPUBuffer) return this.gateBias;
    if (this._gateBiasGPU) return this._gateBiasGPU;

    if (!(this.gateBias instanceof Float32Array)) {
      throw new Error('Unsupported gateBias type');
    }

    const buf = device.createBuffer({
      label: 'moe_gate_bias',
      size: this.gateBias.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, this.gateBias as GPUAllowSharedBufferSource);
    this._gateBiasGPU = buf;
    return buf;
  }

  /**
   * Add per-expert bias to logits in-place.
   * logits layout: [numTokens, numExperts], bias layout: [numExperts]
   * @private
   */
  private async _addBiasInPlace(
    logits: GPUBuffer,
    bias: GPUBuffer,
    numTokens: number,
    device: GPUDevice
  ): Promise<void> {
    if (!this._biasAddPipeline) {
      const code = `
        struct Uniforms {
          numTokens: u32,
          numExperts: u32,
          _pad0: u32,
          _pad1: u32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read_write> logits: array<f32>;
        @group(0) @binding(2) var<storage, read> bias: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
          let idx = gid.x;
          let total = uniforms.numTokens * uniforms.numExperts;
          if (idx >= total) { return; }
          let e = idx % uniforms.numExperts;
          logits[idx] = logits[idx] + bias[e];
        }
      `;
      const module = device.createShaderModule({ code });
      this._biasAddPipeline = device.createComputePipeline({
        label: 'moe_router_bias_add',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }

    const uniformData = new ArrayBuffer(16);
    const uniformView = new DataView(uniformData);
    uniformView.setUint32(0, numTokens, true);
    uniformView.setUint32(4, this.numExperts, true);

    const uniformBuffer = device.createBuffer({
      label: 'moe_router_bias_uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: this._biasAddPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logits } },
        { binding: 2, resource: { buffer: bias } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'moe_router_bias_add_encoder' });
    const pass = encoder.beginComputePass({ label: 'moe_router_bias_add_pass' });
    pass.setPipeline(this._biasAddPipeline);
    pass.setBindGroup(0, bindGroup);
    const total = numTokens * this.numExperts;
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    device.queue.submit([encoder.finish()]);

    uniformBuffer.destroy();
  }

  /**
   * Route tokens using GPU and read back results
   * @param hiddenStates - Hidden states on GPU
   * @param numTokens - Number of tokens
   * @returns Expert selections for each token
   */
  async routeGPU(hiddenStates: GPUBuffer, numTokens: number): Promise<ExpertSelection[]> {
    // Compute router logits on GPU
    const logitsBuffer = await this.computeRouterLogitsGPU(hiddenStates, numTokens);

    // Read back logits to CPU for top-k selection
    // (GPU top-k is complex and not always faster for small numExperts)
    const logitsData = await readBuffer(logitsBuffer);
    const logits = new Float32Array(logitsData);

    const selections: ExpertSelection[] = [];
    this.activeExperts.clear();

    for (let t = 0; t < numTokens; t++) {
      const tokenLogits = logits.subarray(
        t * this.numExperts,
        (t + 1) * this.numExperts
      );

      const selection = this.selectExpertsForToken(tokenLogits);
      selections.push(selection);

      for (const idx of selection.indices) {
        this.activeExperts.add(idx);
        this.loadBalanceStats.expertCounts[idx]++;
      }
      this.loadBalanceStats.totalTokens++;
    }

    // Clean up logits buffer
    releaseBuffer(logitsBuffer);

    return selections;
  }

  /**
   * Apply softmax to logits
   * @param logits - Input logits
   * @param size - Size of softmax dimension
   * @returns Softmax probabilities
   */
  softmax(logits: Float32Array, size: number): Float32Array {
    const result = new Float32Array(size);

    // Find max for numerical stability
    let max = -Infinity;
    for (let i = 0; i < size; i++) {
      if (logits[i] > max) max = logits[i];
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < size; i++) {
      result[i] = Math.exp(logits[i] - max);
      sum += result[i];
    }

    // Normalize
    for (let i = 0; i < size; i++) {
      result[i] /= sum;
    }

    return result;
  }

  /**
   * Select top-k experts for a single token
   * @param logits - Router logits for one token [numExperts]
   * @returns Selected experts with weights
   */
  selectExpertsForToken(logits: Float32Array): ExpertSelection {
    // Apply softmax to get probabilities
    const probs = this.softmax(logits, this.numExperts);

    // Find top-k experts
    const indexed: Array<{ index: number; prob: number }> = [];
    for (let i = 0; i < this.numExperts; i++) {
      indexed.push({ index: i, prob: probs[i] });
    }
    indexed.sort((a, b) => b.prob - a.prob);

    const topKExperts = indexed.slice(0, this.topK);
    const indices = topKExperts.map(e => e.index);
    const weights = new Float32Array(topKExperts.map(e => e.prob));

    // Renormalize weights if configured
    if (this.normalizeWeights) {
      let weightSum = 0;
      for (let i = 0; i < this.topK; i++) {
        weightSum += weights[i];
      }
      for (let i = 0; i < this.topK; i++) {
        weights[i] /= weightSum;
      }
    }

    return {
      indices,
      weights,
      routerLogits: new Float32Array(logits)
    };
  }

  /**
   * Route a batch of tokens to experts
   * @param hiddenStates - Input hidden states [numTokens, hiddenSize]
   * @param numTokens - Number of tokens
   * @returns Expert selections for each token
   */
  route(hiddenStates: Float32Array, numTokens: number): ExpertSelection[] {
    // Compute router logits
    const allLogits = this.computeRouterLogitsCPU(hiddenStates, numTokens);

    const selections: ExpertSelection[] = [];
    this.activeExperts.clear();

    for (let t = 0; t < numTokens; t++) {
      // Extract logits for this token
      const tokenLogits = allLogits.subarray(
        t * this.numExperts,
        (t + 1) * this.numExperts
      );

      const selection = this.selectExpertsForToken(tokenLogits);
      selections.push(selection);

      // Track active experts
      for (const idx of selection.indices) {
        this.activeExperts.add(idx);
      }

      // Update load balance stats
      for (const idx of selection.indices) {
        this.loadBalanceStats.expertCounts[idx]++;
      }
      this.loadBalanceStats.totalTokens++;
    }

    return selections;
  }

  /**
   * Get currently active expert indices
   * @returns Array of active expert indices
   */
  getActiveExperts(): number[] {
    return Array.from(this.activeExperts).sort((a, b) => a - b);
  }

  /**
   * Compute auxiliary load balancing loss
   * Used during training to encourage balanced expert utilization.
   * @returns Load balancing loss value
   */
  computeLoadBalanceLoss(): number {
    if (this.loadBalanceStats.totalTokens === 0) return 0;

    const numTokens = this.loadBalanceStats.totalTokens;
    const expertProbs = new Float32Array(this.numExperts);

    // Compute fraction of tokens routed to each expert
    for (let i = 0; i < this.numExperts; i++) {
      expertProbs[i] = this.loadBalanceStats.expertCounts[i] / numTokens;
    }

    // Load balance loss: sum of (expert_prob * expert_fraction)
    // Ideally each expert gets 1/numExperts of tokens
    let loss = 0;
    const idealFraction = 1 / this.numExperts;
    for (let i = 0; i < this.numExperts; i++) {
      // Squared deviation from ideal
      const deviation = expertProbs[i] - idealFraction;
      loss += deviation * deviation;
    }

    return loss * this.numExperts;
  }

  /**
   * Reset load balancing statistics
   */
  resetStats(): void {
    this.loadBalanceStats.expertCounts.fill(0);
    this.loadBalanceStats.totalTokens = 0;
    this.activeExperts.clear();
  }

  /**
   * Get expert utilization statistics
   * @returns Utilization stats per expert
   */
  getUtilizationStats(): UtilizationStats {
    const total = this.loadBalanceStats.totalTokens;
    if (total === 0) {
      return { experts: [], totalTokens: 0, loadBalanceLoss: 0 };
    }

    const experts: Array<{ index: number; count: number; percentage: number }> = [];
    for (let i = 0; i < this.numExperts; i++) {
      experts.push({
        index: i,
        count: this.loadBalanceStats.expertCounts[i],
        percentage: (this.loadBalanceStats.expertCounts[i] / total) * 100
      });
    }

    return {
      experts,
      totalTokens: total,
      loadBalanceLoss: this.computeLoadBalanceLoss()
    };
  }
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
export function createExpertExecutionPlan(
  selections: ExpertSelection[],
  numExperts: number
): Map<number, ExpertExecutionPlanEntry> {
  const plan = new Map<number, ExpertExecutionPlanEntry>();

  // Initialize empty plans for each expert
  for (let e = 0; e < numExperts; e++) {
    plan.set(e, { tokenIndices: [], weights: [] as unknown as Float32Array });
  }

  // Group tokens by expert
  for (let t = 0; t < selections.length; t++) {
    const sel = selections[t];
    for (let k = 0; k < sel.indices.length; k++) {
      const expertIdx = sel.indices[k];
      const weight = sel.weights[k];
      const entry = plan.get(expertIdx)!;
      entry.tokenIndices.push(t);
      (entry.weights as unknown as number[]).push(weight);
    }
  }

  // Convert weight arrays to Float32Array
  for (const [expertIdx, data] of plan) {
    plan.set(expertIdx, {
      tokenIndices: data.tokenIndices,
      weights: new Float32Array(data.weights as unknown as number[])
    });
  }

  return plan;
}

/**
 * Combine expert outputs with routing weights
 *
 * @param expertOutputs - Output from each expert [numTokens, hiddenSize]
 * @param selections - Original routing decisions
 * @param numTokens - Number of tokens
 * @param hiddenSize - Hidden dimension
 * @returns Combined output [numTokens, hiddenSize]
 */
export function combineExpertOutputs(
  expertOutputs: Map<number, Float32Array>,
  selections: ExpertSelection[],
  numTokens: number,
  hiddenSize: number
): Float32Array {
  const output = new Float32Array(numTokens * hiddenSize);

  for (let t = 0; t < numTokens; t++) {
    const sel = selections[t];

    for (let k = 0; k < sel.indices.length; k++) {
      const expertIdx = sel.indices[k];
      const weight = sel.weights[k];
      const expertOut = expertOutputs.get(expertIdx);

      if (!expertOut) continue;

      // Weighted sum: output += weight * expert_output
      for (let h = 0; h < hiddenSize; h++) {
        output[t * hiddenSize + h] += weight * expertOut[t * hiddenSize + h];
      }
    }
  }

  return output;
}

export default MoERouter;
