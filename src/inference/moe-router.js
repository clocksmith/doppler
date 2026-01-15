/**
 * moe-router.ts - Mixture of Experts Router
 *
 * Implements top-k expert selection for Mixtral-style MoE models.
 * Handles gating network computation and expert selection.
 *
 * @module inference/moe-router
 */

import { getDevice } from '../gpu/device.js';
import { getWeightDtype, isWeightBuffer } from '../gpu/weight-buffer.js';
import { runMatmul, runSoftmax } from '../gpu/kernel-selector.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../gpu/buffer-pool.js';
import { createTensor } from '../gpu/tensor.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { f16ToF32Array } from './kv-cache/types.js';

/**
 * @typedef {import('./moe-router.js').MoEConfig} MoEConfig
 * @typedef {import('./moe-router.js').ExpertSelection} ExpertSelection
 * @typedef {import('./moe-router.js').UtilizationStats} UtilizationStats
 */

/**
 * @typedef {Object} GpuContext
 * @property {GPUDevice} device
 */

/**
 * @typedef {Object} LoadBalanceStats
 * @property {Uint32Array} expertCounts
 * @property {number} totalTokens
 */

/**
 * @typedef {Object} ExpertExecutionPlanEntry
 * @property {number[]} tokenIndices
 * @property {Float32Array} weights
 */

export class MoERouter {
  /** @type {number} */
  numExperts;

  /** @type {number} */
  topK;

  /** @type {number} */
  hiddenSize;

  /** @type {boolean} */
  normalizeWeights;

  /** @type {Float32Array | GPUBuffer | null} */
  gateWeight = null;

  /** @type {Float32Array | GPUBuffer | null} */
  gateBias = null;

  /** @type {Set<number>} */
  activeExperts;

  /** @type {LoadBalanceStats} */
  loadBalanceStats;

  /** @type {Map<string, GPUComputePipeline>} */
  _biasAddPipelines = new Map();

  /** @type {GPUBuffer | null} */
  _gateBiasGPU = null;

  /** @type {GPUBuffer | null} */
  _gateWeightGPU = null;

  /** @type {'f16' | 'f32'} */
  lastLogitsDtype = 'f32';

  /**
   * @param {MoEConfig} config
   */
  constructor(config) {
    const runtimeDefaults = getRuntimeConfig().inference.moe.routing;
    this.numExperts = config.numExperts || runtimeDefaults.numExperts;
    this.topK = config.topK || runtimeDefaults.topK;
    this.hiddenSize = config.hiddenSize || 4096;  // hiddenSize is model-specific, no global default
    this.normalizeWeights = config.normalizeWeights ?? runtimeDefaults.normalizeWeights;

    // Track active experts for the current batch
    this.activeExperts = new Set();

    // Auxiliary load balancing stats
    this.loadBalanceStats = {
      expertCounts: new Uint32Array(this.numExperts),
      totalTokens: 0
    };
  }

  /**
   * Load router gate weights from model
   * @param {Float32Array | GPUBuffer} weights - Gate weight matrix [hidden_size, num_experts]
   * @param {Float32Array | GPUBuffer | null} [bias=null] - Optional gate bias vector [num_experts]
   * @returns {void}
   */
  loadWeights(weights, bias = null) {
    this.gateWeight = weights;
    this.gateBias = bias;
    // Clear cached GPU uploads when swapping router parameters (e.g., per-layer routers).
    this._gateBiasGPU = null;
    this._gateWeightGPU = null;
  }

  /**
   * Compute router logits from hidden states (CPU fallback)
   * @param {Float32Array} hiddenStates - Input tensor [batchSize * seqLen, hiddenSize]
   * @param {number} numTokens - Number of tokens
   * @returns {Float32Array} Router logits [numTokens, numExperts]
   */
  computeRouterLogitsCPU(hiddenStates, numTokens) {
    if (!this.gateWeight) {
      throw new Error('Router gate weights not loaded');
    }

    if (this.gateWeight instanceof GPUBuffer || isWeightBuffer(this.gateWeight)) {
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
   * @param {GPUBuffer} hiddenStates - Input tensor on GPU [numTokens, hiddenSize]
   * @param {number} numTokens - Number of tokens
   * @param {GpuContext | null} [gpuContext=null] - GPU context (optional, uses global device if not provided)
   * @returns {Promise<GPUBuffer>} Router logits on GPU [numTokens, numExperts]
   */
  async computeRouterLogitsGPU(hiddenStates, numTokens, gpuContext = null, options = {}) {
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
    if (!gateWeightBuffer) {
      throw new Error('Router gate weights not loaded');
    }
    if (!isWeightBuffer(gateWeightBuffer) && !(gateWeightBuffer instanceof GPUBuffer)) {
      const uploaded = device.createBuffer({
        label: 'moe_gate_weight',
        size: gateWeightBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uploaded, 0, /** @type {GPUAllowSharedBufferSource} */ (gateWeightBuffer));
      this._gateWeightGPU = uploaded;
      this.gateWeight = uploaded;
      gateWeightBuffer = uploaded;
    }

    const { inputDtype = 'f32', outputDtype = 'f32' } = options;
    const gateWeightDtype = isWeightBuffer(gateWeightBuffer) ? getWeightDtype(gateWeightBuffer) : null;

    // Matrix multiply: hidden_states [numTokens, hiddenSize] @ gate_weight [hiddenSize, numExperts]
    // Result: [numTokens, numExperts]
    const hiddenStatesTensor = createTensor(hiddenStates, inputDtype, [numTokens, this.hiddenSize], 'moe_hidden_states');
    const logitsTensor = await runMatmul(
      hiddenStatesTensor,
      gateWeightBuffer,
      numTokens,           // M
      this.numExperts,     // N
      this.hiddenSize,     // K
      {
        preferF16: outputDtype === 'f16',
        transposeB: true,
        outputDtype,
        bDtype: gateWeightDtype ?? undefined,
        role: 'moe_router',
      }
    );

    // Add bias on GPU if present (GPT-OSS style)
    if (this.gateBias) {
      const biasBuffer = await this._getGateBiasBuffer(device);
      const biasDtype = this._inferBiasDtype(biasBuffer);
      await this._addBiasInPlace(logitsTensor.buffer, biasBuffer, numTokens, device, logitsTensor.dtype, biasDtype);
    }

    this.lastLogitsDtype = logitsTensor.dtype;
    return logitsTensor.buffer;
  }

  /**
   * Ensure router bias is available as a GPUBuffer.
   * @param {GPUDevice} device
   * @returns {Promise<GPUBuffer>}
   * @private
   */
  async _getGateBiasBuffer(device) {
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
    device.queue.writeBuffer(buf, 0, /** @type {GPUAllowSharedBufferSource} */ (this.gateBias));
    this._gateBiasGPU = buf;
    return buf;
  }

  /**
   * Add per-expert bias to logits in-place.
   * logits layout: [numTokens, numExperts], bias layout: [numExperts]
   * @param {GPUBuffer} logits
   * @param {GPUBuffer} bias
   * @param {number} numTokens
   * @param {GPUDevice} device
   * @returns {Promise<void>}
   * @private
   */
  _inferBiasDtype(bias) {
    if (bias instanceof Float32Array) return 'f32';
    if (bias instanceof GPUBuffer) {
      const bytesPerElement = Math.round(bias.size / this.numExperts);
      return bytesPerElement <= 2 ? 'f16' : 'f32';
    }
    return 'f32';
  }

  _getBiasAddPipeline(logitsDtype, biasDtype, device) {
    const key = `${logitsDtype}_${biasDtype}`;
    const cached = this._biasAddPipelines.get(key);
    if (cached) return cached;

    const useF16 = logitsDtype === 'f16' || biasDtype === 'f16';
    const logitsType = logitsDtype === 'f16' ? 'f16' : 'f32';
    const biasType = biasDtype === 'f16' ? 'f16' : 'f32';
    const logitsRead = logitsDtype === 'f16' ? 'f32(logits[idx])' : 'logits[idx]';
    const biasRead = biasDtype === 'f16' ? 'f32(bias[e])' : 'bias[e]';
    const logitsWrite = logitsDtype === 'f16' ? 'logits[idx] = f16(value);' : 'logits[idx] = value;';
    const enableF16 = useF16 ? 'enable f16;\n' : '';

    const code = `
        ${enableF16}
        struct Uniforms {
          numTokens: u32,
          numExperts: u32,
          _pad0: u32,
          _pad1: u32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read_write> logits: array<${logitsType}>;
        @group(0) @binding(2) var<storage, read> bias: array<${biasType}>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
          let idx = gid.x;
          let total = uniforms.numTokens * uniforms.numExperts;
          if (idx >= total) { return; }
          let e = idx % uniforms.numExperts;
          let value = ${logitsRead} + ${biasRead};
          ${logitsWrite}
        }
      `;
    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({
      label: `moe_router_bias_add_${key}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this._biasAddPipelines.set(key, pipeline);
    return pipeline;
  }

  async _addBiasInPlace(logits, bias, numTokens, device, logitsDtype, biasDtype) {
    const pipeline = this._getBiasAddPipeline(logitsDtype, biasDtype, device);

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
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logits } },
        { binding: 2, resource: { buffer: bias } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'moe_router_bias_add_encoder' });
    const pass = encoder.beginComputePass({ label: 'moe_router_bias_add_pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const total = numTokens * this.numExperts;
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    device.queue.submit([encoder.finish()]);

    uniformBuffer.destroy();
  }

  /**
   * Route tokens using GPU and read back results
   * @param {GPUBuffer} hiddenStates - Hidden states on GPU
   * @param {number} numTokens - Number of tokens
   * @returns {Promise<ExpertSelection[]>} Expert selections for each token
   */
  async routeGPU(hiddenStates, numTokens) {
    // Compute router logits on GPU
    const logitsBuffer = await this.computeRouterLogitsGPU(hiddenStates, numTokens);

    // Read back logits to CPU for top-k selection
    // (GPU top-k is complex and not always faster for small numExperts)
    const logitsData = await readBuffer(logitsBuffer);
    const logits = this.lastLogitsDtype === 'f16'
      ? f16ToF32Array(new Uint16Array(logitsData))
      : new Float32Array(logitsData);

    /** @type {ExpertSelection[]} */
    const selections = [];
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
   * @param {Float32Array} logits - Input logits
   * @param {number} size - Size of softmax dimension
   * @returns {Float32Array} Softmax probabilities
   */
  softmax(logits, size) {
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
   * @param {Float32Array} logits - Router logits for one token [numExperts]
   * @returns {ExpertSelection} Selected experts with weights
   */
  selectExpertsForToken(logits) {
    // Apply softmax to get probabilities
    const probs = this.softmax(logits, this.numExperts);

    // Find top-k experts
    /** @type {Array<{ index: number; prob: number }>} */
    const indexed = [];
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
   * @param {Float32Array} hiddenStates - Input hidden states [numTokens, hiddenSize]
   * @param {number} numTokens - Number of tokens
   * @returns {ExpertSelection[]} Expert selections for each token
   */
  route(hiddenStates, numTokens) {
    // Compute router logits
    const allLogits = this.computeRouterLogitsCPU(hiddenStates, numTokens);

    /** @type {ExpertSelection[]} */
    const selections = [];
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
   * @returns {number[]} Array of active expert indices
   */
  getActiveExperts() {
    return Array.from(this.activeExperts).sort((a, b) => a - b);
  }

  /**
   * Compute auxiliary load balancing loss
   * Used during training to encourage balanced expert utilization.
   * @returns {number} Load balancing loss value
   */
  computeLoadBalanceLoss() {
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
   * @returns {void}
   */
  resetStats() {
    this.loadBalanceStats.expertCounts.fill(0);
    this.loadBalanceStats.totalTokens = 0;
    this.activeExperts.clear();
  }

  /**
   * Get expert utilization statistics
   * @returns {UtilizationStats} Utilization stats per expert
   */
  getUtilizationStats() {
    const total = this.loadBalanceStats.totalTokens;
    if (total === 0) {
      return { experts: [], totalTokens: 0, loadBalanceLoss: 0 };
    }

    /** @type {Array<{ index: number; count: number; percentage: number }>} */
    const experts = [];
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
 * Create a grouped expert execution plan
 * Groups tokens by their selected experts for efficient batched computation
 *
 * @param {ExpertSelection[]} selections - Expert selections for all tokens
 * @param {number} numExperts - Total number of experts
 * @returns {Map<number, ExpertExecutionPlanEntry>} Map of expert index to token indices and weights
 */
export function createExpertExecutionPlan(selections, numExperts) {
  /** @type {Map<number, ExpertExecutionPlanEntry>} */
  const plan = new Map();

  // Initialize empty plans for each expert
  for (let e = 0; e < numExperts; e++) {
    plan.set(e, { tokenIndices: [], weights: /** @type {Float32Array} */ (/** @type {unknown} */ ([])) });
  }

  // Group tokens by expert
  for (let t = 0; t < selections.length; t++) {
    const sel = selections[t];
    for (let k = 0; k < sel.indices.length; k++) {
      const expertIdx = sel.indices[k];
      const weight = sel.weights[k];
      const entry = /** @type {ExpertExecutionPlanEntry} */ (plan.get(expertIdx));
      entry.tokenIndices.push(t);
      /** @type {number[]} */ (/** @type {unknown} */ (entry.weights)).push(weight);
    }
  }

  // Convert weight arrays to Float32Array
  for (const [expertIdx, data] of plan) {
    plan.set(expertIdx, {
      tokenIndices: data.tokenIndices,
      weights: new Float32Array(/** @type {number[]} */ (/** @type {unknown} */ (data.weights)))
    });
  }

  return plan;
}

/**
 * Combine expert outputs with routing weights
 *
 * @param {Map<number, Float32Array>} expertOutputs - Output from each expert [numTokens, hiddenSize]
 * @param {ExpertSelection[]} selections - Original routing decisions
 * @param {number} numTokens - Number of tokens
 * @param {number} hiddenSize - Hidden dimension
 * @returns {Float32Array} Combined output [numTokens, hiddenSize]
 */
export function combineExpertOutputs(expertOutputs, selections, numTokens, hiddenSize) {
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
