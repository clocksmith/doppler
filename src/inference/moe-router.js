

import { getDevice } from '../gpu/device.js';
import { getWeightDtype, isWeightBuffer } from '../gpu/weight-buffer.js';
import { runMatmul } from '../gpu/kernel-selector.js';
import { createTensor } from '../gpu/tensor.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { getShaderModule } from '../gpu/kernels/shader-cache.js';

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function isRouterVector(value) {
  return value instanceof Float32Array || isGpuBufferInstance(value) || isWeightBuffer(value);
}

function requireRouterDtype(value, label) {
  if (value !== 'f16' && value !== 'f32') {
    throw new Error(`MoERouter requires ${label} to be "f16" or "f32", got ${value}.`);
  }
  return value;
}

function inferRouterBufferDtypeFromElementCount(buffer, elementCount, label) {
  const byteLength = Number.isFinite(buffer?.size) ? buffer.size : buffer?.byteLength;
  if (!Number.isFinite(byteLength)) {
    throw new Error(`MoERouter requires ${label} byte size to resolve dtype.`);
  }
  const bytesPerElement = byteLength / elementCount;
  if (bytesPerElement !== 2 && bytesPerElement !== 4) {
    throw new Error(`MoERouter cannot infer ${label} dtype from bytesPerElement=${bytesPerElement}.`);
  }
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });
}








export class MoERouter {
  
  numExperts;

  
  topK;

  
  hiddenSize;

  
  normalizeWeights;

  
  gateWeight = null;

  
  gateBias = null;

  gateScale = null;

  perExpertScale = null;

  
  activeExperts;

  
  loadBalanceStats;

  
  _biasAddPipelines = new Map();

  
  _gateBiasGPU = null;

  
  _gateWeightGPU = null;

  _gateWeightDtype = null;

  
  lastLogitsDtype = null;

  
  constructor(config) {
    if (config.numExperts == null) {
      throw new Error('MoERouter requires numExperts in config.');
    }
    if (!Number.isFinite(config.numExperts) || config.numExperts <= 0) {
      throw new Error(`MoERouter requires numExperts to be a positive number, got ${config.numExperts}.`);
    }
    if (config.topK == null) {
      throw new Error('MoERouter requires topK in config.');
    }
    if (!Number.isFinite(config.topK) || config.topK <= 0) {
      throw new Error(`MoERouter requires topK to be a positive number, got ${config.topK}.`);
    }
    if (config.hiddenSize == null) {
      throw new Error('MoERouter requires hiddenSize in config.');
    }
    if (!Number.isFinite(config.hiddenSize) || config.hiddenSize <= 0) {
      throw new Error(`MoERouter requires hiddenSize to be a positive number, got ${config.hiddenSize}.`);
    }
    if (config.normalizeWeights == null) {
      throw new Error('MoERouter requires normalizeWeights in config.');
    }
    this.numExperts = config.numExperts;
    this.topK = config.topK;
    this.hiddenSize = config.hiddenSize;
    this.normalizeWeights = config.normalizeWeights;

    // Track active experts for the current batch
    this.activeExperts = new Set();

    // Auxiliary load balancing stats
    this.loadBalanceStats = {
      expertCounts: new Uint32Array(this.numExperts),
      totalTokens: 0
    };
  }

  
  loadWeights(weights, bias = null, scale = null, perExpertScale = null) {
    if (!weights) {
      throw new Error('MoERouter.loadWeights requires non-null weights.');
    }
    if (bias != null && !(bias instanceof Float32Array) && !isGpuBufferInstance(bias)) {
      throw new Error('MoERouter.loadWeights bias must be a Float32Array or GPUBuffer.');
    }
    if (scale != null && !isRouterVector(scale)) {
      throw new Error('MoERouter.loadWeights scale must be a Float32Array, GPUBuffer, or WeightBuffer.');
    }
    if (perExpertScale != null && !isRouterVector(perExpertScale)) {
      throw new Error('MoERouter.loadWeights perExpertScale must be a Float32Array, GPUBuffer, or WeightBuffer.');
    }
    if (this._gateBiasGPU) {
      this._gateBiasGPU.destroy();
    }
    if (this._gateWeightGPU) {
      this._gateWeightGPU.destroy();
    }
    this._gateWeightDtype = isWeightBuffer(weights)
      ? getWeightDtype(weights)
      : inferRouterBufferDtypeFromElementCount(weights, this.numExperts * this.hiddenSize, 'gate weight');
    this.gateWeight = weights;
    this.gateBias = bias;
    this.gateScale = scale;
    this.perExpertScale = perExpertScale;
    // Clear cached GPU uploads when swapping router parameters (e.g., per-layer routers).
    this._gateBiasGPU = null;
    this._gateWeightGPU = null;
  }

  destroy() {
    if (isGpuBufferInstance(this._gateBiasGPU)) {
      this._gateBiasGPU.destroy();
    }
    if (isGpuBufferInstance(this._gateWeightGPU)) {
      this._gateWeightGPU.destroy();
    }
    this._gateBiasGPU = null;
    this._gateWeightGPU = null;
    this._gateWeightDtype = null;
    this.gateWeight = null;
    this.gateBias = null;
    this.gateScale = null;
    this.perExpertScale = null;
    this._biasAddPipelines.clear();
  }

  
  computeRouterLogitsCPU(hiddenStates, numTokens) {
    void hiddenStates;
    void numTokens;
    throw new Error(
      'MoERouter.computeRouterLogitsCPU is unavailable: runtime routing mathematics requires the WebGPU path.'
    );
  }

  
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
    if (!isWeightBuffer(gateWeightBuffer) && !isGpuBufferInstance(gateWeightBuffer)) {
      const sourceWeightDtype = this._gateWeightDtype
        ?? inferRouterBufferDtypeFromElementCount(gateWeightBuffer, this.numExperts * this.hiddenSize, 'gate weight');
      const uploaded = device.createBuffer({
        label: 'moe_gate_weight',
        size: gateWeightBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      try {
        device.queue.writeBuffer(uploaded, 0, gateWeightBuffer);
      } catch (error) {
        uploaded.destroy();
        throw error;
      }
      this._gateWeightGPU = uploaded;
      this._gateWeightDtype = sourceWeightDtype;
      this.gateWeight = uploaded;
      gateWeightBuffer = uploaded;
    }

    const inputDtype = requireRouterDtype(options.inputDtype, 'options.inputDtype');
    const outputDtype = requireRouterDtype(options.outputDtype, 'options.outputDtype');
    const gateWeightDtype = isWeightBuffer(gateWeightBuffer)
      ? getWeightDtype(gateWeightBuffer)
      : this._gateWeightDtype;

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

  
  async _getGateBiasBuffer(device) {
    if (isGpuBufferInstance(this.gateBias)) return this.gateBias;
    if (this._gateBiasGPU) return this._gateBiasGPU;

    if (!(this.gateBias instanceof Float32Array)) {
      throw new Error('Unsupported gateBias type');
    }

    const buf = device.createBuffer({
      label: 'moe_gate_bias',
      size: this.gateBias.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    try {
      device.queue.writeBuffer(buf, 0, this.gateBias);
    } catch (error) {
      buf.destroy();
      throw error;
    }
    this._gateBiasGPU = buf;
    return buf;
  }

  
  _inferBiasDtype(bias) {
    if (bias instanceof Float32Array) return 'f32';
    if (isGpuBufferInstance(bias)) {
      const bytesPerElement = Math.round(bias.size / this.numExperts);
      return inferRouterBufferDtypeFromElementCount(bias, this.numExperts, 'gate bias');
    }
    throw new Error('MoERouter cannot infer gate bias dtype.');
  }

  async _getBiasAddPipeline(logitsDtype, biasDtype, device) {
    const key = `${logitsDtype}_${biasDtype}`;
    const cached = this._biasAddPipelines.get(key);
    if (cached) return cached;

    const { shaderFile } = selectRuleValue('inference', 'moe', 'biasAddCode', {
      logitsDtype,
      biasDtype,
    });
    const module = await getShaderModule(device, shaderFile, `moe_router_bias_add_${key}`);
    const pipeline = device.createComputePipeline({
      label: `moe_router_bias_add_${key}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this._biasAddPipelines.set(key, pipeline);
    return pipeline;
  }

  async _addBiasInPlace(logits, bias, numTokens, device, logitsDtype, biasDtype) {
    const pipeline = await this._getBiasAddPipeline(logitsDtype, biasDtype, device);

    const uniformData = new ArrayBuffer(16);
    const uniformView = new DataView(uniformData);
    uniformView.setUint32(0, numTokens, true);
    uniformView.setUint32(4, this.numExperts, true);

    const uniformBuffer = device.createBuffer({
      label: 'moe_router_bias_uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
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
    } finally {
      uniformBuffer.destroy();
    }
  }

  
  async routeGPU(hiddenStates, numTokens, options = {}) {
    void hiddenStates;
    void numTokens;
    void options;
    throw new Error(
      'MoERouter.routeGPU cannot read routing tensors back to JavaScript; use the GPU MoE executor.'
    );
  }

  
  softmax(logits, size) {
    void logits;
    void size;
    throw new Error('MoERouter.softmax is unavailable: runtime softmax requires the WebGPU path.');
  }

  
  selectExpertsForToken(logits) {
    void logits;
    throw new Error(
      'MoERouter.selectExpertsForToken is unavailable: top-k routing requires the WebGPU path.'
    );
  }

  
  route(hiddenStates, numTokens) {
    void hiddenStates;
    void numTokens;
    throw new Error('MoERouter.route is unavailable: runtime MoE routing requires WebGPU.');
  }

  
  getActiveExperts() {
    return Array.from(this.activeExperts).sort((a, b) => a - b);
  }

  
  computeLoadBalanceLoss() {
    if (this.loadBalanceStats.totalTokens === 0) return 0;

    const numTokens = this.loadBalanceStats.totalTokens;
    let loss = 0;
    const idealFraction = 1 / this.numExperts;
    for (let i = 0; i < this.numExperts; i++) {
      const expertFraction = this.loadBalanceStats.expertCounts[i] / numTokens;
      const deviation = expertFraction - idealFraction;
      loss += deviation * deviation;
    }

    return loss * this.numExperts;
  }

  
  resetStats() {
    this.loadBalanceStats.expertCounts.fill(0);
    this.loadBalanceStats.totalTokens = 0;
    this.activeExperts.clear();
  }

  
  getUtilizationStats() {
    const total = this.loadBalanceStats.totalTokens;
    if (total === 0) {
      return { experts: [], totalTokens: 0, loadBalanceLoss: 0 };
    }

    
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


export function createExpertExecutionPlan(selections, numExperts) {
  
  const plan = new Map();

  // Initialize empty plans for each expert
  for (let e = 0; e < numExperts; e++) {
    plan.set(e, { tokenIndices: [], weights:  ( ([])) });
  }

  // Group tokens by expert
  for (let t = 0; t < selections.length; t++) {
    const sel = selections[t];
    for (let k = 0; k < sel.indices.length; k++) {
      const expertIdx = sel.indices[k];
      const weight = sel.weights[k];
      const entry =  (plan.get(expertIdx));
      entry.tokenIndices.push(t);
       ( (entry.weights)).push(weight);
    }
  }

  // Convert weight arrays to Float32Array
  for (const [expertIdx, data] of plan) {
    plan.set(expertIdx, {
      tokenIndices: data.tokenIndices,
      weights: new Float32Array( ( (data.weights)))
    });
  }

  return plan;
}


export function combineExpertOutputs(expertOutputs, selections, numTokens, hiddenSize) {
  void expertOutputs;
  void selections;
  void numTokens;
  void hiddenSize;
  throw new Error('combineExpertOutputs is unavailable: expert aggregation requires WebGPU.');
}

export default MoERouter;
