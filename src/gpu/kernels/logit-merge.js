

import { getDevice } from '../device.js';
import { log } from '../../debug/index.js';







// WGSL shader for weighted logit merging
const WEIGHTED_MERGE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits_a: array<f32>;
@group(0) @binding(1) var<storage, read> logits_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged: array<f32>;
@group(0) @binding(3) var<uniform> params: MergeParams;

struct MergeParams {
  vocab_size: u32,
  weight_a: f32,
  weight_b: f32,
  temperature: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.vocab_size) {
    return;
  }

  // Weighted average of logits
  let a = logits_a[idx];
  let b = logits_b[idx];
  var result = params.weight_a * a + params.weight_b * b;

  // Apply temperature scaling
  if (params.temperature != 1.0) {
    result = result / params.temperature;
  }

  merged[idx] = result;
}
`;

// WGSL shader for max logit merging
const MAX_MERGE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits_a: array<f32>;
@group(0) @binding(1) var<storage, read> logits_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged: array<f32>;
@group(0) @binding(3) var<uniform> params: MergeParams;

struct MergeParams {
  vocab_size: u32,
  weight_a: f32,
  weight_b: f32,
  temperature: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.vocab_size) {
    return;
  }

  // Max of logits
  var result = max(logits_a[idx], logits_b[idx]);

  // Apply temperature scaling
  if (params.temperature != 1.0) {
    result = result / params.temperature;
  }

  merged[idx] = result;
}
`;

// WGSL shader for geometric mean merging (in log space)
const GEOMETRIC_MERGE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits_a: array<f32>;
@group(0) @binding(1) var<storage, read> logits_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged: array<f32>;
@group(0) @binding(3) var<uniform> params: MergeParams;

struct MergeParams {
  vocab_size: u32,
  weight_a: f32,
  weight_b: f32,
  temperature: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.vocab_size) {
    return;
  }

  // Geometric mean in log space: weight_a * log(a) + weight_b * log(b)
  // Since logits are already in log space, this is just weighted sum
  // but with weights that must sum to 1 for proper geometric mean
  let a = logits_a[idx];
  let b = logits_b[idx];
  var result = params.weight_a * a + params.weight_b * b;

  // Apply temperature scaling
  if (params.temperature != 1.0) {
    result = result / params.temperature;
  }

  merged[idx] = result;
}
`;


export class LogitMergeKernel {
  
  #device;

  
  #pipelines = new Map();

  
  #bindGroupLayout;

  
  #initialized = false;

  constructor() {
    this.#device = getDevice();
  }

  
  async init() {
    if (this.#initialized) return;

    this.#device = getDevice();

    // Create bind group layout
    this.#bindGroupLayout = this.#device.createBindGroupLayout({
      label: 'logit-merge-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipelineLayout = this.#device.createPipelineLayout({
      label: 'logit-merge-pipeline-layout',
      bindGroupLayouts: [this.#bindGroupLayout],
    });

    // Create pipelines for each strategy
    const strategies = [
      { name: 'weighted', shader: WEIGHTED_MERGE_SHADER },
      { name: 'max', shader: MAX_MERGE_SHADER },
      { name: 'geometric', shader: GEOMETRIC_MERGE_SHADER },
    ];

    for (const { name, shader } of strategies) {
      const module = this.#device.createShaderModule({
        label: `logit-merge-${name}`,
        code: shader,
      });

      const pipeline = await this.#device.createComputePipelineAsync({
        label: `logit-merge-${name}-pipeline`,
        layout: pipelineLayout,
        compute: {
          module,
          entryPoint: 'main',
        },
      });

      this.#pipelines.set( (name), pipeline);
    }

    this.#initialized = true;
    log.info('LogitMerge', 'Kernel initialized');
  }

  
  async merge(logitsA, logitsB, vocabSize, config = {}) {
    if (!this.#initialized) {
      await this.init();
    }

    if (!config?.strategy) {
      throw new Error('LogitMerge strategy is required.');
    }
    if (!config?.weights) {
      throw new Error('LogitMerge weights are required.');
    }
    if (config?.temperature == null) {
      throw new Error('LogitMerge temperature is required.');
    }
    const strategy = config.strategy;
    const weights = normalizeWeights(config.weights, 2);
    const temperature = config.temperature;

    const pipeline = this.#pipelines.get(strategy);
    if (!pipeline) {
      throw new Error(`Unknown merge strategy: ${strategy}`);
    }

    // Create output buffer
    const mergedBuffer = this.#device.createBuffer({
      label: 'logit-merge-output',
      size: vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create params uniform buffer
    const paramsData = new Float32Array([
      vocabSize, // vocab_size (will be reinterpreted as u32)
      weights[0],
      weights[1],
      temperature,
    ]);
    // Fix: vocab_size needs to be u32
    const paramsView = new DataView(paramsData.buffer);
    paramsView.setUint32(0, vocabSize, true);

    const paramsBuffer = this.#device.createBuffer({
      label: 'logit-merge-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Create bind group
    const bindGroup = this.#device.createBindGroup({
      label: 'logit-merge-bindgroup',
      layout: this.#bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: logitsA } },
        { binding: 1, resource: { buffer: logitsB } },
        { binding: 2, resource: { buffer: mergedBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    // Dispatch
    const encoder = this.#device.createCommandEncoder({ label: 'logit-merge' });
    const pass = encoder.beginComputePass({ label: 'logit-merge-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(vocabSize / 256));
    pass.end();

    this.#device.queue.submit([encoder.finish()]);

    // Cleanup temporary buffer
    paramsBuffer.destroy();

    return mergedBuffer;
  }

  
  async mergeMultiple(logitBuffers, vocabSize, config = {}) {
    if (logitBuffers.length === 0) {
      throw new Error('No logit buffers to merge');
    }

    if (logitBuffers.length === 1) {
      return logitBuffers[0];
    }

    if (logitBuffers.length === 2) {
      return this.merge(logitBuffers[0], logitBuffers[1], vocabSize, config);
    }

    // For more than 2 buffers, do pairwise reduction
    if (!config?.weights) {
      throw new Error('LogitMerge weights are required.');
    }
    if (config?.temperature == null) {
      throw new Error('LogitMerge temperature is required.');
    }
    const weights = normalizeWeights(config.weights, logitBuffers.length);

    // Normalize weights for pairwise reduction
    let totalWeight = 0;
    let current = logitBuffers[0];
    let currentWeight = weights[0];

    for (let i = 1; i < logitBuffers.length; i++) {
      totalWeight = currentWeight + weights[i];
      const normalizedWeightA = currentWeight / totalWeight;
      const normalizedWeightB = weights[i] / totalWeight;

      const merged = await this.merge(current, logitBuffers[i], vocabSize, {
        ...config,
        weights: [normalizedWeightA, normalizedWeightB],
        temperature: i === logitBuffers.length - 1 ? config.temperature : 1.0,
      });

      // Destroy intermediate buffer if not the original
      if (i > 1) {
        current.destroy();
      }

      current = merged;
      currentWeight = totalWeight;
    }

    return current;
  }
}

function normalizeWeights(weights, expectedCount) {
  if (!Array.isArray(weights) || weights.length !== expectedCount) {
    throw new Error(`LogitMerge weights must have length ${expectedCount}`);
  }

  let sum = 0;
  for (const weight of weights) {
    if (!Number.isFinite(weight)) {
      throw new Error('LogitMerge weights must be finite numbers');
    }
    if (weight < 0) {
      throw new Error('LogitMerge weights must be non-negative');
    }
    sum += weight;
  }

  if (sum <= 0) {
    throw new Error('LogitMerge weights must sum to a positive value');
  }

  return weights.map((weight) => weight / sum);
}

// Singleton instance
let _instance = null;


export function getLogitMergeKernel() {
  if (!_instance) {
    _instance = new LogitMergeKernel();
  }
  return _instance;
}


export async function mergeLogits(logitsA, logitsB, vocabSize, weights, temperature) {
  if (!weights) {
    throw new Error('LogitMerge weights are required.');
  }
  if (temperature == null) {
    throw new Error('LogitMerge temperature is required.');
  }
  const kernel = getLogitMergeKernel();
  return kernel.merge(logitsA, logitsB, vocabSize, {
    strategy: 'weighted',
    weights,
    temperature,
  });
}


export async function mergeMultipleLogits(logitBuffers, vocabSize, weights, temperature) {
  if (!weights) {
    throw new Error('LogitMerge weights are required.');
  }
  if (temperature == null) {
    throw new Error('LogitMerge temperature is required.');
  }
  const kernel = getLogitMergeKernel();
  return kernel.mergeMultiple(logitBuffers, vocabSize, {
    strategy: 'weighted',
    weights,
    temperature,
  });
}
