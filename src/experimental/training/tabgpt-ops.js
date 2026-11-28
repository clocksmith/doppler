import { compileShader, loadShaderSource } from '../../gpu/kernels/shader-cache.js';

/** @param {number} n @param {number} width */
const groups = (n, width) => Math.ceil(n / width);

/**
 * @param {string} module @param {string} entryPoint @param {number[]} bindings
 * @param {number} workgroups @param {Record<string,number>} constants
 */
function kernel(module, entryPoint, bindings, workgroups, constants) {
  return { module, entryPoint, bindings, workgroups, constants };
}

/**
 * @param {import('./tabgpt-model.js').TabGPTConfig} config
 * @returns {Record<string,(dims: import('./tabgpt-ops.js').TabGPTOpDimensions) => ReturnType<typeof kernel>>}
 */
function operationSpecs({ B, T, C, NH, V }) {
  const rows = B * T;
  const shape = { BATCH: B, LENGTH: T, HEADS: NH, HEAD_WIDTH: C / NH };
  const elementCount = rows * C;
  /** @param {string} entry @param {number[]} bindings @param {number} count */
  const pointwise = (entry, bindings, count) =>
    kernel('elementwise', entry, bindings, groups(count, 256), { COUNT: count });
  /** @param {string} entry @param {number[]} bindings */
  const lookup = (entry, bindings) =>
    kernel('elementwise', entry, bindings, groups(elementCount, 256), { COUNT: elementCount, WIDTH: C });
  /** @param {string} entry @param {number[]} bindings @param {number} n */
  const attention = (entry, bindings, n) => kernel('attention', entry, bindings, n, shape);
  const positionGroups = groups(rows * 2 * NH, 64);
  const activationGroups = groups(elementCount, 256);
  return {
    embed: () => lookup('embed', [3, 0, 2]),
    embedGrad: () => lookup('embed_grad', [3, 0, 4]),
    rms: ({ width = C }) => kernel('norm', 'forward', [0, 1, 2, 3], rows, { WIDTH: width }),
    rmsDx: ({ width = C }) => kernel('norm', 'input_grad', [4, 0, 1, 3, 5], rows, { WIDTH: width }),
    rmsDw: ({ width = C }) => kernel('norm', 'weight_grad', [4, 0, 3, 6], groups(width, 128), { ROWS: rows, WIDTH: width }),
    position: () => attention('position', [0, 1], positionGroups),
    positionGrad: () => attention('position_grad', [0, 1, 2], positionGroups),
    scores: () => kernel('attention', 'scores', [0, 3], rows * NH,
      { LENGTH: T, HEADS: NH, HEAD_WIDTH: C / NH }),
    context: () => attention('context', [3, 0, 4], activationGroups),
    scoresGrad: () => attention('scores_grad', [5, 4, 3, 0, 6], groups(rows * NH * T, 256)),
    qkvGrad: () => attention('qkv_grad', [6, 3, 0, 5, 2], activationGroups),
    relu2: ({ count = 0 }) => pointwise('relu_squared', [0, 2], count),
    relu2Grad: ({ count = 0 }) => pointwise('relu_squared_grad', [0, 1, 2], count),
    add: ({ count = 0 }) => pointwise('add', [0, 1, 2], count),
    accumulate: ({ count = 0 }) => pointwise('accumulate', [0, 2], count),
    loss: () => kernel('loss', 'loss', [0, 1, 2, 3], rows, { VOCAB: V, ROWS: rows }),
    normPart: ({ count = 0, offset = 0 }) => kernel('optimizer', 'norm_part', [1, 5], groups(count, 1024), { COUNT: count, PARTIAL_OFFSET: offset }),
    normFinal: ({ count = 0 }) => kernel('optimizer', 'norm_final', [5, 4], 1, { COUNT: count }),
    adam: ({ count = 0 }) => kernel('optimizer', 'adam', [0, 1, 2, 3, 4, 6], groups(count, 256), { COUNT: count }),
  };
}

// Each graph owns its specialized pipelines; bind groups refer to its fixed tensor slices.
/**
 * @param {GPUDevice} device @param {import('./tabgpt-model.js').TabGPTConfig} config
 * @returns {Promise<import('./tabgpt-ops.js').TabGPTOps>}
 */
export async function createTabGPTOps(device, config) {
  const specs = operationSpecs(config);
  /** @type {Map<string, GPUShaderModule>} */
  const modules = new Map();
  const files = {
    elementwise: 'src/gpu/kernels/training/tabgpt_elementwise.wgsl',
    norm: 'src/gpu/kernels/training/tabgpt_norm.wgsl',
    attention: 'src/gpu/kernels/training/tabgpt_attention.wgsl',
    loss: 'src/gpu/kernels/training/tabgpt_loss.wgsl',
    optimizer: 'src/gpu/kernels/training/tabgpt_optimizer.wgsl',
  };
  await Promise.all(Object.entries(files).map(async ([name, file]) => {
    modules.set(name, await compileShader(device, await loadShaderSource(file), file));
  }));
  /** @type {Map<string, GPUComputePipeline>} */
  const pipelines = new Map();

  /** @type {import('./tabgpt-ops.js').TabGPTOps['compile']} */
  const compile = (name, buffers, dims = {}) => {
    if (!Object.hasOwn(specs, name)) throw new Error(`Unknown TabGPT operation: ${name}`);
    const spec = specs[name](dims);
    if (buffers.length !== spec.bindings.length) throw new Error(`Wrong number of buffers for TabGPT ${name}`);
    const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
    const groupCount = spec.workgroups;
    if (!Number.isSafeInteger(groupCount) || groupCount <= 0) throw new Error(`Invalid TabGPT ${name} dispatch size`);
    if (groupCount > maxGroups && name !== 'adam' && name !== 'normPart') {
      throw new Error(`TabGPT ${name} requires ${groupCount} workgroups on one axis; limit is ${maxGroups}`);
    }
    const gridX = Math.min(groupCount, maxGroups);
    const gridY = groups(groupCount, gridX);
    if (gridY > maxGroups) throw new Error(`TabGPT ${name} exceeds the device dispatch grid`);
    const key = JSON.stringify([name, spec.constants]);
    let pipeline = pipelines.get(key);
    if (!pipeline) {
      const module = modules.get(spec.module);
      if (!module) throw new Error(`Missing TabGPT shader module: ${spec.module}`);
      pipeline = device.createComputePipeline({
        label: `tabgpt-${name}`, layout: 'auto',
        compute: { module, entryPoint: spec.entryPoint, constants: spec.constants },
      });
      pipelines.set(key, pipeline);
    }
    const bindGroup = device.createBindGroup({
      label: `tabgpt-${name}`, layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((value, i) => ({
        binding: spec.bindings[i],
        resource: 'buffer' in value ? value : { buffer: value },
      })),
    });
    return (pass) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(gridX, gridY);
    };
  };
  return { compile };
}
