import { buildTabGPTGraph } from './tabgpt-graph.js';
import { createTabGPTMatmul } from './tabgpt-matmul.js';
import { createTabGPTOps } from './tabgpt-ops.js';

/** @typedef {import('./tabgpt-model.js').TabGPTConfig} TabGPTConfig */
/** @typedef {import('./tabgpt-model.js').TabGPTTrainer} TabGPTTrainer */
/** @typedef {'one' | 'zero' | 'normal'} InitialFill */
/** @typedef {{name:string, count:number, offset:number, fill:InitialFill}} ParameterEntry */

/** @param {TabGPTConfig} config */
function validateConfig(config) {
  for (const key of /** @type {const} */ (['B', 'T', 'C', 'L', 'NH', 'V'])) {
    if (!Number.isSafeInteger(config[key]) || config[key] <= 0) {
      throw new RangeError(`TabGPT requires a positive integer ${key}.`);
    }
  }
  if (config.C % config.NH || (config.C / config.NH) % 2) {
    throw new RangeError('TabGPT requires an even, integral RoPE head width.');
  }
  if (config.f16 !== false || !Number.isFinite(config.lr) || !Number.isFinite(Math.fround(config.lr)) ||
      Math.fround(config.lr) <= 0 ||
      !Number.isSafeInteger(config.warmup) || config.warmup < 0) {
    throw new RangeError('TabGPT requires f16:false, positive lr and a nonnegative warmup.');
  }
}

/** @param {TabGPTConfig} config @param {number} alignment @param {number} maxElements */
function parameterLayout({ C, V, L }, alignment, maxElements) {
  /** @type {Array<[string, number, InitialFill]>} */
  const layers = [
    ['ln1.weight', C, 'one'],
    ['attn.qkv.weight', 3 * C * C, 'normal'],
    ['attn.proj.weight', C * C, 'zero'],
    ['ln2.weight', C, 'one'],
    ['mlp.fc.weight', 4 * C * C, 'normal'],
    ['mlp.proj.weight', 4 * C * C, 'zero'],
  ];
  /** @param {number} size */
  const padded = size => Math.ceil(size / alignment) * alignment;
  const layerSize = layers.reduce((sum, [, size]) => sum + padded(size), 0);
  const paddedCount = padded(C * V) + L * layerSize + padded(C);
  if (!Number.isSafeInteger(paddedCount) || paddedCount > maxElements) {
    throw new RangeError('TabGPT parameter pool exceeds the GPU binding limit.');
  }
  /** @type {ParameterEntry[]} */
  const entries = [];
  let offset = 0;
  /** @param {string} name @param {number} count @param {InitialFill} fill */
  function add(name, count, fill) {
    entries.push({ name, count, offset, fill });
    offset += padded(count);
  }
  add('wte', C * V, 'normal');
  for (let i = 0; i < L; i++) {
    for (const [name, count, fill] of layers) add(`h.${i}.${name}`, count, fill);
  }
  add('lnf.weight', C, 'one');
  return { entries, paddedCount };
}

function normalInitializer() {
  // Match the public TabGPT initializer: two Mulberry32(1337) draws per weight.
  let state = 1337;
  function uniform() {
    state = (state + 0x6d2b79f5) | 0;
    let bits = Math.imul(state ^ (state >>> 15), 1 | state);
    bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
    return ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
  }
  return () => Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-12))) *
    Math.cos(2 * Math.PI * uniform()) * 0.02;
}

/** @param {number} count @param {'one' | 'normal'} fill @param {() => number} normal */
function materializeInitialParameter(count, fill, normal) {
  const data = new Float32Array(count);
  if (fill === 'one') data.fill(1);
  else for (let j = 0; j < count; j++) data[j] = normal();
  return data;
}

/** @param {TabGPTConfig} config @param {number} step */
function learningRate(config, step) {
  if (step < config.warmup) return config.lr * step / config.warmup;
  const progress = Math.min((step - config.warmup) / 8000, 1);
  return config.lr * (0.05 + 0.475 * (1 + Math.cos(Math.PI * progress)));
}

/** @param {Uint32Array} x @param {Uint32Array} y @param {number} count @param {number} vocab */
function checkBatch(x, y, count, vocab) {
  for (const tokens of [x, y]) {
    if (!(tokens instanceof Uint32Array) || tokens.length !== count) {
      throw new RangeError(`Expected Uint32Array with ${count} token IDs.`);
    }
    for (const token of tokens) {
      if (token >= vocab) throw new RangeError(`Token ID ${token} exceeds the vocabulary.`);
    }
  }
}

// FP32 training from scratch with one master parameter per tied embedding/head
// element. All queues, pipelines and buffers belong to the supplied device.
/** @param {GPUDevice} device
 * @param {TabGPTConfig} configuration
 * @param {Map<string, Float32Array>} [initialWeights]
 * @returns {Promise<TabGPTTrainer>}
 */
export async function createTabGPTTrainer(device, configuration, initialWeights) {
  const config = Object.freeze({ ...configuration });
  validateConfig(config);
  const { B, T, V } = config;
  const alignment = Math.max(128, device.limits.minStorageBufferOffsetAlignment / 4);
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  const { entries, paddedCount } = parameterLayout(config, alignment, limit / 4);
  const largestActivation = B * T * Math.max(V, 4 * config.C, config.NH * T);
  if (!Number.isSafeInteger(largestActivation) || largestActivation > limit / 4) {
    throw new RangeError('TabGPT activation exceeds the GPU binding limit.');
  }
  if (initialWeights != null && (!(initialWeights instanceof Map) || initialWeights.size !== entries.length)) {
    throw new TypeError('initialWeights must contain exactly one Float32Array per named parameter.');
  }
  if (initialWeights) {
    for (const { name, count } of entries) {
      const data = initialWeights.get(name);
      if (!(data instanceof Float32Array) || data.length !== count || !data.every(Number.isFinite)) {
        throw new RangeError(`Expected ${count} finite FP32 weights for ${name}.`);
      }
    }
  }
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  /** @type {Set<GPUBuffer>} */
  const buffers = new Set();
  const memory = { ownedBufferBytes: 0, peakOwnedBufferBytes: 0 };
  /** @param {number} size @param {GPUBufferUsageFlags} usage @param {string} label */
  function allocateBuffer(size, usage, label) {
    if (!Number.isSafeInteger(size) || size <= 0 || size > limit) {
      throw new RangeError(`TabGPT buffer ${label} exceeds the binding limit or has an invalid size.`);
    }
    const buffer = device.createBuffer({ label: `tabgpt.${label}`, usage, size });
    buffers.add(buffer);
    memory.ownedBufferBytes += size;
    memory.peakOwnedBufferBytes = Math.max(memory.peakOwnedBufferBytes, memory.ownedBufferBytes);
    return buffer;
  }
  /** @param {number} elements @param {string} label */
  const alloc = (elements, label) => allocateBuffer(elements * 4, storageUsage, label);
  /** @param {GPUBuffer} buffer */
  function release(buffer) {
    if (!buffers.delete(buffer)) return;
    memory.ownedBufferBytes -= buffer.size;
    buffer.destroy();
  }
  let destroyed = false;
  let busy = false;
  function dispose() {
    destroyed = true;
    for (const buffer of buffers) release(buffer);
  }
  function destroy() {
    if (busy) throw new Error('TabGPT has an outstanding GPU operation.');
    dispose();
  }

  /** @param {import('./tabgpt-ops.js').TabGPTOps} ops
   * @param {import('./tabgpt-matmul.js').TabGPTMatmul} matmul
   * @returns {TabGPTTrainer}
   */
  function prepare(ops, matmul) {
    const values = alloc(paddedCount, 'weights');
    const gradients = alloc(paddedCount, 'gradients');
    const moments = alloc(paddedCount, 'adam-first');
    const variances = alloc(paddedCount, 'adam-second');
    /** @type {Map<string, {value: Required<GPUBufferBinding>, gradient: Required<GPUBufferBinding>}>} */
    const params = new Map();
    const normal = normalInitializer();
    for (const { name, count, offset, fill } of entries) {
      const slice = { offset: offset * 4, size: count * 4 };
      params.set(name, {
        value: { buffer: values, ...slice }, gradient: { buffer: gradients, ...slice },
      });
      if (initialWeights != null) {
        const data = initialWeights.get(name);
        if (!data) throw new Error(`Missing initial weights for ${name}.`);
        device.queue.writeBuffer(values, slice.offset, data);
      } else if (fill !== 'zero') {
        device.queue.writeBuffer(values, slice.offset, materializeInitialParameter(count, fill, normal));
      }
    }
    const graph = buildTabGPTGraph(config, params, alloc, (name, bindings, dims) => ops.compile(name, bindings, dims), matmul);
    const partialCount = Math.ceil(paddedCount / 1024);
    const partials = alloc(partialCount, 'gradient-norm-partials');
    const scale = alloc(1, 'gradient-scale');
    const stepUniform = allocateBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'step');
    const update = [
      ops.compile('normPart', [gradients, partials], { count: paddedCount, offset: 0 }),
      ops.compile('normFinal', [partials, scale], { count: partialCount }),
      ops.compile('adam', [values, gradients, moments, variances, scale, stepUniform], { count: paddedCount }),
    ];
    const stageUsage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    const lossReadback = allocateBuffer(B * T * 4, stageUsage, 'loss-readback');
    let step = 0;
    /** @template T @param {() => Promise<T>} work @returns {Promise<T>} */
    async function exclusive(work) {
      if (destroyed || busy) throw new Error(destroyed ? 'TabGPT trainer was destroyed.' : 'TabGPT trainer is busy.');
      busy = true;
      try { return await work(); } finally { busy = false; }
    }
    /** @param {import('./tabgpt-ops.js').TabGPTDispatch[]} dispatches */
    function submitDispatches(dispatches) {
      // Bound the GPU queue's command-buffer granularity on the browser thread.
      for (let i = 0; i < dispatches.length; i += 16) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let j = i; j < Math.min(i + 16, dispatches.length); j++) dispatches[j](pass);
        pass.end();
        device.queue.submit([encoder.finish()]);
      }
    }
    /** @param {GPUBuffer} source @param {number} offset @param {number} size @param {GPUBuffer} stage */
    async function read(source, offset, size, stage) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, offset, stage, 0, size);
      device.queue.submit([encoder.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      try { return new Float32Array(stage.getMappedRange(0, size)).slice(); }
      finally { stage.unmap(); }
    }
    /** @param {Uint32Array} x @param {Uint32Array} y @param {boolean} training */
    async function executeBatch(x, y, training) {
      device.queue.writeBuffer(graph.ids, 0, x);
      device.queue.writeBuffer(graph.targets, 0, y);
      if (training) {
        const iteration = step + 1;
        device.queue.writeBuffer(stepUniform, 0, new Float32Array([iteration, learningRate(config, iteration), 0, 0]));
      }
      submitDispatches(graph.forward);
      if (training) {
        submitDispatches(graph.backward);
        submitDispatches(update);
      }
      const losses = await read(graph.outputLoss, 0, B * T * 4, lossReadback);
      const loss = losses.reduce((sum, l) => sum + l, 0) / losses.length;
      if (!Number.isFinite(loss)) throw new Error(`TabGPT returned a nonfinite loss at step ${training ? step + 1 : step}.`);
      if (training) step++;
      return loss;
    }
    /** @param {Uint32Array} x @param {Uint32Array} y @param {boolean} training */
    function run(x, y, training) {
      return exclusive(async () => {
        checkBatch(x, y, B * T, V);
        try {
          return await executeBatch(x, y, training);
        } catch (error) {
          // A rejected readback can arrive after the optimizer was submitted.
          // Retire this instance so a retry never updates an unknown state twice.
          dispose();
          throw error;
        }
      });
    }
    return {
      get step() { return step; },
      parameterNames: Object.freeze(entries.map(({ name }) => name)),
      numParams: entries.reduce((sum, { count }) => sum + count, 0),
      get memory() { return { ...memory }; },
      trainStep: (x, y) => run(x, y, true),
      evaluate: (x, y) => run(x, y, false),
      readParameter(name) {
        return exclusive(async () => {
          const binding = params.get(name)?.value;
          if (!binding) throw new RangeError(`Unknown TabGPT parameter: ${name}`);
          const stage = allocateBuffer(binding.size, stageUsage, 'parameter-readback');
          try { return await read(values, binding.offset, binding.size, stage); }
          finally { release(stage); }
        });
      },
      destroy,
    };
  }
  try {
    const ops = await createTabGPTOps(device, config);
    const matmul = await createTabGPTMatmul(device, allocateBuffer);
    // Resource and bind-group creation report errors asynchronously. Submit the
    // entire synchronous build between the pushes and pops on this device.
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    try {
      return prepare(ops, matmul);
    } finally {
      const pending = [device.popErrorScope(), device.popErrorScope()];
      const errors = (await Promise.all(pending)).filter(e => e !== null);
      if (errors.length) throw new Error(`TabGPT graph creation failed: ${errors.map(e => e.message).join(', ')}`);
    }
  } catch (error) {
    dispose();
    throw error;
  }
}
