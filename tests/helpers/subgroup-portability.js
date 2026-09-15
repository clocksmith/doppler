import { readbackBuffer } from '../../src/gpu/readback-buffer.js';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function execute(device, source, width, values, workgroups, outputs) {
  const buffers = [];
  try {
    const module = device.createShaderModule({ code: source });
    const info = await module.getCompilationInfo();
    assert(info.messages.every(message => message.type !== 'error'), JSON.stringify(info.messages));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: width } } });
    for (const [index, value] of values.entries()) {
      const buffer = device.createBuffer({ size: value.byteLength,
        usage: (index === 0 ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE)
          | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      buffers.push(buffer);
      device.queue.writeBuffer(buffer, 0, value);
    }
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(...workgroups); pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return await Promise.all(outputs.map(async index => new Float32Array(
      await readbackBuffer(device, buffers[index], values[index].byteLength))));
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

function assertClose(actual, expected, label) {
  assert(actual.length === expected.length, `${label}: output length mismatch`);
  for (let index = 0; index < actual.length; index++) {
    const error = Math.abs(actual[index] - expected[index]);
    assert(error <= 3e-5 * Math.max(1, Math.abs(expected[index])),
      `${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

function permuteLocalIndex(source, name) {
  const original = `let ${name} = local_id.x;`;
  assert(source.split(original).length === 2, `Missing unique data-index assignment: ${name}`);
  // Bijection changes the relation between data indices and physical subgroups.
  // This is instrumented geometry coverage, not a claimed hardware lane mapping.
  return source.replace(original,
    `let ${name} = (local_id.x % 8u) * (WORKGROUP_SIZE / 8u) + local_id.x / 8u;`);
}

export async function runSubgroupPortabilityCases(device, { stats, portableStats, attention }) {
let rmsCases = 0, attentionCases = 0;
  for (const width of [8, 32, 128, 256]) {
    for (const hidden of [1, 17, 65, 257, 513]) {
      const tokens = 3;
      const input = Float32Array.from({ length: hidden * tokens }, (_, i) => Math.sin(i * 0.37));
      const residual = Float32Array.from(input, (_, i) => Math.cos(i * 0.13));
      const sum = Float32Array.from(input, (v, i) => v + residual[i]);
      const expected = Float32Array.from({ length: tokens }, (_, token) => {
        const row = sum.subarray(token * hidden, (token + 1) * hidden);
        return 1 / Math.sqrt(row.reduce((total, value) => total + value * value, 0) / hidden + 1e-5);
      });
      const uniform = new Uint32Array([hidden, tokens, 0, 1]);
      new Float32Array(uniform.buffer)[2] = 1e-5;
      for (const [kind, source] of [['subgroup', stats], ['permuted', permuteLocalIndex(stats, 'thread_index')], ['portable', portableStats]]) {
        const [observedSum, observedRms] = await execute(device, source, width,
          [uniform, input, residual, new Float32Array(sum.length), new Float32Array(tokens)], [tokens, 1, 1], [3, 4]);
        assertClose(observedSum, sum, `prenorm:${width}:${hidden}:${kind}`);
        assertClose(observedRms, expected, `inv_rms:${width}:${hidden}:${kind}`); rmsCases++;
      }
    }
  }
  for (const width of [32, 128, 256]) {
    for (const headDim of [1, 17, Math.min(65, width)]) {
      for (const kvLen of [1, 5, 129]) {
        const heads = 2, scale = 1 / Math.sqrt(headDim);
        const q = Float32Array.from({ length: heads * headDim }, (_, i) => Math.sin(i * 0.17));
        const k = Float32Array.from({ length: kvLen * headDim }, (_, i) => Math.cos(i * 0.23));
        const v = Float32Array.from(k, (_, i) => Math.sin(i * 0.41));
        const uniform = new Uint32Array(16);
        uniform.set([heads, 1, headDim, kvLen, 1, 0, 0, kvLen - 1]);
        new Float32Array(uniform.buffer)[5] = scale;
        const expected = new Float32Array(q.length);
        for (let h = 0; h < heads; h++) {
          const scores = Array.from({ length: kvLen }, (_, t) => {
            let dot = 0;
            for (let d = 0; d < headDim; d++) dot += q[h * headDim + d] * k[t * headDim + d];
            return dot * scale;
          });
          const max = Math.max(...scores), weights = scores.map(score => Math.exp(score - max));
          const denom = weights.reduce((a, b) => a + b, 0);
          for (let d = 0; d < headDim; d++) {
            let result = 0;
            for (let t = 0; t < kvLen; t++) result += weights[t] / denom * v[t * headDim + d];
            expected[h * headDim + d] = result;
          }
        }
        for (const source of [attention, permuteLocalIndex(attention, 'tid')]) {
          const [observed] = await execute(device, source, width,
            [uniform, q, k, v, new Float32Array(q.length), new Uint32Array([kvLen]), new Uint32Array([0])], [heads, 1, 1], [4]);
          assertClose(observed, expected, `attention:${width}:${headDim}:${kvLen}`); attentionCases++;
        }
      }
    }
  }
  return { rmsCases, attentionCases };
}
