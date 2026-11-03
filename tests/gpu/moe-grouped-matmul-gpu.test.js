import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// moe_grouped_matmul_f32.wgsl at its registered geometries against a dequantized CPU
// reference: Q4_0 split gate/up (token input) and Q4_1 down (route input), plus a Q8_0
// split case. Groups of up to eight routes per expert as moe_grouped_sort lays them out:
// [first, count, expert, 0] per group, then the route ids at pair_offset.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`moe-grouped-matmul-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, destroyDevice } = await import('../../src/gpu/device.js');
const { getKernelConfig } = await import('../../src/gpu/kernels/kernel-configs.js');

const device = getDevice();
const code = await readFile(new URL('../../src/gpu/kernels/moe_grouped_matmul_f32.wgsl', import.meta.url), 'utf8');
const module = device.createShaderModule({ code });
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const owned = [];
const upload = (data) => {
  const bytes = new Uint8Array(Math.ceil(Math.max(data.byteLength, 16) / 4) * 4);
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = device.createBuffer({ size: bytes.byteLength, usage });
  device.queue.writeBuffer(buffer, 0, bytes);
  owned.push(buffer);
  return buffer;
};
let seed = 0x1b873593;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
};
const f16 = (bits) => {
  const e = (bits >> 10) & 31, m = bits & 1023, s = bits >> 15 ? -1 : 1;
  return s * (e === 0 ? m / 1024 * 2 ** -14 : (1 + m / 1024) * 2 ** (e - 15));
};
const FORMATS = { 0: { bytes: 18, values: (b, o) => Array.from({ length: 32 }, (_, i) => f16(b[o] | b[o + 1] << 8)
  * ((i < 16 ? b[o + 2 + i] & 15 : b[o + 2 + i - 16] >> 4) - 8)) },
1: { bytes: 20, values: (b, o) => Array.from({ length: 32 }, (_, i) => f16(b[o] | b[o + 1] << 8)
  * (i < 16 ? b[o + 4 + i] & 15 : b[o + 4 + i - 16] >> 4) + f16(b[o + 2] | b[o + 3] << 8)) },
2: { bytes: 34, values: (b, o) => Array.from({ length: 32 }, (_, i) => f16(b[o] | b[o + 1] << 8)
  * ((b[o + 2 + i] << 24) >> 24)) } };

function bank(experts, rows, blocks, format) {
  const { bytes } = FORMATS[format];
  const data = Uint8Array.from({ length: experts * rows * blocks * bytes }, () => (random() * 256) | 0);
  for (let block = 0; block < experts * rows * blocks; block++) {
    data[block * bytes + 1] = 0x20 | (data[block * bytes + 1] & 3);
    if (format === 1) data[block * bytes + 3] = 0xa0 | (data[block * bytes + 3] & 3);
  }
  return data;
}
function row(data, format, index, blocks) {
  const { bytes, values } = FORMATS[format];
  return Array.from({ length: blocks }, (_, b) => values(data, (index * blocks + b) * bytes)).flat();
}

async function check({ variant, op, format, tokens, topK, experts, K, N }) {
  const config = getKernelConfig(op, variant);
  const overrides = { ...config.wgslOverrides, QUANT_FORMAT: format, BLOCK_BYTES: FORMATS[format].bytes };
  const { ROWS_PER_THREAD: rows, GROUP_SIZE: groupSize, INPUT_IS_TOKEN: inputIsToken, SPLIT_GATE_UP: split } = overrides;
  const routes = tokens * topK, blocks = K / 32, bankRows = split ? N / 2 : N;
  const byExpert = Array.from({ length: experts }, () => []);
  for (let route = 0; route < routes; route++) byExpert[(random() * experts) | 0].push(route);
  const groups = [], order = [];
  byExpert.forEach((ids, expert) => {
    for (let i = 0; i < ids.length; i += groupSize) {
      groups.push(order.length, Math.min(groupSize, ids.length - i), expert, 0);
      order.push(...ids.slice(i, i + groupSize));
    }
  });
  const pairOffset = groups.length;
  const inputRows = inputIsToken ? tokens : routes;
  const input = Float32Array.from({ length: inputRows * K }, () => random() * 2 - 1);
  const gate = bank(experts, bankRows, blocks, format);
  const up = split ? bank(experts, bankRows, blocks, format) : new Uint8Array(16);
  const output = device.createBuffer({ size: routes * N * 4, usage });
  owned.push(output);
  const uniforms = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  owned.push(uniforms);
  device.queue.writeBuffer(uniforms, 0, new Uint32Array([routes, experts, topK, pairOffset, N, K, 0, 0]));
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: config.entryPoint,
    constants: Object.fromEntries(Object.entries(overrides).map(([key, value]) => [key, Number(value)])) } });
  const buffers = [uniforms, upload(input), upload(gate), upload(up), upload(new Uint32Array([...groups, ...order])), output];
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
  pass.dispatchWorkgroups(Math.ceil(N / (32 * rows)), groups.length / 4);
  pass.end();
  const staging = device.createBuffer({ size: routes * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  encoder.copyBufferToBuffer(output, 0, staging, 0, routes * N * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const actual = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  let maxAbs = 0;
  groups.forEach((_, g) => {
    if (g % 4) return;
    const [first, count, expert] = groups.slice(g, g + 3);
    for (const route of order.slice(first, first + count)) {
      const x = input.subarray((inputIsToken ? Math.floor(route / topK) : route) * K, (inputIsToken ? Math.floor(route / topK) : route) * K + K);
      for (let r = 0; r < N; r++) {
        const useUp = split && r >= bankRows;
        const w = row(useUp ? up : gate, format, expert * bankRows + (split ? r % bankRows : r), blocks);
        const expected = w.reduce((sum, value, k) => sum + value * x[k], 0);
        const error = Math.abs(actual[route * N + r] - expected);
        maxAbs = Math.max(maxAbs, error);
        assert.ok(error <= 1e-4 + 1e-5 * Math.abs(expected), `${op}/${variant} Q${format} route ${route} row ${r}: ${actual[route * N + r]} vs ${expected}`);
      }
    }
  });
  console.log(JSON.stringify({ op, variant, format, tokens, experts, K, N, groups: groups.length / 4, maxAbs }));
}

const errors = [];
device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
try {
  await check({ variant: 'native_f32', op: 'moe_grouped_gu', format: 0, tokens: 21, topK: 4, experts: 6, K: 512, N: 128 });
  await check({ variant: 'native_f32', op: 'moe_grouped_down', format: 1, tokens: 21, topK: 4, experts: 6, K: 288, N: 256 });
  await check({ variant: 'native_f32', op: 'moe_grouped_gu', format: 2, tokens: 9, topK: 3, experts: 4, K: 256, N: 96 });
  // Odd block counts exercise native half-word alignment across row boundaries.
  await check({ variant: 'native_f32', op: 'moe_grouped_gu', format: 0, tokens: 9, topK: 3, experts: 5, K: 288, N: 96 });
  await check({ variant: 'native_f32', op: 'moe_grouped_gu', format: 2, tokens: 9, topK: 3, experts: 5, K: 288, N: 96 });
  assert.deepEqual(errors, []);
  console.log('moe-grouped-matmul-gpu.test: ok');
} finally {
  for (const buffer of owned) buffer.destroy();
  destroyDevice();
}
