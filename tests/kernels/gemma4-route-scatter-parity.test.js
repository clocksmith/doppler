import assert from 'node:assert/strict';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runScatterAddDynamic } from '../../src/gpu/kernels/moe.js';
import { runScatterAddRoutesF16ExpertScale } from '../../src/gpu/kernels/gemma4-route-expert.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`gemma4-route-scatter-parity.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const device = getDevice();

function makeStorageBuffer(data, label) {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function readF16AsF32(raw) {
  const bits = new Uint16Array(raw);
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) {
    out[i] = f16ToF32Bits(bits[i]);
  }
  return out;
}

const numTokens = 3;
const hiddenSize = 5;
const numExperts = 4;
const topK = 2;
const maxTokensPerExpert = 3;
const routeCount = numTokens * topK;

const indices = new Uint32Array([
  2, 0,
  1, 2,
  3, 1,
]);
const weightsF16 = f32ToF16Array(new Float32Array([
  0.50, 0.25,
  0.75, 0.10,
  0.40, 0.35,
]));
const expertScales = new Float32Array([0.75, 1.25, 0.90, 1.10]);

const routeOutputsF32 = new Float32Array(routeCount * hiddenSize);
for (let i = 0; i < routeOutputsF32.length; i += 1) {
  routeOutputsF32[i] = ((i % 7) - 3) * 0.125 + Math.floor(i / hiddenSize) * 0.03125;
}
const routeOutputsF16 = f32ToF16Array(routeOutputsF32);

const expertCounts = new Uint32Array(numExperts);
const tokenOffsets = new Uint32Array(routeCount);
const expertOutputsF16 = new Uint16Array(numExperts * maxTokensPerExpert * hiddenSize);
for (let route = 0; route < routeCount; route += 1) {
  const expert = indices[route];
  const slot = expertCounts[expert];
  assert.ok(slot < maxTokensPerExpert, 'fixture exceeded maxTokensPerExpert');
  expertCounts[expert] += 1;
  const flatSlot = expert * maxTokensPerExpert + slot;
  tokenOffsets[route] = flatSlot;
  expertOutputsF16.set(
    routeOutputsF16.subarray(route * hiddenSize, (route + 1) * hiddenSize),
    flatSlot * hiddenSize
  );
}

const routeOutputsBuffer = makeStorageBuffer(routeOutputsF16, 'route_scatter_route_outputs');
const expertOutputsBuffer = makeStorageBuffer(expertOutputsF16, 'route_scatter_dynamic_outputs');
const indicesBuffer = makeStorageBuffer(indices, 'route_scatter_indices');
const weightsBuffer = makeStorageBuffer(weightsF16, 'route_scatter_weights');
const tokenOffsetsBuffer = makeStorageBuffer(tokenOffsets, 'route_scatter_offsets');
const expertScalesBuffer = makeStorageBuffer(expertScales, 'route_scatter_expert_scales');

let dynamicOutput = null;
let routeOutput = null;
try {
  dynamicOutput = await runScatterAddDynamic(
    createTensor(expertOutputsBuffer, 'f16', [numExperts, maxTokensPerExpert, hiddenSize], 'dynamic_expert_outputs'),
    indicesBuffer,
    weightsBuffer,
    tokenOffsetsBuffer,
    numTokens,
    hiddenSize,
    topK,
    {
      weightsDtype: 'f16',
      perExpertScale: expertScalesBuffer,
    }
  );
  routeOutput = await runScatterAddRoutesF16ExpertScale(
    createTensor(routeOutputsBuffer, 'f16', [routeCount, hiddenSize], 'route_outputs'),
    indicesBuffer,
    weightsBuffer,
    expertScalesBuffer,
    numTokens,
    hiddenSize,
    topK
  );

  const dynamic = readF16AsF32(await readBuffer(dynamicOutput.buffer, numTokens * hiddenSize * 2));
  const route = readF16AsF32(await readBuffer(routeOutput.buffer, numTokens * hiddenSize * 2));

  let maxAbsDiff = 0;
  let maxIndex = -1;
  for (let i = 0; i < dynamic.length; i += 1) {
    const diff = Math.abs(dynamic[i] - route[i]);
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      maxIndex = i;
    }
  }

  console.log(`gemma4-route-scatter-parity: max_abs_diff=${maxAbsDiff.toExponential(3)} max_index=${maxIndex}`);
  assert.ok(
    maxAbsDiff === 0,
    `route scatter diverged from dynamic scatter: max_abs_diff=${maxAbsDiff} max_index=${maxIndex}`
  );
} finally {
  if (dynamicOutput?.buffer) releaseBuffer(dynamicOutput.buffer);
  if (routeOutput?.buffer) releaseBuffer(routeOutput.buffer);
  routeOutputsBuffer.destroy();
  expertOutputsBuffer.destroy();
  indicesBuffer.destroy();
  weightsBuffer.destroy();
  tokenOffsetsBuffer.destroy();
  expertScalesBuffer.destroy();
  destroyDevice();
}

console.log('gemma4-route-scatter-parity.test: ok');
