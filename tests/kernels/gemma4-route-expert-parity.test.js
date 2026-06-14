import assert from 'node:assert/strict';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { runMoEBuildTokenOffsets, runMoEGather, runScatterAddDynamic } from '../../src/gpu/kernels/moe.js';
import { runSiLURowSplit } from '../../src/gpu/kernels/silu.js';
import {
  runGemma4RouteQ4MatmulF16A,
  runScatterAddRoutesF16ExpertScale,
} from '../../src/gpu/kernels/gemma4-route-expert.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';
import { acquireBuffer, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`gemma4-route-expert-parity.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const device = getDevice();

function makePrng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

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

function makeWeights(rows, cols, rng, scale) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = (rng() - 0.5) * scale;
  }
  return out;
}

const rng = makePrng(19);

const numTokens = 2;
const numExperts = 4;
const topK = 2;
const hiddenSize = 2816;
const intermediateSize = 1408;
const gateUpOutDim = intermediateSize * 2;
const maxTokensPerExpert = 2;
const routeCount = numTokens * topK;
const bytesPerF16 = 2;
const q4BlockBytes = 144;

const inputF32 = new Float32Array(numTokens * hiddenSize);
for (let i = 0; i < inputF32.length; i += 1) {
  inputF32[i] = (rng() - 0.5) * 0.25;
}
const inputF16 = f32ToF16Array(inputF32);

const gateUpF32 = makeWeights(numExperts * gateUpOutDim, hiddenSize, rng, 0.15);
const downF32 = makeWeights(numExperts * hiddenSize, intermediateSize, rng, 0.15);
const gateUpQ4 = quantizeToQ4KMRowWise(gateUpF32, [numExperts, gateUpOutDim, hiddenSize]).quantized;
const downQ4 = quantizeToQ4KMRowWise(downF32, [numExperts, hiddenSize, intermediateSize]).quantized;

const indices = new Uint32Array([
  2, 0,
  1, 3,
]);
const weightsF16 = f32ToF16Array(new Float32Array([
  0.50, 0.25,
  0.75, 0.10,
]));
const expertScales = new Float32Array([0.75, 1.25, 0.90, 1.10]);
const countsForOffsets = new Uint32Array(numExperts);
const expectedTokenOffsets = new Uint32Array(routeCount);
const expertRouteSlots = Array.from({ length: numExperts }, () => []);
for (let route = 0; route < routeCount; route += 1) {
  const expert = indices[route];
  expectedTokenOffsets[route] = expert * maxTokensPerExpert + countsForOffsets[expert];
  expertRouteSlots[expert].push(route);
  countsForOffsets[expert] += 1;
}

const inputBuffer = makeStorageBuffer(inputF16, 'route_expert_input');
const gateUpBuffer = makeStorageBuffer(gateUpQ4, 'route_expert_gate_up_q4');
const downBuffer = makeStorageBuffer(downQ4, 'route_expert_down_q4');
const indicesBuffer = makeStorageBuffer(indices, 'route_expert_indices');
const weightsBuffer = makeStorageBuffer(weightsF16, 'route_expert_weights');
const expertScalesBuffer = makeStorageBuffer(expertScales, 'route_expert_scales');

const inputTensor = createTensor(inputBuffer, 'f16', [numTokens, hiddenSize], 'route_expert_input');
const gateUpWeight = createWeightBuffer(
  gateUpBuffer,
  'q4k',
  'row',
  [numExperts, gateUpOutDim, hiddenSize],
  'route_expert_gate_up'
);
const downWeight = createWeightBuffer(
  downBuffer,
  'q4k',
  'row',
  [numExperts, hiddenSize, intermediateSize],
  'route_expert_down'
);

let gathered = null;
let tokenCounts = null;
let tokenMap = null;
let tokenOffsets = null;
let expertOutputs = null;
let dynamicOutput = null;
let routeGateUp = null;
let routeActivated = null;
let routeDown = null;
let routeOutput = null;
const dynamicGateUpByRoute = new Float32Array(routeCount * gateUpOutDim);
const dynamicActivatedByRoute = new Float32Array(routeCount * intermediateSize);

try {
  ({ gathered, tokenCounts, tokenMap } = await runMoEGather(
    inputTensor,
    indicesBuffer,
    numTokens,
    hiddenSize,
    numExperts,
    topK,
    { maxTokensPerExpert }
  ));
  tokenOffsets = await runMoEBuildTokenOffsets(
    tokenCounts,
    tokenMap,
    numTokens,
    numExperts,
    topK,
    maxTokensPerExpert
  );

  expertOutputs = acquireBuffer(
    numExperts * maxTokensPerExpert * hiddenSize * bytesPerF16,
    undefined,
    'route_expert_dynamic_outputs'
  );

  const gateUpStrideBytes = gateUpOutDim * Math.ceil(hiddenSize / 256) * q4BlockBytes;
  const downStrideBytes = hiddenSize * Math.ceil(intermediateSize / 256) * q4BlockBytes;
  const expertStrideBytes = maxTokensPerExpert * hiddenSize * bytesPerF16;
  const counts = new Uint32Array(numExperts);
  for (const expert of indices) {
    counts[expert] += 1;
  }

  for (let expert = 0; expert < numExperts; expert += 1) {
    const count = counts[expert];
    if (count === 0) continue;
    const inputOffset = expert * expertStrideBytes;
    const outputOffset = expert * expertStrideBytes;
    const gateUp = await runMatmul(
      gathered,
      gateUpWeight,
      count,
      gateUpOutDim,
      hiddenSize,
      {
        transposeB: true,
        aOffset: inputOffset,
        bOffset: expert * gateUpStrideBytes,
        outputDtype: 'f16',
        role: 'moe_gate_up',
      }
    );
    const gateUpData = readF16AsF32(await readBuffer(gateUp.buffer, count * gateUpOutDim * bytesPerF16));
    for (let slot = 0; slot < count; slot += 1) {
      const route = expertRouteSlots[expert][slot];
      dynamicGateUpByRoute.set(
        gateUpData.subarray(slot * gateUpOutDim, (slot + 1) * gateUpOutDim),
        route * gateUpOutDim
      );
    }
    const activated = await runSiLURowSplit(gateUp, {
      numTokens: count,
      dim: intermediateSize,
      activation: 'gelu',
      swigluLimit: null,
    });
    const activatedData = readF16AsF32(await readBuffer(activated.buffer, count * intermediateSize * bytesPerF16));
    for (let slot = 0; slot < count; slot += 1) {
      const route = expertRouteSlots[expert][slot];
      dynamicActivatedByRoute.set(
        activatedData.subarray(slot * intermediateSize, (slot + 1) * intermediateSize),
        route * intermediateSize
      );
    }
    releaseBuffer(gateUp.buffer);

    await runMatmul(
      activated,
      downWeight,
      count,
      hiddenSize,
      intermediateSize,
      {
        transposeB: true,
        bOffset: expert * downStrideBytes,
        outputBuffer: expertOutputs,
        cOffset: outputOffset,
        outputDtype: 'f16',
        role: 'moe_down',
      }
    );
    releaseBuffer(activated.buffer);
  }

  dynamicOutput = await runScatterAddDynamic(
    createTensor(expertOutputs, 'f16', [numExperts, maxTokensPerExpert, hiddenSize], 'route_expert_dynamic_outputs'),
    indicesBuffer,
    weightsBuffer,
    tokenOffsets,
    numTokens,
    hiddenSize,
    topK,
    {
      weightsDtype: 'f16',
      perExpertScale: expertScalesBuffer,
    }
  );

  routeGateUp = await runGemma4RouteQ4MatmulF16A(
    inputTensor,
    indicesBuffer,
    gateUpWeight,
    {
      numRoutes: routeCount,
      topK,
      N: gateUpOutDim,
      K: hiddenSize,
      inputMode: 'token',
      label: 'route_expert_route_gate_up',
    }
  );
  const routeGateUpData = readF16AsF32(await readBuffer(routeGateUp.buffer, routeCount * gateUpOutDim * bytesPerF16));
  let maxGateUpAbsDiff = 0;
  let maxGateUpIndex = -1;
  for (let i = 0; i < dynamicGateUpByRoute.length; i += 1) {
    const diff = Math.abs(dynamicGateUpByRoute[i] - routeGateUpData[i]);
    if (diff > maxGateUpAbsDiff) {
      maxGateUpAbsDiff = diff;
      maxGateUpIndex = i;
    }
  }
  console.log(`gemma4-route-expert-parity: gate_up_max_abs_diff=${maxGateUpAbsDiff.toExponential(3)} gate_up_max_index=${maxGateUpIndex}`);

  routeActivated = await runSiLURowSplit(routeGateUp, {
    numTokens: routeCount,
    dim: intermediateSize,
    activation: 'gelu',
    swigluLimit: null,
  });
  const routeActivatedData = readF16AsF32(await readBuffer(routeActivated.buffer, routeCount * intermediateSize * bytesPerF16));
  let maxActivatedAbsDiff = 0;
  let maxActivatedIndex = -1;
  for (let i = 0; i < dynamicActivatedByRoute.length; i += 1) {
    const diff = Math.abs(dynamicActivatedByRoute[i] - routeActivatedData[i]);
    if (diff > maxActivatedAbsDiff) {
      maxActivatedAbsDiff = diff;
      maxActivatedIndex = i;
    }
  }
  console.log(`gemma4-route-expert-parity: activated_max_abs_diff=${maxActivatedAbsDiff.toExponential(3)} activated_max_index=${maxActivatedIndex}`);

  routeDown = await runGemma4RouteQ4MatmulF16A(
    routeActivated,
    indicesBuffer,
    downWeight,
    {
      numRoutes: routeCount,
      topK,
      N: hiddenSize,
      K: intermediateSize,
      inputMode: 'route',
      label: 'route_expert_route_down',
    }
  );
  routeOutput = await runScatterAddRoutesF16ExpertScale(
    routeDown,
    indicesBuffer,
    weightsBuffer,
    expertScalesBuffer,
    numTokens,
    hiddenSize,
    topK
  );

  const dynamicDown = readF16AsF32(await readBuffer(
    expertOutputs,
    numExperts * maxTokensPerExpert * hiddenSize * bytesPerF16
  ));
  const routeDownData = readF16AsF32(await readBuffer(routeDown.buffer, routeCount * hiddenSize * bytesPerF16));
  let maxDownAbsDiff = 0;
  let maxDownIndex = -1;
  for (let route = 0; route < routeCount; route += 1) {
    const dynamicBase = expectedTokenOffsets[route] * hiddenSize;
    const routeBase = route * hiddenSize;
    for (let dim = 0; dim < hiddenSize; dim += 1) {
      const diff = Math.abs(dynamicDown[dynamicBase + dim] - routeDownData[routeBase + dim]);
      if (diff > maxDownAbsDiff) {
        maxDownAbsDiff = diff;
        maxDownIndex = routeBase + dim;
      }
    }
  }
  console.log(`gemma4-route-expert-parity: down_max_abs_diff=${maxDownAbsDiff.toExponential(3)} down_max_index=${maxDownIndex}`);

  const dynamic = readF16AsF32(await readBuffer(dynamicOutput.buffer, numTokens * hiddenSize * bytesPerF16));
  const route = readF16AsF32(await readBuffer(routeOutput.buffer, numTokens * hiddenSize * bytesPerF16));

  let maxAbsDiff = 0;
  let maxIndex = -1;
  for (let i = 0; i < dynamic.length; i += 1) {
    const diff = Math.abs(dynamic[i] - route[i]);
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      maxIndex = i;
    }
  }

  console.log(`gemma4-route-expert-parity: max_abs_diff=${maxAbsDiff.toExponential(3)} max_index=${maxIndex}`);
  assert.ok(
    maxAbsDiff < 2e-3,
    `route expert path diverged from dynamic path: max_abs_diff=${maxAbsDiff} max_index=${maxIndex}`
  );
} finally {
  if (routeOutput?.buffer) releaseBuffer(routeOutput.buffer);
  if (routeDown?.buffer) releaseBuffer(routeDown.buffer);
  if (routeActivated?.buffer) releaseBuffer(routeActivated.buffer);
  if (routeGateUp?.buffer) releaseBuffer(routeGateUp.buffer);
  if (dynamicOutput?.buffer) releaseBuffer(dynamicOutput.buffer);
  if (expertOutputs) releaseBuffer(expertOutputs);
  if (tokenOffsets) releaseBuffer(tokenOffsets);
  if (tokenCounts) releaseBuffer(tokenCounts);
  if (tokenMap) releaseBuffer(tokenMap);
  if (gathered?.buffer) releaseBuffer(gathered.buffer);
  inputBuffer.destroy();
  gateUpBuffer.destroy();
  downBuffer.destroy();
  indicesBuffer.destroy();
  weightsBuffer.destroy();
  expertScalesBuffer.destroy();
  destroyDevice();
}

console.log('gemma4-route-expert-parity.test: ok');
