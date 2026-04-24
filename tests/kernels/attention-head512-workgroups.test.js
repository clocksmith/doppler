import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runAttention } from '../../src/gpu/kernels/attention.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '../..');
const registry = JSON.parse(readFileSync(join(rootDir, 'src/config/kernels/registry.json'), 'utf8'));
const head512 = registry.operations.attention.variants.prefill_head512_f16kv;

assert.equal(head512.variantMetadata.queryBlockSize, 16);
assert.match(
  readFileSync(join(rootDir, 'src/gpu/kernels/attention_head512_f16kv.wgsl'), 'utf8'),
  /const BLOCK_SIZE:\s*u32\s*=\s*16u;/
);

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`attention-head512-workgroups.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const seqLen = 19;
const kvLen = 19;
const numHeads = 32;
const numKVHeads = 4;
const headDim = 512;
const qValues = new Float32Array(seqLen * numHeads * headDim);
const kValues = new Float32Array(kvLen * numKVHeads * headDim);
const vValues = new Float32Array(kvLen * numKVHeads * headDim);

for (let pos = 0; pos < kvLen; pos += 1) {
  for (let kvHead = 0; kvHead < numKVHeads; kvHead += 1) {
    for (let dim = 0; dim < headDim; dim += 1) {
      vValues[(pos * numKVHeads + kvHead) * headDim + dim] = (pos + 1) * 0.01 + kvHead * 0.001 + dim * 0.000001;
    }
  }
}

const vF16 = f32ToF16Array(vValues);
const kF16 = f32ToF16Array(kValues);
const device = getDevice();

assert.ok(device, 'WebGPU device should be initialized by probeNodeGPU');

function makeBuffer(data, label) {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function expectedUniformCausal(token, head, dim) {
  const kvHead = Math.floor(head / (numHeads / numKVHeads));
  let sum = 0;
  for (let pos = 0; pos <= token; pos += 1) {
    const raw = vF16[(pos * numKVHeads + kvHead) * headDim + dim];
    sum += f16ToF32Bits(raw);
  }
  return sum / (token + 1);
}

const qBuffer = makeBuffer(qValues, 'attention_head512_q');
const kBuffer = makeBuffer(kF16, 'attention_head512_k');
const vBuffer = makeBuffer(vF16, 'attention_head512_v');
const outputBuffer = makeBuffer(new Float32Array(seqLen * numHeads * headDim), 'attention_head512_output');

try {
  const kernelPath = {
    id: 'attention-head512-workgroups-test',
    name: 'attention-head512-workgroups-test',
    activationDtype: 'f32',
    prefill: {
      steps: [
        { op: 'attention', kernel: 'attention_head512_f16kv.wgsl', entry: 'main' },
      ],
    },
    decode: { steps: [] },
  };
  const output = await runAttention(
    createTensor(qBuffer, 'f32', [seqLen, numHeads, headDim], 'attention_head512_q'),
    createTensor(kBuffer, 'f16', [kvLen, numKVHeads * headDim], 'attention_head512_k'),
    createTensor(vBuffer, 'f16', [kvLen, numKVHeads * headDim], 'attention_head512_v'),
    null,
    numHeads,
    headDim,
    {
      seqLen,
      kvLen,
      numKVHeads,
      causal: true,
      scale: 1,
      layerIdx: 5,
      kernelPath,
      outputBuffer,
    }
  );
  const actual = new Float32Array(await readBuffer(output.buffer, seqLen * numHeads * headDim * 4));
  const checkedToken = 18;
  const checkedOffset = (checkedToken * numHeads) * headDim;
  const expected = expectedUniformCausal(checkedToken, 0, 0);
  assert.ok(
    Math.abs(actual[checkedOffset] - expected) < 1e-3,
    `head512 prefill must dispatch tail query block: got ${actual[checkedOffset]}, expected ${expected}`
  );
} finally {
  qBuffer.destroy();
  kBuffer.destroy();
  vBuffer.destroy();
  outputBuffer.destroy();
  destroyDevice();
}

console.log('attention-head512-workgroups.test: ok');
