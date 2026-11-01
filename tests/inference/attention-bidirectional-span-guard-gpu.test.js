import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { runAttention } from '../../src/gpu/kernels/attention.js';
import { createTensor } from '../../src/gpu/tensor.js';

// A bidirectional span is honored only by attention kernels whose uniform contract
// declares it. attention_small (prefill_small) has no span mask, so a span selected onto
// it fails instead of being masked causally; the span-capable query tile accepts it.
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`attention-bidirectional-span-guard: skipped (${probe.reason})`);
  process.exit(0);
}
const device = getDevice();
const owned = [];
const tensor = (shape) => {
  const elements = shape.reduce((a, b) => a * b, 1);
  const buffer = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(buffer, 0, new Float32Array(elements).fill(0.01));
  owned.push(buffer);
  return createTensor(buffer, 'f32', shape);
};
const kernelPath = (kernel) => ({ activationDtype: 'f32', kvDtype: 'f32', prefill: { steps: [{ op: 'attention', kernel, entry: 'main' }] } });
const [T, K, H, D] = [9, 24, 16, 256];
const Q = tensor([T, H, D]), Kt = tensor([K, 2, D]), V = tensor([K, 2, D]);
const options = { seqLen: T, kvLen: K, numKVHeads: 2, scale: 1 / 16, causal: true, startPos: 8,
  bidirectionalSpanStart: 10, bidirectionalSpanLength: 12 };
try {
  await assert.rejects(() => runAttention(Q, Kt, V, null, H, D, { ...options, kernelPath: kernelPath('attention_small.wgsl') }),
    /"prefill_small" has no bidirectional-span mask/);
  const output = await runAttention(Q, Kt, V, null, H, D, { ...options, kernelPath: kernelPath('attention_prefill_query_tile_f32.wgsl') });
  assert.deepEqual(output.shape, [T, H, D]);
  owned.push(output.buffer);
  console.log('attention-bidirectional-span-guard: ok');
} finally {
  for (const buffer of owned) buffer.destroy();
  destroyDevice();
}
