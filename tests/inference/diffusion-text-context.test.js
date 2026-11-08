import assert from 'node:assert/strict';
import { planSd3TextContext, assembleSd3TextContext, bytesPerElementForDtype } from '../../src/experimental/diffusion/text-context.js';

// SD3 joint context: [CLIP-L | CLIP-G | zeros] rows, then the T5 rows. The
// pipeline used to hand the transformer only the T5 hidden state.
{
  const plan = planSd3TextContext({ clipRows: 77, clipDims: [768, 1280], t5Rows: 77, jointDim: 4096, bytesPerElement: 2 });
  assert.equal(plan.rows, 154);
  assert.equal(plan.clipWidth, 2048);
  assert.equal(plan.paddedPerClipRow, 2048);
  assert.equal(plan.totalBytes, 154 * 4096 * 2);
  assert.equal(plan.copies.length, 77 * 2 + 1);
  assert.deepEqual(plan.copies[0], { source: 0, sourceOffset: 0, destOffset: 0, byteLength: 768 * 2 });
  assert.deepEqual(plan.copies[1], { source: 0, sourceOffset: 768 * 2, destOffset: 4096 * 2, byteLength: 768 * 2 });
  assert.deepEqual(plan.copies[77], { source: 1, sourceOffset: 0, destOffset: 768 * 2, byteLength: 1280 * 2 });
  assert.deepEqual(plan.copies[154], { source: 2, sourceOffset: 0, destOffset: 77 * 4096 * 2, byteLength: 77 * 4096 * 2 });
  for (const copy of plan.copies) {
    assert.equal(copy.sourceOffset % 4, 0);
    assert.equal(copy.destOffset % 4, 0);
    assert.equal(copy.byteLength % 4, 0);
  }
}
assert.throws(() => planSd3TextContext({ clipRows: 77, clipDims: [3000, 1280], t5Rows: 77, jointDim: 4096, bytesPerElement: 2 }), /CLIP width 4280 exceeds the joint width 4096/);
assert.throws(() => planSd3TextContext({ clipRows: 77, clipDims: [], t5Rows: 77, jointDim: 4096, bytesPerElement: 2 }), /at least one CLIP hidden size/);
assert.throws(() => planSd3TextContext({ clipRows: 1, clipDims: [3], t5Rows: 1, jointDim: 8, bytesPerElement: 2 }), /multiple of 4 bytes/);
assert.equal(bytesPerElementForDtype('f16'), 2);
assert.equal(bytesPerElementForDtype('f32'), 4);
assert.throws(() => bytesPerElementForDtype('bf16'), /f16 or f32/);

// GPU assembly: the copies land where the plan says and the pad is zero even
// when the pool returns a dirty buffer.
const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`diffusion-text-context.test: planner ok; GPU assembly skipped (${gpuProbe.reason})`);
  process.exit(0);
}
const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { destroyDevice } = await import('../../src/gpu/device.js');

function upload(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}
try {
  const clipRows = 2;
  const clipDims = [4, 6];
  const t5Rows = 3;
  const jointDim = 12;
  const clip = Float32Array.from({ length: clipRows * clipDims[0] }, (_, i) => 100 + i);
  const clip2 = Float32Array.from({ length: clipRows * clipDims[1] }, (_, i) => 200 + i);
  const t5 = Float32Array.from({ length: t5Rows * jointDim }, (_, i) => 300 + i);
  // Dirty a pooled buffer of the output size first so the clear is exercised.
  const dirty = upload(Float32Array.from({ length: (clipRows + t5Rows) * jointDim }, () => -1), 'dirty');
  releaseBuffer(dirty);
  const context = await assembleSd3TextContext({
    clipHidden: [
      createTensor(upload(clip, 'clip'), 'f32', [clipRows, clipDims[0]], 'clip'),
      createTensor(upload(clip2, 'clip2'), 'f32', [clipRows, clipDims[1]], 'clip2'),
    ],
    t5Hidden: createTensor(upload(t5, 't5'), 'f32', [t5Rows, jointDim], 't5'),
  });
  assert.deepEqual(context.shape, [clipRows + t5Rows, jointDim]);
  assert.equal(context.dtype, 'f32');
  const out = new Float32Array(await readBuffer(context.buffer, (clipRows + t5Rows) * jointDim * 4));
  const expected = [];
  for (let row = 0; row < clipRows; row += 1) {
    expected.push(...clip.subarray(row * 4, row * 4 + 4), ...clip2.subarray(row * 6, row * 6 + 6), 0, 0);
  }
  expected.push(...t5);
  assert.deepEqual(Array.from(out), expected);
  releaseBuffer(context.buffer);
  await assert.rejects(
    () => assembleSd3TextContext({
      clipHidden: [createTensor(upload(clip, 'clip'), 'f16', [clipRows, clipDims[0]], 'clip')],
      t5Hidden: createTensor(upload(t5, 't5'), 'f32', [t5Rows, jointDim], 't5'),
    }),
    /dtype "f16" differs from T5 dtype "f32"/
  );
  console.log('diffusion-text-context.test: ok (GPU)');
} finally {
  await destroyDevice?.();
}
