import assert from 'node:assert/strict';

// The fused decode linear-attention input norm + a/b kernel against the three dispatches
// it replaces: rmsnorm/default without residual, and matmul/gemv_f32_subgroup for
// in_proj_a and in_proj_b. f32 reproduces all three bit for bit; f32_single_pass
// reproduces the normalized row and a/b within rounding.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`linear-norm-ab-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
if (!getKernelCapabilities()?.hasSubgroups) {
  console.log('linear-norm-ab-gpu.test: skipped (no subgroups)');
  destroyDevice();
  process.exit(0);
}
const { unifiedKernelWrapper } = await import('../../src/gpu/kernels/kernel-execution.js');
const { recordLinearNormAB } = await import('../../src/gpu/kernels/linear-norm-ab.js');
const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { createTensor } = await import('../../src/gpu/tensor.js');

const device = getDevice();
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const buffer = (data, label) => {
  const out = device.createBuffer({ label, size: Math.max(16, data.byteLength), usage });
  device.queue.writeBuffer(out, 0, data);
  return out;
};
const empty = (bytes, label) => device.createBuffer({ label, size: Math.max(16, bytes), usage });
async function readWords(source, count) {
  const staging = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return words;
}

let seed = 0x7f4a7c15;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296 - 0.5;
};
const values = (count, scale) => Float32Array.from({ length: count }, () => random() * scale);

async function checkCase({ hiddenSize, rows, offset, variant = 'f32' }) {
  const eps = 1e-6;
  const input = buffer(values(hiddenSize, 8), 'layer_in');
  const normWeight = buffer(Float32Array.from({ length: hiddenSize }, () => (offset ? 0 : 1) + random() * 0.4), 'norm_w');
  const aWeight = buffer(values(rows * hiddenSize, 0.1), 'a_w');
  const bWeight = buffer(values(rows * hiddenSize, 0.1), 'b_w');

  // Reference: the three dispatches the fused kernel replaces.
  const refNormed = empty(hiddenSize * 4, 'ref_normed');
  // rmsnorm.js binds the input as the unused residual and a placeholder prenorm output.
  await unifiedKernelWrapper('rmsnorm', device, 'default', [input, normWeight, refNormed, input], {
    size: hiddenSize, num_tokens: 1, eps, has_residual: 0, token_stride: 1, output_scale: 1, _pad1: 0, _pad2: 0,
  }, [1, 1, 1], { RMS_NORM_OFFSET: offset, WEIGHT_IS_F16: false, PRE_RESIDUAL: false, OUTPUT_PRENORM: false },
  [{ binding: 5, buffer: empty(4, 'prenorm_placeholder') }]);
  const gemv = async (weights, output) => {
    const workgroupsX = Math.ceil(rows / 4);
    await unifiedKernelWrapper('matmul', device, 'gemv_f32_subgroup', [refNormed, weights, output], {
      M: 1, N: rows, K: hiddenSize, alpha: 1, transpose_b: 1, workgroups_x: workgroupsX, num_blocks_per_row: 0,
    }, [workgroupsX, 1, 1], { WORKGROUP_SIZE: 128, COLS_PER_WG: 4 });
  };
  const refA = empty(rows * 4, 'ref_a');
  const refB = empty(rows * 4, 'ref_b');
  await gemv(aWeight, refA);
  await gemv(bWeight, refB);

  const normed = empty(hiddenSize * 4, 'normed');
  const a = empty(rows * 4, 'a');
  const b = empty(rows * 4, 'b');
  const recorder = new CommandRecorder(device, 'linear_norm_ab_test');
  await recordLinearNormAB(recorder, {
    variant, input: createTensor(input, 'f32', [1, hiddenSize], 'layer_in'),
    normWeight, rmsNormWeightOffset: offset, eps, aWeight, bWeight, hiddenSize, rows,
    normedOutput: normed, aOutput: a, bOutput: b,
  });
  await recorder.submitAndWait();

  for (const [label, actual, expected, count] of [
    ['normed', normed, refNormed, hiddenSize], ['a', a, refA, rows], ['b', b, refB, rows],
  ]) {
    const got = await readWords(actual, count);
    const want = await readWords(expected, count);
    assert.ok(want.some((word) => word !== 0), `${label} reference was never written`);
    if (variant === 'f32_single_pass' && label !== 'normed') {
      const g = new Float32Array(got.buffer), w = new Float32Array(want.buffer);
      w.forEach((value, i) => assert.ok(Math.abs(g[i] - value) <= 1e-5 + 1e-5 * Math.abs(value),
        `${label}[${i}] ${g[i]} vs ${value} (${variant})`));
      continue;
    }
    const first = got.findIndex((word, i) => word !== want[i]);
    assert.equal(first, -1, `${label} differs at ${first} (${variant}, hidden ${hiddenSize}, rows ${rows}, offset ${offset}): `
      + `${got[first]?.toString(16)} vs ${want[first]?.toString(16)}`);
  }
}

await checkCase({ hiddenSize: 2048, rows: 32, offset: false });
await checkCase({ hiddenSize: 2048, rows: 32, offset: true });
await checkCase({ hiddenSize: 1024, rows: 16, offset: false });
await checkCase({ hiddenSize: 2052, rows: 5, offset: false });
await checkCase({ hiddenSize: 2048, rows: 32, offset: false, variant: 'f32_single_pass' });
await checkCase({ hiddenSize: 2048, rows: 32, offset: true, variant: 'f32_single_pass' });
await checkCase({ hiddenSize: 2052, rows: 5, offset: false, variant: 'f32_single_pass' });
destroyDevice();
console.log('linear-norm-ab-gpu.test: ok');
