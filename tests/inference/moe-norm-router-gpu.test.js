import assert from 'node:assert/strict';

// The fused decode norm + router kernel must reproduce, bit for bit, the three
// dispatches it replaces: rmsnorm/default with the folded attention residual, and
// matmul/gemv_f32_subgroup for the expert router and the shared-expert gate.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`moe-norm-router-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
if (!getKernelCapabilities()?.hasSubgroups) {
  console.log('moe-norm-router-gpu.test: skipped (no subgroups)');
  destroyDevice();
  process.exit(0);
}
const { unifiedKernelWrapper } = await import('../../src/gpu/kernels/kernel-execution.js');
const { recordMoENormRouter } = await import('../../src/gpu/kernels/moe-norm-router.js');
const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { createTensor } = await import('../../src/gpu/tensor.js');

const device = getDevice();
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

function buffer(data, label) {
  const out = device.createBuffer({ label, size: Math.max(16, data.byteLength), usage });
  device.queue.writeBuffer(out, 0, data);
  return out;
}

function empty(bytes, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
}

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

let seed = 0x9e3779b9;
function random() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296 - 0.5;
}
function values(count, scale) {
  return Float32Array.from({ length: count }, () => random() * scale);
}

async function checkCase({ hiddenSize, numExperts, offset, variant = 'f32' }) {
  const eps = 1e-6;
  const input = buffer(values(hiddenSize, 4), 'attn_out');
  const residual = buffer(values(hiddenSize, 8), 'layer_in');
  const normWeight = buffer(Float32Array.from({ length: hiddenSize }, () => (offset ? 0 : 1) + random() * 0.4), 'norm_w');
  const routerWeight = buffer(values(numExperts * hiddenSize, 0.1), 'router_w');
  const sharedWeight = buffer(values(hiddenSize, 0.1), 'shared_w');

  // Reference: the three dispatches the fused kernel replaces.
  const refNormed = empty(hiddenSize * 4, 'ref_normed');
  const refPrenorm = empty(hiddenSize * 4, 'ref_prenorm');
  await unifiedKernelWrapper('rmsnorm', device, 'default', [input, normWeight, refNormed, residual], {
    size: hiddenSize, num_tokens: 1, eps, has_residual: 0, token_stride: 1, output_scale: 1, _pad1: 0, _pad2: 0,
  }, [1, 1, 1], { RMS_NORM_OFFSET: offset, WEIGHT_IS_F16: false, PRE_RESIDUAL: true, OUTPUT_PRENORM: true },
  [{ binding: 5, buffer: refPrenorm }]);
  const gemv = async (weights, columns, output) => {
    const workgroupsX = Math.ceil(columns / 4);
    await unifiedKernelWrapper('matmul', device, 'gemv_f32_subgroup', [refNormed, weights, output], {
      M: 1, N: columns, K: hiddenSize, alpha: 1, transpose_b: 1, workgroups_x: workgroupsX, num_blocks_per_row: 0,
    }, [workgroupsX, 1, 1], { WORKGROUP_SIZE: 128, COLS_PER_WG: 4 });
  };
  const refLogits = empty(numExperts * 4, 'ref_logits');
  const refGate = empty(4, 'ref_gate');
  await gemv(routerWeight, numExperts, refLogits);
  await gemv(sharedWeight, 1, refGate);

  const normed = empty(hiddenSize * 4, 'normed');
  const prenorm = empty(hiddenSize * 4, 'prenorm');
  const logits = empty(numExperts * 4, 'logits');
  const gate = empty(4, 'gate');
  const recorder = new CommandRecorder(device, 'moe_norm_router_test');
  await recordMoENormRouter(recorder, {
    variant, input: createTensor(input, 'f32', [1, hiddenSize], 'attn_out'),
    residual: createTensor(residual, 'f32', [1, hiddenSize], 'layer_in'),
    normWeight, rmsNormWeightOffset: offset, eps,
    routerWeight, sharedRouterWeight: sharedWeight, hiddenSize, numExperts,
    normedOutput: normed, prenormOutput: prenorm, logitsOutput: logits, sharedGateOutput: gate,
  });
  await recorder.submitAndWait();

  // The single pass keeps rmsnorm's bits for the row and scales its own row dots.
  const exact = variant === 'f32'
    ? ['prenorm', 'normed', 'logits', 'shared gate'] : ['prenorm', 'normed'];
  for (const [label, actual, expected, count] of [
    ['prenorm', prenorm, refPrenorm, hiddenSize], ['normed', normed, refNormed, hiddenSize],
    ['logits', logits, refLogits, numExperts], ['shared gate', gate, refGate, 1],
  ]) {
    const a = await readWords(actual, count);
    const e = await readWords(expected, count);
    assert.ok(e.some((word) => word !== 0), `${label} reference was never written`);
    if (!exact.includes(label)) {
      const av = new Float32Array(a.buffer), ev = new Float32Array(e.buffer);
      ev.forEach((value, i) => assert.ok(Math.abs(av[i] - value) <= 1e-5 + 1e-5 * Math.abs(value),
        `${label}[${i}] ${av[i]} vs ${value} (${variant})`));
      continue;
    }
    const first = a.findIndex((word, i) => word !== e[i]);
    assert.equal(first, -1, `${label} differs at ${first} (hidden ${hiddenSize}, offset ${offset}): `
      + `${a[first]?.toString(16)} vs ${e[first]?.toString(16)}`);
  }
}

await checkCase({ hiddenSize: 2048, numExperts: 256, offset: false });
await checkCase({ hiddenSize: 2048, numExperts: 256, offset: true });
await checkCase({ hiddenSize: 1024, numExperts: 64, offset: false });
await checkCase({ hiddenSize: 2048, numExperts: 256, offset: false, variant: 'f32_single_pass' });
await checkCase({ hiddenSize: 2048, numExperts: 256, offset: true, variant: 'f32_single_pass' });
await checkCase({ hiddenSize: 1024, numExperts: 63, offset: false, variant: 'f32_single_pass' });
destroyDevice();
console.log('moe-norm-router-gpu.test: ok');
