import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { destroyDevice, getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { observeKernelDispatches } from '../helpers/kernel-dispatch-observer.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  if (process.env.DOPPLER_REQUIRE_MATMUL_GEOMETRY_GPU === '1') throw new Error(probe.reason);
  console.log(`matmul-workgroup-geometry-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const source = JSON.parse(await readFile(new URL('../../src/config/kernels/registry.json', import.meta.url), 'utf8'));
const report = { provider: probe.providerReceipt, capabilities: getKernelCapabilities(), cases: [] };

async function runCase(variant, overrides, M, transpose) {
  const schema = structuredClone(source.operations.matmul.variants[variant]);
  // The sole JSON change is shader override geometry, never the duplicate hand fields.
  Object.assign(schema.wgslOverrides, overrides);
  const restore = enterKernelRegistry(createKernelRegistry({ extensions: { matmul: { variants: { [variant]: schema } } } }));
  const N = 35, K = 67;
  const buffers = [];
  let recorder = null;
  const own = buffer => { buffers.push(buffer); return buffer; };
  const upload = values => {
    const buffer = own(acquireBuffer(values.byteLength));
    uploadData(buffer, values);
    return buffer;
  };
  try {
    const inputValues = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.17) * 0.125);
    const weightValues = new Float32Array(N * K);
    for (let column = 0; column < N; column++) for (let k = 0; k < K; k++) {
      weightValues[transpose ? column * K + k : k * N + column] = Math.cos(column * 0.13 + k * 0.07) * 0.125;
    }
    const reference = new Float64Array(M * N);
    for (let row = 0; row < M; row++) for (let column = 0; column < N; column++) {
      for (let k = 0; k < K; k++) reference[row * N + column] += inputValues[row * K + k]
        * weightValues[transpose ? column * K + k : k * N + column];
    }
    const input = createTensor(upload(inputValues), 'f32', [M, K]);
    const weight = createWeightBuffer(upload(weightValues), 'f32', transpose ? 'row' : 'column',
      transpose ? [N, K] : [K, N]);
    const step = { op: 'q_proj', kernel: schema.wgsl, entry: 'main' };
    const kernelPath = { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };
    const options = { role: 'q_proj', kernelPath, transposeB: transpose, outputDtype: 'f32' };
    const immediate = await observeKernelDispatches(async () => {
      const result = await runMatmul(input, weight, M, N, K, options);
      own(result.buffer);
      return result;
    });
    const config = getMatmulConfig(variant, null);
    const expectedGrid = overrides.TILE_SIZE
      ? [Math.ceil(M / overrides.TILE_SIZE), Math.ceil(N / overrides.TILE_SIZE), 1]
      : [Math.ceil(N / overrides.COLS_PER_WG), 1, 1];
    assert.equal(immediate.dispatches.length, 1);
    assert.deepEqual(immediate.dispatches[0].workgroups, expectedGrid);
    assert.equal(immediate.dispatches[0].pipeline.variant, variant);
    for (const [name, value] of Object.entries(overrides)) {
      assert.equal(immediate.dispatches[0].pipeline.constants[name], value);
    }
    const values = new Float32Array(await readBuffer(immediate.result.buffer, M * N * 4));
    let maxAbsoluteError = 0;
    for (let index = 0; index < values.length; index++) {
      const error = Math.abs(values[index] - reference[index]);
      assert.ok(Number.isFinite(values[index]) && error <= 1e-6 + 1e-5 * Math.abs(reference[index]),
        `${variant}[${index}] missed the CPU reference after JSON geometry change`);
      maxAbsoluteError = Math.max(maxAbsoluteError, error);
    }
    const recorded = await observeKernelDispatches(async () => {
      recorder = new CommandRecorder(null, 'json_matmul_geometry');
      const result = await recordMatmul(recorder, input, weight, M, N, K, options);
      own(result.buffer);
      await recorder.submitAndWait();
      return result;
    });
    assert.deepEqual(recorded.dispatches, immediate.dispatches);
    assert.deepEqual(new Float32Array(await readBuffer(recorded.result.buffer, M * N * 4)), values);
    report.cases.push({ variant, M, N, K, transpose, overrides, workgroupSize: config.workgroupSize,
      immediateDispatches: immediate.dispatches, recordedDispatches: recorded.dispatches, maxAbsoluteError });
  } finally {
    if (recorder && !recorder.getStats().submitted) await recorder.abort();
    await getDevice().queue.onSubmittedWorkDone();
    for (const buffer of new Set(buffers)) releaseBuffer(buffer);
    restore();
  }
}

try {
  for (const transpose of [true, false]) {
    await runCase('f32_compensated', { TILE_SIZE: 8 }, 17, transpose);
    await runCase('gemv_f32_compensated', { THREADS_PER_COL: 7, COLS_PER_WG: 3 }, 1, transpose);
    await runCase('gemv_f32', { THREADS_PER_COL: 7, COLS_PER_WG: 3 }, 1, transpose);
  }
  console.log(`matmul-workgroup-geometry-gpu.test: ok ${JSON.stringify(report)}`);
} finally {
  await destroyDevice();
  await releaseNodeWebGPU();
}
