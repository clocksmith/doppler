import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture, decodePacked, projectReference } from '../helpers/q4_0-fixture.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { unifiedKernelWrapper } from '../../src/gpu/kernels/kernel-execution.js';
import { q40Dispatch } from '../../src/gpu/kernels/q4_0.js';
import { ggufBlockFormat } from '../../src/formats/gguf/block-storage.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice(), capabilities = getKernelCapabilities();
assert.match(JSON.stringify(capabilities.adapterInfo).toLowerCase(), /apple|m3/);
const errors = [], cases = [], allocated = new Set(), recorders = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = values => {
  const bytes = new Uint8Array(Math.ceil(values.byteLength / 4) * 4);
  bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  const buffer = acquireBuffer(bytes.byteLength);
  allocated.add(buffer);
  uploadData(buffer, bytes);
  return buffer;
};
const run = async (dtype, M, N, K, mode, forcedLimit = null) => {
  const source = packedFixture(dtype, N, K), decoded = decodePacked(dtype, source, N, K);
  const input = new Float32Array(M * K).fill(0.125);
  const expectedRow = projectReference(input.subarray(0, K), decoded, 1, N, K);
  const outputValues = new Float32Array(M * N + 16).fill(-12345);
  const inputBuffer = upload(input), weightsBuffer = upload(source), outputBuffer = upload(outputValues);
  const tiles = Math.ceil(M / 16) * Math.ceil(N / 16);
  const grid = q40Dispatch(forcedLimit ? { limits: { maxComputeWorkgroupsPerDimension: forcedLimit } } : device, tiles);
  assert.ok(grid[0] * grid[1] > tiles, 'fixture must exercise padded groups');
  const recorder = mode === 'recorded' ? createCommandRecorder('native_prefill_padded_grid') : null;
  if (recorder) recorders.push(recorder);
  const format = ggufBlockFormat(dtype);
  await unifiedKernelWrapper('matmul', recorder, `${dtype}_prefill`, [inputBuffer, weightsBuffer, outputBuffer],
    { M, N, K, alpha: 1, blocks: K / format.values, a_offset: 0, b_row: 0, c_offset: 0 }, grid);
  if (recorder) await recorder.submitAndWait();
  const actual = new Float32Array(await readBuffer(outputBuffer, outputValues.byteLength));
  let maxAbs = 0;
  for (let i = 0; i < M * N; i++) {
    assert.ok(Number.isFinite(actual[i]));
    maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expectedRow[i % N]));
  }
  assert.ok(maxAbs <= 0.000001, `${dtype}/${mode}: ${maxAbs}`);
  for (let i = M * N; i < actual.length; i++) assert.equal(actual[i], -12345, 'tail stores must remain in bounds');
  const result = { dtype, M, N, K, mode, grid, requiredGroups: tiles,
    paddingGroups: grid[0] * grid[1] - tiles, maxAbs, canaryElements: 16 };
  cases.push(result);
  console.log(JSON.stringify(result));
};
try {
  for (const dtype of ['q4_0', 'q4_1', 'q5_0', 'q8_0', 'q6k']) {
    for (const mode of ['immediate', 'recorded']) await run(dtype, 17, 33, dtype === 'q6k' ? 256 : 32, mode, 4);
  }
  await run('q4_0', 2287, 8192, 32, 'recorded');
  assert.equal(cases.length, 11);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ timeUTC: new Date().toISOString(), status: 'passed', cases: cases.length,
    sourceSHA256: createHash('sha256').update(readFileSync(new URL('../../src/gpu/kernels/matmul_q4_0.wgsl', import.meta.url))).digest('hex'),
    providerReceipt: probe.providerReceipt, adapter: capabilities.adapterInfo,
    scope: 'Bounded padded-grid physical numerical/bounds/barrier-uniformity qualification; not skipped-work measurement, full-K2048 AMD watchdog proof or full-model performance' }));
} finally {
  for (const recorder of recorders) await recorder.abort();
  await device.queue.onSubmittedWorkDone();
  for (const buffer of allocated) releaseBuffer(buffer);
  destroyDevice();
}
