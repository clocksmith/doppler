import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { observeKernelDispatches } from '../helpers/kernel-dispatch-observer.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { f16ToF32Array } from '../../src/inference/kv-cache/types.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const file = process.env.DOPPLER_FLASH_NEXT_ROUTER_PROBE;
if (!file) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_ROUTER_PROBE.');
  console.log('Flash-Next router diagnostic: skipped (no provisioned captured input)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(file), run = JSON.parse(runBytes);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.manifestSha256);
  const manifest = JSON.parse(manifestBytes), descriptor = manifest.tensors[run.tensor];
  assert(descriptor, `Declared tensor is absent: ${run.tensor}`);
  const [N, K] = descriptor.shape, [M, inputWidth] = run.inputShape;
  assert.equal(inputWidth, K);
  assert(run.rows.every(row => Number.isSafeInteger(row) && row >= 0 && row < M));
  assert.equal(descriptor.dtype, 'F16');
  const weightsBytes = Buffer.alloc(descriptor.size);
  let cursor = 0;
  for (const span of descriptor.spans ?? [{ shardIndex: descriptor.shard, offset: descriptor.offset, size: descriptor.size }]) {
    const handle = await fs.open(path.join(run.modelDir, manifest.shards[span.shardIndex].filename));
    try {
      const { bytesRead } = await handle.read(weightsBytes, cursor, span.size, span.offset);
      assert.equal(bytesRead, span.size); cursor += bytesRead;
    } finally { await handle.close(); }
  }
  assert.equal(cursor, descriptor.size);
  const raw = await fs.readFile(run.inputFile); assert.equal(hash(raw), run.inputSha256);
  const all = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  assert.equal(all.length, M * K);
  const input = all;
  const weights = f16ToF32Array(new Uint16Array(weightsBytes.buffer, weightsBytes.byteOffset, N * K));
  const exact = new Float64Array(run.rows.length * N);
  for (let m = 0; m < run.rows.length; m++) for (let n = 0; n < N; n++) {
    let sum = 0, correction = 0;
    for (let k = 0; k < K; k++) {
      const product = input[run.rows[m] * K + k] * weights[n * K + k], next = sum + product;
      correction += Math.abs(sum) >= Math.abs(product) ? (sum - next) + product : (product - next) + sum;
      sum = next;
    }
    exact[m * N + n] = sum + correction;
  }
  const probe = await probeNodeGPU({ installFileFetchShim: true }); assert(probe.ready, probe.reason);
  const buffers = [], gpuErrors = [];
  getDevice().addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const own = buffer => { buffers.push(buffer); return buffer; };
  const upload = values => { const buffer = own(acquireBuffer(values.byteLength)); uploadData(buffer, values); return buffer; };
  const role = run.role ?? 'moe_router';
  const report = { scope: run.scope ?? 'Captured learned router input; existing kernels only, no model or speed qualification',
    process: { pid: process.pid, node: process.versions.node },
    runSha256: hash(runBytes), manifestSha256: hash(manifestBytes), weightsSha256: hash(weightsBytes),
    inputSha256: hash(raw), M, N, K, oracle: 'Neumaier Float64 sum of exact F32 inputs and expanded F16 weights',
    provider: probe.providerReceipt, cases: [], gpuErrors };
  await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
  let primaryError = null;
  try {
    const a = createTensor(upload(input), 'f32', [M, K]);
    const candidates = [
      ['matmul_f32.wgsl', 'f32', weights],
      ['matmul_f16w_f32a_tiled.wgsl', 'f16', weightsBytes],
      ['matmul_compensated_f32.wgsl', 'f32', weights],
      ['matmul_f32_row_subgroup.wgsl', 'f32', weights],
    ];
    const requested = run.kernels ?? candidates.slice(0, 3).map(row => row[0]);
    assert(requested.length && new Set(requested).size === requested.length);
    assert(requested.every(kernel => candidates.some(row => row[0] === kernel)), 'only declared diagnostic kernels are allowed');
    for (const [kernel, dtype, data] of candidates.filter(row => requested.includes(row[0]))) {
      const step = { op: role, kernel, entry: 'main' };
      const b = createWeightBuffer(upload(data), dtype, 'row', [N, K]);
      const observed = await observeKernelDispatches(async () => {
        const output = await runMatmul(a, b, M, N, K, { role, transposeB: true,
          outputDtype: 'f32', phaseOverride: 'prefill', kernelPath: { activationDtype: 'f32',
            prefill: { steps: [step] }, decode: { steps: [step] } } });
        own(output.buffer); return output;
      });
      assert.equal(observed.dispatches[0].pipeline.shaderFile, kernel);
      const values = new Float32Array(await readBuffer(observed.result.buffer, M * N * 4));
      let maxAbsoluteError = 0, nearestF32Mismatches = 0;
      for (let m = 0; m < run.rows.length; m++) for (let n = 0; n < N; n++) {
        const actual = values[run.rows[m] * N + n], expected = exact[m * N + n];
        assert(Number.isFinite(actual));
        maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual - expected));
        if (actual !== Math.fround(expected)) nearestF32Mismatches++;
      }
      // The runtime ranks stored F32 logits. Distinct exact dots can round to
      // the same logit, so an F64 ordering is not an F32 top-k oracle.
      const roundedExact = Float32Array.from(exact);
      const order = values => Array.from({ length: N }, (_, n) => n).sort((a, b) => values[b] - values[a] || a - b).slice(0, 10);
      const rows = run.rows.map((position, m) => ({ position,
        expected: order(roundedExact.subarray(m * N, (m + 1) * N)), actual: order(values.subarray(position * N, (position + 1) * N)),
        ...(run.retainRows === true ? { logits: Array.from(values.subarray(position * N, (position + 1) * N)) } : {}) }));
      report.cases.push({ kernel, maxAbsoluteError, nearestF32Mismatches, rows, dispatches: observed.dispatches });
      if (kernel === 'matmul_compensated_f32.wgsl') {
        assert.equal(nearestF32Mismatches, 0);
        for (const row of rows) assert.deepEqual(row.actual, row.expected);
      } else if (kernel === 'matmul_f32.wgsl' && run.requireBaselineReversal !== false) {
        assert(rows.some(row => JSON.stringify(row.actual) !== JSON.stringify(row.expected)),
          'the captured full-batch baseline must reproduce an expert-choice reversal');
      }
    }
    await getDevice().queue.onSubmittedWorkDone(); assert.deepEqual(gpuErrors, []); report.completed = true;
  } catch (error) {
    primaryError = error;
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    const failures = [];
    try { await getDevice()?.queue.onSubmittedWorkDone(); } catch (error) { failures.push(error); }
    for (const buffer of new Set(buffers)) {
      try { releaseBuffer(buffer); } catch (error) { failures.push(error); }
    }
    try { destroyDevice(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU(); } catch (error) { failures.push(error); }
    report.cleanupFailures = failures.map(error => String(error));
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(primaryError ? [primaryError, ...failures] : failures,
      'Router diagnostic cleanup failed', { cause: primaryError });
  }
}
