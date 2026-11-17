import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { runQ40Matmul, recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';
import { runHyperConnectionUpMixBranch, recordHyperConnectionUpMixBranch } from '../../src/gpu/kernels/hyper-connection-up.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { resetUniformCache, getCompletedUniformCache } from '../../src/gpu/uniform-cache.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import {
  runHyperConnectionActivation, recordHyperConnectionActivation, runHyperConnectionBranch, recordHyperConnectionBranch,
  runHyperConnectionMixBranch, recordHyperConnectionMixBranch,
} from '../../src/gpu/kernels/hyper-connection.js';

// Short synthetic component check, never a learned-model/performance gate.
// Exact bits match q4_0_gemv + separate refined sigmoid/branch AND the v20 fused mix.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, `Physical mix-branch check unavailable: ${probe.reason}`);
assert.ok(!/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(getKernelCapabilities()?.adapterInfo)));
const device = getDevice(), owned = new Set(), errors = [], rows = [];
const allocationCapBytes = 8 * 1024 * 1024;
setDeviceMemoryBudget(device, allocationCapBytes);
const workspace = new PreparedDecodeWorkspace(device);
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = (values, shape) => {
  const buffer = acquireBuffer(values.byteLength);
  owned.add(buffer);
  uploadData(buffer, values);
  return createTensor(buffer, 'f32', shape);
};
try {
  for (const [width, rank] of [[65, 32], [257, 352], [2560, 320]]) {
    const tokens = 1, streams = 4;
    const options = { numTokens: tokens, streamCount: streams, streamWidth: width };
    const count = streams * width;
    const values = Float32Array.from({ length: rank }, (_, i) => Math.fround(Math.sin(i * 0.731) * (i % 13 ? 1 : 100)));
    const norm = Float32Array.from({ length: count }, (_, i) => Math.fround(Math.sin(i * 0.417) * (i % 11 ? 1 : 1e5)));
    const packed = new Uint8Array(count * (rank / 32) * 18);
    const packedView = new DataView(packed.buffer);
    for (let block = 0; block < count * rank / 32; block++) {
      packedView.setUint16(block * 18, [0, 0x8000, 0x0400, 0x3800, 0x3c00, 0xbc00][block % 6], true);
      for (let i = 0; i < 16; i++) packed[block * 18 + 2 + i] = (block * 13 + i * 37) & 255;
    }
    const x = upload(values, [1, rank]), n = upload(norm, [1, count]);
    const weightBuffer = acquireBuffer(packed.byteLength);
    owned.add(weightBuffer);
    uploadData(weightBuffer, packed);
    const weights = createWeightBuffer(weightBuffer, 'q4_0', 'row', [count, rank]);
    let prior = null;
    for (const mode of ['immediate', 'recorded', 'prepared', 'prepared-repeat']) {
      const recorder = mode === 'immediate' ? null : new CommandRecorder(device, 'hc_up_mix_branch_component');
      if (mode.startsWith('prepared')) recorder.decodeWorkspace = workspace;
      const outputBuffer = mode.startsWith('prepared') ? workspace.getBuffer(`component:${tokens}:${streams}:${width}`, tokens * width * 4) : null;
      try {
        const matmulOptions = { variant: 'q4_0_gemv', outputDtype: 'f32', transposeB: true };
        const up = recorder ? await recordQ40Matmul(recorder, x, weights, 1, count, rank, matmulOptions)
          : await runQ40Matmul(x, weights, 1, count, rank, matmulOptions);
        owned.add(up.buffer);
        const sigmoidOptions = { mode: 'mix_sigmoid', rows: tokens, columns: count, streamCount: streams };
        const mix = recorder ? await recordHyperConnectionActivation(recorder, up, sigmoidOptions)
          : await runHyperConnectionActivation(up, sigmoidOptions);
        owned.add(mix.buffer);
        const separate = recorder ? await recordHyperConnectionBranch(recorder, mix, n, options)
          : await runHyperConnectionBranch(mix, n, options);
        owned.add(separate.buffer);
        const current = recorder ? await recordHyperConnectionMixBranch(recorder, up, n, options)
          : await runHyperConnectionMixBranch(up, n, options);
        owned.add(current.buffer);
        const fused = recorder ? await recordHyperConnectionUpMixBranch(recorder, x, weights, n, { ...options, projectionRank: rank, outputBuffer })
          : await runHyperConnectionUpMixBranch(x, weights, n, { ...options, projectionRank: rank, outputBuffer });
        if (!outputBuffer) owned.add(fused.buffer);
        else assert.equal(fused.buffer, outputBuffer, 'supplied output stays caller-owned');
        if (recorder) await recorder.submitAndWait();
        const expected = new Uint32Array(await readBuffer(separate.buffer, tokens * width * 4));
        const actual = new Uint32Array(await readBuffer(fused.buffer, tokens * width * 4));
        const v20 = new Uint32Array(await readBuffer(current.buffer, width * 4));
        assert.deepEqual(actual, v20, `${mode}: fused up must match v20 projection + fused mix`);
        assert.deepEqual(actual, expected, `${mode}: fused arithmetic must match separate kernels bit for bit`);
        if (mode === 'prepared') prior = outputBuffer;
        if (mode === 'prepared-repeat') assert.equal(outputBuffer, prior, 'same prepared slot is reused');
        rows.push({ tokens, streams, width, rank, mode, elements: actual.length, exact: true });
      } finally { await recorder?.abort(); }
    }
  }
  assert.deepEqual(errors, []);

} finally {
  try {
    await device.queue.onSubmittedWorkDone();
    workspace.release();
    assert.equal(workspace.getStats().bytes, 0);
    for (const buffer of owned) releaseBuffer(buffer);
    getCompletedUniformCache(device).clear();
    resetUniformCache(device);
    const pool = getBufferPool(device);
    pool.destroy();
    await pool.settleDeferredDestruction();
    const retirement = getDeviceMemorySnapshot(device);
    assert.ok(retirement.peakBytes <= allocationCapBytes, 'all observed GPUBuffer allocations must fit the 8 MiB cap');
    assert.equal(retirement.liveBytes, 0, 'retirement must reach zero before device destruction/loss');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ test: 'hyper-connection-up-gpu', adapter: getKernelCapabilities()?.adapterInfo, rows,
      allocationCapBytes, peakBytes: retirement.peakBytes, retiredLiveBytes: retirement.liveBytes,
      scope: 'synthetic exact operator parity and bounded retirement, no model or timing claim' }));
  } finally {
    destroyDevice();
    await releaseNodeWebGPU();
  }
}
