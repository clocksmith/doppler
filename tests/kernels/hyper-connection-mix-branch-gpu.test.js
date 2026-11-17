import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
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
// Exact bits must match the actual separate refined-sigmoid + branch kernels.
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
  for (const [tokens, streams, width] of [[1, 4, 65], [3, 5, 257], [1, 4, 2560]]) {
    const options = { numTokens: tokens, streamCount: streams, streamWidth: width };
    const count = tokens * streams * width;
    const values = Float32Array.from({ length: count }, (_, i) => [-100, -80, -18, -1e-7, 0, 1e-7, 1, 16, 80, 100][i % 10]);
    const norm = Float32Array.from({ length: count }, (_, i) => Math.fround(Math.sin(i * 0.417) * (i % 11 ? 1 : 1e5)));
    const x = upload(values, [tokens, streams * width]), n = upload(norm, [tokens, streams * width]);
    let prior = null;
    for (const mode of ['immediate', 'recorded', 'prepared', 'prepared-repeat']) {
      const recorder = mode === 'immediate' ? null : new CommandRecorder(device, 'hc_mix_branch_component');
      if (mode.startsWith('prepared')) recorder.decodeWorkspace = workspace;
      const outputBuffer = mode.startsWith('prepared') ? workspace.getBuffer(`component:${tokens}:${streams}:${width}`, tokens * width * 4) : null;
      try {
        const sigmoidOptions = { mode: 'mix_sigmoid', rows: tokens, columns: streams * width, streamCount: streams };
        const mix = recorder ? await recordHyperConnectionActivation(recorder, x, sigmoidOptions)
          : await runHyperConnectionActivation(x, sigmoidOptions);
        owned.add(mix.buffer);
        const separate = recorder ? await recordHyperConnectionBranch(recorder, mix, n, options)
          : await runHyperConnectionBranch(mix, n, options);
        owned.add(separate.buffer);
        const fused = recorder ? await recordHyperConnectionMixBranch(recorder, x, n, { ...options, outputBuffer })
          : await runHyperConnectionMixBranch(x, n, options);
        if (!outputBuffer) owned.add(fused.buffer);
        else assert.equal(fused.buffer, outputBuffer, 'supplied output stays caller-owned');
        if (recorder) await recorder.submitAndWait();
        const expected = new Uint32Array(await readBuffer(separate.buffer, tokens * width * 4));
        const actual = new Uint32Array(await readBuffer(fused.buffer, tokens * width * 4));
        assert.deepEqual(actual, expected, `${mode}: fused arithmetic must match separate kernels bit for bit`);
        if (mode === 'prepared') prior = outputBuffer;
        if (mode === 'prepared-repeat') assert.equal(outputBuffer, prior, 'same prepared slot is reused');
        rows.push({ tokens, streams, width, mode, elements: actual.length, exact: true });
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
    console.log(JSON.stringify({ test: 'hyper-connection-mix-branch-gpu', adapter: getKernelCapabilities()?.adapterInfo, rows,
      allocationCapBytes, peakBytes: retirement.peakBytes, retiredLiveBytes: retirement.liveBytes,
      scope: 'synthetic exact operator parity and bounded retirement, no model or timing claim' }));
  } finally {
    destroyDevice();
    await releaseNodeWebGPU();
  }
}
