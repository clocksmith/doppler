import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';
import { recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { runGatedResidual } from '../../src/inference/pipelines/text/hyper-connection/decoder-layer.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { resetUniformCache, getCompletedUniformCache } from '../../src/gpu/uniform-cache.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

// Explicit physical recipe: run only in a coordinated clean GPU window.
// Synthetic weights, exact prepared/unprepared bits, no model or timing claim.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, `Physical HC workspace check unavailable: ${probe.reason}`);
assert.ok(!/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(getKernelCapabilities()?.adapterInfo)));
const device = getDevice(), errors = [], rows = [];
const allocationCapBytes = 8 * 1024 * 1024;
setDeviceMemoryBudget(device, allocationCapBytes);
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const workspaces = new Set(), owned = new Set();
const upload = (values, shape) => {
  const buffer = acquireBuffer(values.byteLength);
  owned.add(buffer); uploadData(buffer, values);
  return createTensor(buffer, 'f32', shape);
};
const packed = (n, k) => {
  const values = new Uint8Array(n * k / 32 * 18), view = new DataView(values.buffer);
  for (let block = 0; block < n * k / 32; block++) {
    view.setUint16(block * 18, [0x1400, 0x1800, 0x1c00, 0x9400][block % 4], true);
    for (let i = 0; i < 16; i++) values[block * 18 + 2 + i] = (block * 13 + i * 37) & 255;
  }
  const buffer = acquireBuffer(values.byteLength);
  owned.add(buffer); uploadData(buffer, values);
  return createWeightBuffer(buffer, 'q4_0', 'row', [n, k]);
};
try {
  for (const [streamWidth, projectionRank] of [[64, 32], [256, 64], [2560, 320]]) {
    const streamCount = 4, width = streamCount * streamWidth;
    const streams = upload(Float32Array.from({ length: width }, (_, i) => Math.fround(Math.sin(i * 0.417))), [1, width]);
    const norm = upload(Float32Array.from({ length: width }, (_, i) => Math.fround(0.5 + (i % 13) / 16)), [width]);
    const down = packed(projectionRank, width), up = packed(width, projectionRank);
    const inject = upload(Float32Array.from({ length: streamCount * width }, (_, i) => Math.fround(Math.sin(i * 0.113) / 512)), [streamCount, width]);
    const weights = { norm, down, up, inject };
    const hyper = { streamCount, streamWidth, projectionRank, normEps: 1e-6, normWeightOffset: false };
    for (const role of ['attn_hc', 'mlp_hc']) {
      const workspace = new PreparedDecodeWorkspace(device);
      workspaces.add(workspace);
      let reference;
      for (const mode of ['unprepared', 'prepared', 'prepared-repeat']) {
        const recorder = new CommandRecorder(device, 'hc_workspace_component');
        if (mode !== 'unprepared') recorder.decodeWorkspace = workspace;
        let result;
        try {
          const project = (_role, input, weight, n, k) => weight.dtype === 'q4_0'
            ? recordQ40Matmul(recorder, input, weight, 1, n, k,
              { variant: 'q4_0_gemv', outputDtype: 'f32', transposeB: true })
            : recordMatmul(recorder, input, weight, 1, n, k,
              { outputDtype: 'f32', transposeB: true, bDtype: 'f32' });
          result = await runGatedResidual(recorder, { streams, weights, hyper, numTokens: 1,
            layerIdx: 3, role, project, mixBranch: true });
          await recorder.submitAndWait();
          const bits = { branch: new Uint32Array(await readBuffer(result.branchInput.buffer, streamWidth * 4)),
            gate: new Uint32Array(await readBuffer(result.gate.buffer, streamCount * 4)) };
          if (!reference) reference = bits;
          else assert.deepEqual(bits, reference, `${streamWidth}/${projectionRank}/${role}/${mode}: exact HC output bits`);
          if (mode !== 'unprepared') {
            for (const [stage, bytes] of [['normed', width * 4], ['low_rank_silu', projectionRank * 4], ['inject_gate', streamCount * 4]]) {
              const buffer = workspace.getBuffer(`hc:3:${role}:${stage}`, bytes);
              assert.ok(workspace.ownsBuffer(buffer));
            }
            assert.equal(workspace.getStats().buffers, 4, 'three intermediate slots and the existing fused branch slot');
          }
          rows.push({ streamWidth, projectionRank, role, mode, exact: true, workspace: workspace.getStats() });
        } finally {
          await recorder.abort();
          for (const tensor of [result?.branchInput, result?.gate]) {
            if (tensor && !workspace.ownsBuffer(tensor.buffer)) releaseBuffer(tensor.buffer);
          }
        }
      }
      await device.queue.onSubmittedWorkDone();
      workspace.release(); workspaces.delete(workspace);
    }
    for (const buffer of owned) releaseBuffer(buffer);
    owned.clear();
  }
  assert.deepEqual(errors, []);
} finally {
  try {
    await device.queue.onSubmittedWorkDone();
    for (const workspace of workspaces) workspace.release();
    for (const buffer of owned) releaseBuffer(buffer);
    getCompletedUniformCache(device).clear(); resetUniformCache(device);
    const pool = getBufferPool(device);
    pool.destroy(); await pool.settleDeferredDestruction();
    const retirement = getDeviceMemorySnapshot(device);
    assert.ok(retirement.peakBytes <= allocationCapBytes);
    assert.equal(retirement.liveBytes, 0, 'owned buffers retire before device destruction');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ test: 'hyper-connection-workspace-gpu', rows, allocationCapBytes,
      peakBytes: retirement.peakBytes, retiredLiveBytes: retirement.liveBytes,
      scope: 'synthetic exact HC storage parity; no model/performance qualification' }));
  } finally { destroyDevice(); await releaseNodeWebGPU(); }
}
