import assert from 'node:assert/strict';
import { test } from 'node:test';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { getBufferPool, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { getCompletedUniformCache, resetUniformCache } from '../../src/gpu/uniform-cache.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { executeSharedSlotDown } from '../../src/gpu/kernels/moe-shared-slot.js';
import { runResidualAdd, recordResidualAdd } from '../../src/gpu/kernels/residual.js';

const H = 2560, I = 640, TOP_K = 10, E = 10, CAP = 128 * 1024 * 1024;
const FUSED = 'pair_combine_f32', SPLIT = 'pair_f32';

// This is an operator diagnostic, not model performance or peer equivalence.
test('Flash-shaped split down preserves routed/shared math within a bounded physical ABBA diagnostic', { timeout: 30000 }, async t => {
  const owned = new Set(), errors = [], parity = [], batches = [];
  let device, pool, baseline, memory, provider;
  try {
    const probe = await probeNodeGPU({ installFileFetchShim: true });
    assert(probe.ready, `Physical shared-slot diagnostic requires WebGPU: ${probe.reason}`);
    device = getDevice(); pool = getBufferPool();
    assert.doesNotMatch(JSON.stringify(getKernelCapabilities().adapterInfo), /software|swiftshader|llvmpipe|lavapipe/i);
    provider = probe.providerReceipt?.selectedProviderId ?? null;
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    setDeviceMemoryBudget(device, CAP);
    baseline = pool.getStats().activeBuffers;
    const allocate = (size, label) => {
      const buffer = device.createBuffer({ size, label,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.add(buffer); return buffer;
    };
    const upload = (data, label) => {
      const buffer = allocate(data.byteLength, label); uploadData(buffer, data); return buffer;
    };
    const activated = createTensor(upload(Float32Array.from({ length: (TOP_K + 1) * I }, (_, n) => Math.sin(n * 0.37) * 0.01),
      'split_down_input'), 'f32', [TOP_K + 1, I]);
    const residual = createTensor(upload(Float32Array.from({ length: H }, (_, n) => Math.cos(n * 0.13) * 0.01),
      'split_down_residual'), 'f32', [1, H]);
    const indices = upload(new Uint32Array([0, 3, 3, 9, 1, 7, 5, 0xffffffff, 2, 8]), 'split_down_indices');
    const routeWeights = upload(Float32Array.from({ length: TOP_K }, (_, n) => n === 4 ? 0 : (n + 1) / 55), 'split_down_route_weights');
    const gate = createTensor(upload(new Float32Array(1), 'split_down_gate'), 'f32', [1, 1]);
    const sharedDown = createWeightBuffer(upload(packedFixture('q8_0', H, I), 'split_down_shared_q8'), 'q8_0', 'row', [H, I]);
    const slots = allocate((TOP_K + 1) * H * 4, 'split_down_slots');
    const outputs = [allocate(H * 4, 'split_down_fused'), allocate(H * 4, 'split_down_split')];
    const outer = [allocate(H * 4, 'split_down_fused_outer'), allocate(H * 4, 'split_down_split_outer')];
    const geometry = { numTokens: 1, numExperts: E, topK: TOP_K, hiddenSize: H,
      intermediateSize: I, sharedSize: I, swigluLimit: null, indices, routeWeights, slotsBuffer: slots };
    const compare = (a, b) => {
      const fa = new Float32Array(a), fb = new Float32Array(b), ua = new Uint32Array(a), ub = new Uint32Array(b);
      let differingBits = 0, maxAbs = 0;
      for (let n = 0; n < H; n++) {
        if (ua[n] !== ub[n]) differingBits++;
        const delta = Math.abs(fa[n] - fb[n]); maxAbs = Math.max(maxAbs, delta);
        assert(Number.isFinite(fa[n]) && Number.isFinite(fb[n]) && delta <= 2e-4 + 2e-5 * Math.abs(fa[n]),
          `Split down differs at ${n}: ${fa[n]} vs ${fb[n]}`);
      }
      return { exact: differingBits === 0, differingBits, maxAbs };
    };
    for (const dtype of ['q4_0', 'q8_0']) {
      t.signal.throwIfAborted();
      const downBuffer = upload(packedFixture(dtype, E * H, I), `split_down_routed_${dtype}`);
      const weights = { down: createWeightBuffer(downBuffer, dtype, 'row', [E, H, I]), sharedDown };
      try {
        uploadData(indices, new Uint32Array([0, 3, 3, 9, 1, 7, 5, 0xffffffff, 2, 8]));
        uploadData(routeWeights, Float32Array.from({ length: TOP_K }, (_, n) => n === 4 ? 0 : (n + 1) / 55));
        for (const scalarGate of [-20, 0, 20]) for (const mode of ['immediate', 'recorded']) {
          uploadData(gate.buffer, Float32Array.of(scalarGate));
          for (const [index, variant] of [FUSED, SPLIT].entries()) {
            const recorder = mode === 'recorded' ? createCommandRecorder('split_down_parity') : null;
            try {
              const result = await executeSharedSlotDown(recorder, activated, weights, gate,
                { ...geometry, variant, outputBuffer: outputs[index] });
              // Both variants hand the same result to the existing caller residual kernel.
              if (recorder) await recordResidualAdd(recorder, result, residual, H, { outputBuffer: outer[index] });
              else await runResidualAdd(result, residual, H, { outputBuffer: outer[index] });
              if (recorder) await recorder.submitAndWait();
              else await device.queue.onSubmittedWorkDone();
            } finally { await recorder?.abort(); }
          }
          const down = compare(await readBuffer(outputs[0], H * 4), await readBuffer(outputs[1], H * 4));
          const residualAdded = compare(await readBuffer(outer[0], H * 4), await readBuffer(outer[1], H * 4));
          parity.push({ dtype, scalarGate, mode, down, residualAdded });
          t.signal.throwIfAborted();
        }
        // Both variants are compiled and executed above. Eight fixed batches:
        // ABBAABBA, ten down operators each, no allocations of outputs or slots.
        uploadData(indices, Uint32Array.from({ length: TOP_K }, (_, n) => n));
        uploadData(routeWeights, Float32Array.from({ length: TOP_K }, (_, n) => (n + 1) / 55));
        uploadData(gate.buffer, Float32Array.of(0));
        for (const [batch, variant] of [FUSED, SPLIT, SPLIT, FUSED, FUSED, SPLIT, SPLIT, FUSED].entries()) {
          t.signal.throwIfAborted();
          const started = performance.now(), recorder = createCommandRecorder('split_down_abba');
          try {
            for (let repeat = 0; repeat < 10; repeat++) {
              await executeSharedSlotDown(recorder, activated, weights, gate,
                { ...geometry, variant, outputBuffer: outputs[variant === FUSED ? 0 : 1] });
            }
            await recorder.submitAndWait();
            batches.push({ dtype, batch, variant, operators: 10, wallMs: performance.now() - started });
          } finally { await recorder.abort(); }
        }
      } finally { await device.queue.onSubmittedWorkDone(); downBuffer.destroy(); owned.delete(downBuffer); }
    }
    assert.deepEqual(errors, []);
    await device.queue.onSubmittedWorkDone();
    for (const buffer of owned) buffer.destroy(); owned.clear();
    await pool.settleDeferredDestruction();
    assert.equal(pool.getStats().activeBuffers, baseline, 'All operation-owned pooled buffers return to baseline');
    getCompletedUniformCache(device).clear(); resetUniformCache(device);
    pool.destroy(); await pool.settleDeferredDestruction();
    memory = getDeviceMemorySnapshot(device);
    assert.equal(memory.maxBytes, CAP); assert(memory.peakBytes <= CAP); assert.equal(memory.rejected, 0);
    assert.equal(memory.liveBytes, 0, 'Owned fixture, uniforms and readback pool retire completely');
    console.log(JSON.stringify({ scope: 'Physical operator only; ABBA wall includes JS recording, submission and queue completion; not GPU timestamps or peer comparison',
      provider, geometry: { tokens: 1, hidden: H, intermediate: I, shared: I, topK: TOP_K, experts: E },
      budgetBytes: CAP, peakApiBytes: memory.peakBytes, finalApiBytes: memory.liveBytes,
      poolBaseline: baseline, finalActiveBuffers: pool.getStats().activeBuffers, uncapturedErrors: errors.length, parity, batches }));
  } finally {
    try {
      if (device) {
        try { await device.queue.onSubmittedWorkDone(); }
        finally {
          for (const buffer of owned) buffer.destroy();
          getCompletedUniformCache(device).clear(); resetUniformCache(device);
          pool?.destroy(); await pool?.settleDeferredDestruction(); destroyDevice();
        }
      }
    } finally { await releaseNodeWebGPU(); }
  }
});
