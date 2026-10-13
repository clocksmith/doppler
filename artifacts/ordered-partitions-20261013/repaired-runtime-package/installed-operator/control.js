// Exact installed wrapper control. Fixture/reference remain the original F64 equations.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { cases, reference } from './fixture.js';

const [root, destination] = process.argv.slice(2);
assert(root && destination);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const reservation = JSON.parse(readFileSync(new URL('./reservation.json', import.meta.url)));
const cg = readFileSync('/proc/self/cgroup', 'utf8').trim().split(':').at(-1);
assert(cg.endsWith('/reploid-installed-operator-20261013.scope'));
writeFileSync('/sys/fs/cgroup' + cg + '/memory.oom.group', '1');
for (const [key, value] of Object.entries({ 'memory.high': '10737418240',
  'memory.max': '12884901888', 'memory.swap.max': '0', 'memory.oom.group': '1' })) {
  assert.equal(readFileSync('/sys/fs/cgroup' + cg + '/' + key, 'utf8').trim(), value);
}
const descriptor = readFileSync(resolve(root, 'native-delivery.json'));
assert.equal(hash(descriptor), reservation.nativeDescriptorSha256);
for (const entry of JSON.parse(descriptor).entries) {
  const bytes = readFileSync(resolve(root, entry.path));
  assert.equal(bytes.length, entry.bytes); assert.equal(hash(bytes), entry.sha256, entry.path);
}
for (const file of reservation.runtimeFiles) {
  const bytes = readFileSync(resolve(root, 'node_modules/doppler-gpu', file.path));
  assert.equal(bytes.length, file.bytes); assert.equal(hash(bytes), file.sha256, file.path);
}
assert.equal(hash(readFileSync(resolve(root, 'node_modules/webgpu/dist/linux-x64.dawn.node'))), reservation.providerSha256);
const fixtureSha256 = hash(JSON.stringify(cases));
assert.equal(fixtureSha256, 'da5fd98a93245a53911d6c6a685749fd79e958dafcc4612a23fe0c068bbdcce5');
const receipt = { schema: 'doppler.installed-paired-operator/v1', passed: false,
  scope: 'Installed immediate and recorded paired operator; no full-model qualification',
  reservation, fixtureSha256, tolerance: 2e-11, pid: process.pid,
  node: process.version, events: [], cases: [], gpuErrors: [], peakAllocatedGpuBytes: 0 };
const mark = (phase, fields = {}) => {
  receipt.events.push({ phase, ...fields, at: new Date().toISOString() });
  writeFileSync(destination, JSON.stringify(receipt, null, 2) + '\n');
};
const load = name => import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src', name)));
const { createNativeGpuHost } = await import(pathToFileURL(resolve(root, 'server/native-participant.js')));
const host = createNativeGpuHost(() => import(pathToFileURL(resolve(root, 'node_modules/webgpu/index.js'))));
let device, pool, pools, uniforms, allocated = 0;
try {
  await host.open(); mark('host-open');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter && !adapter.info.isFallbackAdapter);
  receipt.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info[k]]));
  device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 9 } });
  device.lost.then(info => mark('device-lost', { reason: info.reason, message: info.message }));
  device.addEventListener('uncapturederror', event => receipt.gpuErrors.push(event.error.message));
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = descriptor => {
    assert(allocated + descriptor.size <= 1420000000, 'Unchanged complete allocation cap');
    const buffer = createBuffer(descriptor), destroy = buffer.destroy.bind(buffer);
    allocated += buffer.size; receipt.peakAllocatedGpuBytes = Math.max(receipt.peakAllocatedGpuBytes, allocated);
    let live = true;
    buffer.destroy = () => { destroy(); if (live) { allocated -= buffer.size; live = false; } };
    return buffer;
  };
  const createPipeline = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = async descriptor => {
    mark('pipeline-start', { label: descriptor.label });
    const result = await createPipeline(descriptor);
    mark('pipeline-complete', { label: descriptor.label }); return result;
  };
  (await load('gpu/device.js')).setDevice(device, { platformConfig: null });
  pools = await load('memory/buffer-pool.js'); pool = pools.getBufferPool(device);
  uniforms = await load('gpu/uniform-cache.js');
  const { createTensor } = await load('gpu/tensor.js');
  const { CommandRecorder } = await load('gpu/command-recorder.js');
  const { preparePairedLinearAttentionPipelines, executePairedLinearAttention } = await load('gpu/kernels/paired-linear-attention.js');
  await preparePairedLinearAttentionPipelines(device); mark('prepared');
  for (const recorded of [false, true]) for (const f of cases) {
    const owned = [], mapped = []; let output, recorder;
    const storage = values => {
      const buffer = device.createBuffer({ size: values.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.push(buffer); device.queue.writeBuffer(buffer, 0, new Float32Array(values)); return buffer;
    };
    const tensor = buffer => createTensor(buffer, 'f32', [buffer.size / 8], 'fixture',
      { precision: 'two-f32', components: 2, elementStrideBytes: 8 });
    try {
      const qkv = tensor(storage(f.qkv)), z = f.packed ? qkv : tensor(storage(f.z));
      const a = tensor(storage(f.packed ? [...f.a, ...f.b] : f.a)), b = f.packed ? a : tensor(storage(f.b));
      const state = { convDim: f.convDim, convKernelSize: f.kernel, valueDim: f.valueDim,
        numVHeads: f.nv, numKHeads: f.nk, headKDim: f.kd, headVDim: f.vd,
        qSize: f.nk * f.kd, kSize: f.nk * f.kd, qRep: f.nv / f.nk,
        normMode: 'shared', rmsNormEps: f.eps, normWeightOffset: f.offset,
        convStatePrecision: 'two-f32', convStateComponents: 2,
        recurrentStatePrecision: 'two-f32', recurrentStateComponents: 2,
        convStateGPU: storage(f.convState), recurrentStateGPU: storage(f.state),
        convWeightGPU: storage(f.weight), dtBiasGPU: storage(f.bias),
        aLogGPU: storage(f.log), normWeightGPU: storage(f.norm) };
      recorder = recorded ? new CommandRecorder(device, 'installed-operator') : null;
      output = await executePairedLinearAttention(qkv, z, a, b, state, {
        numTokens: f.tokens, outputDtype: 'f32', qkL2NormEps: f.qeps, recorder,
        abPacked: f.packed, qkvzPacked: f.packed, bProjOffsetElements: f.tokens * f.nv });
      if (recorder) await recorder.submitAndWait();
      await device.queue.onSubmittedWorkDone();
      const expected = reference(f), rows = {};
      for (const [name, buffer] of [['convState', state.convStateGPU], ['state', state.recurrentStateGPU], ['output', output.buffer]]) {
        const bytes = expected[name].length * 8;
        const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        owned.push(staging);
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ); mapped.push(staging);
        const actual = new Float32Array(staging.getMappedRange()); let failures = 0, maxError = 0;
        for (let i = 0; i < expected[name].length; i++) {
          const error = Math.abs(actual[i * 2] + actual[i * 2 + 1] - expected[name][i]) / Math.max(1, Math.abs(expected[name][i]));
          if (!Number.isFinite(error) || error > receipt.tolerance) failures++;
          maxError = Math.max(maxError, error);
        }
        rows[name] = { count: expected[name].length, failures, maxError };
      }
      receipt.cases.push({ recorded, packed: f.packed, rows, dispatches: recorder?.getStats() ?? null });
    } finally {
      await recorder?.abort();
      for (const buffer of mapped) buffer.unmap();
      await device.queue.onSubmittedWorkDone();
      if (output) pools.releaseBuffer(output.buffer);
      for (const buffer of owned) buffer.destroy();
    }
    assert.equal(pool.getStats().activeBuffers, 0); mark('case-settled', { recorded, packed: f.packed });
  }
  receipt.numericalPassed = receipt.gpuErrors.length === 0 && receipt.cases.length === 4
    && receipt.cases.every(c => Object.values(c.rows).every(row => row.failures === 0));
  assert(receipt.numericalPassed, 'Installed paired operator differs from reference');
} catch (error) {
  receipt.error = error.stack; process.exitCode = 1;
} finally {
  try {
    if (device) {
      await device.queue.onSubmittedWorkDone();
      receipt.poolBeforeDestroy = pool?.getStats();
      uniforms?.resetUniformCache(device); pools?.destroyBufferPool(device);
      await device.queue.onSubmittedWorkDone();
      receipt.poolAfterDestroy = pool?.getStats(); receipt.remainingTrackedGpuBytes = allocated;
      assert.equal(allocated, 0, 'All observed allocations must settle before host close');
      receipt.operationSettled = true;
    }
    await host.close(); receipt.hostClosed = true; mark('host-close-complete');
  } catch (error) { receipt.cleanupError = error.stack; process.exitCode = 1; }
  receipt.passed = receipt.numericalPassed === true && receipt.operationSettled === true
    && receipt.hostClosed === true && !receipt.error && !receipt.cleanupError;
  mark('finished');
}
console.log(JSON.stringify({ passed: receipt.passed, operationSettled: receipt.operationSettled, hostClosed: receipt.hostClosed }));
