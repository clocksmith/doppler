#!/usr/bin/env node
// Launch only through gdn-gguf-prefill-locked.sh on the physical host.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { probeNodeGPU } from '../../tests/helpers/gpu-probe.js';
import { makeGdnFixture, sliceGdnFixture, packGdnInputs, gdnReference, GDN_REFERENCE_TOLERANCE } from '../../tests/helpers/gdn-gguf-prefill-fixture.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getKernelBindGroupLayout } from '../../src/gpu/kernels/kernel-bindings.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';

const { values: options } = parseArgs({ options: { 'adapter-match': { type: 'string' }, bench: { type: 'boolean', default: false } } });
assert(options['adapter-match'], '--adapter-match is required');
const emit = value => console.log(JSON.stringify(value));
const canary = 12345.5;
const bits = values => new Uint32Array(values.buffer, values.byteOffset, values.length);
const exact = (a, b, label) => {
  assert.equal(a.length, b.length, label);
  const aBits = bits(a), bBits = bits(b);
  for (let i = 0; i < a.length; i++) assert.equal(aBits[i], bBits[i], `${label}[${i}] (${a[i]} vs ${b[i]})`);
};
function close(a, b, tolerance, label) {
  let maxAbs = 0;
  assert.equal(a.length, b.length, label);
  for (let i = 0; i < a.length; i++) {
    const error = Math.abs(a[i] - b[i]); maxAbs = Math.max(error, maxAbs);
    assert(Number.isFinite(a[i]) && error <= tolerance, `${label}[${i}]: ${a[i]} vs ${b[i]}, error=${error}`);
  }
  return maxAbs;
}
let device, completed = 0;
const errors = [];
async function compile(variant) {
  const config = getKernelConfig('gated_delta_recurrent', variant);
  const code = await readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
  device.pushErrorScope('validation');
  try {
    const module = device.createShaderModule({ code, label: config.shaderFile });
    const messages = (await module.getCompilationInfo()).messages;
    assert.deepEqual(messages.filter(message => message.type === 'error'), []);
    await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [getKernelBindGroupLayout(config, device)] }),
      compute: { module, entryPoint: config.entryPoint, constants: { WORKGROUP_SIZE: 128, RECURRENT_STATE_COMPONENTS: 1 } },
    });
    emit({ event: 'compiled', variant, shader: config.shaderFile,
      sha256: createHash('sha256').update(code).digest('hex'), messages: messages.map(m => ({ type: m.type, message: m.message })) });
  } finally { assert.equal(await device.popErrorScope(), null); }
}

async function run(f, enabled, chunks, packed, recorded, profile = false) {
  const timings = [];
  const owned = [], recorders = [];
  const upload = values => { const buffer = acquireBuffer(values.byteLength); owned.push(buffer); uploadData(buffer, values); return buffer; };
  const guardedState = new Float32Array(f.state.length + 4); guardedState.set(f.state); guardedState.fill(canary, f.state.length);
  const guardedConv = new Float32Array(f.convState.length + 4); guardedConv.set(f.convState); guardedConv.fill(canary, f.convState.length);
  const layer = { sourceLayout: 'gguf-v1', convDim: f.C, convKernelSize: f.window,
    numVHeads: f.NV, numKHeads: f.NK, headKDim: f.K, headVDim: f.V, valueDim: f.D,
    qSize: f.Q, kSize: f.Q, qRep: f.NV / f.NK, normMode: f.normMode, rmsNormEps: f.rmsEps,
    recurrentStatePrecision: 'f32', recurrentStateComponents: 1,
    convWeightGPU: upload(f.convWeight), convStateGPU: upload(guardedConv), recurrentStateGPU: upload(guardedState),
    dtBiasGPU: upload(f.bias), aLogGPU: upload(f.aLog), normWeightGPU: upload(f.norm) };
  const output = new Float32Array(f.T * f.D);
  let start = 0;
  device.pushErrorScope('validation');
  try {
    for (const count of chunks) {
      const part = sliceGdnFixture(f, start, count), data = packGdnInputs(part, ...packed);
      const cached = new Map();
      const tensor = values => {
        if (!cached.has(values)) cached.set(values, createTensor(upload(values), 'f32', [values.length]));
        return cached.get(values);
      };
      const recorder = recorded ? new CommandRecorder(device, 'gdn-prefill-continuation', { profile, profilePhase: 'prefill' }) : null;
      if (recorder) recorders.push(recorder);
      const result = await runLinearAttentionCoreGPU(tensor(data.qkv), tensor(data.z), tensor(data.a), tensor(data.b), layer, {
        numTokens: count, outputDtype: 'f32', normGateActivation: 'silu', qkL2NormEps: f.eps,
        abPacked: packed[0], qkvzPacked: packed[1], bProjOffsetElements: data.bOffset, recorder,
        runtimeSession: { useLinearAttentionRegisterPrefill: enabled, useLinearAttentionRegisterDecode: false, useLinearAttentionFusedDecodeCore: false },
      });
      owned.push(result.buffer);
      if (recorder) {
        await recorder.submitAndWait();
        if (profile) timings.push(await recorder.resolveProfileTimings({ requireComplete: true }));
      }
      output.set(new Float32Array(await readBuffer(result.buffer, count * f.D * 4)), start * f.D);
      start += count;
    }
    assert.equal(start, f.T);
    const state = new Float32Array(await readBuffer(layer.recurrentStateGPU, guardedState.byteLength));
    const convState = new Float32Array(await readBuffer(layer.convStateGPU, guardedConv.byteLength));
    for (const [buffer, length] of [[state, f.state.length], [convState, f.convState.length]]) {
      assert(buffer.subarray(length).every(value => value === canary), 'state canary modified');
    }
    assert.equal(errors.length, 0, errors.join('\n'));
    return { output, state: state.slice(0, f.state.length), convState: convState.slice(0, f.convState.length), timings };
  } finally {
    await device.queue.onSubmittedWorkDone();
    for (const recorder of recorders) await recorder.abort();
    for (const buffer of owned) releaseBuffer(buffer);
    assert.equal(await device.popErrorScope(), null);
  }
}

try {
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason); device = getDevice();
  const adapter = getKernelCapabilities().adapterInfo, identity = JSON.stringify(adapter);
  emit({ event: 'start', adapter, provider: probe.providerReceipt, referenceTolerance: GDN_REFERENCE_TOLERANCE,
    scope: 'Synthetic operator correctness/compile/continuation only; no performance or model qualification' });
  assert(!/llvmpipe|lavapipe|swiftshader|software/i.test(identity), 'Software adapter cannot establish physical evidence');
  assert(new RegExp(options['adapter-match'], 'i').test(identity), `Unexpected physical adapter: ${identity}`);
  device.addEventListener('uncapturederror', event => errors.push(String(event.error)));
  await compile('gguf_f32'); await compile('gguf_register_prefill');
  const cases = [
    { T: 5, V: 17, packed: [false, false] },
    { T: 2, V: 128, window: 1, scale: 1e-5, eps: 1e-4, normMode: 'shared', packed: [true, false] },
    { T: 7, V: 33, zeroState: true, packed: [false, true] },
    { T: 9, V: 128, packed: [true, true], chunks: [2, 3, 1, 3] },
    { T: 257, V: 17, packed: [true, true], chunks: [128, 128, 1] },
    { T: 2, V: 128, NK: 16, NV: 32, packed: [true, true] },
    { T: 256, V: 128, NK: 16, NV: 32, packed: [true, true] },
  ];
  for (const spec of cases) {
    emit({ event: 'case-start', spec });
    const f = makeGdnFixture(spec), reference = gdnReference(f);
    const baseline = await run(f, false, [f.T], spec.packed, false);
    const candidate = await run(f, true, [f.T], spec.packed, false);
    const maxima = {};
    for (const key of ['output', 'state', 'convState']) {
      exact(candidate[key], baseline[key], `A/B ${key}`);
      maxima[key] = close(candidate[key], reference[key], GDN_REFERENCE_TOLERANCE[key] ?? 0, `reference ${key}`);
    }
    if (spec.chunks) {
      const continuation = await run(f, true, spec.chunks, spec.packed, true);
      for (const key of ['output', 'state', 'convState']) exact(continuation[key], candidate[key], `recorded chunk/decode continuation ${key}`);
    }
    completed++; emit({ event: 'case-pass', spec, maxima, completed });
  }
  if (options.bench) {
    assert(device.features.has('timestamp-query'), 'Physical component timing requires timestamp-query');
    const f = makeGdnFixture({ T: 256, V: 128, NK: 16, NV: 32, packed: [true, true] });
    for (let pair = -1; pair < 4; pair++) {
      const order = pair % 2 === 0 ? [false, true] : [true, false];
      const results = {};
      for (const enabled of order) {
        const start = performance.now();
        const result = await run(f, enabled, [f.T], [true, true], true, true);
        results[enabled ? 'candidate' : 'baseline'] = result;
        emit({ event: 'component-timing', pair, warmup: pair < 0, enabled, order,
          timings: result.timings, hostCallMs: performance.now() - start,
          scope: 'Timestamp-instrumented synthetic operator; host call includes uploads/readbacks/cleanup, not pure GPU or model time' });
      }
      for (const key of ['output', 'state', 'convState']) exact(results.candidate[key], results.baseline[key], 'Timed A/B ' + key);
    }
  }
  emit({ event: 'passed', completed, errors });
} catch (error) {
  emit({ event: 'failed', completed, message: error.message, stack: error.stack }); process.exitCode = 1;
} finally { if (device) await destroyDevice(); }
