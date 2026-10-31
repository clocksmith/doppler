#!/usr/bin/env node
// Root controller runs this on the physical host while holding the real timing lock.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { probeNodeGPU } from '../../tests/helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getKernelBindGroupLayout } from '../../src/gpu/kernels/kernel-bindings.js';
import { resolvePipelineIdentity } from '../../src/gpu/kernels/pipeline-cache.js';
import { UNIFORM_WRITERS } from '../../src/gpu/kernels/generated/uniform-writers.js';
import { validateAttentionVariant, calculateAttentionWorkgroups } from '../../src/gpu/kernels/attention/plan.js';
import { runAttention, recordAttention } from '../../src/gpu/kernels/attention.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';

const { values: options } = parseArgs({ options: { 'adapter-match': { type: 'string' }, bench: { type: 'boolean', default: false } } });
assert(options['adapter-match'], '--adapter-match is required');
assert(process.env.DOPPLER_F32_PREFILL_LOCK_HELD, 'Use the locked runner');
const variants = ['prefill_small', 'prefill_query_tile_f32'];
const emit = value => console.log(JSON.stringify(value));
const canary = 12345.5;
const tolerance = 0.001; // abs only, 0 relative; never a full-model gate substitute.
let device;
const pipelines = new Map(), errors = [];
function close(a, b, label) {
  assert.equal(a.length, b.length);
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const error = Math.abs(a[i] - b[i]);
    assert(Number.isFinite(error) && error <= tolerance, `${label}[${i}]: ${a[i]} vs ${b[i]} (${error})`);
    maxAbs = Math.max(maxAbs, error);
  }
  return maxAbs;
}
function fixture(spec) {
  const s = { T: 9, K: 17, start: 8, kvStart: 0, window: 0, layout: 0, pageSize: 16,
    scale: 1 / 16, softcap: 0, causal: true, magnitude: 0.25, zero: false, dynamicLength: false, ...spec };
  const physicalRows = s.layout === 1 ? s.window : Math.ceil((s.kvStart + s.K) / s.pageSize) * s.pageSize;
  const pages = Uint32Array.from({ length: Math.max(1, Math.ceil((s.kvStart + s.K) / s.pageSize)) }, (_, i) => i);
  if (s.layout === 2) pages.reverse();
  const row = key => s.layout === 1 ? (s.kvStart + key) % s.window
    : s.layout === 2 ? pages[Math.floor((s.kvStart + key) / s.pageSize)] * s.pageSize + (s.kvStart + key) % s.pageSize
      : s.kvStart + key;
  const q = Float32Array.from({ length: s.T * 16 * 256 }, (_, i) => s.zero ? 0 : Math.sin(i * 0.017 + 0.3) * s.magnitude);
  const k = new Float32Array(Math.max(1, physicalRows) * 2 * 256), v = new Float32Array(k.length);
  for (let key = 0; key < s.K; key++) for (let h = 0; h < 2; h++) for (let d = 0; d < 256; d++) {
    const i = (row(key) * 2 + h) * 256 + d;
    k[i] = Math.sin((key * 509 + h * 251 + d) * 0.013) * s.magnitude;
    v[i] = Math.cos((key * 503 + h * 239 + d) * 0.019) * 0.75;
  }
  return { ...s, q, k, v, pages, row };
}
// Independent scalar Float64 oracle; all rows for short cases, boundary rows for
// long cases. GPU A/B still compares every output component in every case.
function reference(f, actual) {
  const queries = f.T <= 17 ? [...Array(f.T).keys()] : [...new Set([0, 1, Math.floor(f.T / 2), f.T - 1])];
  let maxAbs = 0, checked = 0;
  for (const t of queries) for (const h of [0, 7, 8, 15]) {
    const scores = new Float64Array(f.K).fill(-Infinity);
    let max = -Infinity;
    for (let key = 0; key < f.K; key++) {
      const absolute = f.kvStart + key, position = f.start + t;
      if ((f.causal && absolute > position) || (f.window && position >= f.window && absolute < position - f.window + 1)) continue;
      let score = 0;
      for (let d = 0; d < 256; d++) score += f.q[(t * 16 + h) * 256 + d] * f.k[(f.row(key) * 2 + Math.floor(h / 8)) * 256 + d];
      score *= f.scale;
      if (f.softcap) score = Math.tanh(score / f.softcap) * f.softcap;
      scores[key] = score; max = Math.max(max, score);
    }
    let sum = 0;
    for (let key = 0; key < f.K; key++) {
      scores[key] = Number.isFinite(scores[key]) ? Math.exp(scores[key] - max) : 0;
      sum += scores[key];
    }
    for (let d = 0; d < 256; d++) {
      let result = 0;
      for (let key = 0; key < f.K; key++) result += scores[key] * f.v[(f.row(key) * 2 + Math.floor(h / 8)) * 256 + d];
      const expected = sum > 0 ? result / sum : 0;
      maxAbs = Math.max(maxAbs, close([actual[(t * 16 + h) * 256 + d]], [expected], 'Float64 oracle'));
      checked++;
    }
  }
  return { maxAbs, checked };
}
async function compile(variant) {
  const config = getKernelConfig('attention', variant);
  validateAttentionVariant(variant, false, false, false, getKernelCapabilities(), 256, 2287, device.limits.maxComputeWorkgroupStorageSize);
  const source = await readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
  const module = device.createShaderModule({ code: source, label: config.shaderFile });
  const messages = (await module.getCompilationInfo()).messages;
  assert.deepEqual(messages.filter(m => m.type === 'error'), []);
  const layout = getKernelBindGroupLayout(config, device);
  const { constants } = resolvePipelineIdentity({ operation: 'attention', variant });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: config.entryPoint, constants: constants ?? {} } });
  pipelines.set(variant, { config, pipeline, layout });
  emit({ event: 'compiled', variant, sha256: createHash('sha256').update(source).digest('hex'), constants, messages });
}
async function run(f, variant, chunks = [f.T], timed = false, wrapper = null) {
  const { config, pipeline, layout } = pipelines.get(variant);
  const owned = [], recorders = [];
  const upload = (data, usage = GPUBufferUsage.STORAGE) => {
    const b = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    owned.push(b); if (data.byteLength) device.queue.writeBuffer(b, 0, data); return b;
  };
  const K = upload(f.k), V = upload(f.v), pages = upload(f.pages), length = upload(new Uint32Array([f.K]));
  const result = new Float32Array(f.T * 16 * 256), samples = [];
  let offset = 0;
  device.pushErrorScope('validation');
  try {
    for (const count of chunks) {
      const Q = upload(f.q.subarray(offset * 4096, (offset + count) * 4096));
      const guardElements = device.limits.minStorageBufferOffsetAlignment / 4;
      const guarded = new Float32Array(count * 4096 + guardElements * 2).fill(canary);
      const output = upload(guarded);
      let actual;
      if (wrapper) {
        const args = [createTensor(Q, 'f32', [count, 16, 256]), createTensor(K, 'f32', [f.K, 2, 256]), createTensor(V, 'f32', [f.K, 2, 256]), null, 16, 256, {
          seqLen: count, kvLen: f.K, numKVHeads: 2, scale: f.scale, causal: f.causal, startPos: f.start + offset,
          kvStart: f.kvStart, slidingWindow: f.window, attnSoftcap: f.softcap, kvLayout: ['contiguous', 'ring', 'paged'][f.layout],
          kvPageTable: pages, kvPageSize: f.pageSize, kvLenBuffer: f.dynamicLength ? length : null,
          outputBuffer: output, kernelPath: { activationDtype: 'f32', kvDtype: 'f32', prefill: { steps: [{ op: 'attention', kernel: config.shaderFile, entry: 'main' }] } },
        }];
        if (wrapper === 'immediate') await runAttention(...args);
        else {
          const recorder = new CommandRecorder(device, 'f32-prefill-adapter'); recorders.push(recorder);
          await recordAttention(recorder, ...args);
          if (wrapper === 'cancel') {
            await recorder.abort();
            const unchanged = new Float32Array(await readBuffer(output, guarded.byteLength));
            assert(unchanged.every(v => v === canary), 'cancel before submit wrote output');
            continue;
          }
          await recorder.submitAndWait();
        }
        const bytes = new Float32Array(await readBuffer(output, guarded.byteLength));
        assert(bytes.subarray(count * 4096).every(v => v === canary));
        actual = bytes.subarray(0, count * 4096);
      } else {
        const writer = UNIFORM_WRITERS[`attention/${variant}`];
        const data = new ArrayBuffer(Math.ceil(writer.size / 16) * 16);
        writer.write(new DataView(data), { num_heads: 16, num_kv_heads: 2, head_dim: 256, seq_len: f.K, query_len: count, scale: f.scale,
          is_causal: Number(f.causal), start_pos: f.start + offset, attn_softcap: f.softcap, sliding_window: f.window,
          kv_len_source: Number(f.dynamicLength), kv_start: f.kvStart, page_size: f.pageSize, kv_layout: f.layout,
          bidirectional_span_start: 0, bidirectional_span_length: 0 }, variant);
        const uniforms = upload(new Uint8Array(data), GPUBufferUsage.UNIFORM);
        const bindGroup = device.createBindGroup({ layout, entries: [uniforms, Q, K, V, output, length, pages].map((buffer, binding) => ({ binding, resource: binding === 4
          ? { buffer, offset: guardElements * 4, size: count * 4096 * 4 } : { buffer } })) });
        const encoder = device.createCommandEncoder();
        let queries, resolve, staging;
        if (timed) {
          queries = device.createQuerySet({ type: 'timestamp', count: 2 }); owned.push(queries);
          resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); owned.push(resolve);
          staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); owned.push(staging);
        }
        const pass = encoder.beginComputePass(timed ? { timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(calculateAttentionWorkgroups('tiled_small', count, 16, variant)); pass.end();
        if (timed) { encoder.resolveQuerySet(queries, 0, 2, resolve, 0); encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16); }
        device.queue.submit([encoder.finish()]);
        if (timed) {
          // Dedicated query readback retains raw nanoseconds, including zero ticks.
          try {
            await staging.mapAsync(GPUMapMode.READ);
            const ticks = new BigUint64Array(staging.getMappedRange());
            samples.push({ startNs: String(ticks[0]), endNs: String(ticks[1]), gpuMs: Number(ticks[1] - ticks[0]) / 1e6, zeroOrBelowResolution: ticks[1] === ticks[0] });
            assert(ticks[1] >= ticks[0]);
          } finally { if (staging.mapState === 'mapped') staging.unmap(); }
        }
        const bytes = new Float32Array(await readBuffer(output, guarded.byteLength));
        assert(bytes.subarray(0, guardElements).every(v => v === canary), 'prefix output guard');
        assert(bytes.subarray(guardElements + count * 4096).every(v => v === canary), 'suffix output guard');
        actual = bytes.subarray(guardElements, guardElements + count * 4096);
      }
      result.set(actual, offset * 4096); offset += count;
    }
    assert.equal(errors.length, 0, errors.join('\n'));
    return { output: result, samples };
  } finally {
    await device.queue.onSubmittedWorkDone();
    for (const recorder of recorders) await recorder.abort();
    owned.forEach(b => b.destroy());
    assert.equal(await device.popErrorScope(), null);
  }
}
try {
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason); device = getDevice();
  const caps = getKernelCapabilities(), identity = JSON.stringify(caps.adapterInfo);
  assert(!/llvmpipe|lavapipe|swiftshader|software/i.test(identity), 'Software adapter rejected');
  assert(new RegExp(options['adapter-match'], 'i').test(identity), identity);
  device.addEventListener('uncapturederror', e => errors.push(String(e.error)));
  emit({ event: 'start', caps, limits: { shared: device.limits.maxComputeWorkgroupStorageSize }, provider: probe.providerReceipt,
    tolerance: { abs: tolerance, rel: 0 }, scope: 'Physical component only; no full-model qualification or speed claim' });
  for (const variant of variants) await compile(variant);
  const cases = [
    { T: 9, K: 17, start: 8 }, { T: 9, K: 0, start: 0 }, { T: 9, K: 1, start: 0, zero: true },
    { T: 9, K: 33, start: 0, kvStart: 64 }, // Entirely masked causal rows.
    { T: 9, K: 33, start: 100, kvStart: 64, causal: false, scale: 0 },
    { T: 17, K: 71, start: 54, softcap: 2, magnitude: 8, scale: 0.5 },
    { T: 9, K: 17, start: 8, scale: -0.125 },
    { T: 17, K: 31, start: 94, kvStart: 80, window: 31, layout: 1, dynamicLength: true },
    { T: 17, K: 71, start: 86, kvStart: 32, window: 19, layout: 2 },
    { T: 256, K: 2287, start: 2031 }, { T: 239, K: 2287, start: 2048, dynamicLength: true },
    { T: 256, K: 2543, start: 2287 },
  ];
  for (const spec of cases) {
    emit({ event: 'case-start', spec });
    const f = fixture(spec);
    const a = await run(f, variants[0]), b = await run(f, variants[1]);
    const maxAbs = close(a.output, b.output, 'baseline/candidate');
    const oracle = reference(f, b.output), baselineOracle = reference(f, a.output);
    const split = Math.floor(f.T / 2);
    const chunked = await run(f, variants[1], [split, f.T - split]);
    const chunkMaxAbs = close(b.output, chunked.output, 'whole/chunks');
    emit({ event: 'case-pass', spec, maxAbs, oracle, baselineOracle, chunkMaxAbs });
  }
  const f = fixture(cases[0]);
  const raw = await run(f, variants[1]);
  for (const mode of ['immediate', 'recorded']) close(raw.output, (await run(f, variants[1], [f.T], false, mode)).output, mode);
  await run(f, variants[1], [f.T], false, 'cancel');
  close(raw.output, (await run(f, variants[1], [f.T], false, 'recorded')).output, 'reuse after cancellation');
  emit({ event: 'correctness-passed', cases: cases.length, adapters: ['immediate', 'recorded', 'cancel-before-submit', 'reuse'] });
  if (options.bench) {
    assert(device.features.has('timestamp-query'), 'timestamp-query required');
    const f = fixture({ T: 256, K: 2287, start: 2031 });
    for (const variant of variants) await run(f, variant); // Explicit untimed warmup.
    for (let pair = 0; pair < 4; pair++) {
      const order = pair % 2 ? [...variants].reverse() : variants;
      const outputs = [];
      for (const variant of order) {
        const result = await run(f, variant, [f.T], true); outputs.push(result.output);
        emit({ event: 'component-timing', pair, order, variant, samples: result.samples, T: f.T, K: f.K,
          scope: 'One timestamp-instrumented attention dispatch; uploads/readback excluded; zero samples retained' });
      }
      close(outputs[0], outputs[1], 'timed A/B');
    }
  }
  emit({ event: 'passed' });
} catch (error) {
  emit({ event: 'failed', message: error.message, stack: error.stack }); process.exitCode = 1;
} finally { if (device) await destroyDevice(); }
