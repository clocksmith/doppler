import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { mock } from 'node:test';
import { routedPrefillWaits } from '../../src/inference/pipelines/text/expert-loading-stats.js';
import { beginRoutedExpertTrace, finishRoutedExpertTrace } from '../../src/loader/experts/routed-expert-trace.js';

const target = new URL('../../src/inference/pipelines/text/moe/routed-expert-chunks.js', import.meta.url).href;
const state = { reads: [], prepares: [], executions: [] };
globalThis.routedTraceFixture = {
  read(buffer, offset, length, recorder) {
    state.reads.push({ offset, length, recorder });
    return Promise.resolve(buffer.data.slice(offset, offset + length));
  },
  device: { queue: { submit() {}, async onSubmittedWorkDone() {} }, createCommandEncoder() {
    return { copyBufferToBuffer(src, from, dst, to, size) {
      new Uint8Array(dst.data, to, size).set(new Uint8Array(src.data, from, size));
    }, finish() { return {}; } };
  } },
  run(options, recorder = null) {
    state.executions.push({ options, recorder });
    return { buffer: { data: new ArrayBuffer(options.numTokens * options.hiddenSize * 4) } };
  },
};
const substitutions = {
  '../../../../config/runtime.js': 'export const getRuntimeConfig = () => ({shared:{debug:{profiler:{enabled:routedTraceFixture.profile === true}}}});',
  '../../../../memory/buffer-pool.js': 'export const acquireBuffer = n => ({data:new ArrayBuffer(n)}); export const releaseBuffer = () => {}; export const readBufferSlice = (...a) => routedTraceFixture.read(...a);',
  '../../../../gpu/device.js': 'export const getDevice = () => routedTraceFixture.device;',
  '../../../../gpu/tensor.js': 'export const createTensor = buffer => ({buffer});',
  '../../../../gpu/kernels/moe-route.js': 'export const runRouteRemap = () => { throw new Error("Unexpected remap"); };',
  './qwen-route-executor.js': 'export const runQwenRouteExperts = o => routedTraceFixture.run(o); export const recordQwenRouteExperts = (r,o) => routedTraceFixture.run(o,r);',
};
const hooks = registerHooks({ resolve(name, context, next) {
  if (context.parentURL === target && substitutions[name]) return { url: 'data:text/javascript,' + encodeURIComponent(substitutions[name]), shortCircuit: true };
  return next(name, context);
} });
try {
  const { executeRoutedExpertChunks } = await import(target);
  const resident = {};
  const loader = { getResidentRoutedExpertWeights: () => resident,
    async prepareRoutedExperts(layer, ids) { state.prepares.push({ layer, ids }); return {}; } };
  const options = { activationDtype: 'f32', phase: 'decode', numTokens: 1, topK: 2, hiddenSize: 1, layerIdx: 5,
    indicesBuffer: { data: new Uint32Array([9, 2]).buffer }, weightsBuffer: { data: new ArrayBuffer(8) },
    inputTensor: { buffer: { data: new ArrayBuffer(4) } } };
  const recorder = {};
  await executeRoutedExpertChunks(options, loader, { storageMode: 'tiered-compact', routeChunkTokens: 2 }, undefined, recorder);
  assert.equal(state.reads.length, 0, 'Default resident path adds no readback');
  beginRoutedExpertTrace(loader, { requestId: 'resident', maxRecords: 1, maxIndices: 2 });
  await executeRoutedExpertChunks(options, loader, { storageMode: 'tiered-compact', routeChunkTokens: 2 }, undefined, recorder);
  assert.equal(state.reads[0].recorder, recorder, 'Resident diagnostic respects the routing recorder');
  assert.equal(state.executions.at(-1).recorder, recorder);
  assert.deepEqual(finishRoutedExpertTrace(loader, 'resident').records[0].indices, [9, 2]);

  const paged = { ...options, phase: 'prefill', numTokens: 3,
    indicesBuffer: { data: new Uint32Array([9, 2, 9, 4, 7, 4]).buffer },
    weightsBuffer: { data: new ArrayBuffer(24) }, inputTensor: { buffer: { data: new ArrayBuffer(12) } } };
  beginRoutedExpertTrace(loader, { requestId: 'paged', maxRecords: 2, maxIndices: 6 });
  const before = state.reads.length;
  await executeRoutedExpertChunks(paged, loader, { storageMode: 'sparse', routeChunkTokens: 2 });
  const trace = finishRoutedExpertTrace(loader, 'paged');
  assert.equal(state.reads.length - before, 2, 'Exactly the two existing paging reads');
  assert.deepEqual(trace.records.map(r => r.indices), [[9, 2, 9, 4], [7, 4]]);
  assert.deepEqual(trace.records.map(r => r.tokenOffset), [0, 2]);
  assert.deepEqual(trace.records.map(r => r.layer), [5, 5]);
  assert.deepEqual(state.prepares.map(r => r.ids), [[9, 2, 4], [7, 4]], 'Trace retains duplicates before loader deduplication');
  assert.deepEqual(routedPrefillWaits(loader), { selectionAwaitMs: 0, preparationAwaitMs: 0 });
  // A virtual clock and explicit gates distinguish a hidden read from time actually blocking the next chunk.
  let now = 0;
  const clock = mock.method(performance, 'now', () => now);
  const fixture = globalThis.routedTraceFixture;
  const originalRead = fixture.read, originalSubmit = fixture.device.queue.submit;
  try {
    fixture.profile = true;
    for (const selectionLast of [false, true]) {
      now = 0;
      const selects = Promise.withResolvers();
      const prep = [Promise.withResolvers(), Promise.withResolvers()];
      const started = [Promise.withResolvers(), Promise.withResolvers()];
      let reads = 0, prepares = 0, submits = 0, readbackWall = 0;
      fixture.read = async (buffer, offset, length) => {
        if (++reads === 1) now = 5;
        else await selects.promise;
        return buffer.data.slice(offset, offset + length);
      };
      fixture.device.queue.submit = () => { if (++submits === 1) now = 10; };
      const measured = { ...loader, recordRoutedReadback: (_phase, elapsed) => { readbackWall += elapsed; },
        async prepareRoutedExperts() {
          const i = prepares++;
          started[i].resolve();
          await prep[i].promise;
          return {};
        } };
      const running = executeRoutedExpertChunks(paged, measured, { storageMode: 'sparse', routeChunkTokens: 2 });
      await started[0].promise;
      if (!selectionLast) {
        now = 15; selects.resolve();
        await new Promise(resolve => setImmediate(resolve));
      }
      now = 20; prep[0].resolve();
      if (selectionLast) {
        await new Promise(resolve => setImmediate(resolve));
        now = 29; selects.resolve();
      }
      await started[1].promise;
      now += 5; prep[1].resolve();
      await running;
      assert.deepEqual(routedPrefillWaits(measured), { selectionAwaitMs: selectionLast ? 14 : 5, preparationAwaitMs: 15 });
      assert.equal(readbackWall, selectionLast ? 29 : 15, 'Readback wall includes overlap with preparation and input copies');
    }
    now = 0;
    fixture.device.queue.submit = originalSubmit;
    fixture.read = async (buffer, offset, length) => {
      now += 4;
      return buffer.data.slice(offset, offset + length);
    };
    const direct = { ...loader, async prepareRoutedExperts() { now += 7; return {}; } };
    await executeRoutedExpertChunks(paged, direct, { storageMode: 'sparse', routeChunkTokens: paged.numTokens });
    assert.deepEqual(routedPrefillWaits(direct), { selectionAwaitMs: 4, preparationAwaitMs: 7 });
  } finally {
    clock.mock.restore();
    fixture.read = originalRead;
    fixture.device.queue.submit = originalSubmit;
  }
} finally {
  hooks.deregister();
  delete globalThis.routedTraceFixture;
}
console.log('Routed trace real JS wiring with CPU GPU stubs: resident recorder and chunked paging passed');
