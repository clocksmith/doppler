import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/moe/routed-expert-chunks.js', import.meta.url).href;
const calls = [], released = [], buffers = [];
const buffer = (size, label = 'test') => {
  const value = { size, label, data: new Uint8Array(size) };
  buffers.push(value);
  return value;
};
const state = globalThis.routedAdapterTest = {
  buffer, released, fail: false, aborted: false, reads: 0,
  device: { queue: { submit() {}, async onSubmittedWorkDone() {} }, createCommandEncoder() {
    return { copyBufferToBuffer(source, from, destination, to, length) {
      destination.data.set(source.data.subarray(from, from + length), to);
    }, finish() { return {}; } };
  } },
  async read(source, start, size) { this.reads++; return source.data.slice(start, start + size).buffer; },
  async execute(recorder, options) {
    calls.push({ recorder, options });
    if (this.fail) throw Error('injected dispatch failure');
    const output = buffer(options.numTokens * options.hiddenSize * 4, 'result');
    output.data.set(options.inputTensor.buffer.data.subarray(0, output.size));
    return { buffer: output };
  },
  async remap(indices, map, count, sourceCount, slots, recorder) {
    calls.push({ remap: { indices, map, count, sourceCount, slots, recorder } });
    const result = buffer(count * 4, 'remapped');
    const sourceIds = new Uint32Array(indices.data.buffer, 0, count), mapping = new Uint32Array(map.data.buffer);
    new Uint32Array(result.data.buffer).set(Array.from(sourceIds, id => mapping[id]));
    return result;
  },
};
const modules = {
  '../../../../config/runtime.js': 'export const getRuntimeConfig=()=>({shared:{debug:{profiler:{enabled:false}}}});',
  '../../../../memory/buffer-pool.js': `export const acquireBuffer=(n,_u,l)=>routedAdapterTest.buffer(n,l);
    export const releaseBuffer=b=>routedAdapterTest.released.push(b);
    export const readBufferSlice=(...a)=>routedAdapterTest.read(...a);`,
  '../../../../gpu/device.js': 'export const getDevice=()=>routedAdapterTest.device;',
  '../../../../gpu/tensor.js': 'export const createTensor=(buffer,dtype,shape)=>({buffer,dtype,shape});',
  '../../../../gpu/kernels/moe-route.js': 'export const runRouteRemap=(...a)=>routedAdapterTest.remap(...a);',
  './qwen-route-executor.js': `export const runQwenRouteExperts=o=>routedAdapterTest.execute(null,o);
    export const recordQwenRouteExperts=(r,o)=>routedAdapterTest.execute(r,o);`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { executeRoutedExpertChunks: execute } = await import(target);
  const input = buffer(3 * 2 * 4, 'input');
  input.data.set(Uint8Array.from({ length: input.size }, (_, i) => i + 1));
  const indices = buffer(3 * 4, 'indices');
  new Uint32Array(indices.data.buffer).set([3, 1, 2]);
  const weightsBuffer = buffer(3 * 4, 'routes');
  const weights = { numExperts: 4 };
  const prepared = [];
  const loader = { async prepareRoutedExperts(layer, ids, signal, phase) {
    prepared.push({ layer, ids, phase }); return weights;
  } };
  const options = { inputTensor: { buffer: input }, indicesBuffer: indices, weightsBuffer,
    numTokens: 3, topK: 1, hiddenSize: 2, layerIdx: 0, phase: 'prefill',
    activationDtype: 'f32', groupedPrefill: true, kernelPath: {} };
  const policy = { routeChunkTokens: 2, storageMode: 'compact-decode' };
  const output = await execute(options, loader, policy);
  assert.deepEqual(output.buffer.data, input.data);
  assert.deepEqual(prepared.map(row => row.ids), [[3, 1], [2]]);
  assert.deepEqual(calls.map(row => row.options.numTokens), [2, 1]);
  assert(calls.every(row => row.options.groupedPrefill && row.options.kernelPath === options.kernelPath));
  assert(!released.includes(input) && !released.includes(indices) && !released.includes(weightsBuffer));
  calls.length = 0;
  weights.sourceExpertToSlot = buffer(16, 'map'); weights.sourceNumExperts = 4; weights.numExperts = 2;
  const sharedSlot = { weights: { layerOwned: true }, residual: null, gate: { layerOwned: true } };
  const recorder = { submitPartial() {}, trackTemporaryBuffer(value) { released.push(value); } };
  await execute({ ...options, numTokens: 1, phase: 'decode', groupedPrefill: false, sharedSlot }, loader, policy, undefined, recorder);
  assert.equal(calls[0].remap.sourceCount, 4);
  assert.equal(calls[1].options.sharedSlot, sharedSlot);
  assert.equal(calls[1].recorder, recorder);
  assert.equal(calls[1].options.indicesBuffer.label, 'remapped');
  assert(!released.includes(sharedSlot.weights) && !released.includes(weights.sourceExpertToSlot));
  for (const unsupported of [
    { ...options, sharedSlot },
    { ...options, groupedPrefill: true, phase: 'decode', numTokens: 1 },
    { ...options, groupedPrefill: false, phase: 'decode', numTokens: 1, sharedSlot: { residual: {} } },
  ]) await assert.rejects(execute(unsupported, loader, policy), /Grouped routed experts|Paged shared slots/);
  delete weights.sourceExpertToSlot; delete weights.sourceNumExperts;
  const tieredPolicy = { routeChunkTokens: 256, storageMode: 'tiered-compact' };
  const residentWeights = { numExperts: 4 };
  const residentLoader = { getResidentRoutedExpertWeights: () => residentWeights,
    getRoutedExpertCapacity: () => 2, prepareRoutedExperts: async () => { throw Error('resident must bypass preparation'); } };
  const readsBefore = state.reads;
  calls.length = 0;
  await execute(options, residentLoader, tieredPolicy);
  assert.equal(state.reads, readsBefore, 'complete resident bank bypasses route readback');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.indicesBuffer, indices, 'resident execution uses original GPU source IDs');
  assert.equal(calls[0].options.numTokens, 3);
  const tieredInput = buffer(5 * 2 * 4, 'tiered input');
  tieredInput.data.set(Uint8Array.from({ length: tieredInput.size }, (_, i) => i + 1));
  const tieredIndices = buffer(5 * 4, 'tiered indices');
  new Uint32Array(tieredIndices.data.buffer).set([3, 1, 2, 0, 3]);
  const map = buffer(4 * 4, 'tiered source map');
  const tieredPrepared = [];
  const streamed = { getResidentRoutedExpertWeights: () => null, getRoutedExpertCapacity: () => 2,
    prepareRoutedExperts: async (layer, ids, signal, phase) => {
      tieredPrepared.push({ layer, ids: [...ids], phase });
      const slots = new Uint32Array(map.data.buffer); slots.fill(0xffffffff);
      ids.forEach((id, slot) => { slots[id] = slot; });
      return { numExperts: 2, sourceNumExperts: 4, sourceExpertToSlot: map };
    } };
  const tieredOptions = { ...options, numTokens: 5, inputTensor: { buffer: tieredInput }, indicesBuffer: tieredIndices,
    weightsBuffer: buffer(5 * 4, 'tiered route weights') };
  calls.length = 0;
  const tieredOutput = await execute(tieredOptions, streamed, tieredPolicy);
  assert.deepEqual(tieredOutput.buffer.data, tieredInput.data);
  assert.deepEqual(tieredPrepared.map(row => row.ids), [[3, 1], [2, 0], [3]]);
  assert.ok(tieredPrepared.every(row => row.phase === 'prefill'));
  const chunks = calls.filter(row => row.options);
  assert.deepEqual(chunks.map(row => row.options.numTokens), [2, 2, 1], 'capacity/topK bounds streamed prefill');
  assert.deepEqual(chunks.map(row => [...new Uint32Array(row.options.indicesBuffer.data.buffer)]), [[0, 1], [0, 1], [0]],
    'every multi-token chunk receives compact slot IDs, never original source IDs');
  assert.ok(chunks.every(row => released.includes(row.options.indicesBuffer)), 'temporary remaps follow chunk ownership');
  await assert.rejects(execute(tieredOptions, { ...streamed, getRoutedExpertCapacity: undefined }, tieredPolicy), /explicit admitted capacity/);
  calls.length = 0;
  state.fail = true;
  const allocationStart = buffers.length;
  await assert.rejects(execute(options, loader, policy), /injected dispatch failure/);
  assert(buffers.slice(allocationStart).every(value => released.includes(value)), 'Failed chunk and output allocations settle');
  const tieredFailureStart = buffers.length;
  await assert.rejects(execute(tieredOptions, streamed, tieredPolicy), /injected dispatch failure/);
  assert.ok(buffers.slice(tieredFailureStart).every(value => released.includes(value)), 'failed tiered chunk retains no copied inputs/remap/output');
  assert.ok(!released.includes(map), 'loader owns the stable source-to-slot map');
  const aborted = new AbortController(); aborted.abort();
  const count = prepared.length;
  await assert.rejects(execute(options, loader, policy, aborted.signal), /abort/i);
  assert.equal(prepared.length, count);
  console.log('routed grouped prefill and non-residual shared-slot decode adapters passed (CPU contract doubles)');
} finally { hooks.deregister(); delete globalThis.routedAdapterTest; }
