import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128, UNIFORM: 64 };
const { setDevice } = await import('../../src/gpu/device.js');
const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { getBufferPool, destroyBufferPool } = await import('../../src/memory/buffer-pool.js');
const target = new URL('../../src/inference/pipelines/text/partition-attempt.js', import.meta.url).href;
const replayURL = new URL('../../src/inference/pipelines/text/partition-replay.js', import.meta.url).href;
const buffers = [], caches = [];
let submitted = 0, wait = async () => {};
const device = { features: new Set(), limits: { maxStorageBufferBindingSize: 1e6, maxBufferSize: 1e6 },
  queue: {
    submit(commands) { submitted++; for (const command of commands) for (const copy of command) copy(); },
    writeBuffer(buffer, offset, data) { buffer.bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset); },
    onSubmittedWorkDone: () => wait(),
  },
  createBuffer({ size, usage }) {
    const buffer = { size, usage, bytes: new Uint8Array(size), __dopplerFakeGPUBuffer: true, destroyed: false, destroy() { this.destroyed = true; } };
    buffers.push(buffer); return buffer;
  },
  createBindGroup() { return {}; },
  createCommandEncoder() {
    const commands = [];
    return { copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
      assert(src.usage & GPUBufferUsage.COPY_SRC, 'Real WebGPU requires COPY_SRC for every cloned buffer');
      assert(dst.usage & GPUBufferUsage.COPY_DST);
      commands.push(() => dst.bytes.set(src.bytes.subarray(srcOffset, srcOffset + size), dstOffset));
    }, finish: () => commands };
  } };
setDevice(device, { platformConfig: null });
globalThis.cloneTestCreateCache = config => {
  const cache = new KVCache({ numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: config.session.kvcache.maxSeqLen,
    pageSize: 2, kvDtype: 'f32', layout: 'paged', useGPU: true });
  caches.push(cache); return cache;
};
const modules = {
  './init.js': 'export const createKVCache = (_model, _gpu, _debug, config) => cloneTestCreateCache(config);',
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {};',
  './layer-partition-contract.js': 'export const resolveLayerPartition = () => ({layerRange:[0,0]});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === replayURL && specifier === './partition-execution.js') return {
    url: 'data:text/javascript,export const executePartitionLayers = (...args) => runPartitionReplayFixture(...args);', shortCircuit: true };
  if (context.parentURL === replayURL && specifier === '../../../memory/buffer-pool.js') return {
    url: 'data:text/javascript,export const releaseBuffer = buffer => { buffer.released = true; };', shortCircuit: true };
  if (context.parentURL === target && modules[specifier]) return { url: 'data:text/javascript,' + encodeURIComponent(modules[specifier]), shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { createPartitionAttempt, clonePartitionAttempt } = await import(target);
  const owner = { manifest: {}, modelConfig: { maxSeqLen: 8 }, modelPartition: {
    index: 0, plan: { activationDtype: 'f32', hiddenSize: 2, vocabSize: 100 } }, gpuContext: { device },
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 8 } } } }, executionPlanState: {}, weights: new Map(), useGPU: true };
  const parent = createPartitionAttempt(owner, 8);
  const layer = parent.state.kvCache.layers[0];
  layer.keysGPU.bytes.fill(13); layer.valuesGPU.bytes.fill(27);
  layer.pageTable.set([3, 1, 0, 2]); device.queue.writeBuffer(layer.pageTableGPU, 0, layer.pageTable);
  layer.allocatedPages = 3; layer.seqLen = 5;
  parent.state.kvCache.currentSeqLen = 5; parent.state.kvCache.totalTokensSeen = 9;
  parent.state.currentSeqLen = 5; parent.state.currentTokenIds = [1, 2, 3, 4, 5]; parent.state.decodeStepCount = 4;
  const recurrent = { layerIdx: 0, seqLen: 5, numVHeads: 1, headKDim: 1, headVDim: 1,
    recurrentStatePrecision: 'two-f32', recurrentStateComponents: 2, recurrentState: new Float32Array([1, 2 ** -25]),
    convDim: 1, convKernelSize: 2, convStatePrecision: 'two-f32', convStateComponents: 2, normWeightOffset: false,
    convState: new Float32Array([3, 2 ** -26, 4, -(2 ** -27)]),
    convWeight: new Float32Array([1]), dtBias: new Float32Array([1]), aLog: new Float32Array([1]), normWeight: new Float32Array([1]) };
  parent.state.linearAttentionRuntime.layers.set(0, recurrent);
  const child = await clonePartitionAttempt(parent, new AbortController().signal);
  assert.equal(child.state.weights, parent.state.weights, 'immutable weights are shared');
  assert.equal(child.state.currentSeqLen, 5); assert.equal(child.state.decodeStepCount, 4);
  assert.equal(child.state.kvCache.totalTokensSeen, 9);
  const copied = child.state.kvCache.layers[0];
  for (const field of ['keysGPU', 'valuesGPU', 'pageTableGPU']) {
    assert.deepEqual(copied[field].bytes, layer[field].bytes); assert.notEqual(copied[field], layer[field]);
    copied[field].bytes.fill(0); assert(layer[field].bytes.some(value => value !== 0));
  }
  assert.deepEqual(copied.pageTable, layer.pageTable); copied.pageTable.fill(0); assert.equal(layer.pageTable[0], 3);
  const state = child.state.linearAttentionRuntime.layers.get(0);
  assert.deepEqual(state.recurrentState, recurrent.recurrentState); assert.deepEqual(state.convState, recurrent.convState);
  state.recurrentState.fill(0); state.convState.fill(0); child.state.currentTokenIds[0] = 99;
  assert.equal(recurrent.recurrentState[0], 1); assert.equal(recurrent.convState[0], 3); assert.equal(parent.state.currentTokenIds[0], 1);
  child.close();
  const abort = new AbortController();
  wait = async () => { if (submitted > 1) abort.abort(Error('cancelled after copy')); };
  await assert.rejects(clonePartitionAttempt(parent, abort.signal), /cancelled after copy/);
  assert(caches.at(-1).layers.every(item => item.keysGPU === null && item.valuesGPU === null));
  assert.equal(layer.keysGPU.destroyed, false, 'cancelled child never releases parent');
  wait = async () => { if (submitted > 2) throw Error('device completion unknown'); };
  let unsettled;
  await assert.rejects(clonePartitionAttempt(parent, new AbortController().signal), error => {
    unsettled = error.unsettledAttempt; return !!unsettled;
  });
  assert.equal(unsettled.state.kvCache.layers[0].keysGPU.destroyed, false, 'unknown completion retains its allocation owner');
  wait = async () => {}; unsettled.close();
  const { replayPartitionAttempt } = await import(replayURL);
  const calls = [], logits = [];
  globalThis.runPartitionReplayFixture = async (state, request) => {
    calls.push({ position: state.currentSeqLen, ...structuredClone(request) });
    if (state.modelPartition.index === 0) return { activationBytes: new Uint8Array(8).fill(request.tokenIds[0]).buffer };
    const buffer = { released: false }; logits.push(buffer);
    return { logits: { logitsBuffer: buffer, vocabSize: 100 } };
  };
  const boundaryRows = [new Uint8Array(8).fill(3), new Uint8Array(8).fill(4)];
  const replayed = await replayPartitionAttempt(parent, { tokenIds: [3, 4], boundaryRows }, new AbortController().signal);
  assert.equal(replayed.state.currentSeqLen, 7); assert.equal(parent.state.currentSeqLen, 5);
  assert.deepEqual(calls.map(item => item.position), [5, 6]); replayed.close();
  await assert.rejects(replayPartitionAttempt(parent, { tokenIds: [3], boundaryRows: [new Uint8Array(8).fill(99)] },
    new AbortController().signal), error => error.code === 'REPLAY_DIVERGENCE');
  assert.equal(layer.keysGPU.destroyed, false);
  const outputParent = createPartitionAttempt({ ...owner, modelPartition: { ...owner.modelPartition, index: 1 } }, 8);
  const activationRows = [new Uint8Array(8).fill(41), new Uint8Array(8).fill(42)];
  const operation = replayPartitionAttempt(outputParent, { activationRows }, new AbortController().signal);
  activationRows[0].fill(99);
  const outputChild = await operation;
  assert.equal(outputChild.state.currentSeqLen, 2);
  assert.deepEqual(calls.slice(-2).map(item => [...item.activationBytes]), [Array(8).fill(41), Array(8).fill(42)]);
  assert(calls.slice(-2).every(item => !('tokenIds' in item)), 'B replays original activations, never tokens alone');
  assert(logits.every(buffer => buffer.released), 'Reconstruction releases unsampled logits');
  await assert.rejects(replayPartitionAttempt(outputParent, { tokenIds: [1], boundaryRows: [new Uint8Array(8)] },
    new AbortController().signal), /unknown contract field/);
  outputChild.close(); outputParent.close();
  parent.close();
  const pool = getBufferPool();
  assert.equal(pool.getStats().activeBuffers, 0, 'all cloned recurrent buffers are released');
  destroyBufferPool();
  await pool.settleDeferredDestruction();
  assert(buffers.every(buffer => buffer.destroyed));
} finally {
  hooks.deregister(); delete globalThis.cloneTestCreateCache; delete globalThis.runPartitionReplayFixture;
  setDevice(null, { platformConfig: null });
}
console.log('partition clone/replay: paged bytes, recurrent/conv high-low state, parent isolation, cancellation, divergence and retained B rows passed');
