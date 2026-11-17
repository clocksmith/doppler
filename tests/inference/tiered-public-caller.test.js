import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';

// Composed CPU caller check: real public initialization, loader bindings,
// packed-bank planner, TieredExpertCache, mapped cache, compact cache and routed
// chunk adapter. Synthetic source/device; kernel arithmetic alone is substituted.
globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_WRITE: 2 };
globalThis.GPUMapMode = { WRITE: 2 };
const state = globalThis.tieredPublicCaller = { live: 0, reads: 0, readbacks: 0, releases: 0, device: null };
const device = state.device = { limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 }, lost: new Promise(() => {}),
  createBuffer(descriptor) {
    state.live += descriptor.size;
    return { ...descriptor, data: new Uint8Array(descriptor.size), mapped: descriptor.mappedAtCreation === true, destroyed: false,
      getMappedRange() { assert(this.mapped); return this.data.buffer; },
      async mapAsync(mode) { assert.equal(mode, GPUMapMode.WRITE); assert(!this.destroyed && !this.mapped); this.mapped = true; },
      unmap() { this.data = new Uint8Array(structuredClone(this.data.buffer, { transfer: [this.data.buffer] })); this.mapped = false; },
      destroy() { if (!this.destroyed) state.live -= this.size; this.destroyed = true; } };
  },
  createCommandEncoder() { const ops = []; return {
    copyBufferToBuffer(source, from, target, to, size) { ops.push(() => { assert(!source.destroyed && !source.mapped); target.data.set(source.data.subarray(from, from + size), to); }); },
    finish: () => ops,
  }; },
  queue: { submit(commands) { for (const ops of commands) for (const op of ops) op(); },
    writeBuffer(target, offset, bytes) { target.data.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), offset); },
    onSubmittedWorkDone: async () => {} },
};
state.make = bytes => device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
state.remap = (indices, map, count) => { const result = state.make(count * 4), ids = new Uint32Array(indices.data.buffer), slots = new Uint32Array(map.data.buffer);
  new Uint32Array(result.data.buffer).set(Array.from(ids.subarray(0, count), id => slots[id])); return result; };
state.execute = options => {
  const slots = new Uint32Array(options.indicesBuffer.data.buffer);
  for (let i = 0; i < options.numTokens * options.topK; i++) {
    const expert = state.expectedIds[state.tokenCursor * options.topK + i];
    for (const [plane, role] of ['gate', 'up', 'down'].entries()) {
      const start = options.layerIdx * 13824 + plane * 4608 + expert * 576;
      assert.deepEqual(options.weights[role].buffer.data.slice(slots[i] * 576, slots[i] * 576 + 576),
        state.source.slice(start, start + 576), 'remapped slots preserve exact original expert bytes');
    }
  }
  state.tokenCursor += options.numTokens;
  const result = state.make(options.numTokens * options.hiddenSize * 4);
  result.data.set(options.inputTensor.buffer.data.subarray(0, result.size)); return { buffer: result };
};
const root = new URL('../../src/', import.meta.url);
const initTarget = new URL('inference/pipelines/text/loading/init-weights.js', root).href;
const bindingTarget = new URL('loader/experts/loader-bindings.js', root).href;
const chunkTarget = new URL('inference/pipelines/text/moe/routed-expert-chunks.js', root).href;
const initMocks = {
  '../../../../loader/doppler-loader.js': 'export const createDopplerLoader=()=>{throw Error("unexpected factory");};',
  '../../../../debug/index.js': 'export const log={debug(){},verbose(){}};',
  '../generator/session/session-context.js': 'export const resolvePerLayerInputsSession=()=>null;',
  './load-policy.js': 'export const resolveShardHashVerification=()=>({verifyHashes:false});',
  './weight-loading-config.js': 'export const resolveWeightLoadingConfig=()=>({});',
};
const hook = registerHooks({ resolve(specifier, context, next) {
  let source = context.parentURL === initTarget ? initMocks[specifier] : null;
  if ([bindingTarget, chunkTarget].includes(context.parentURL) && specifier.endsWith('/gpu/device.js')) source = 'export const getDevice=()=>tieredPublicCaller.device;';
  if (context.parentURL === bindingTarget && specifier.endsWith('/memory/buffer-pool.js')) source = 'export const getBufferPool=()=>({getCurrentBytesAllocated:()=>0});';
  if (context.parentURL === chunkTarget && specifier.endsWith('/memory/buffer-pool.js')) source = `export const acquireBuffer=bytes=>tieredPublicCaller.make(bytes);
    export const releaseBuffer=buffer=>{buffer.destroy();};
    export const readBufferSlice=async(buffer,offset,bytes)=>{tieredPublicCaller.readbacks++;return buffer.data.slice(offset,offset+bytes).buffer;};`;
  if (context.parentURL === chunkTarget && specifier.endsWith('/gpu/kernels/moe-route.js')) source = 'export const runRouteRemap=(indices,map,count)=>tieredPublicCaller.remap(indices,map,count);';
  if (context.parentURL === chunkTarget && specifier === './qwen-route-executor.js') source = 'export const runQwenRouteExperts=async options=>tieredPublicCaller.execute(options);export const recordQwenRouteExperts=async(_recorder,options)=>tieredPublicCaller.execute(options);';
  return source ? { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true } : next(specifier, context);
} });
try {
  const api = await import(bindingTarget), { loadWeights } = await import(initTarget), { executeRoutedExpertChunks } = await import(chunkTarget);
  const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true, storageMode: 'tiered-compact',
    tiered: { residentLayers: [0], parkedLayers: [1], mappedCacheBytes: 6912, mappedProtectedShare: 0 }, compactLayerBytes: { 1: 6944, 2: 6944 },
    sourceReadMode: 'expert-lease-batch', cacheBytes: 48448, stagingBytes: 65536, readConcurrency: 2, gpuBudgetBytes: 1 << 20, reserveBytes: 4096 };
  const manifest = { modelId: 'synthetic', shards: [], architecture: { numLayers: 3, hiddenSize: 32 },
    moeConfig: { expertFormat: 'qwen3_5', numExperts: 8, numExpertsPerToken: 2, expertIntermediateSize: 32 } };
  const source = Uint8Array.from({ length: 3 * 13824 }, (_, i) => 1 + i % 250), locations = new Map();
  for (let layer = 0; layer < 3; layer++) for (const [plane, role] of ['gate', 'up', 'down'].entries()) locations.set(`blk.${layer}.ffn_${role}_exps.weight`,
    { shardIndex: 0, offset: layer * 13824 + plane * 4608, size: 4608, shape: [8, 32, 32], dtype: 'Q4_0', layout: 'row' });
  const rangeLease = { maxPartBytes: 1728, memoryBytes: 4096, concurrency: 2,
    async readMany(ranges) { state.reads++; const bytes = new Uint8Array(ranges.reduce((sum, range) => sum + range.length, 0)); let offset = 0;
      for (const range of ranges) { bytes.set(source.subarray(range.offset, range.offset + range.length), offset); offset += range.length; }
      return { bytes, release() { state.releases++; structuredClone(bytes.buffer, { transfer: [bytes.buffer] }); } }; },
    read: async () => { throw Error('unexpected unbatched source read'); } };
  const loader = { manifest, tensorLocations: locations, layers: new Map([[2, {}], [1, {}], [0, {}]]), routedExpertCache: null,
    shardCache: { rangeLease, hasCustomRangeIntoLoader: false },
    setQ4KConfig() {}, setLoaderDebugConfig() {}, setPerLayerInputSession() {}, setTensorsJsonUrl() {}, setTensorsJsonLoader() {},
    init: async () => {}, load: async () => {}, getLayerWeights: layer => loader.layers.get(layer), getLoadTiming: () => null,
    getRoutedExpertPolicy: () => policy, prepareRoutedExpertResidency: signal => api.prepareRoutedExpertResidency(loader, signal),
    getResidentRoutedExpertWeights: layer => api.getResidentRoutedExpertWeights(loader, layer), getRoutedExpertCapacity: (layer, phase) => api.getRoutedExpertCapacity(loader, layer, phase),
    prepareRoutedExperts: (layer, ids, signal, phase) => api.prepareRoutedExperts(loader, layer, ids, signal, phase),
    unload: async () => { const errors = await api.disposeRoutedExperts(loader); if (errors.length) throw new AggregateError(errors); } };
  const loaded = await loadWeights(manifest, { numLayers: 3, useMoE: true }, { loader, storageContext: {} });
  assert.equal(loaded.loader, loader); assert.equal(state.reads, 16, 'public setup reads resident and parked experts once');
  const input = state.make(3 * 32 * 4), indices = state.make(6 * 4), routes = state.make(6 * 4);
  state.source = source; state.expectedIds = [7, 1, 3, 2, 0, 4];
  input.data.fill(9); new Uint32Array(indices.data.buffer).set(state.expectedIds);
  for (const layerIdx of [0, 1, 2]) {
    state.tokenCursor = 0;
    const beforeReadbacks = state.readbacks, beforeReads = state.reads;
    const result = await executeRoutedExpertChunks({ layerIdx, phase: 'prefill', numTokens: 3, topK: 2, hiddenSize: 32,
      inputTensor: { buffer: input }, indicesBuffer: indices, weightsBuffer: routes, activationDtype: 'f32', groupedPrefill: true }, loader, policy);
    assert.deepEqual(result.buffer.data, input.data); result.buffer.destroy();
    assert.equal(state.readbacks - beforeReadbacks, layerIdx === 0 ? 0 : 2);
    if (layerIdx < 2) assert.equal(state.reads, beforeReads, 'resident and parked requests never reread source');
  }
  await loader.unload(); input.destroy(); indices.destroy(); routes.destroy();
  assert.equal(state.reads, state.releases); assert.equal(state.live, 0);
} finally { hook.deregister(); delete globalThis.tieredPublicCaller; }
console.log('composed CPU tiered caller: public preload -> real factory/cache -> resident bypass/parked/streamed prefill -> zero retirement passed');
