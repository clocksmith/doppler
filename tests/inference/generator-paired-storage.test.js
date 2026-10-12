import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
import { decodeReadback } from '../../src/inference/pipelines/text/debug-utils/index.js';

// Import-only executor substitutions exercise the real generation orchestrators.
// No GPU computation, shader accuracy or whole-model acceptance is claimed.
const prefillURL = new URL('../../src/inference/pipelines/text/generator/prefill-runtime.js', import.meta.url).href;
const decodeURL = new URL('../../src/inference/pipelines/text/generator/decode.js', import.meta.url).href;
const pair = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
const probe = globalThis.generatorStorageProbe = {
  calls: [], reads: [], copies: [], live: new Set(), failure: null,
  allocate(size, usage = 0, label = 'fixture') {
    const buffer = { __dopplerFakeGPUBuffer: true, size, usage, label, bytes: new Uint8Array(size) };
    this.live.add(buffer); return buffer;
  },
  release(buffer) { assert(this.live.delete(buffer), 'owned buffer released exactly once'); },
  async embed(ids, weight, options) {
    this.calls.push(options);
    const bytes = this.paired ? 8 : 4;
    const buffer = options.outputBuffer ?? this.allocate(ids.length * options.hiddenSize * bytes);
    const words = new Float32Array(buffer.bytes.buffer);
    for (let i = 0; i < words.length; i++) words[i] = this.paired && i % 2 ? 2 ** -26 : 1;
    return createTensor(buffer, 'f32', [ids.length, options.hiddenSize], 'fixture', this.paired ? pair : undefined);
  },
  read(buffer, size, offset = 0) {
    this.reads.push({ size, offset });
    if (this.failure === 'readback') throw Error('injected readback');
    return buffer.bytes.slice(offset, offset + size).buffer;
  },
  decode(bytes, dtype, representation) {
    assert.equal(representation.precision, this.paired ? 'two-f32' : undefined);
    const values = decodeReadback(bytes, dtype, representation);
    assert.equal(values[0], this.paired ? 1 + 2 ** -26 : 1);
    return values;
  },
};
const shared = {
  '../../../../gpu/device.js': 'export const getDevice=()=>generatorStorageProbe.device; export const setTrackSubmits=()=>{};',
  '../../../../memory/buffer-pool.js': `export const acquireBuffer=(...a)=>generatorStorageProbe.allocate(...a);
    export const releaseBuffer=b=>generatorStorageProbe.release(b);
    export const readBuffer=(b,n)=>generatorStorageProbe.read(b,n);
    export const readBufferSlice=(b,o,n)=>generatorStorageProbe.read(b,n,o);
    export const getBufferPool=()=>({settleDeferredDestruction:async()=>{generatorStorageProbe.cleanup.push('settle');}}); export const uploadData=()=>{};`,
  '../../../../gpu/command-recorder.js': `export const createCommandRecorder=()=>generatorStorageProbe.recorder;
    export const createProfilingRecorder=createCommandRecorder; export class CommandRecorder {}`,
  '../embed.js': 'export const embed=(...a)=>generatorStorageProbe.embed(...a);',
  '../layer.js': `export const hasConvLayers=()=>false;
    export const processLayer=async(l,b)=>{if(generatorStorageProbe.failure==='layer')throw Error('injected layer');return b;};`,
  '../per-layer-inputs.js': `export const preparePerLayerInputs=async()=>{if(generatorStorageProbe.failure==='ple')throw Error('injected ple');return null;};
    export const prefetchPerLayerRow=()=>{};`,
  '../debug-utils/index.js': 'export const decodeReadback=(...a)=>generatorStorageProbe.decode(...a); export const getLogitsHealth=()=>({});',
};
const onlyPrefill = {
  './session-context.js': 'export const buildLayerContext=()=>({useGPU:true}); export const debugCheckBuffer=()=>{};',
  './prefill-observation.js': `export const tracePrefillEmbeddingIds=()=>{};
    export const probePrefillEmbedding=async name=>{if(generatorStorageProbe.failure===name)throw Error('injected '+name);};`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  const source = [prefillURL, decodeURL].includes(context.parentURL)
    ? (context.parentURL === prefillURL ? onlyPrefill[specifier] : null) || shared[specifier] : null;
  if (source) return { url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true };
  return next(specifier, context);
} });
const failures = [];
async function check(name, action) {
  try { await action(); console.log('PASS ' + name); }
  catch (error) { failures.push({ name, error }); console.error('FAIL ' + name + ': ' + error.message); }
  finally { probe.live.clear(); }
}
function reset(paired, recorded = false) {
  probe.paired = paired; probe.failure = null; probe.cleanup = []; probe.abortFailure = false; probe.calls.length = 0; probe.reads.length = 0; probe.copies.length = 0;
  const tracked = new Set();
  probe.recorder = {
    getEncoder: () => ({ copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
      probe.copies.push(size); target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset);
    } }),
    trackTemporaryBuffer: buffer => tracked.add(buffer),
    async submitAndWait() { for (const buffer of tracked) probe.release(buffer); tracked.clear(); },
    async abort() { probe.cleanup.push('abort'); for (const buffer of tracked) if (probe.live.has(buffer)) probe.release(buffer); tracked.clear(); if (probe.abortFailure) throw Error('injected cleanup'); },
    getStats: () => ({}), isProfilingEnabled: () => false,
  };
  probe.device = recorded ? { queue: { onSubmittedWorkDone: async () => {} } } : null;
  const plan = { activationDtype: 'f32', ...(paired ? { activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 } : {}) };
  const state = { modelConfig: { hiddenSize: 4, vocabSize: 8, numLayers: 1 }, weights: new Map([['embed', new Float32Array(32)]]),
    convLayerStates: new Map(), stats: {}, currentSeqLen: 0, useGPU: true, decodeStepCount: 0,
    runtimeConfig: { inference: { compute: { activationDtype: 'f32' }, session: { prefillChunkLayers: 1, prefillTokenChunkSize: null } },
      shared: { debug: { probes: null, pipeline: { readbackSampleSize: 63 } } } },
    decodeBuffers: { resetPingPong() {}, swapPingPong() {}, getHiddenBuffer: () => null, getOutputHiddenBuffer: () => null, ownsBuffer: () => false },
  };
  return { state, plan, host: { _state: state, _getEffectiveActivationDtype: () => 'f32' },
    helpers: { buildLayerContext: () => ({ decodeBuffers: state.decodeBuffers }), releaseSharedAttentionState() {} } };
}
try {
  const { _prefill, _commitPrefillHiddenChunk } = await import(prefillURL);
  const { runDecodeLayers } = await import(decodeURL);
  for (const paired of [false, true]) {
    const label = paired ? 'paired' : 'scalar';
    await check(label + ' recorded prefill carries full storage and commits cleanup', async () => {
      const { host, plan } = reset(paired, true);
      const output = await _prefill.call(host, [1, 2], { executionPlan: plan, _returnHidden: true });
      const bytes = paired ? 8 : 4;
      assert.equal(probe.calls[0].intermediatePrecision, paired ? 'two-f32' : undefined);
      assert.equal(output.activationBytes, bytes); assert.equal(output.representation.precision, paired ? 'two-f32' : undefined);
      assert.deepEqual(probe.copies, [2 * 4 * bytes]); assert.equal(output.currentHiddenBuffer.size, 2 * 4 * bytes);
      await _commitPrefillHiddenChunk.call(host, output);
      assert.equal(host._state.currentSeqLen, 2); assert.equal(probe.live.size, 0);
    });
    await check(label + ' debug prefill reads logical last row with correct stride', async () => {
      const { host, plan } = reset(paired);
      const output = await _prefill.call(host, [1, 2], { executionPlan: plan, _returnHidden: true, debug: true });
      const bytes = paired ? 8 : 4;
      assert.deepEqual(probe.reads, [{ size: Math.floor(Math.min(63, 8 * bytes) / bytes) * bytes, offset: 0 }, { size: 4 * bytes, offset: 4 * bytes }]);
      probe.release(output.currentHiddenBuffer); assert.equal(probe.live.size, 0);
    });
    for (const failure of ['layer', 'ple', 'prefill_base_embedding_out', 'readback']) {
      await check(label + ' prefill cleans ' + failure + ' failure', async () => {
        const { host, plan } = reset(paired); probe.failure = failure;
        await assert.rejects(_prefill.call(host, [1, 2], { executionPlan: plan, _returnHidden: true, debug: failure === 'readback' }), /injected/);
        assert.equal(probe.live.size, 0, 'prefill must release owned embedding/hidden on failure');
      });
    }
    for (const failure of ['prefill_base_embedding_out', 'prefill_embedding_override_out', 'ple']) {
      await check(label + ' recorded preparation failure settles recorder ' + failure, async () => {
        const { host, plan } = reset(paired, true); probe.failure = failure;
        await assert.rejects(_prefill.call(host, [1, 2], { executionPlan: plan, _returnHidden: true }), /injected/);
        assert.deepEqual(probe.cleanup, ['abort', 'settle']); assert.equal(probe.live.size, 0);
      });
    }
    await check(label + ' failed preparation cleanup retains original cause', async () => {
      const { host, plan } = reset(paired, true); probe.failure = 'prefill_base_embedding_out'; probe.abortFailure = true;
      await assert.rejects(_prefill.call(host, [1, 2], { executionPlan: plan, _returnHidden: true }), error => {
        assert(error instanceof AggregateError); assert.match(error.cause.message, /injected prefill_base/);
        assert.match(error.errors[1].message, /injected cleanup/); return true;
      });
      assert.deepEqual(probe.cleanup, ['abort', 'settle']); assert.equal(probe.live.size, 0);
    });
    for (const failure of [null, 'layer', 'ple']) {
      await check(label + ' ordinary decode ' + (failure || 'success'), async () => {
        const { state, plan, helpers } = reset(paired); probe.failure = failure;
        const work = runDecodeLayers(state, 1, { executionPlan: plan }, helpers);
        if (failure) await assert.rejects(work, /injected/);
        else { const result = await work; assert.equal(result.hiddenStates.size, 4 * (paired ? 8 : 4)); probe.release(result.hiddenStates); }
        assert.equal(probe.calls[0].intermediatePrecision, paired ? 'two-f32' : undefined);
        assert.equal(probe.live.size, 0);
      });
    }
  }
} finally { hooks.deregister(); delete globalThis.generatorStorageProbe; }
assert.equal(failures.length, 0, failures.map(item => item.name + ': ' + item.error.message).join('\n'));
console.log('generator-paired-storage: passed');
