import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
import { decodeReadback } from '../../src/inference/pipelines/text/debug-utils/readback-decoding.js';

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
  sample(options) { if (this.failure==='record-sample') throw Error('injected record-sample'); const buffer=options.outputBuffer ?? this.allocate(4);new Uint32Array(buffer.bytes.buffer)[0]=2;return buffer; },
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
    if (!representation) return decodeReadback(bytes, dtype);
    assert.equal(representation.precision, this.paired ? 'two-f32' : undefined);
    const values = decodeReadback(bytes, dtype, representation);
    assert.equal(values[0], this.paired ? 1 + 2 ** -26 : 1);
    return values;
  },
};
const shared = {
  '../../../../gpu/device.js': 'export const getDevice=()=>generatorStorageProbe.device; export const setTrackSubmits=()=>{};',
  '../../../../memory/buffer-pool.js': `export const acquireBuffer=(...a)=>generatorStorageProbe.allocate(...a);
    export const releaseBuffer=b=>generatorStorageProbe.release(b); export const discardBuffer=b=>generatorStorageProbe.release(b);
    export const readBuffer=(b,n)=>generatorStorageProbe.read(b,n);
    export const readBufferSlice=(b,o,n)=>generatorStorageProbe.read(b,n,o);
    export const getBufferPool=()=>({settleDeferredDestruction:async()=>{generatorStorageProbe.cleanup.push('settle');}}); export const uploadData=()=>{};`,
  '../../../../gpu/command-recorder.js': `export const createCommandRecorder=()=>generatorStorageProbe.recorder;
    export const createProfilingRecorder=createCommandRecorder; export class CommandRecorder {}`,
  '../../../../gpu/kernels/sample.js': `export const isGPUSamplingAvailable=()=>generatorStorageProbe.gpuSampling; export const recordArgmax=(r,b,n,o)=>generatorStorageProbe.sample(o); export const recordGPUSample=recordArgmax;`,
  '../../../../gpu/kernels/rep-penalty.js': `export const recordHistoryPenalties=async()=>{if(generatorStorageProbe.failure==='penalty')throw Error('injected penalty');};`,
  '../sampling.js': `export const sample=()=>{if(generatorStorageProbe.failure==='sampling')throw Error('injected sampling');return 2;}; export const applyRepetitionPenalty=()=>{}; export const applyPresencePenalty=()=>{}; export const logitsSanity=()=>{};`,
  '../logits/index.js': `export const computeLogits=async()=>{if(generatorStorageProbe.failure==='logits')throw Error('injected logits');return new Float32Array(8);};
    export const computeLogitsGPU=async()=>({logitsBuffer:generatorStorageProbe.allocate(32),vocabSize:8,logitsDtype:'f32'}); export const recordLogitsGPU=computeLogitsGPU; export const extractLastPositionLogits=v=>v;`,
  '../embed.js': 'export const embed=(...a)=>generatorStorageProbe.embed(...a);',
  '../layer.js': `export const hasConvLayers=()=>false;
    export const processLayer=async(l,b)=>{if(generatorStorageProbe.failure==='layer'||(l===1&&generatorStorageProbe.failure==='layer-second'))throw Error('injected layer');return generatorStorageProbe.replaceLayer?generatorStorageProbe.allocate(b.size):b;};`,
  '../per-layer-inputs.js': `export const preparePerLayerInputs=async()=>{if(generatorStorageProbe.failure==='ple')throw Error('injected ple');return null;};
    export const prefetchPerLayerRow=()=>{};`,
  '../debug-utils/index.js': 'export const decodeReadback=(...a)=>generatorStorageProbe.decode(...a); export const getLogitsHealth=()=>({});',
};
const onlyPrefill = {
  './session-context.js': 'export const buildLayerContext=()=>({useGPU:true}); export const debugCheckBuffer=()=>{};',
  './prefill-observation.js': `export const logPrefillCacheState=()=>{}; export const tracePrefillEmbeddingIds=()=>{};
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
  probe.replaceLayer=false;probe.gpuSampling = false; probe.paired = paired; probe.failure = null; probe.cleanup = []; probe.abortFailure = false; probe.calls.length = 0; probe.reads.length = 0; probe.copies.length = 0;
  const tracked = new Set();
  probe.recorder = {
    getEncoder: () => ({ copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
      probe.copies.push(size); target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), targetOffset);
    } }),
    trackTemporaryBuffer: buffer => tracked.add(buffer),
    submit(){probe.cleanup.push('submit');},
    completeDeferredCleanup:async()=>{probe.cleanup.push('deferred');for(const b of tracked)probe.release(b);tracked.clear();},
    async submitAndWait() { for (const buffer of tracked) probe.release(buffer); tracked.clear(); },
    async abort() { probe.cleanup.push('abort'); for (const buffer of tracked) if (probe.live.has(buffer)) probe.release(buffer); tracked.clear(); if (probe.abortFailure) throw Error('injected cleanup'); },
    getStats: () => ({}), isProfilingEnabled: () => false,
  };
  probe.device = recorded ? { queue: { onSubmittedWorkDone: async () => {} } } : null;
  const plan = { activationDtype: 'f32', ...(paired ? { activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 } : {}) };
  const state = { modelConfig: { hiddenSize: 4, vocabSize: 8, numLayers: 1 }, weights: new Map([['embed', new Float32Array(32)]]),
    convLayerStates: new Map(), stats: {}, currentSeqLen: 0, useGPU: true, decodeStepCount: 0,
    runtimeConfig: { inference: { sampling: {greedyThreshold:0.01}, compute: { activationDtype: 'f32' }, session: { prefillChunkLayers: 1, prefillTokenChunkSize: null } },
      shared: { debug: { probes: null, pipeline: { readbackSampleSize: 63 } } } },
    decodeBuffers: { resetPingPong() {}, swapPingPong() {}, getHiddenBuffer: () => null, getOutputHiddenBuffer: () => null, ownsBuffer: () => false },
  };
  return { state, plan, host: { _state: state, _getEffectiveActivationDtype: () => 'f32' },
    helpers: { buildLayerContext: () => ({ decodeBuffers: state.decodeBuffers }), releaseSharedAttentionState() {}, getLogitsWeights:()=>({}), getLogitsConfig:()=>({}) } };
}
try {
  const { _prefill, _commitPrefillHiddenChunk } = await import(prefillURL);
  const { runDecodeLayers, decodeStep } = await import(decodeURL);
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
    for (const failure of [null, 'ple', 'readback', 'layer', 'logits', 'sampling']) {
      await check(label + ' actual decodeStep ' + (failure || 'success'), async () => {
        const {state,plan,helpers} = reset(paired); probe.failure = failure;
        const work = decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[],debug:failure==='readback'},helpers);
        if (failure) await assert.rejects(work,/injected/); else assert.equal(await work,2);
        assert.equal(probe.calls[0].intermediatePrecision,paired?'two-f32':undefined);
        assert.equal(probe.live.size,0,'decodeStep must release owned activations on every exit');
      });
    }
    await check(label + ' actual decodeStep GPU logits readback failure', async()=>{
      const {state,plan,helpers} = reset(paired); probe.gpuSampling=true;probe.failure='readback';
      await assert.rejects(decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers),/injected/);
      assert.equal(probe.live.size,0,'both owned hidden and logits must be released');
    });
    for (const failure of ['ple','layer','logits','penalty','record-sample']) {
      await check(label+' recorded actual decodeStep '+failure,async()=>{
        const {state,plan,helpers}=reset(paired,true);probe.failure=failure;probe.gpuSampling=['penalty','record-sample'].includes(failure);
        await assert.rejects(decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[],temperature:0,topK:1},helpers),/injected/);
        assert.equal(probe.live.size,0); assert.deepEqual(probe.cleanup,['abort','deferred']);
      });
    }
    for (const failure of [null,'ple','layer','logits']) {
      await check(label+' actual decodeStep preserves borrowed hidden '+(failure||'success'),async()=>{
        const {state,plan,helpers}=reset(paired); probe.failure=failure;
        const borrowed=probe.allocate(4*(paired?8:4));state.decodeBuffers.getHiddenBuffer=()=>borrowed;state.decodeBuffers.ownsBuffer=b=>b===borrowed;
        const work=decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers);
        if(failure)await assert.rejects(work,/injected/);else assert.equal(await work,2);
        assert.deepEqual([...probe.live],[borrowed]);probe.release(borrowed);
      });
    }
    await check(label+' actual decodeStep failed recorder cleanup retains cause',async()=>{
      const {state,plan,helpers}=reset(paired,true);probe.failure='ple';probe.abortFailure=true;
      await assert.rejects(decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers),error=>{
        assert(error instanceof AggregateError);assert.match(error.cause.message,/injected ple/);assert.match(error.errors[1].message,/cleanup/);return true;
      });
      assert.equal(probe.live.size,0);assert.deepEqual(probe.cleanup,['abort','deferred']);
    });
    for (const recorded of [false,true]) {
      await check(label+' decode retires replaced activations '+recorded,async()=>{
        const {state,plan,helpers}=reset(paired,recorded);state.modelConfig.numLayers=2;probe.replaceLayer=true;probe.failure='layer-second';
        await assert.rejects(decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers),/injected layer/);
        assert.equal(probe.live.size,0);
      });
    }
    await check(label+' decode pending PLE rejection',async()=>{
      const {state,plan,helpers}=reset(paired);state.plePrefetchPending=Promise.reject(Error('injected pending PLE'));
      await assert.rejects(decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers),/injected pending PLE/);assert.equal(probe.live.size,0);
    });
    for (const failure of [null,'sampling']) {
      await check(label+' GPU logits decode '+(failure||'success'),async()=>{
        const {state,plan,helpers}=reset(paired);probe.gpuSampling=true;probe.failure=failure;
        const work=decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[]},helpers);
        if(failure)await assert.rejects(work,/injected sampling/);else assert.equal(await work,2);
        assert.equal(probe.live.size,0);
      });
    }
    for (const failure of [null,'map']) {
      await check(label+' fused actual decodeStep '+(failure||'success'),async()=>{
        const {state,plan,helpers}=reset(paired,true);probe.gpuSampling=true;probe.failure=failure;
        globalThis.GPUBufferUsage={MAP_READ:1,COPY_DST:2};globalThis.GPUMapMode={READ:1};
        probe.device.queue.writeBuffer=()=>{};
        probe.device.createBuffer=({size})=>{
          const buffer=probe.allocate(size);buffer.mapAsync=async()=>{if(failure==='map')throw Error('injected map');};
          buffer.getMappedRange=()=>buffer.bytes.buffer;buffer.unmap=()=>{};buffer.destroy=()=>probe.release(buffer);return buffer;
        };
        const finiteness={bytes:new Uint8Array(16)};state.finitenessBuffer=finiteness;
        const work=decodeStep(state,[1],{executionPlan:plan,suppressTokenIds:[],temperature:0,topK:1},helpers);
        if(failure)await assert.rejects(work,/injected map/);else assert.equal(await work,2);
        assert.deepEqual([...probe.live],[state.sampleReadbackBuffer],'only retained session staging remains');
        state.sampleReadbackBuffer.destroy();
      });
    }
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
