import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const owner = new URL('../../src/inference/pipelines/text/embedding-normalization.js', import.meta.url).href;
const calls = [];
const embedOwner = new URL('../../src/inference/pipelines/text/embed.js', import.meta.url).href;
const released = [];
class FakeBuffer {}
globalThis.GPUBuffer = FakeBuffer;
globalThis.embeddingProbeState = {released, gather:{buffer:new FakeBuffer(),dtype:'f32',shape:[2,4]},
  paired:{buffer:new FakeBuffer(),dtype:'f32',shape:[2,4],precision:'two-f32',components:2,elementStrideBytes:8}};
globalThis.embeddingProbeTest = (...args) => { calls.push(args); };
const hooks = registerHooks({resolve(name, context, next) {
  if([owner,embedOwner].includes(context.parentURL) && name === './probes.js') return {
    url:'data:text/javascript,export const runProbes=(...args)=>globalThis.embeddingProbeTest(...args)', shortCircuit:true,
  };
  if(context.parentURL === embedOwner) {
    const sources = {
      '../../../gpu/device.js': 'export const getDevice=()=>({});export const getKernelCapabilities=()=>({hasF16:true})',
      '../../../memory/buffer-pool.js': 'export const acquireBuffer=()=>{throw Error("unexpected allocation")};export const readBuffer=()=>{};export const releaseBuffer=b=>embeddingProbeState.released.push(b)',
      '../../../gpu/kernel-selector.js': 'export const runGather=async()=>embeddingProbeState.gather;export const recordGather=runGather;export const runGatherSplit=runGather;export const recordGatherSplit=runGather;export const runScale=()=>{};export const recordScale=runScale',
      '../../../gpu/kernels/paired-cast.js': 'export const expandF32ToPaired=async()=>embeddingProbeState.paired;export const recordExpandF32ToPaired=expandF32ToPaired',
    };
    if(sources[name]) return {url:'data:text/javascript,'+encodeURIComponent(sources[name]),shortCircuit:true};
  }
  return next(name, context);
}});
try {
  const { finalizeEmbeddingOutput } = await import(owner);
  const tensor = {buffer:{}, dtype:'f32', shape:[2,4], precision:'two-f32', components:2, elementStrideBytes:8};
  assert.equal(await finalizeEmbeddingOutput(tensor, null, {numTokens:2,hiddenSize:4}),tensor);
  assert.equal(calls[0][0],'embed_out');
  for(const field of ['precision','components','elementStrideBytes']) assert.equal(calls[0][2][field],tensor[field]);
  const { embed } = await import(embedOwner);
  for(const recorded of [false,true]) for(const borrowed of [false,true]) {
    const tracked=[];released.length=0;
    globalThis.embeddingProbeTest = stage => {if(stage==='embed_out') throw Error('injected probe failure');};
    const pair=globalThis.embeddingProbeState.paired.buffer;
    const gather=globalThis.embeddingProbeState.gather.buffer;
    await assert.rejects(embed(new FakeBuffer(),new FakeBuffer(),{
      hiddenSize:4,vocabSize:8,numTokens:2,scaleEmbeddings:false,embeddingScale:null,
      embeddingNormalization:null,activationDtype:'f32',embeddingDtype:'f32',intermediatePrecision:'two-f32',
      outputBuffer:borrowed?pair:null,recorder:recorded?{trackTemporaryBuffer:b=>tracked.push(b)}:null,
    }),/injected probe failure/);
    const owned=recorded?tracked:released;
    assert.equal(owned.filter(b=>b===gather).length,1);
    assert.equal(owned.filter(b=>b===pair).length,borrowed?0:1);
  }
} finally { hooks.deregister(); delete globalThis.embeddingProbeTest; delete globalThis.embeddingProbeState; delete globalThis.GPUBuffer; }
console.log('paired embedding observation preserves logical and stored layout');
