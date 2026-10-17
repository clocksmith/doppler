import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { destroyBufferPool, isBufferActive } from '../../src/memory/buffer-pool.js';
import { verifySpeculativeTokens } from '../../src/inference/pipelines/text/generator/speculative-runtime.js';
const savedBuffer = globalThis.GPUBuffer, savedUsage = globalThis.GPUBufferUsage;
globalThis.GPUBuffer = class GPUBuffer { constructor(size) { this.size = size; this.bytes = new Uint8Array(size); } destroy() {} };
globalThis.GPUBufferUsage = { STORAGE:128, COPY_SRC:4, COPY_DST:8 };
const allocated = [];
const device = { createBindGroup() { return {}; }, features: new Set(), limits:{maxBufferSize:1048576,maxStorageBufferBindingSize:1048576},
 createBuffer(d) { const b = new GPUBuffer(d.size); allocated.push(b); return b; },
 createCommandEncoder() { const ops = []; return { copyBufferToBuffer(a,ao,b,bo,n) { ops.push(() => b.bytes.set(a.bytes.subarray(ao,ao+n),bo)); }, finish() { return ops; } }; },
 queue: { submit(commands) { for(const command of commands) for(const op of command) op(); }, async onSubmittedWorkDone() {} } };
setDevice(device,{platformConfig:null});
try {
 for(const failure of ['none','verification','replay','cancel']) {
  const layer={convStateGPU:new GPUBuffer(4),recurrentStateGPU:new GPUBuffer(4),seqLen:2,warnedSeqMismatch:false};
  layer.convStateGPU.bytes.fill(7); layer.recurrentStateGPU.bytes.fill(9);
  const state={modelConfig:{vocabSize:8,useMoE:false,numKvSharedLayers:0}, currentSeqLen:2,
   kvCache:{layout:'contiguous',maxSeqLen:16,truncate(n){this.length=n;}},convLayerStates:new Map(),linearAttentionRuntime:{layers:new Map([[0,layer]])}};
  let calls=0; const abort=new AbortController();
  const generator={_state:state,_sampleNextTokenFromLogits:row=>row.indexOf(Math.max(...row)),async _prefill(tokens){
   calls++; state.currentSeqLen+=tokens.length; layer.seqLen=state.currentSeqLen;
   layer.convStateGPU.bytes.fill(31);layer.recurrentStateGPU.bytes.fill(42);
   if(failure==='verification' && calls===1 || failure==='replay' && calls===2) throw new Error('injected '+failure);
   if(failure==='cancel') abort.abort();
   // Accept first draft token 3; replace second proposal 4 with target 5.
   const rows=new Float32Array(tokens.length*8); for(let i=0;i<tokens.length;i++) rows[i*8+(i===0?3:5)]=1;
   return rows;
  }};
  const run=()=>verifySpeculativeTokens(generator,[1,2,2],[3,4],{temperature:0},abort.signal,()=>false);
  if(failure==='none') {
   assert.deepEqual(await run(),{tokens:[3,5],accepted:1});assert.equal(calls,2);assert.equal(state.currentSeqLen,4);
  } else {
   await assert.rejects(run());assert.equal(state.currentSeqLen,2);assert.equal(layer.seqLen,2);
   assert.deepEqual([...layer.convStateGPU.bytes],[7,7,7,7]);assert.deepEqual([...layer.recurrentStateGPU.bytes],[9,9,9,9]);
  }
  assert(allocated.every(b=>!isBufferActive(b)),'checkpoint buffers released on every exit');
 }
} finally { destroyBufferPool(device);setDevice(null);globalThis.GPUBuffer=savedBuffer;globalThis.GPUBufferUsage=savedUsage; }
console.log('speculative-state-recovery: rejection, verification failure, replay failure and cancellation passed');
