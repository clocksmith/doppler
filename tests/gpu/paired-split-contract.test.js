import assert from 'node:assert/strict';
import { runSplitQG } from '../../src/gpu/kernels/split_qg.js';
import { runSplitQKV } from '../../src/gpu/kernels/split_qkv.js';
const pair={dtype:'f32',precision:'two-f32',components:2,elementStrideBytes:8,shape:[2,12],buffer:{size:192}};
await assert.rejects(runSplitQG({...pair,components:1},{numTokens:2,numHeads:3,headDim:2}),/Explicit interleaved two-F32/);
await assert.rejects(runSplitQKV({...pair,elementStrideBytes:4},{numTokens:2,qSize:6,kSize:3,vSize:3}),/Explicit interleaved two-F32/);
await assert.rejects(runSplitQG(pair,{numTokens:2,numHeads:3,headDim:2,qTensor:{...pair,shape:[2,6],buffer:pair.buffer}}),/alias/);
await assert.rejects(runSplitQKV(pair,{numTokens:2,qSize:6,kSize:3,vSize:3,kTensor:{dtype:'f32',shape:[2,3],buffer:{size:48}}}),/Explicit interleaved two-F32/);
await assert.rejects(runSplitQG(pair,{numTokens:0,numHeads:3,headDim:2}),/positive/);
await assert.rejects(runSplitQKV(pair,{numTokens:2,qSize:6,kSize:4,vSize:3}),/shape/);
console.log('paired split strict representation, geometry and alias guards passed');

// A recorded failure retains every owned output until recorder settlement;
// allocation failure must also unwind outputs acquired before the failure.
const { setDevice } = await import('../../src/gpu/device.js');
const { getBufferPool, destroyBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { recordSplitQG } = await import('../../src/gpu/kernels/split_qg.js');
const { recordSplitQKV } = await import('../../src/gpu/kernels/split_qkv.js');
const oldUsage=globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage={MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
const make=size=>({size,destroyed:false,destroy(){this.destroyed=true;}});
try {
  for(const kind of ['qg','qkv'])for(const recorded of [false,true])for(const borrowed of [false,true])for(const failure of ['bind','allocation']){
    if(borrowed&&failure==='allocation')continue;
    let allocations=0;
    destroyBufferPool();
    const device={features:new Set(),limits:{maxBufferSize:1<<20,maxStorageBufferBindingSize:1<<20,maxComputeWorkgroupSizeX:256,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupStorageSize:16384,maxStorageBuffersPerShaderStage:8,maxUniformBufferBindingSize:65536,maxComputeWorkgroupsPerDimension:65535},queue:{submit(){},writeBuffer(){},onSubmittedWorkDone:async()=>{}},createBuffer:({size,label})=>{if(failure==='allocation'&&label?.startsWith('paired_split')&&++allocations===2)throw new Error('controlled split allocation failure');return make(size);},createBindGroupLayout:()=>({}),createPipelineLayout:()=>({}),createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),createBindGroup(){throw new Error('controlled split bind failure');}};
    setDevice(device,{platformConfig:null});getBufferPool().configure({enablePooling:false});
    const held=[],uniforms=[],recorder={device,beginComputePass(){throw new Error('unexpected recording');},trackTemporaryBuffer:b=>held.push(b),createUniformBuffer(){const b=make(16);uniforms.push(b);return b;}};
    const input={...pair,buffer:make(192)},out=width=>({...pair,shape:[2,width],buffer:make(width*16)});
    const opts=kind==='qg'?{numTokens:2,numHeads:3,headDim:2,...(borrowed?{qTensor:out(6),gTensor:out(6)}:{})}:{numTokens:2,qSize:6,kSize:3,vSize:3,...(borrowed?{qTensor:out(6),kTensor:out(3),vTensor:out(3)}:{})};
    const promise=kind==='qg'?(recorded?recordSplitQG(recorder,input,opts):runSplitQG(input,opts)):(recorded?recordSplitQKV(recorder,input,opts):runSplitQKV(input,opts));
    await assert.rejects(promise,/controlled split (bind|allocation) failure/);
    assert.equal(held.length,recorded&&!borrowed?(failure==='allocation'?1:(kind==='qg'?2:3)):0);
    assert.equal(getBufferPool().getStats().activeBuffers,held.length);
    assert.equal(input.buffer.destroyed,false);
    for(const tensor of Object.values(opts).filter(value=>value?.buffer))assert.equal(tensor.buffer.destroyed,false);
    for(const b of held)releaseBuffer(b);for(const b of uniforms)b.destroy();await getBufferPool().settleDeferredDestruction();
    assert.equal(getBufferPool().getStats().activeBuffers,0);
  }
} finally {destroyBufferPool();setDevice(null,{platformConfig:null});if(oldUsage===undefined)delete globalThis.GPUBufferUsage;else globalThis.GPUBufferUsage=oldUsage;}
console.log('paired split recorded/immediate owned/borrowed failure settlement passed');
