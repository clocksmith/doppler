import assert from 'node:assert/strict';
import { executePairedMultiply, executePairedSiLU } from '../../src/gpu/kernels/paired-activation.js';
import { runResidualAdd, recordResidualAdd } from '../../src/gpu/kernels/residual.js';
import { runScale } from '../../src/gpu/kernels/scale.js';
import { runSiLU, runSiLURowSplit } from '../../src/gpu/kernels/silu.js';
const pair = { dtype:'f32', precision:'two-f32', components:2, elementStrideBytes:8, shape:[4], buffer:{size:32} };
const scalar = { dtype:'f32', shape:[4], buffer:{size:16} };
for (const change of [{components:1},{elementStrideBytes:4},{dtype:'f16'},{shape:[2]},{buffer:{size:16}}]) {
  await assert.rejects(executePairedMultiply(null,{...pair,...change},pair,4),/tensor|required|undersized/);
}
await assert.rejects(runResidualAdd(pair,scalar,4),/two-F32/);
await assert.rejects(runResidualAdd({...scalar,elementStrideBytes:8},scalar,4),/two-F32/);
await assert.rejects(runSiLU({...scalar,elementStrideBytes:8},{size:4,swigluLimit:null}),/two-F32/);
await assert.rejects(recordResidualAdd({},pair,scalar,4),/two-F32/);
await assert.rejects(runResidualAdd(pair,pair,4,{outputBuffer:pair.buffer}),/distinct/);
await assert.rejects(runScale(pair,Infinity,{count:4}),/Finite/);
await assert.rejects(runScale(pair,1,{count:5}),/undersized/);
await assert.rejects(runScale(pair,1,{count:4,inplace:true,outputBuffer:{size:32}}),/another output/);
await assert.rejects(runSiLU(pair,{size:4,swigluLimit:3}),/swigluLimit/);
await assert.rejects(runSiLU(pair,{size:4,swigluLimit:null,gate:pair}),/inputActivation/);
await assert.rejects(executePairedSiLU(null,pair,4,{swigluLimit:null,inputActivation:'bad',gateActivation:'silu'}),/activation/);
await assert.rejects(runSiLURowSplit(pair,{numTokens:1,dim:2,swigluLimit:null}),/declared paired/);
console.log('paired pointwise metadata, mixed precision, size, alias and declared-option guards passed');

// Injected bind failure verifies orchestration ownership, not GPU arithmetic.
const { setDevice } = await import('../../src/gpu/device.js');
const { getBufferPool, destroyBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const oldUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128 };
const buffer = size => ({ size, destroyed:false, destroy() { this.destroyed=true; } });
try {
  for(const recorded of [false,true]) for(const borrowed of [false,true]) {
    destroyBufferPool();
    const device={features:new Set(),limits:{maxBufferSize:1<<20,maxStorageBufferBindingSize:1<<20,maxComputeWorkgroupSizeX:256,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupStorageSize:16384,maxStorageBuffersPerShaderStage:8,maxUniformBufferBindingSize:65536,maxComputeWorkgroupsPerDimension:65535},
      queue:{submit(){},writeBuffer(){},onSubmittedWorkDone:async()=>{}},createBuffer:({size})=>buffer(size),
      createBindGroupLayout:()=>({}),createPipelineLayout:()=>({}),createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
      createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),createBindGroup(){throw new Error('controlled pointwise bind failure');}};
    setDevice(device,{platformConfig:null});getBufferPool().configure({enablePooling:false});
    const retained=[], uniforms=[];
    const recorder={device,beginComputePass(){throw new Error('unexpected recording');},
      createUniformBuffer(){const owned=buffer(16);uniforms.push(owned);return owned;},
      trackTemporaryBuffer(owned){retained.push(owned);}};
    const a={...pair,buffer:buffer(32)},b={...pair,buffer:buffer(32)},supplied=borrowed?buffer(32):undefined;
    await assert.rejects(executePairedMultiply(recorded?recorder:null,a,b,4,{outputBuffer:supplied}),/controlled pointwise bind failure/);
    assert.equal(getBufferPool().getStats().activeBuffers,recorded&&!borrowed?1:0);
    assert.equal(retained.length,recorded&&!borrowed?1:0,'failed recorded output remains with recorder');
    assert.equal(a.buffer.destroyed,false);assert.equal(b.buffer.destroyed,false);
    if(supplied)assert.equal(supplied.destroyed,false);
    for(const owned of retained)releaseBuffer(owned);
    for(const owned of uniforms)owned.destroy();
    await getBufferPool().settleDeferredDestruction();
    assert.equal(getBufferPool().getStats().activeBuffers,0);
  }
} finally {
  destroyBufferPool();setDevice(null,{platformConfig:null});
  if(oldUsage===undefined)delete globalThis.GPUBufferUsage;else globalThis.GPUBufferUsage=oldUsage;
}
console.log('paired pointwise owned/borrowed immediate/recorded failure settlement passed');
