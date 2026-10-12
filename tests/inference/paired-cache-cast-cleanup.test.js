import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { destroyBufferPool, getBufferPool, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createRecordedResourceScope } from '../../src/inference/resource-scope.js';
import { recordAttentionCacheCast } from '../../src/inference/pipelines/text/attention/cache-cast.js';
const oldUsage=globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage={MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
const make=size=>({size,destroyed:false,destroy(){this.destroyed=true;}});
const device={features:new Set(['shader-f16']),limits:{maxBufferSize:1<<20,maxStorageBufferBindingSize:1<<20,maxComputeWorkgroupSizeX:256,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupStorageSize:16384,maxStorageBuffersPerShaderStage:8,maxUniformBufferBindingSize:65536,maxComputeWorkgroupsPerDimension:65535},queue:{submit(){},writeBuffer(){},onSubmittedWorkDone:async()=>{}},createBuffer:({size})=>make(size),createBindGroupLayout:()=>({}),createPipelineLayout:()=>({}),createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),createBindGroup:()=>({})};
try {
  setDevice(device,{platformConfig:null});getBufferPool().configure({enablePooling:false});
  const held=[],uniforms=[];
  const recorder={device,beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};},trackTemporaryBuffer:b=>held.push(b),createUniformBuffer(){const b=make(16);uniforms.push(b);return b;}};
  const scope=createRecordedResourceScope(recorder);
  const input={dtype:'f32',precision:'two-f32',components:2,elementStrideBytes:8,shape:[4],buffer:make(32)};
  const first=await recordAttentionCacheCast(recorder,input,scope,'first-K');
  assert.equal(first.dtype,'f16');
  await assert.rejects(recordAttentionCacheCast(recorder,{...input,components:1},scope,'second-V'),/two-F32|paired/);
  scope.close('failure');
  assert.equal(held.filter(b=>b===first.buffer).length,1,'successful first conversion remains owned after second fails');
  assert.equal(input.buffer.destroyed,false);
  for(const b of held)releaseBuffer(b);for(const b of uniforms)b.destroy();await getBufferPool().settleDeferredDestruction();
  assert.equal(getBufferPool().getStats().activeBuffers,0);
}finally{destroyBufferPool();setDevice(null,{platformConfig:null});if(oldUsage===undefined)delete globalThis.GPUBufferUsage;else globalThis.GPUBufferUsage=oldUsage;}
console.log('paired cache conversion retains first output across second conversion failure and settles once');
