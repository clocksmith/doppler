import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { packFP8Source } from '../../src/converter/fp8-source.js';
const contract=JSON.parse(await readFile(new URL('../fixtures/fp8-operator-contract.json',import.meta.url)));
globalThis.GPUBufferUsage={MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128,INDIRECT:256,QUERY_RESOLVE:512};
const {setDevice}=await import('../../src/gpu/device.js');
const {destroyBufferPool,getBufferPool,releaseBuffer}=await import('../../src/memory/buffer-pool.js');
const {loadTensorToGPU,loadTensorToCPU}=await import('../../src/loader/tensors/tensor-loader.js');
const {runFP8Matmul}=await import('../../src/gpu/kernels/fp8.js');
const {compileWeightRepresentationPlan}=await import('../../src/loader/weight-representation-plan.js');
let failUpload=false;
const device={features:new Set(),limits:{maxStorageBufferBindingSize:1<<26,maxBufferSize:1<<26,maxComputeWorkgroupsPerDimension:65535,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupSizeY:256,maxComputeWorkgroupSizeZ:64,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupStorageSize:32768},
 createBindGroup:()=>({}),
 createBuffer:({size,usage})=>({__dopplerFakeGPUBuffer:true,size,usage,bytes:new Uint8Array(size),destroy(){this.destroyed=true;}}),
 queue:{submit:()=>{},onSubmittedWorkDone:async()=>{},writeBuffer(buffer,offset,data,dataOffset=0,length=data.byteLength){if(failUpload)throw Error('injected FP8 upload failure');const bytes=data instanceof ArrayBuffer?new Uint8Array(data):new Uint8Array(data.buffer,data.byteOffset,data.byteLength);buffer.bytes.set(bytes.subarray(dataOffset,dataOffset+length),offset);}}};
destroyBufferPool();setDevice(device,{platformConfig:null});getBufferPool().configure({enablePooling:true});
try {
 for(const encoding of ['e4m3fn','e5m2']) {
  const policy={...contract.policy,encoding,blockShape:[2,4]};
  const packed=packFP8Source(new Uint8Array([0,1,0x38,0xb8,2,3,4,5,6,7]),[2,5],policy,new Float32Array([0.5,2]));
  const dtype=encoding==='e4m3fn'?'FP8_E4M3FN':'FP8_E5M2';
  const weightDtype=encoding==='e4m3fn'?'fp8e4':'fp8e5';
  const location={dtype,role:'matmul',layout:'row',shape:[2,5],size:packed.quantized.length,storage:packed.storage};
  const name='model.layers.0.self_attn.q_proj.weight';
  const step={op:'q_proj',kernel:'matmul_fp8.wgsl',entry:'main',constants:{E4M3FN:encoding==='e4m3fn',WORKGROUP_SIZE:64}};
  const selected={id:'fp8',prefill:{steps:[step]},decode:{steps:[step]},preLayer:[],postLayer:[]};
  const defaults={useFusedQ4K:false,keepF32Weights:false,q4kLayout:'row',q4kMaterializationMode:'mixed'};
  const locations=new Map([[name,location]]);
  const requirement=compileWeightRepresentationPlan(locations,{architecture:{numLayers:1}},[selected],defaults)[name];
  assert.equal(requirement.mode,'packed-only');assert.equal(requirement.complete,true);assert.deepEqual(requirement.dtypes,[weightDtype]);
  const denseStep={op:'q_proj',kernel:'matmul_f32.wgsl',entry:'main'};
  assert.throws(()=>compileWeightRepresentationPlan(locations,{architecture:{numLayers:1}},[{...selected,prefill:{steps:[denseStep]}}],defaults),/every selected consumer/);
  const config={fp8Policy:policy,representationRequirement:requirement,allowF32UpcastNonMatmul:false};
  const loaded=await loadTensorToGPU(packed.quantized,location,'fp8-fixture',config);
  assert.equal(loaded.data.dtype,weightDtype);assert.deepEqual(loaded.data.metadata,packed.storage);
  assert.deepEqual(loaded.data.buffer.bytes.slice(0,packed.quantized.length),packed.quantized);
  assert.equal(loaded.allocatedBuffers.length,1);assert.deepEqual(Object.keys(loaded.data.materializations),[weightDtype]);
  const input={dtype:'f32',shape:[1,5],buffer:{size:20,usage:128}};
  await assert.rejects(runFP8Matmul(input,loaded.data,1,2,5,{outputDtype:'f16'}),/F32/);
  await assert.rejects(runFP8Matmul(input,loaded.data,1,2,5,{outputDtype:'f32',constants:{E4M3FN:encoding!=='e4m3fn'}}),/constants/);
  await assert.rejects(runFP8Matmul(input,loaded.data,1,2,6,{outputDtype:'f32'}),/shape/);
  await assert.rejects(runFP8Matmul(input,loaded.data,1,2,5,{outputDtype:'f32',outputBuffer:loaded.data.buffer}),/distinct/);
  releaseBuffer(loaded.data.buffer);
  assert.equal(getBufferPool().getStats().activeBuffers,0);
  for(const bad of [{...location,role:'embedding'},{...location,layout:'column'},{...location,storage:null}]) {
   await assert.rejects(loadTensorToGPU(packed.quantized,bad,'bad',config),/FP8/);
   assert.equal(getBufferPool().getStats().activeBuffers,0);
  }
  await assert.rejects(loadTensorToGPU(packed.quantized,location,'bad',{...config,fp8Policy:null}),/explicit/);
  await assert.rejects(loadTensorToGPU(packed.quantized,location,'bad',{...config,representationRequirement:{mode:'dense-only',dtypes:['f32']}}),/consumer/);
  assert.throws(()=>loadTensorToCPU(packed.quantized,location,'bad'),/Unsupported packed/);
  await assert.rejects(loadTensorToGPU(packed.quantized,location,'bad',{...config,representationRequirement:null}),/consumer/);
  for (const code of (encoding==='e4m3fn'?[0x7f,0xff]:[0x7c,0xfc,0x7d,0xff])) {
    const special=packFP8Source(new Uint8Array([code]),[1,1],policy,new Float32Array([1]));
    await assert.rejects(loadTensorToGPU(special.quantized,{...location,shape:[1,1],size:special.quantized.length,storage:special.storage},'special',config),/NaN\/infinity/);
    assert.equal(getBufferPool().getStats().activeBuffers,0);
  }
  const maximal=packFP8Source(new Uint8Array([encoding==='e4m3fn'?0x7e:0x7b]),[1,1],policy,new Float32Array([1]));
  const maxLoaded=await loadTensorToGPU(maximal.quantized,{...location,shape:[1,1],size:maximal.quantized.length,storage:maximal.storage},'max-finite',config);
  releaseBuffer(maxLoaded.data.buffer);
  failUpload=true;
  await assert.rejects(loadTensorToGPU(packed.quantized,location,'bad',config),/injected FP8/);
  assert.equal(getBufferPool().getStats().activeBuffers,0);failUpload=false;
 }
} finally {destroyBufferPool();setDevice(null,{platformConfig:null});}
console.log('FP8 generic loader retains exact packed bytes, rejects dense/CPU/wrong consumer paths, cleans failed upload and validates matmul admission.');
