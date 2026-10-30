import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice,destroyDevice } from '../../src/gpu/device.js';
import { loadTensorToGPU,loadTensorToCPU } from '../../src/loader/tensors/tensor-loader.js';
import { readBuffer,releaseBuffer,getBufferPool,acquireBuffer,uploadData } from '../../src/memory/buffer-pool.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';
const probe=await probeNodeGPU({installFileFetchShim:true});assert.ok(probe.ready,probe.reason);
const config={useFusedQ4K:false,keepF32Weights:false,allowF32UpcastNonMatmul:false,q4kLayout:null,gpuCapabilities:null,
 representationRequirement:{mode:'packed-only',dtypes:['q4_0'],complete:true}};
const device=getDevice();let cases=0;
try {
 for(const [dtype,stored] of [['q4_0','Q4_0'],['q4_1','Q4_1'],['q5_0','Q5_0'],['q8_0','Q8_0'],['q6k','Q6_K']]) {
  const K=256,bytes=packedFixture(dtype,3,K),source=Uint8Array.from([9,9,...bytes,9,9]).subarray(2,bytes.length+2);
  const location={dtype:stored,shape:[3,K],layout:'row',size:bytes.length,role:'matmul'};
  const settings={...config,representationRequirement:{...config.representationRequirement,dtypes:[dtype]}};
  const result=await loadTensorToGPU(source,location,'native-source',settings);
  try {assert.equal(result.data.dtype,dtype);assert.deepEqual(new Uint8Array(await readBuffer(result.data.buffer,Math.ceil(bytes.length/4)*4)).slice(0,bytes.length),bytes);assert.deepEqual(source,bytes);cases++;}
  finally {for(const b of result.allocatedBuffers)releaseBuffer(b);}
  await assert.rejects(Promise.resolve().then(()=>loadTensorToGPU(source.subarray(1),location,'truncated',settings)),/length/);cases++;
  // Streamed payload: large and multi-span tensors arrive already uploaded; adopt the exact bytes.
  const streamedBuffer=acquireBuffer(Math.ceil(bytes.length/4)*4);
  const padded=new Uint8Array(Math.ceil(bytes.length/4)*4);padded.set(bytes);uploadData(streamedBuffer,padded);
  const streamed=await loadTensorToGPU(streamedBuffer,location,'native-streamed',settings);
  try {assert.equal(streamed.data.buffer,streamedBuffer);assert.deepEqual(streamed.allocatedBuffers,[streamedBuffer]);
   assert.deepEqual(new Uint8Array(await readBuffer(streamedBuffer,padded.length)).slice(0,bytes.length),bytes);cases++;}
  finally {releaseBuffer(streamedBuffer);}
  const shortBuffer=acquireBuffer(16);
  try {await assert.rejects(Promise.resolve().then(()=>loadTensorToGPU(shortBuffer,{...location,size:bytes.length},'short-streamed',settings)),/length/);cases++;}
  finally {releaseBuffer(shortBuffer);}
  await assert.rejects(Promise.resolve().then(()=>loadTensorToGPU(source,{...location,layout:'column'},'wrong-layout',settings)),/row/);cases++;
  await assert.rejects(Promise.resolve().then(()=>loadTensorToGPU(source,location,'wrong-consumer',{...settings,representationRequirement:{...settings.representationRequirement,dtypes:['f32']}})),/representation/);cases++;
 }
 const before=getBufferPool().getStats().activeBuffers,write=device.queue.writeBuffer.bind(device.queue);
 device.queue.writeBuffer=()=>{throw Error('Injected upload failure');};
 try {await assert.rejects(Promise.resolve().then(()=>loadTensorToGPU(packedFixture('q4_0',2,32),{dtype:'Q4_0',shape:[2,32],layout:'row',role:'matmul'},'failure',config)),/Injected upload failure/);}
 finally {device.queue.writeBuffer=write;}
 assert.equal(getBufferPool().getStats().activeBuffers,before);cases++;
 assert.throws(()=>loadTensorToCPU(new Uint8Array(18),{dtype:'Q4_0'}),/Unsupported packed/);cases++;
 console.log(JSON.stringify({status:'passed',cases}));
} finally {destroyDevice();}
