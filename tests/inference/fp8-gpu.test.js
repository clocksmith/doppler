import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { packFP8Source } from '../../src/converter/fp8-source.js';
import { loadTensorToGPU } from '../../src/loader/tensors/tensor-loader.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { observeKernelDispatches } from '../helpers/kernel-dispatch-observer.js';

const oracleBytes=await readFile(new URL('../fixtures/fp8-code-oracle.json',import.meta.url));
const oracle=JSON.parse(oracleBytes);
const contract=JSON.parse(await readFile(new URL('../fixtures/fp8-operator-contract.json',import.meta.url)));
const probe=await probeNodeGPU({installFileFetchShim:true});
if(!probe.ready){if(process.env.DOPPLER_REQUIRE_FP8_GPU==='1')throw Error(probe.reason);console.log(`fp8-gpu.test: skipped (${probe.reason})`);process.exit(0);}
const report={provider:probe.providerReceipt,capabilities:getKernelCapabilities(),oracleSha256:createHash('sha256').update(oracleBytes).digest('hex'),cases:[],claim:'Packed FP8 storage/fused dequant with portable F32 arithmetic, no native FP8 or timing claim.'};
const scratch=new Uint32Array(1),float=new Float32Array(scratch.buffer);
function value(bits){scratch[0]=bits;return float[0];}
function reference(input,codes,scales,M,N,K,policy){
 const output=new Float64Array(M*N),absolute=new Float64Array(M*N);
 const blocksK=Math.ceil(K/policy.blockShape[1]);
 for(let m=0;m<M;m++)for(let n=0;n<N;n++)for(let k=0;k<K;k++){
  const scale=scales[Math.floor(n/policy.blockShape[0])*blocksK+Math.floor(k/policy.blockShape[1])];
  const weight=Math.fround(value(oracle.formats[policy.encoding][codes[n*K+k]])*scale);
  const product=input[m*K+k]*weight;output[m*N+n]+=product;absolute[m*N+n]+=Math.abs(product);
 }
 return {output,absolute};
}
function check(expected,actual){
 let maximum=0,maximumBoundRatio=0;
 for(let i=0;i<actual.length;i++){
  const error=Math.abs(actual[i]-expected.output[i]);
  const limit=contract.numeric.absoluteTolerance+contract.numeric.sumAbsoluteTolerance*expected.absolute[i];
  assert.ok(Number.isFinite(actual[i])&&error<=limit,`FP8[${i}]:${actual[i]} !=${expected.output[i]}, error${error}>${limit}`);
  maximum=Math.max(maximum,error);maximumBoundRatio=Math.max(maximumBoundRatio,error/limit);
 }
 return {maximum,maximumBoundRatio};
}
async function runCase(encoding,{M,N,K},allCodes=false){
 const policy={...contract.policy,encoding};
 const finite=oracle.formats[encoding].map((bits,code)=>({bits,code})).filter(x=>(x.bits&0x7fffffff)<0x7f800000).map(x=>x.code);
 if(allCodes)N=finite.length;
 const codes=Uint8Array.from({length:N*K},(_,i)=>finite[allCodes?i:(i*73+19)%finite.length]);
 const scales=Float32Array.from({length:Math.ceil(N/3)*Math.ceil(K/128)},(_,i)=>allCodes?1:Math.fround(0.125+(i%7)*0.17));
 const inputValues=Float32Array.from({length:M*K},(_,i)=>allCodes?1:Math.sin(i*0.17)*0.125+Math.cos(i*0.03)*0.0625);
 const expected=reference(inputValues,codes,scales,M,N,K,policy);
 const packed=packFP8Source(codes,[N,K],policy,scales);
 const dtype=encoding==='e4m3fn'?'FP8_E4M3FN':'FP8_E5M2',weightDtype=encoding==='e4m3fn'?'fp8e4':'fp8e5';
 const location={dtype,role:'matmul',layout:'row',shape:[N,K],storage:packed.storage,size:packed.quantized.length};
 const loaded=await loadTensorToGPU(packed.quantized,location,'fp8-physical',{fp8Policy:policy,representationRequirement:{mode:'packed-only',complete:true,dtypes:[weightDtype]},allowF32UpcastNonMatmul:false});
 const inputBuffer=acquireBuffer(inputValues.byteLength);uploadData(inputBuffer,inputValues);
 const input=createTensor(inputBuffer,'f32',[M,K]);
 const variant=encoding==='e4m3fn'?'fp8_e4m3fn_f32':'fp8_e5m2_f32';
 const step={op:'q_proj',kernel:'matmul_fp8.wgsl',entry:'main',constants:{E4M3FN:encoding==='e4m3fn',WORKGROUP_SIZE:64}};
 const options={role:'q_proj',outputDtype:'f32',transposeB:true,kernelPath:{activationDtype:'f32',decode:{steps:[step]},prefill:{steps:[step]}}};
 let immediate,recorded,recorder;
 try{
  immediate=await observeKernelDispatches(()=>runMatmul(input,loaded.data,M,N,K,options));
  const actual=new Float32Array(await readBuffer(immediate.result.buffer,M*N*4));
  const checked=check(expected,actual);const maxAbsoluteError=checked.maximum;
  recorded=await observeKernelDispatches(async()=>{recorder=new CommandRecorder(null,'fp8-physical');const output=await recordMatmul(recorder,input,loaded.data,M,N,K,options);await recorder.submitAndWait();return output;});
  const replay=new Float32Array(await readBuffer(recorded.result.buffer,M*N*4));
  const recordedChecked=check(expected,replay);const recordedMaxAbsoluteError=recordedChecked.maximum;assert.deepEqual(replay,actual);
  for(const observed of [immediate,recorded]){
   assert.equal(observed.dispatches.length,1);const dispatch=observed.dispatches[0];
   assert.equal(dispatch.pipeline?.shaderFile,'matmul_fp8.wgsl');assert.equal(dispatch.pipeline?.variant,variant);
   assert.match(dispatch.pipeline.shaderHash,/^sha256:[a-f0-9]{64}$/);assert.deepEqual(dispatch.workgroups,[N,M,1]);
  }
  report.cases.push({encoding,M,N,K,allFiniteCodes:allCodes,packedBytes:packed.quantized.length,actualWeightBufferBytes:loaded.data.buffer.size,materializations:Object.keys(loaded.data.materializations),denseF32Bytes:N*K*4,maxAbsoluteError,recordedMaxAbsoluteError,maximumBoundRatio:checked.maximumBoundRatio,recordedMaximumBoundRatio:recordedChecked.maximumBoundRatio,immediate:immediate.dispatches,recorded:recorded.dispatches});
 }finally{
  if(recorder&&!recorder.getStats().submitted)await recorder.submitAndWait();
  for(const buffer of [inputBuffer,loaded.data.buffer,immediate?.result.buffer,recorded?.result.buffer])if(buffer)releaseBuffer(buffer);
 }
}
try{
 for(const encoding of ['e4m3fn','e5m2']){
  await runCase(encoding,{M:1,N:1,K:1},true);
  for(const scenario of contract.cases)await runCase(encoding,scenario);
 }
 console.log(`fp8-gpu.test: ok ${JSON.stringify(report)}`);
}finally{await destroyDevice();await releaseNodeWebGPU();}
