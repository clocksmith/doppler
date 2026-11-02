import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runGather, recordGather } from '../../src/gpu/kernels/gather.js';
import { q40Fixture, decodeQ40, projectReference,packedFixture,decodePacked } from '../helpers/q4_0-fixture.js';
const probe=await probeNodeGPU({installFileFetchShim:true});assert.ok(probe.ready,probe.reason);
const device=getDevice(), owned=[], errors=[], cases=[];
device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
const upload=data=>{const b=acquireBuffer(Math.ceil(data.byteLength/4)*4);owned.push(b);const bytes=new Uint8Array(Math.ceil(data.byteLength/4)*4);bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));uploadData(b,bytes);return b;};
try {
  for(const [M,N,K] of [[1,1,32],[1,7,96],[1,129,256],[3,19,64],[17,21,96],[33,5,2048]]) {
    const input=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*0.13)*0.1);
    const bytes=q40Fixture(N,K), weights=createWeightBuffer(upload(bytes),'q4_0','row',[N,K]);
    const A=createTensor(upload(input),'f32',[M,K]), expected=projectReference(input,decodeQ40(bytes,N,K),M,N,K);
    for(const recorded of [false,true]) {
      const recorder=recorded?createCommandRecorder('q4_0-test'):null;
      const result=recorder?await recordMatmul(recorder,A,weights,M,N,K,{outputDtype:'f32'}):await runMatmul(A,weights,M,N,K,{outputDtype:'f32'});
      owned.push(result.buffer);if(recorder)await recorder.submitAndWait();
      const actual=new Float32Array(await readBuffer(result.buffer,M*N*4));let maxAbs=0;
      for(let i=0;i<actual.length;i++) {assert.ok(Number.isFinite(actual[i]));maxAbs=Math.max(maxAbs,Math.abs(actual[i]-expected[i]));assert.ok(Math.abs(actual[i]-expected[i])<0.0002,`${M},${N},${K} at ${i}: ${actual[i]} vs ${expected[i]}`);}
      cases.push({M,N,K,recorded,maxAbs});
    }
    await assert.rejects(runMatmul(A,weights,M,N,K,{outputDtype:'f16'}),/F32/);
    await assert.rejects(runMatmul(A,weights,M,N,K,{outputDtype:'f32',outputBuffer:A.buffer}),/alias|undersized/);
  }
  {
    const N=7,K=96,bytes=q40Fixture(N,K), decoded=decodeQ40(bytes,N,K);
    const weights=upload(bytes), ids=upload(new Uint32Array([6,2,0,7]));
    for(const recorded of [false,true]) {
      const recorder=recorded?createCommandRecorder('q40-gather'):null;
      const options={embeddingDtype:'q4_0',outputDtype:'f32',indexOffset:1};
      const out=recorder?await recordGather(recorder,ids,weights,3,K,N,options):await runGather(ids,weights,3,K,N,options);
      owned.push(out.buffer);if(recorder)await recorder.submitAndWait();
      const result=new Float32Array(await readBuffer(out.buffer,3*K*4));
      assert.deepEqual(Array.from(result),[...decoded.slice(2*K,3*K),...decoded.slice(0,K),...new Float32Array(K)]);
      cases.push({gather:true,recorded,indexOffset:1,maxAbs:0});
    }
  }
  for(const dtype of ['q4_1','q5_0','q8_0','q6k'])for(const M of [1,17]) {
    const N=7,K=256,input=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*0.21)*0.1);
    const packed=packedFixture(dtype,N,K), decoded=decodePacked(dtype,packed,N,K), B=createWeightBuffer(upload(packed),dtype,'row',[N,K]);
    const A=createTensor(upload(input),'f32',[M,K]), expected=projectReference(input,decoded,M,N,K);
    for(const recorded of [false,true]) {
      const recorder=recorded?createCommandRecorder(dtype):null;
      const out=recorder?await recordMatmul(recorder,A,B,M,N,K,{outputDtype:'f32'}):await runMatmul(A,B,M,N,K,{outputDtype:'f32'});
      owned.push(out.buffer);if(recorder)await recorder.submitAndWait();
      const actual=new Float32Array(await readBuffer(out.buffer,M*N*4));let maxAbs=0;
      for(let i=0;i<actual.length;i++){maxAbs=Math.max(maxAbs,Math.abs(actual[i]-expected[i]));assert.ok(Math.abs(actual[i]-expected[i])<0.0002,`${dtype} ${actual[i]} vs ${expected[i]}`);}
      cases.push({dtype,M,recorded,maxAbs});
    }
    const ids=upload(new Uint32Array([6,0]));const out=await runGather(ids,B.buffer,2,K,N,{embeddingDtype:dtype,outputDtype:'f32'});owned.push(out.buffer);
    assert.deepEqual(Array.from(new Float32Array(await readBuffer(out.buffer,2*K*4))),[...decoded.slice(6*K,7*K),...decoded.slice(0,K)]);
  }
  for(const dtype of ['q4_0','q4_1','q5_0','q8_0','q6k'])for(const [M,N,K] of [[17,21,256],[67,69,256]]) {
    const input=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*0.17)*0.1);
    const packed=dtype==='q4_0'?q40Fixture(N,K):packedFixture(dtype,N,K);
    const B=createWeightBuffer(upload(packed),dtype,'row',[N,K]), A=createTensor(upload(input),'f32',[M,K]);
    const refOut=await runMatmul(A,B,M,N,K,{outputDtype:'f32',variant:`${dtype}_prefill_tiled`});owned.push(refOut.buffer);
    const tileOut=await runMatmul(A,B,M,N,K,{outputDtype:'f32',variant:`${dtype}_prefill_tile64`});owned.push(tileOut.buffer);
    const ref=new Uint32Array(await readBuffer(refOut.buffer,M*N*4));
    const actual=new Uint32Array(await readBuffer(tileOut.buffer,M*N*4));
    assert.deepEqual(Array.from(actual),Array.from(ref),`${dtype}_prefill_tile64 (${M},${N},${K}) must be bitwise identical to ${dtype}_prefill_tiled`);
    cases.push({dtype,variant:`${dtype}_prefill_tile64`,M,N,K,bitwiseIdentical:true});
  }
  for(const [M,N,K] of [[19,35,68],[65,1,2048]]) {
    const input=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*0.11)*0.25);
    const wRow=Float32Array.from({length:N*K},(_,i)=>Math.cos(i*0.07)*0.25);
    const wCol=new Float32Array(K*N);
    for(let n=0;n<N;n++)for(let k=0;k<K;k++)wCol[k*N+n]=wRow[n*K+k];
    const expected=projectReference(input,wRow,M,N,K), A=createTensor(upload(input),'f32',[M,K]);
    const bRow=createWeightBuffer(upload(wRow),'f32','row',[N,K]);
    const bCol=createWeightBuffer(upload(wCol),'f32','column',[K,N]);
    const outRow=await runMatmul(A,bRow,M,N,K,{outputDtype:'f32',transposeB:true,variant:'f32'});owned.push(outRow.buffer);
    const outCol=await runMatmul(A,bCol,M,N,K,{outputDtype:'f32',transposeB:false,variant:'f32'});owned.push(outCol.buffer);
    const actualRow=new Float32Array(await readBuffer(outRow.buffer,M*N*4));
    const actualCol=new Float32Array(await readBuffer(outCol.buffer,M*N*4));
    assert.deepEqual(Array.from(new Uint32Array(actualRow.buffer)),Array.from(new Uint32Array(actualCol.buffer)),`matmul_f32 (${M},${N},${K}) transposeB=true vs false must be bitwise identical`);
    let maxAbs=0;
    for(let i=0;i<actualRow.length;i++){maxAbs=Math.max(maxAbs,Math.abs(actualRow[i]-expected[i]));assert.ok(Math.abs(actualRow[i]-expected[i])<1e-5,`matmul_f32 (${M},${N},${K}) at ${i}: ${actualRow[i]} vs ${expected[i]}`);}
    cases.push({variant:'f32',M,N,K,maxAbs,transposeBitwiseIdentical:true});
  }
  assert.deepEqual(errors,[]);console.log(JSON.stringify({provider:probe.providerReceipt,cases}));
} finally {for(const b of owned)releaseBuffer(b);destroyDevice();}
