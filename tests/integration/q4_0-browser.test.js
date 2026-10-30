import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs/promises';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createProbeServer,closeProbeServer,probeOrigin } from '../../tools/pooled-equal-weights/probe-server.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';
import { buildGGUFFixture } from '../helpers/gguf-fixture.js';
const root=fileURLToPath(new URL('../../',import.meta.url)),temp=await fs.mkdtemp(path.join(os.tmpdir(),'q40-browser-'));
const file=path.join(temp,'weights.gguf');await fs.writeFile(file,buildGGUFFixture('blk.0.attn_q.weight',[7,256],2,packedFixture('q4_0',7,256)));
const server=await createProbeServer({entry:'unused',config:{},routes:[['/src/',path.join(root,'src')],['/tests/',path.join(root,'tests')]],files:{'/weights.gguf':file}});
let browser;
try {
 browser=await chromium.launch({headless:true,args:['--enable-unsafe-webgpu'],...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page=await browser.newPage();await page.goto(`${probeOrigin(server)}/probe-config.json`);
 const receipt=await page.evaluate(async()=>{
  const {initDevice,getDevice,destroyDevice}=await import('/src/gpu/device.js');
  const {acquireBuffer,uploadData,readBuffer,releaseBuffer}=await import('/src/memory/buffer-pool.js');
  const {createTensor}=await import('/src/gpu/tensor.js');
  const {createWeightBuffer}=await import('/src/gpu/weight-buffer.js');
  const {runMatmul,recordMatmul}=await import('/src/gpu/kernels/matmul.js');
  const {runGather}=await import('/src/gpu/kernels/gather.js');
  const {createCommandRecorder}=await import('/src/gpu/command-recorder.js');
  const {packedFixture,decodePacked,projectReference}=await import('/tests/helpers/q4_0-fixture.js');
  const {parseGGUFHeader}=await import('/src/formats/gguf/types.js');
  const {parseGGUFModel}=await import('/src/converter/parsers/gguf.js');
  const {resolveSourceRuntimeBundleFromParsedArtifact}=await import('/src/tooling/source-artifact-adapter.js');
  const {loadTensorToGPU}=await import('/src/loader/tensors/tensor-loader.js');
  await initDevice();const device=getDevice(),owned=[],cases=[],errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const upload=data=>{const b=acquireBuffer(Math.ceil(data.byteLength/4)*4),bytes=new Uint8Array(Math.ceil(data.byteLength/4)*4);bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));uploadData(b,bytes);owned.push(b);return b;};
  try {
   const response=await fetch('/weights.gguf'),raw=await response.arrayBuffer(),source={size:raw.byteLength,readRange:async(offset,length)=>raw.slice(offset,offset+length)};
   const parsed=await parseGGUFModel({file:source,normalizeTensorSource:s=>s,parseGGUFHeaderFromSource:async s=>parseGGUFHeader(await s.readRange(0,s.size))});
   const tensor=parsed.tensors[0];if(JSON.stringify(tensor.shape)!=='[7,256]')throw Error('GGUF wire shape was not normalized');
   const bundle=await resolveSourceRuntimeBundleFromParsedArtifact({parsedArtifact:{sourceKind:'gguf',config:{model_type:'llama',eos_token_id:1},
    tensors:parsed.tensors.map(t=>({...t,sourcePath:'weights.gguf'})),sourceFiles:[{path:'weights.gguf',size:raw.byteLength}],auxiliaryFiles:[],sourceQuantization:'Q4_0',sourcePathForModelId:'weights.gguf',
    architecture:{numLayers:1,hiddenSize:256,intermediateSize:1024,numAttentionHeads:4,numKeyValueHeads:4,headDim:64,vocabSize:7,maxSeqLen:128}},
    async hashFileEntries(entries,hashAlgorithm){const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(x=>x.toString(16).padStart(2,'0')).join('');return entries.map(e=>({...e,hash,hashAlgorithm}));}});
   const payload=new Uint8Array(raw.slice(tensor.offset,tensor.offset+tensor.size));
   const loaded=await loadTensorToGPU(payload,bundle.model.tensors[tensor.name],tensor.name,{useFusedQ4K:false,keepF32Weights:false,allowF32UpcastNonMatmul:false,q4kLayout:null,gpuCapabilities:null});owned.push(...loaded.allocatedBuffers);
   const rawGpu=new Uint8Array(await readBuffer(loaded.data.buffer,Math.ceil(payload.length/4)*4));if(payload.some((x,i)=>x!==rawGpu[i]))throw Error('Direct GGUF GPU bytes differ');
   for(const dtype of ['q4_0','q4_1','q8_0','q6k'])for(const M of [1,17]) {
    const N=7,K=256,bytes=packedFixture(dtype,N,K),decoded=decodePacked(dtype,bytes,N,K),weights=dtype==='q4_0'?loaded.data:createWeightBuffer(upload(bytes),dtype,'row',[N,K]);
    const values=Float32Array.from({length:M*K},(_,i)=>Math.sin(i*0.13)*0.1),input=createTensor(upload(values),'f32',[M,K]),expected=projectReference(values,decoded,M,N,K);
    for(const recorded of [false,true]) {
     const recorder=recorded?createCommandRecorder('browser-native-gguf'):null;
     const out=recorder?await recordMatmul(recorder,input,weights,M,N,K,{outputDtype:'f32'}):await runMatmul(input,weights,M,N,K,{outputDtype:'f32'});owned.push(out.buffer);if(recorder)await recorder.submitAndWait();
     const actual=new Float32Array(await readBuffer(out.buffer,M*N*4));let maxAbs=0;
     for(let i=0;i<actual.length;i++){maxAbs=Math.max(maxAbs,Math.abs(actual[i]-expected[i]));if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>0.0002)throw Error(`${dtype} browser projection mismatch`);}
     cases.push({dtype,M,recorded,maxAbs});
    }
    const ids=upload(new Uint32Array([6,0])),out=await runGather(ids,weights.buffer,2,K,N,{embeddingDtype:dtype,outputDtype:'f32'});owned.push(out.buffer);
    const actual=new Float32Array(await readBuffer(out.buffer,2*K*4)),expectedGather=[...decoded.slice(6*K,7*K),...decoded.slice(0,K)];if(expectedGather.some((v,i)=>v!==actual[i]))throw Error('Browser gather mismatch');
   }
   if(errors.length)throw Error(errors.join('\n'));
   return {status:'passed',directGGUF:{bytes:payload.length,identity:true,sourceHash:bundle.model.shards[0].hash},cases,adapter:device.adapterInfo??null,features:[...device.features],userAgent:navigator.userAgent};
  } finally {for(const b of owned)releaseBuffer(b);destroyDevice();}
 });assert.equal(receipt.status,'passed');console.log(JSON.stringify(receipt));
} finally {if(browser)await browser.close();await closeProbeServer(server);await fs.rm(temp,{recursive:true,force:true});}
