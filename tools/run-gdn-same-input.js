import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const [root,fixtureDirectory,runPath,out]=process.argv.slice(2);
assert(root&&fixtureDirectory&&runPath&&out,'Usage: run-gdn-same-input.js RUNTIME_ROOT FIXTURE_DIRECTORY RUN_JSON NEW_OUTPUT');
const sha=b=>createHash('sha256').update(b).digest('hex');
const fixtureBytes=await fs.readFile(path.join(fixtureDirectory,'fixture.json')),fixture=JSON.parse(fixtureBytes);
const originalRun=JSON.parse(await fs.readFile(runPath));
assert.equal(fixture.manifestSha256,originalRun.expectedManifestSha256);
const fromRoot=file=>import(pathToFileURL(path.join(root,file)).href);
const gpu=await fromRoot('tests/helpers/gpu-probe.js'),pool=await fromRoot('src/memory/buffer-pool.js');
const {getDevice,destroyDevice}=await fromRoot('src/gpu/device.js');
const {releaseNodeWebGPU}=await fromRoot('src/tooling/node-webgpu.js');
const {createTensor}=await fromRoot('src/gpu/tensor.js');
const {runLinearAttentionCoreGPU}=await fromRoot('src/gpu/kernels/linear-attention-core.js');
const {setRuntimeConfig,resetRuntimeConfig,getRuntimeConfig}=await fromRoot('src/config/runtime.js');
const probe=await gpu.probeNodeGPU({installFileFetchShim:true});assert(probe.ready,probe.reason);
const device=getDevice(),errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
const held=[];const keep=b=>{held.push(b);return b;};
async function bytes(record,relative=false){const file=relative?path.join(fixtureDirectory,record.file):record.file;const b=await fs.readFile(file);assert.equal(sha(b),record.sha256);if(record.bytes!==undefined)assert.equal(b.length,record.bytes);return b;}
function view(b){return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
function upload(data,label){const buffer=keep(pool.acquireBuffer(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label));device.queue.writeBuffer(buffer,0,data);return buffer;}
function compare(a,b){assert.equal(a.length,b.length);let max=0,sq=0,ref=0,exact=0;const ai=new Uint32Array(a.buffer,a.byteOffset,a.length),bi=new Uint32Array(b.buffer,b.byteOffset,b.length);for(let i=0;i<a.length;i++){assert(Number.isFinite(a[i])&&Number.isFinite(b[i]));const d=a[i]-b[i];max=Math.max(max,Math.abs(d));sq+=d*d;ref+=b[i]*b[i];exact+=ai[i]===bi[i]?1:0;}return{values:a.length,exact,maxAbsolute:max,differenceRms:Math.sqrt(sq/a.length),relativeRms:Math.sqrt(sq/ref)};}
const result={scope:'Standalone learned GDN ordinary-F32 candidate on fixed projections; not model qualification or performance',normalModelQualified:false,fixtureSha256:sha(fixtureBytes),runSha256:sha(await fs.readFile(runPath)),producerSha256:sha(await fs.readFile(new URL(import.meta.url))),rows:[],gpuErrors:errors};
try{
 result.runtimeSourceSha256={};
 for(const file of originalRun.sourceFiles)result.runtimeSourceSha256[file]=sha(await fs.readFile(path.join(root,file)));
 setRuntimeConfig(originalRun.runtimeConfig);result.resolvedRuntimeSession=getRuntimeConfig().inference.session;
 const g=fixture.geometry,T=fixture.numTokens;
 const tensors={};
 for(const key of ['qkv','z','a','b']){const data=await bytes(fixture.projections[key]);const width=key==='qkv'?g.convDim:key==='z'?g.valueDim:g.numVHeads;assert.equal(data.length,T*width*4);tensors[key]=createTensor(upload(data,key),'f32',[T,width],key);}
 const weights={};for(const [key,record]of Object.entries(fixture.weights))weights[key]=upload(await bytes(record,true),key);
 const expected=view(await bytes(fixture.projections.core));const reference=view(await bytes(fixture.sourceCore,true));
 for(const precision of ['f32']){
  const components=precision==='f32'?1:2,mark=held.length;
  const state={...g,sourceLayout:'hf',recurrentStatePrecision:precision,recurrentStateComponents:components,
   convWeightGPU:weights.convWeight,dtBiasGPU:weights.dtBias,aLogGPU:weights.aLog,normWeightGPU:weights.normWeight,
   convStateGPU:upload(new Float32Array(g.convDim*g.convKernelSize),'conv-state'),
   recurrentStateGPU:upload(new Float32Array(g.numVHeads*g.headKDim*g.headVDim*components),'recurrent-state')};
  let output=null,modeFailure;
  try{
   output=await runLinearAttentionCoreGPU(tensors.qkv,tensors.z,tensors.a,tensors.b,state,{numTokens:T,qkL2NormEps:g.qkL2NormEps,outputDtype:'f32',normGateActivation:fixture.normGateActivation,runtimeSession:getRuntimeConfig().inference.session});
   const raw=Buffer.from(await pool.readBuffer(output.buffer,T*g.valueDim*4));const actual=view(raw);
   const captured=compare(actual,expected),source=compare(actual,reference);
   const stateBytes=new Float32Array(await pool.readBuffer(state.recurrentStateGPU,g.numVHeads*g.headKDim*g.headVDim*components*4));
   let nonzeroCorrections=0;if(components===2){for(let i=1;i<stateBytes.length;i+=2)if(stateBytes[i]!==0)nonzeroCorrections++;assert(nonzeroCorrections>0,'Two-component state was not exercised');}
   const file=out+'.'+precision+'.f32';await fs.writeFile(file,raw,{flag:'wx'});
   result.rows.push({precision,components,captured,source,nonzeroCorrections,output:{file,sha256:sha(raw),shape:[T,g.valueDim]},stateGpuBytes:state.recurrentStateGPU.size});
   console.log(JSON.stringify(result.rows.at(-1)));
  }catch(error){modeFailure=error;throw error;}finally{const cleanup=[];try{await device.queue.onSubmittedWorkDone();}catch(e){cleanup.push(e);}if(output)try{pool.releaseBuffer(output.buffer);}catch(e){cleanup.push(e);}for(const b of held.splice(mark))try{pool.releaseBuffer(b);}catch(e){cleanup.push(e);}if(cleanup.length)throw new AggregateError([modeFailure,...cleanup].filter(Boolean),'Core mode cleanup');}
 }
 await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);result.completed=true;
}catch(error){result.failure={name:error.name,message:error.message};throw error;}
finally{
 const failures=[];try{await device.queue.onSubmittedWorkDone();}catch(e){failures.push(String(e));}
 for(const b of held)try{pool.releaseBuffer(b);}catch(e){failures.push(String(e));}
 try{destroyDevice();}catch(e){failures.push(String(e));}
 try{await releaseNodeWebGPU();}catch(e){failures.push(String(e));}resetRuntimeConfig();result.cleanupFailures=failures;
 await fs.writeFile(out,JSON.stringify(result,null,2)+'\n',{flag:'wx'});if(failures.length)throw new AggregateError(failures,'Precision control cleanup');
}
