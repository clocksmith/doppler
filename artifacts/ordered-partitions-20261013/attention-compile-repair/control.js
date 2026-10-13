// Isolated candidate attention dispatch against the existing independent F64 cases.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
const [root,destination]=process.argv.slice(2), here=import.meta.dirname;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const proposal=JSON.parse(readFileSync(resolve(here,'proposal.json')));
const source=readFileSync(resolve(here,'candidate.wgsl'),'utf8');assert.equal(hash(source),proposal.candidate);
assert.equal(hash(readFileSync(resolve(root,'node_modules/webgpu/dist/linux-x64.dawn.node'))),'fbfe599055b469963e4c9bfc900a702843db962dcaba0799b3c81ae33fa43db1');
const cg=readFileSync('/proc/self/cgroup','utf8').trim().split(':').at(-1);assert(cg.endsWith('/reploid-attention-repair-20261013.scope'));
writeFileSync('/sys/fs/cgroup'+cg+'/memory.oom.group','1');
for(const [name,value] of Object.entries({'memory.high':'6442450944','memory.max':'8589934592','memory.swap.max':'0','memory.oom.group':'1'}))assert.equal(readFileSync('/sys/fs/cgroup'+cg+'/'+name,'utf8').trim(),value);
const receipt={schema:'doppler.attention-single-exp-probe/v1',proposal,pid:process.pid,events:[],rows:[],tolerance:2e-11,peakGpuBufferBytes:0};
const persist=()=>writeFileSync(destination,JSON.stringify(receipt,null,2)+'\n');
const mark=(phase,extra={})=>{receipt.events.push({phase,at:new Date().toISOString(),...extra});persist();};
const bounded=async(promise,ms,label)=>{let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error(label+' deadline; ownership unknown')),ms);})]);}finally{clearTimeout(timer);}};
const {createNativeGpuHost}=await import(pathToFileURL(resolve(root,'server/native-participant.js')));
const host=createNativeGpuHost(()=>import(pathToFileURL(resolve(root,'node_modules/webgpu/index.js'))));
let device,aborted=false,operation;const owned=new Set();let allocated=0;
const guard=()=>{if(aborted)throw Error('Stopped after deadline');};
try{
 operation=(async()=>{
  await host.open();guard();mark('host-open');
  const adapter=await navigator.gpu.requestAdapter();guard();assert(adapter&&!adapter.info.isFallbackAdapter&&adapter.features.has('shader-f16'));
  receipt.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description};
  device=await adapter.requestDevice({requiredFeatures:['shader-f16']});guard();
  const {setDevice}=await import(pathToFileURL(resolve(root,'node_modules/doppler-gpu/src/gpu/device.js')));setDevice(device,{platformConfig:null});
  receipt.errors=[];device.addEventListener('uncapturederror',e=>{receipt.errors.push(e.error.message);persist();});
  device.lost.then(info=>mark('device-lost',{reason:info.reason,message:info.message}));
  const module=device.createShaderModule({code:source});const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');assert.equal(errors.length,0,JSON.stringify(errors));guard();
  mark('pipeline-start');const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});guard();mark('pipeline-complete');
  const make=(data,usage)=>{const size=Math.ceil((typeof data==='number'?data:data.byteLength)/4)*4;assert(allocated+size<=proposal.admission.maxGpuBufferBytes);const b=device.createBuffer({size,usage});owned.add(b);allocated+=size;receipt.peakGpuBufferBytes=Math.max(receipt.peakGpuBufferBytes,allocated);if(typeof data!=='number')device.queue.writeBuffer(b,0,data);return b;};
  for(const config of [{tokens:3,kv:7,heads:4,kvheads:2,dim:8,start:4,window:0},{tokens:1,kv:9,heads:2,kvheads:1,dim:256,start:8,window:3},{tokens:1,kv:3,heads:2,kvheads:1,dim:8,start:0,kvStart:4,window:0}]){
   guard();const {tokens,kv,heads,kvheads,dim,start,window}=config,kvStart=config.kvStart??0;
   const q=Float32Array.from(Array.from({length:tokens*heads*dim},(_,i)=>((i%13)-6)/11+2**-27).flatMap(x=>[Math.fround(x),Math.fround(x-Math.fround(x))]));
   const kvals=Array.from({length:kv*kvheads*dim},(_,i)=>((i%9)-4)/8),vvals=Array.from({length:kv*kvheads*dim},(_,i)=>((i%11)-5)/8);
   const half=x=>{if(x===0)return 0;const sign=x<0?0x8000:0,abs=Math.abs(x),e=Math.floor(Math.log2(abs));return sign|((e+15)<<10)|Math.round((abs/2**e-1)*1024);};
   const halfBytes=values=>{const bytes=new Uint16Array(Math.ceil(values.length/2)*2);bytes.set(values.map(half));return bytes;};
   const scale=1/Math.sqrt(dim),uniforms=new ArrayBuffer(48),u=new Uint32Array(uniforms),f=new Float32Array(uniforms);
   u.set([tokens,kv,heads,kvheads,dim,start,kvStart,window]);f[8]=Math.fround(scale);f[9]=Math.fround(scale-f[8]);u[10]=1;
   const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
   const buffers=[make(uniforms,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),make(q,storage),make(halfBytes(kvals),storage),make(halfBytes(vvals),storage),make(tokens*heads*dim*8,storage)];
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
   const readback=make(tokens*heads*dim*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(heads,tokens,1);pass.end();encoder.copyBufferToBuffer(buffers[4],0,readback,0,readback.size);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();guard();await readback.mapAsync(GPUMapMode.READ);const actual=new Float32Array(readback.getMappedRange().slice(0));readback.unmap();
   const expected=[];
   for(let t=0;t<tokens;t++)for(let h=0;h<heads;h++){
    const scores=[],allowed=[];for(let j=0;j<kv;j++){if(kvStart+j>start+t||(window>0&&kvStart+j+window<=start+t))continue;let sum=0;for(let d=0;d<dim;d++){const qi=((t*heads+h)*dim+d)*2;sum+=(q[qi]+q[qi+1])*kvals[(j*kvheads+Math.floor(h/(heads/kvheads)))*dim+d];}scores.push(sum*scale);allowed.push(j);}
    const max=Math.max(...scores),weights=scores.map(x=>Math.exp(x-max)),den=weights.reduce((a,b)=>a+b,0);
    for(let d=0;d<dim;d++)expected.push(allowed.length?allowed.reduce((sum,j,i)=>sum+weights[i]*vvals[(j*kvheads+Math.floor(h/(heads/kvheads)))*dim+d],0)/den:0);
   }
   let maxError=0;for(let i=0;i<expected.length;i++){const error=Math.abs(actual[2*i]+actual[2*i+1]-expected[i])/Math.max(1,Math.abs(expected[i]));assert(Number.isFinite(error)&&error<=receipt.tolerance,JSON.stringify({config,i,error,expected:expected[i],actual:[actual[2*i],actual[2*i+1]]}));maxError=Math.max(maxError,error);}
   receipt.rows.push({config,maxError,passed:true});mark('case-complete',{config,maxError});
   for(const b of owned){allocated-=b.size;b.destroy();}owned.clear();
  }
  assert.equal(receipt.errors.length,0);receipt.operationSettled=true;persist();
 })();
 await bounded(operation,300000,'operation');
}catch(error){aborted=true;receipt.error=error.stack;process.exitCode=1;persist();}
finally{
 try{await bounded((async()=>{if(!receipt.operationSettled)await operation?.catch(()=>{});await device?.queue.onSubmittedWorkDone();for(const b of owned)b.destroy();owned.clear();await host.close();receipt.hostClosed=true;})(),10000,'cleanup');}catch(error){receipt.cleanupError=error.message;process.exitCode=1;}
 receipt.passed=receipt.operationSettled===true&&receipt.hostClosed===true&&!receipt.error;mark('finished',{passed:receipt.passed});
}
