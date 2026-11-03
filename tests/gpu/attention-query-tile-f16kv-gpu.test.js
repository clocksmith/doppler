import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { float32ToFloat16, float16ToFloat32 } from '../../src/formats/float16.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice(), errors = [], cases = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const sources = await Promise.all(['f32', 'f16kv'].map(async suffix => {
  const code = await fs.readFile(new URL(`../../src/gpu/kernels/attention_prefill_query_tile_${suffix}.wgsl`, import.meta.url), 'utf8');
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter(message => message.type === 'error'), []);
  return module;
}));
const pipelines = await Promise.all(sources.flatMap(module => [8,16].map(tile => device.createComputePipelineAsync({
  layout: 'auto', compute: { module, entryPoint: 'main', constants: { HEAD_DIM: 256, WORKGROUP_SIZE: tile * 32, QUERY_TILE: tile } },
}))));
const shapes = [
  { T:1,S:1 }, { T:7,S:17 }, { T:9,S:33,start:24 }, { T:33,S:33,softcap:0.03 },
  { T:17,S:64,start:47,window:16 }, { T:8,S:8,causal:false },
  { T:8,S:17,spanStart:0,spanLength:17 }, { T:17,S:17,kvStart:32 },
  { T:17,S:33,start:24,pageSize:8 }, { T:257,S:257,H:32 },
];
try {
for (const shape of shapes) {
  const {T,S}=shape,H=shape.H??8,KV=H/8,D=256,capacity=Math.ceil((S+(shape.kvStart??0))/8)*8;
  const owned=[],usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const upload=(data,flags=usage)=>{
    const buffer=device.createBuffer({size:Math.max(16,data.byteLength),usage:flags});
    device.queue.writeBuffer(buffer,0,data);owned.push(buffer);return buffer;
  };
  try {
    const q=Float32Array.from({length:T*H*D},(_,i)=>Math.sin(i*.19)*.1);
    const kh=Uint16Array.from({length:capacity*KV*D},(_,i)=>float32ToFloat16(i%29===0?2**-24:Math.cos(i*.13)*.1));
    const vh=Uint16Array.from({length:kh.length},(_,i)=>float32ToFloat16(i%31===0?-(2**-24):Math.sin(i*.11)*.2));
    const k=Float32Array.from(kh,float16ToFloat32),v=Float32Array.from(vh,float16ToFloat32);
    const Q=upload(q),K=[upload(k),upload(kh)],V=[upload(v),upload(vh)],out=upload(new Float32Array(q.length));
    const pages=Uint32Array.from({length:capacity/8},(_,i)=>capacity/8-1-i);
    const page=upload(pages),lens=upload(new Uint32Array([S,0,0,0]));
    const u=new ArrayBuffer(80),view=new DataView(u);
    const fields=[H,KV,D,S,T,0,shape.causal===false?0:1,shape.start??0,0,shape.window??0,1,shape.kvStart??0,shape.pageSize??0,shape.pageSize?2:0,shape.spanStart??0,shape.spanLength??0,0,0,0];
    fields.forEach((x,i)=>view.setUint32(i*4,x,true));view.setFloat32(20,1/16,true);view.setFloat32(32,shape.softcap??0,true);
    const ub=upload(u,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),values=[];
    for(let index=0;index<pipelines.length;index++) {
      const pipeline=pipelines[index],half=index>=2,tile=index%2?16:8;
      const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[ub,Q,K[+half],V[+half],out,lens,page].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const read=device.createBuffer({size:q.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});owned.push(read);
      const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(T/tile)*H);pass.end();enc.copyBufferToBuffer(out,0,read,0,q.byteLength);device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);try { values.push(read.getMappedRange().slice(0)); }finally { read.unmap(); }
    }
    for(const value of values.slice(1)) assert.deepEqual(new Uint32Array(value),new Uint32Array(values[0]),'Half-cache output differs from exactly expanded F32 cache: '+JSON.stringify(shape));
    const actual=new Float32Array(values[2]);let maxAbs=0;
    const physical=logical=>shape.pageSize?pages[Math.floor(logical/8)]*8+logical%8:logical;
    for(const pos of [...new Set([0,Math.floor(T/2),T-1])])for(const head of [0,H-1]) {
      const aq=pos+(shape.start??0),scores=[];
      for(let key=0;key<S;key++) {
        const ak=key+(shape.kvStart??0),inSpan=shape.spanLength&&aq>=shape.spanStart&&aq<shape.spanStart+shape.spanLength&&ak>=shape.spanStart&&ak<shape.spanStart+shape.spanLength;
        if((shape.causal!==false&&ak>aq&&!inSpan)||(shape.window&&aq>=shape.window&&ak<aq-shape.window+1)){scores.push(-Infinity);continue;}
        let dot=0;for(let dim=0;dim<D;dim++)dot+=q[(pos*H+head)*D+dim]*k[(physical(ak)*KV+Math.floor(head/8))*D+dim];
        let score=dot/16;if(shape.softcap)score=Math.tanh(score/shape.softcap)*shape.softcap;scores.push(score);
      }
      const maximum=Math.max(...scores),weights=scores.map(score=>Number.isFinite(maximum)?Math.exp(score-maximum):0),sum=weights.reduce((a,b)=>a+b,0);
      for(let dim=0;dim<D;dim++) {
        let expected=0;for(let key=0;key<S;key++)expected+=weights[key]*v[(physical(key+(shape.kvStart??0))*KV+Math.floor(head/8))*D+dim];expected=sum?expected/sum:0;
        const delta=Math.abs(expected-actual[(pos*H+head)*D+dim]);assert.ok(Number.isFinite(delta)&&delta<=1e-6+1e-5*Math.abs(expected),JSON.stringify({shape,pos,head,dim,delta}));maxAbs=Math.max(maxAbs,delta);
      }
    }
    cases.push({...shape,bitIdenticalAcrossCacheDtypesAndTiles:true,maxAbsIndependentReference:maxAbs});
  }finally{for(const buffer of owned)buffer.destroy();}
}
await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
console.log(JSON.stringify({passed:true,scope:'Attention operator only; no model or browser qualification',adapter:getKernelCapabilities().adapterInfo,cases,errors},null,2));
}finally{destroyDevice();}
