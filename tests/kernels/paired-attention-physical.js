// Actual public wrapper dispatch; no model or runtime policy changes.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const root = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const receipt = { schema: 'doppler-paired-attention-wrapper-physical-v1', sources: {},
  scope: 'Actual split bit permutations, Q/K normalization, RoPE and causal GQA attention against independent F64 equations; no model qualification.' };
for (const path of ['src/gpu/kernels/paired-split.js', 'src/gpu/kernels/split_qg_paired.wgsl', 'src/gpu/kernels/split_qkv_paired.wgsl', 'src/gpu/kernels/paired-rope.js', 'src/gpu/kernels/rope_precompute_paired.wgsl', 'src/gpu/kernels/rope_qk_paired.wgsl', 'src/gpu/kernels/rmsnorm-qk.js', 'src/gpu/kernels/paired-rmsnorm.js', 'src/gpu/kernels/paired-attention.js', 'src/gpu/kernels/attention_paired.wgsl', 'src/config/kernels/registry.json', 'src/rules/generated/rule-bundle.json']) {
  receipt.sources[path] = createHash('sha256').update(await readFile(resolve(root, path))).digest('hex');
}
const server = createServer(async (req, res) => {
  try {
    const path = resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    if (path === root) { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html>'); return; }
    res.setHeader('Content-Type', extname(path) === '.js' ? 'text/javascript' : extname(path) === '.json' ? 'application/json' : 'text/plain');
    res.end(await readFile(path));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}/`);
  receipt.result = await page.evaluate(async () => {
    const { setDevice, destroyDevice } = await import('/src/gpu/device.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { runSplitQG, recordSplitQG } = await import('/src/gpu/kernels/split_qg.js');
    const { runSplitQKV, recordSplitQKV } = await import('/src/gpu/kernels/split_qkv.js');
    const { runRoPEPrecompute } = await import('/src/gpu/kernels/rope-precompute.js');
    const { runRoPEQK, recordRoPEQK } = await import('/src/gpu/kernels/rope-qk.js');
    const { runRMSNormQK, recordRMSNormQK } = await import('/src/gpu/kernels/rmsnorm-qk.js');
    const { runAttention, recordAttention } = await import('/src/gpu/kernels/attention.js');
    const { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } = await import('/src/memory/buffer-pool.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter || !adapter.features.has('shader-f16')) throw new Error('Physical F16 adapter required.');
    const device = await adapter.requestDevice({requiredFeatures:['shader-f16']}); setDevice(device,{platformConfig:null});
    const errors=[],rows=[],owned=new Set(),pooled=new Set();
    device.addEventListener('uncapturederror', event=>errors.push(event.error.message));
    const make=(bytes,upload)=>{const b=device.createBuffer({size:Math.ceil(bytes/4)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});owned.add(b);if(upload)device.queue.writeBuffer(b,0,upload);return b;};
    const rep={precision:'two-f32',components:2,elementStrideBytes:8};
    const pair=(values,shape)=>{const data=Float32Array.from(values.flatMap(x=>[Math.fround(x),Math.fround(x-Math.fround(x))]));return {tensor:createTensor(make(data.byteLength,data),'f32',shape,'paired_fixture',rep),data};};
    const numbers=async(buffer,count)=>Array.from(new Float32Array(await readBufferSlice(buffer,0,count*8)));
    const compare=(actual,expected,tolerance,label)=>{let max=0;for(let i=0;i<expected.length;i++){const value=actual[2*i]+actual[2*i+1],e=Math.abs(value-expected[i])/Math.max(1,Math.abs(expected[i]));if(!Number.isFinite(e)||e>tolerance)throw new Error(JSON.stringify({label,index:i,actual:actual.slice(i*2,i*2+2),expected:expected[i],error:e,tolerance,errors}));max=Math.max(max,e);}return max;};
    const run=async(recorded,label,immediate,record)=>{const recorder=recorded?new CommandRecorder(device,label):null;try{const result=recorder?await record(recorder):await immediate();if(recorder){recorder.submit();await device.queue.onSubmittedWorkDone();}return result;}finally{if(recorder)await recorder.abort();}};
    const collect=result=>{for(const tensor of Object.values(result))pooled.add(tensor.buffer);return result;};
    try {
      for(const kind of ['qg','qkv'])for(const recorded of [false,true])for(const borrowed of [false,true]){
        const bits=Uint32Array.from({length:48},(_,i)=>i%6===0?0x80000000:i%6===1?1:0x3f000000+i*7919);
        const input=createTensor(make(bits.byteLength,bits),'f32',[2,12],'split_bits',rep);
        const output=width=>createTensor(make(2*width*8),'f32',[2,width],'borrowed_split',rep);
        const options=kind==='qg'?{numTokens:2,numHeads:3,headDim:2,...(borrowed?{qTensor:output(6),gTensor:output(6)}:{})}:{numTokens:2,qSize:6,kSize:3,vSize:3,...(borrowed?{qTensor:output(6),kTensor:output(3),vTensor:output(3)}:{})};
        const result=await run(recorded,'paired_split',()=>kind==='qg'?runSplitQG(input,options):runSplitQKV(input,options),r=>kind==='qg'?recordSplitQG(r,input,options):recordSplitQKV(r,input,options));
        if(!borrowed)collect(result);
        for(const [name,tensor] of Object.entries(result)){
          const width=kind==='qg'||name==='Q'?6:3, expected=[];
          for(let t=0;t<2;t++)for(let d=0;d<width;d++){
            const source=kind==='qg'?t*12+Math.floor(d/2)*4+d%2+(name==='G'?2:0):t*12+d+(name==='K'?6:name==='V'?9:0);
            expected.push(bits[source*2],bits[source*2+1]);
          }
          const actual=Array.from(new Uint32Array(await readBufferSlice(tensor.buffer,0,expected.length*4)));
          if(JSON.stringify(actual)!==JSON.stringify(expected)||tensor.precision!=='two-f32')throw new Error('Paired split lost exact bits or metadata.');
        }
        rows.push({kind,recorded,borrowed,exact:true});
      }
      for(const geometry of [{positionCount:65,rotaryDim:64,frequencyBaseDim:64},{positionCount:8192,rotaryDim:4,frequencyBaseDim:64},{positionCount:5,rotaryDim:4,frequencyBaseDim:4,inverseFrequencies:[1,Math.fround(0.3651741272548377)]}]){
        const options={...geometry,maxSeqLen:8192,theta:10000000,ropeScale:1,scalingType:null,inverseFrequencies:geometry.inverseFrequencies??null,intermediatePrecision:'two-f32'};
        const table=await runRoPEPrecompute(options);pooled.add(table.cos);pooled.add(table.sin);
        const count=geometry.positionCount*geometry.rotaryDim/2;
        const expectedSin=[],expectedCos=[];
        for(let p=0;p<geometry.positionCount;p++)for(let d=0;d<geometry.rotaryDim/2;d++){const angle=p*(geometry.inverseFrequencies?.[d]??10000000**(-2*d/geometry.frequencyBaseDim));expectedSin.push(Math.sin(angle));expectedCos.push(Math.cos(angle));}
        const sinError=compare(await numbers(table.sin,count),expectedSin,2e-10,'paired sine table');
        const cosError=compare(await numbers(table.cos,count),expectedCos,2e-10,'paired cosine table');
        rows.push({kind:'precompute',geometry,sinError,cosError,tolerance:2e-10});
        if(geometry.rotaryDim===64)for(const interleaved of [false,true])for(const recorded of [false,true]){
          const tokens=2,qheads=3,kheads=1,dim=80,rot=64,start=61;
          const source=count=>Array.from({length:count},(_,i)=>((i%17)-8)/7+2**-28);
          const q=pair(source(tokens*qheads*dim),[tokens,qheads*dim]),k=pair(source(tokens*kheads*dim),[tokens,kheads*dim]);
          const expected=(data,heads)=>{const a=Array.from({length:data.length/2},(_,i)=>data[2*i]+data[2*i+1]),out=a.slice();for(let t=0;t<tokens;t++)for(let h=0;h<heads;h++)for(let d=0;d<rot/2;d++){const base=(t*heads+h)*dim,first=interleaved?2*d:d,last=interleaved?2*d+1:d+rot/2,angle=(start+t)*10000000**(-2*d/rot),c=Math.cos(angle),s=Math.sin(angle);out[base+first]=a[base+first]*c-a[base+last]*s;out[base+last]=a[base+first]*s+a[base+last]*c;}return out;};
          const options={numQHeads:qheads,numKVHeads:kheads,headDim:dim,rotaryDim:rot,pairSpanDim:rot,interleaved,startPos:start,frequencyPrecision:'two-f32'};
          const result=await run(recorded,'paired_rope',()=>runRoPEQK(q.tensor,k.tensor,table.cos,table.sin,tokens,options),r=>recordRoPEQK(r,q.tensor,k.tensor,table.cos,table.sin,tokens,options));
          const maxError=Math.max(compare(await numbers(result.q.buffer,q.data.length/2),expected(q.data,qheads),2e-11,'rotated Q'),compare(await numbers(result.k.buffer,k.data.length/2),expected(k.data,kheads),2e-11,'rotated K'));
          rows.push({kind:'rotation',recorded,interleaved,maxError,tolerance:2e-11});
        }
      }
      for(const recorded of [false,true]){
        const tokens=2,heads=3,kvheads=1,dim=8,eps=1e-6;
        const source=count=>Array.from({length:count},(_,i)=>((i%13)-6)/9+2**-28);
        const q=pair(source(tokens*heads*dim),[tokens,heads*dim]),k=pair(source(tokens*kvheads*dim),[tokens,kvheads*dim]);
        const wd=Float32Array.from({length:dim},(_,i)=>1+i/32),weight=createTensor(make(wd.byteLength,wd),'f32',[dim],'norm_weight');
        const result=collect(await run(recorded,'paired_qknorm',()=>runRMSNormQK(q.tensor,k.tensor,weight,weight,eps,{numTokens:tokens,numHeads:heads,numKVHeads:kvheads,headDim:dim}),r=>recordRMSNormQK(r,q.tensor,k.tensor,weight,weight,eps,{numTokens:tokens,numHeads:heads,numKVHeads:kvheads,headDim:dim})));
        const expected=data=>{const out=[];for(let r=0;r<data.length/2/dim;r++){const values=Array.from({length:dim},(_,d)=>data[(r*dim+d)*2]+data[(r*dim+d)*2+1]),den=Math.sqrt(values.reduce((s,x)=>s+x*x,0)/dim+eps);out.push(...values.map((x,d)=>x/den*wd[d]));}return out;};
        const maxError=Math.max(compare(await numbers(result.q.buffer,q.data.length/2),expected(q.data),2e-12,'Q norm'),compare(await numbers(result.k.buffer,k.data.length/2),expected(k.data),2e-12,'K norm'));
        rows.push({kind:'qknorm',recorded,maxError,tolerance:2e-12});
      }
      for(const config of [{tokens:3,kv:7,heads:4,kvheads:2,dim:8,start:4,window:0},{tokens:1,kv:9,heads:2,kvheads:1,dim:256,start:8,window:3},{tokens:1,kv:3,heads:2,kvheads:1,dim:8,start:0,kvStart:4,window:0}])for(const recorded of [false,true])for(const borrowed of [false,true]){
        const {tokens,kv,heads,kvheads,dim,start,window}=config,kvStart=config.kvStart??0;
        const q=pair(Array.from({length:tokens*heads*dim},(_,i)=>((i%13)-6)/11+2**-27),[tokens,heads*dim]);
        // These dyadic cache values are exactly representable in F16, independent byte encoding.
        const kvals=Array.from({length:kv*kvheads*dim},(_,i)=>((i%9)-4)/8),vvals=Array.from({length:kv*kvheads*dim},(_,i)=>((i%11)-5)/8);
        const half=x=>{if(x===0)return 0;const sign=x<0?0x8000:0,abs=Math.abs(x),e=Math.floor(Math.log2(abs));return sign|((e+15)<<10)|Math.round((abs/2**e-1)*1024);};
        const halfTensor=(values,label)=>{const bytes=new Uint16Array(Math.ceil(values.length/2)*2);bytes.set(values.map(half));return createTensor(make(bytes.byteLength,bytes),'f16',[kv,kvheads*dim],label);};
        const K=halfTensor(kvals,'K'),V=halfTensor(vvals,'V'),scale=1/Math.sqrt(dim),supplied=borrowed?make(tokens*heads*dim*8):null;
        const options={seqLen:tokens,kvLen:kv,numKVHeads:kvheads,scale,causal:true,startPos:start,kvStart,slidingWindow:window,attnSoftcap:0,kvLayout:'contiguous',outputBuffer:supplied};
        const result=await run(recorded,'paired_attention',()=>runAttention(q.tensor,K,V,null,heads,dim,options),r=>recordAttention(r,q.tensor,K,V,null,heads,dim,options));if(!borrowed)pooled.add(result.buffer);
        const expected=[];
        for(let t=0;t<tokens;t++)for(let h=0;h<heads;h++){
          const scores=[],allowed=[];for(let j=0;j<kv;j++){if(kvStart+j>start+t||(window>0&&kvStart+j+window<=start+t))continue;let sum=0;for(let d=0;d<dim;d++){const qi=((t*heads+h)*dim+d)*2;sum+=(q.data[qi]+q.data[qi+1])*kvals[(j*kvheads+Math.floor(h/(heads/kvheads)))*dim+d];}scores.push(sum*scale);allowed.push(j);}
          const max=Math.max(...scores),weights=scores.map(x=>Math.exp(x-max)),den=weights.reduce((a,b)=>a+b,0);
          for(let d=0;d<dim;d++)expected.push(allowed.length?allowed.reduce((sum,j,i)=>sum+weights[i]*vvals[(j*kvheads+Math.floor(h/(heads/kvheads)))*dim+d],0)/den:0);
        }
        const maxError=compare(await numbers(result.buffer,expected.length),expected,2e-11,'paired attention');
        rows.push({kind:'attention',config,recorded,borrowed,maxError,tolerance:2e-11});
      }
      await device.queue.onSubmittedWorkDone();for(const b of pooled)releaseBuffer(b);pooled.clear();await getBufferPool().settleDeferredDestruction();
      const stats=getBufferPool().getStats();if(stats.activeBuffers!==0||errors.length)throw new Error(JSON.stringify({stats,errors}));
      return {rows,errors,stats,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},settled:true};
    }finally{await device.queue.onSubmittedWorkDone();for(const b of pooled)releaseBuffer(b);for(const b of owned)b.destroy();destroyBufferPool();destroyDevice();}
  });
  receipt.browser = browser.version(); receipt.passed = receipt.result.rows.length === 29 && receipt.result.settled;
} catch (error) { receipt.error = error.stack; throw error; }
finally {
  await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n');
}
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired attention wrappers failed.');
