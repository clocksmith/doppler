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
const receipt = { schema: 'doppler-paired-attention-interpreter-physical-v1', sources: {},
  scope: 'Actual full attention interpreter: input norm, interleaved QG projection, QK norm, partial RoPE, exact F16 KV, causal GQA, sigmoid gate, output projection; independent F64 reference. Synthetic F32 weights; no model qualification.', served: {} };
for (const path of ['src/gpu/kernels/paired-split.js', 'src/gpu/kernels/split_qg_paired.wgsl', 'src/gpu/kernels/split_qkv_paired.wgsl', 'src/gpu/kernels/paired-rope.js', 'src/gpu/kernels/rope_precompute_paired.wgsl', 'src/gpu/kernels/rope_qk_paired.wgsl', 'src/gpu/kernels/rmsnorm-qk.js', 'src/gpu/kernels/paired-rmsnorm.js', 'src/gpu/kernels/paired-attention.js', 'src/gpu/kernels/attention_paired.wgsl', 'src/config/kernels/registry.json', 'src/rules/generated/rule-bundle.json']) {
  receipt.sources[path] = createHash('sha256').update(await readFile(resolve(root, path))).digest('hex');
}
const server = createServer(async (req, res) => {
  try {
    const path = resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    if (path === root) { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html>'); return; }
    res.setHeader('Content-Type', extname(path) === '.js' ? 'text/javascript' : extname(path) === '.json' ? 'application/json' : 'text/plain');
    const bytes=await readFile(path); receipt.served[path.slice(root.length+1)]=createHash('sha256').update(bytes).digest('hex'); res.end(bytes);
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
    const { createWeightBuffer } = await import('/src/gpu/weight-buffer.js');
    const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { getNormWeightBuffer } = await import('/src/inference/pipelines/text/weights.js');
    const { KVCache } = await import('/src/inference/kv-cache/base.js');
    const { runLayerAttentionGPU } = await import('/src/inference/pipelines/text/attention/run.js');
    const { recordLayerAttentionGPU } = await import('/src/inference/pipelines/text/attention/record.js');
    const { runRoPEPrecompute } = await import('/src/gpu/kernels/rope-precompute.js');
    const { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } = await import('/src/memory/buffer-pool.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter || !adapter.features.has('shader-f16')) throw new Error('Physical F16 adapter required.');
    const device = await adapter.requestDevice({requiredFeatures:['shader-f16']}); setDevice(device,{platformConfig:null});
    const errors=[],rows=[],owned=new Set(),pooled=new Set();
    device.addEventListener('uncapturederror', event=>errors.push(event.error.message));
    const H=128, D=256, HEADS=2, KVHEADS=1, ROT=64, EPS=1e-6, THETA=10000000;
    const representation={precision:'two-f32',components:2,elementStrideBytes:8};
    const upload=data=>{const b=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});owned.add(b);device.queue.writeBuffer(b,0,data);return b;};
    const normData=Float32Array.from({length:H},(_,i)=>1+(i%7)/64);
    const qn=Float32Array.from({length:D},(_,i)=>1+(i%5)/32),kn=Float32Array.from({length:D},(_,i)=>1-(i%3)/64);
    const matrix=(N,K,m)=>Float32Array.from({length:N*K},(_,i)=>((i*m%31)-15)/512);
    const qg=matrix(HEADS*D*2,H,7),kw=matrix(KVHEADS*D,H,11),vw=matrix(KVHEADS*D,H,13),ow=matrix(H,HEADS*D,17);
    const weight=(data,shape)=>createWeightBuffer(upload(data),'f32','row',shape,'layer_fixture_weight');
    const weights={inputNorm:weight(normData,[H]),qNorm:weight(qn,[D]),kNorm:weight(kn,[D]),qProj:weight(qg,[HEADS*D*2,H]),kProj:weight(kw,[D,H]),vProj:weight(vw,[D,H]),oProj:weight(ow,[H,HEADS*D])};
    const projection=op=>({op,kernel:'matmul_paired.wgsl',entry:'main',constants:{WEIGHT_F16:false,WEIGHT_Q4K:false,SCALAR_OUTPUT:false}});
    const steps=[{op:'input_norm',kernel:'rmsnorm_paired.wgsl',entry:'main'},...['q_proj','k_proj','v_proj'].map(projection),{op:'attention',kernel:'attention_paired.wgsl',entry:'main'},projection('o_proj')];
    const kernelPath={activationDtype:'f32',prefill:{steps},decode:{steps}};
    const norm=(x,w)=>{const inv=1/Math.sqrt(x.reduce((s,v)=>s+v*v,0)/x.length+EPS);return x.map((v,i)=>v*inv*w[i]);};
    const project=(x,w,N)=>Array.from({length:N},(_,n)=>x.reduce((s,v,k)=>s+v*w[n*x.length+k],0));
    const rotate=(x,position)=>{const out=x.slice();for(let i=0;i<ROT/2;i++){const a=position*THETA**(-2*i/ROT),c=Math.cos(a),s=Math.sin(a);out[i]=x[i]*c-x[i+ROT/2]*s;out[i+ROT/2]=x[i]*s+x[i+ROT/2]*c;}return out;};
    // Independent binary16 nearest-even rounding from binary64 values; no F32 intermediary.
    const half=x=>{const sign=x<0?0x8000:0,a=Math.abs(x);if(a===0)return sign;const e=Math.floor(Math.log2(a)),step=2**Math.max(-24,e-10),v=a/step,f=Math.floor(v),r=v-f,n=f+(r>.5||(r===.5&&f%2)?1:0),rounded=n*step;if(rounded>=65520)return sign|0x7c00;if(rounded<2**-14)return sign|Math.round(rounded*2**24);const exp=Math.floor(Math.log2(rounded));return sign|((exp+15)<<10)|Math.round((rounded/2**exp-1)*1024);};
    const unhalf=b=>{const e=(b>>>10)&31,m=b&1023;return (b&32768?-1:1)*(e?2**(e-15)*(1+m/1024):m*2**-24);};
    const table=await runRoPEPrecompute({positionCount:8,maxSeqLen:8,rotaryDim:ROT,frequencyBaseDim:ROT,theta:THETA,ropeScale:1,scalingType:null,inverseFrequencies:null,intermediatePrecision:'two-f32'});pooled.add(table.cos);pooled.add(table.sin);
    try {
      for(const recorded of [false,true]){
        const cache=new KVCache({numLayers:1,numHeads:KVHEADS,headDim:D,maxSeqLen:8,useGPU:true,layout:'contiguous',pageSize:8,kvDtype:'f16'});
        const state={stats:{},observationContext:{receiptPolicy:'off'},kvCache:cache,ropeFreqsCos:table.cos,ropeFreqsSin:table.sin,resolvedRuntimeSession:{schema:'doppler.resolved-runtime-session/v1',runtime:{session:{intermediatePrecision:'two-f32',activationDtype:'f32',kvDtype:'f16'}}}};
        const expectedK=[],expectedV=[];
        try {
          for(const [tokens,start] of [[3,0],[1,3]]){
            const source=Float32Array.from({length:tokens*H*2},(_,i)=>i%2?((i%11)-5)*2**-30:((i*3+start*7)%29-14)/32);
            const input=createTensor(upload(source),'f32',[tokens,H],'paired_layer_input',representation);
            const queries=[],gates=[];
            for(let t=0;t<tokens;t++){
              const x=norm(Array.from({length:H},(_,i)=>source[(t*H+i)*2]+source[(t*H+i)*2+1]),normData),qall=project(x,qg,HEADS*D*2);
              queries.push(Array.from({length:HEADS},(_,h)=>rotate(norm(qall.slice(h*D*2,h*D*2+D),qn),start+t)));
              gates.push(Array.from({length:HEADS},(_,h)=>qall.slice(h*D*2+D,(h+1)*D*2)));
              expectedK.push(rotate(norm(project(x,kw,D),kn),start+t).map(half));expectedV.push(project(x,vw,D).map(half));
            }
            const expected=[];
            for(let t=0;t<tokens;t++){
              const gated=[];
              for(let h=0;h<HEADS;h++){
                const scores=expectedK.slice(0,start+t+1).map(k=>queries[t][h].reduce((s,x,d)=>s+x*unhalf(k[d]),0)/Math.sqrt(D));
                const mx=Math.max(...scores),ex=scores.map(x=>Math.exp(x-mx)),den=ex.reduce((s,x)=>s+x,0);
                for(let d=0;d<D;d++)gated.push(ex.reduce((s,x,j)=>s+x*unhalf(expectedV[j][d]),0)/den/(1+Math.exp(-gates[t][h][d])));
              }
              expected.push(...project(gated,ow,H));
            }
            const config={layerIdx:0,numTokens:tokens,isPrefill:start===0,numHeads:HEADS,numKVHeads:KVHEADS,headDim:D,hiddenSize:H,rmsNormEps:EPS,normalizationType:'rmsnorm',currentSeqLen:start,slidingWindow:null,layerType:'full_attention',residualTensor:null,attnSoftcap:null,queryPreAttnScalar:D,activationDtype:'f32',inputDtype:'f32',outputDtype:'f32',kvDtype:'f16',causalAttention:true,ropeRotaryDim:ROT,ropeFrequencyBaseDim:ROT,ropeInterleaved:false,queryKeyNorm:true,queryKeyNormType:'rmsnorm',queryKeyNormAxis:'head',queryKeyNormLayers:null,queryKeyNormWeightLayers:null,attentionOutputGate:true,outputGateType:'sigmoid',rmsNormWeightOffset:false,kernelPath};
            const recorder=recorded?new CommandRecorder(device,'paired_full_attention'):null;
            let output;
            try{
              const args=[input,weights,config,state,false,{debugLayers:[]},w=>w,(w,label)=>getNormWeightBuffer(w,label,{},{}),null,null];
              const result=recorder?await recordLayerAttentionGPU(recorder,...args):await runLayerAttentionGPU(...args);output=result.output;
              if(recorder)await recorder.submitAndWait();else await device.queue.onSubmittedWorkDone();
              if(output.precision!=='two-f32'||output.components!==2||output.elementStrideBytes!==8||result.residualFused)throw new Error('Layer lost explicit paired representation.');
              const actual=Array.from(new Float32Array(await readBufferSlice(output.buffer,0,tokens*H*8)));
              let maximum=0;for(let i=0;i<expected.length;i++){const e=Math.abs(actual[2*i]+actual[2*i+1]-expected[i]);maximum=Math.max(maximum,e);if(!Number.isFinite(e)||e>2e-10)throw new Error(JSON.stringify({recorded,start,index:i,actual:actual.slice(2*i,2*i+2),expected:expected[i],error:e,errors}));}
              const cached=cache.getGPUBuffers(0),kb=Array.from(new Uint16Array(await readBufferSlice(cached.keysGPU,0,(start+tokens)*D*2))),vb=Array.from(new Uint16Array(await readBufferSlice(cached.valuesGPU,0,(start+tokens)*D*2)));
              if(JSON.stringify(kb)!==JSON.stringify(expectedK.flat())||JSON.stringify(vb)!==JSON.stringify(expectedV.flat()))throw new Error(JSON.stringify({message:'Full-chain direct F16 cache differs from independent nearest-even reference',recorded,start,kFirst:kb.findIndex((v,i)=>v!==expectedK.flat()[i]),vFirst:vb.findIndex((v,i)=>v!==expectedV.flat()[i])}));
              rows.push({recorded,start,tokens,maximum,tolerance:2e-10,cacheExact:true,source:Array.from(source),expected,actual});
            }finally{if(recorder)await recorder.abort();if(output)releaseBuffer(output.buffer);}
          }
        }finally{await device.queue.onSubmittedWorkDone();cache.destroy();}
      }
      for(const b of pooled)releaseBuffer(b);pooled.clear();await getBufferPool().settleDeferredDestruction();
      const stats=getBufferPool().getStats();if(stats.activeBuffers||errors.length)throw new Error(JSON.stringify({stats,errors}));
      return {rows,errors,stats,geometry:{hidden:H,headDim:D,heads:HEADS,kvheads:KVHEADS,rotary:ROT,theta:THETA,epsilon:EPS},adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},settled:true};
    }finally{await device.queue.onSubmittedWorkDone();for(const b of pooled)releaseBuffer(b);for(const b of owned)b.destroy();destroyBufferPool();destroyDevice();}
  });
  receipt.browser = browser.version(); receipt.passed = receipt.result.rows.length === 4 && receipt.result.settled;
} catch (error) { receipt.error = error.stack; throw error; }
finally {
  await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n');
}
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired full attention interpreter failed.');
