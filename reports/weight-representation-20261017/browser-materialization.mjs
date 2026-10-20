import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat, writeFile, mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
const [rootArg, modelArg, configPath, output] = process.argv.slice(2);
const root = resolve(rootArg), model = resolve(modelArg);
const config = JSON.parse(await readFile(configPath, 'utf8')).request.runtimeConfig;
const inventory = new Map(JSON.parse(await readFile(resolve(root, 'archive-files.json'), 'utf8')).map(row=>[row.path,row.sha256]));
const receipt = { archiveSha256: JSON.parse(await readFile(resolve(root, 'preparation.json'), 'utf8')).archiveSha256,
  scope: 'Physical two-layer Q4 loading fixture; exact unit-valued numerical oracle, representation bytes and independent F16/F32 fallback copies; not model performance',
  modelManifestSha256: createHash('sha256').update(await readFile(resolve(model, 'manifest.json'))).digest('hex'),
  files: {}, console: [], passed: false, driverSha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex') };
let browser, cdp, sampler;
receipt.memorySamples = [];
receipt.configSha256 = createHash('sha256').update(await readFile(configPath)).digest('hex');
const server = createServer(async (req,res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/') { res.setHeader('Content-Type','text/html');res.end('<!doctype html>');return; }
    const base = pathname.startsWith('/model/') ? model : root;
    const relative = pathname.startsWith('/model/') ? pathname.slice(7) : pathname.slice(1);
    const file = resolve(base, relative); assert(file.startsWith(base + sep));
    const info = await stat(file);
    if (base === root && /\.(js|json|wgsl)$/.test(file)) {
      const hash=createHash('sha256').update(await readFile(file)).digest('hex');
      assert.equal(hash,inventory.get(relative),`Runtime file differs from frozen archive: ${relative}`);receipt.files[pathname]=hash;
    }
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    const range = req.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
    if (range) {
      const start = Number(range[1]), end = range[2] ? Number(range[2]) : info.size-1;
      res.writeHead(206, {'Content-Range':`bytes ${start}-${end}/${info.size}`, 'Content-Length':end-start+1});
      createReadStream(file,{start,end}).pipe(res);
    } else {res.setHeader('Content-Length',info.size);createReadStream(file).pipe(res);}
  } catch(error) {res.statusCode=404;res.end(error.message);}
});
try {
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const args=['--enable-unsafe-webgpu','--ignore-gpu-blocklist','--disable-gpu-sandbox',
    ...(process.platform==='darwin'?['--use-angle=metal']:['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface'])];
  browser=await chromium.launch({channel:'chrome',headless:true,args});receipt.browser=browser.version();
  const info=await browser.newBrowserCDPSession();receipt.hardware=await info.send('SystemInfo.getInfo');await info.detach();
  const page=await browser.newPage();page.on('console',m=>receipt.console.push(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  receipt.result=await page.evaluate(async()=>{
    globalThis.__DOPPLER_KERNEL_BASE_PATH__='/src/gpu/kernels/';
    const {initDevice,destroyDevice,getKernelCapabilities,getDevice}=await import('/src/gpu/device.js');
    const {loadTensorToGPU}=await import('/src/loader/tensors/tensor-loader.js');
    const {readBuffer,releaseBuffer,isBufferActive,acquireBuffer}=await import('/src/memory/buffer-pool.js');
    const {compileWeightRepresentationPlan:compile}=await import('/src/loader/weight-representation-plan.js');
    await initDevice();
    const check=(ok,message)=>{if(!ok)throw new Error(message);};
    const bytes=new Uint8Array(144);bytes[1]=0x3c;bytes.fill(1,4,16);bytes.fill(0x11,16);
    const cfg={useFusedQ4K:true,q4kLayout:'row',keepF32Weights:false,allowF32UpcastNonMatmul:false,
      gpuCapabilities:getKernelCapabilities(),q4kMaterializationMode:'mixed',q4kFusedRoles:['q_proj'],consumerRole:'q_proj'};
    const packed={op:'q_proj',kernel:'fused_matmul_q4.wgsl',entry:'main_multicol'};
    const f16={op:'q_proj',kernel:'matmul_f16w_f32a_tiled.wgsl',entry:'main'};
    const f32={op:'q_proj',kernel:'matmul_f32.wgsl',entry:'main'};
    const path={id:'fixture',prefill:{steps:[packed]},decode:{steps:[packed]},preLayer:[],postLayer:[],
      layerOverrides:[{layers:[1],prefill:{steps:[f16]}}]};
    const locations=new Map([0,1].map(i=>[`model.layers.${i}.self_attn.q_proj.weight`,
      {shardIndex:0,offset:i*144,size:144,shape:[1,256],dtype:'Q4_K_M',role:'matmul'}]));
    const manifest={architecture:{numLayers:2}};
    const plan=compile(locations,manifest,[path],cfg),rows=[];
    const verify=async result=>{
      for(const [dtype,material] of Object.entries(result.data.materializations)){
        const n=dtype==='q4k'?144:dtype==='f16'?512:1024;
        const buffer=await readBuffer(material.buffer,n);
        const values=dtype==='q4k'?new Uint8Array(buffer):dtype==='f16'?new Uint16Array(buffer):new Float32Array(buffer);
        check(values.every((value,i)=>value===(dtype==='q4k'?bytes[i]:dtype==='f16'?0x3c00:1)),`Incorrect ${dtype} materialization`);
      }
    };
    try{
      for(const mode of ['legacy-role','per-tensor']){
        let allocatedBytes=0;const tensors=[];
        for(const [name,location] of locations){
          const result=await loadTensorToGPU(bytes,location,name,{...cfg,
            ...(mode==='per-tensor'?{representationRequirement:plan[name]}:{})});
          await verify(result);allocatedBytes+=result.allocatedBuffers.reduce((n,b)=>n+b.size,0);
          tensors.push({name,dtypes:Object.keys(result.data.materializations)});
          for(const buffer of result.allocatedBuffers){releaseBuffer(buffer);check(!isBufferActive(buffer),'released weight remains active');}
        }
        rows.push({mode,allocatedBytes,tensors});
      }
      check(rows[0].allocatedBytes-rows[1].allocatedBytes===512,'Expected exactly one eliminated 256-element F16 shadow');
      const fallback={...path,id:'fallback',layerOverrides:[],prefill:{steps:[f32]},decode:{steps:[f32]}};
      const requirements=compile(locations,manifest,[path,fallback],cfg);
      const [name,location]=[...locations][1];
      const result=await loadTensorToGPU(bytes,location,name,{...cfg,representationRequirement:requirements[name]});
      await verify(result);check(Object.keys(result.data.materializations).sort().join(',')==='f16,f32,q4k','fallback representation omitted');
      rows.push({fallbackCopies:Object.keys(result.data.materializations),allocatedBytes:result.allocatedBuffers.reduce((n,b)=>n+b.size,0)});
      for(const buffer of result.allocatedBuffers)releaseBuffer(buffer);
      const embeddingLocation={...location,offset:0,role:'embedding'};
      const aliases=new Map([['embedding',embeddingLocation],['head',{...embeddingLocation,role:'lm_head'}]]);
      const aliasPath={...path,layerOverrides:[],postLayer:[{...packed,op:'lm_head'},{...f16,op:'lm_head_prefill'}]};
      const aliasPlan=compile(aliases,manifest,[aliasPath],cfg);
      const alias=await loadTensorToGPU(bytes,embeddingLocation,'embedding',{...cfg,representationRequirement:aliasPlan.embedding});
      await verify(alias);check(Object.keys(alias.data.materializations).sort().join(',')==='f16,f32,q4k','aliased endpoint representation omitted');
      rows.push({embeddingAliasCopies:Object.keys(alias.data.materializations),exactValues:true});
      for(const buffer of alias.allocatedBuffers)releaseBuffer(buffer);
      const floatLocation={...location,dtype:'F16',size:512};
      const floatName='model.layers.0.self_attn.q_proj.weight';
      const floatPlan=compile(new Map([[floatName,floatLocation]]),manifest,[path,fallback],cfg);
      const float=await loadTensorToGPU(new Uint8Array(new Uint16Array(256).fill(0x3c00).buffer),floatLocation,floatName,
        {...cfg,keepF32Weights:true,representationRequirement:floatPlan[floatName]});
      await verify(float);check(float.data.dtype==='f16','adaptive fallback primary must be F16');
      const {createTensor}=await import('/src/gpu/tensor.js');
      const {runMatmul}=await import('/src/gpu/kernels/matmul.js');
      const input=acquireBuffer(1024,undefined,'fixture_input');getDevice().queue.writeBuffer(input,0,new Float32Array(256).fill(1));
      const product=await runMatmul(createTensor(input,'f32',[1,256],'fixture_input'),float.data,1,1,256,
        {role:'q_proj',layerIdx:0,kernelPath:path,outputDtype:'f32',transposeB:true});
      const value=new Float32Array(await readBuffer(product.buffer,4))[0];check(value===256,'adaptive fallback matmul changed values');
      rows.push({adaptiveF16Fallback:true,matmulExpected:256,matmulObserved:value,retainedF32:true});
      releaseBuffer(input);releaseBuffer(product.buffer);for(const buffer of float.allocatedBuffers)releaseBuffer(buffer);
      return {passed:true,rows};
    }finally{destroyDevice();}
  });receipt.passed=receipt.result.passed;
}catch(error){receipt.failure=error.stack;process.exitCode=1;}
finally{await browser?.close();await new Promise(r=>server.close(r));await writeFile(output,JSON.stringify(receipt,null,2)+'\n');}
console.log(JSON.stringify({passed:receipt.passed,result:receipt.result,failure:receipt.failure}));
