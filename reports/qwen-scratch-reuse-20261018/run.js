import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat, writeFile, mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
const [rootArg, modelArg, configPath, output, mode = 'lifecycle'] = process.argv.slice(2);
const root = resolve(rootArg), model = resolve(modelArg);
const config = JSON.parse(await readFile(configPath, 'utf8')).request.runtimeConfig;
const inventory = new Map(JSON.parse(await readFile(resolve(root, 'archive-files.json'), 'utf8')).map(row=>[row.path,row.sha256]));
const receipt = { archiveSha256: JSON.parse(await readFile(resolve(root, 'preparation.json'), 'utf8')).archiveSha256,
  scope: mode, bindGroupCaching: false,
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
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface'])];
  receipt.launchArgs=args;
  browser=await chromium.launch({channel:'chrome',headless:true,args});
  receipt.browser=browser.version();
  cdp=await browser.newBrowserCDPSession();receipt.hardware=await cdp.send('SystemInfo.getInfo');
  let sampling=false;
  sampler=setInterval(async()=>{
    if(sampling)return;sampling=true;
    try {
      const {processInfo}=await cdp.send('SystemInfo.getProcessInfo');
      const pids=processInfo.map(p=>p.id);
      const raw=execFileSync('/bin/ps',['-o','pid=,rss=','-p',pids.join(',')],{encoding:'utf8'});
      const processes=raw.trim().split('\n').map(line=>{const [pid,rssKiB]=line.trim().split(/\s+/).map(Number);return {pid,rssBytes:rssKiB*1024};});
      receipt.memorySamples.push({time:Date.now(),processes,cpu:processInfo.map(p=>({id:p.id,type:p.type,cpuTime:p.cpuTime})),rssSumBytes:processes.reduce((n,p)=>n+p.rssBytes,0)});
    }catch(error){receipt.memorySamples.push({time:Date.now(),error:String(error)});}finally{sampling=false;}
  },500);
  const page=await browser.newPage();page.on('console',m=>receipt.console.push(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.exposeFunction('checkpoint',async data=>{receipt.checkpoints??=[];receipt.checkpoints.push(data);await writeFile(output+'.partial',JSON.stringify(receipt,null,2)+'\n');});
  receipt.result=await page.evaluate(async({runtimeConfig,mode})=>{
    globalThis.__DOPPLER_KERNEL_BASE_PATH__='/src/gpu/kernels/';
    const {load}=await import('/src/index.js');
    const {DopplerLoader}=await import('/src/loader/doppler-loader.js');
    const {getDevice,destroyDevice}=await import('/src/gpu/device.js');
    const {getBufferPool,BufferPool}=await import('/src/memory/buffer-pool.js');
    const attribution={acquire:{},create:{},fusedOutputs:{},scratchBuffers:0};
    const scratchBuffers=new Set();
    if(mode==='probe') {
      const acquire=BufferPool.prototype.acquire,create=GPUDevice.prototype.createBuffer,bind=GPUDevice.prototype.createBindGroup;
      const collect=(table,key,start)=>{const row=table[key]??={calls:0,ms:0};row.calls++;row.ms+=performance.now()-start;};
      BufferPool.prototype.acquire=function(size,usage,label){const start=performance.now();try{return acquire.call(this,size,usage,label);}finally{collect(attribution.acquire,label??'default',start);}};
      GPUDevice.prototype.createBuffer=function(descriptor){const start=performance.now();try{return create.call(this,descriptor);}finally{collect(attribution.create,descriptor.label??'default',start);}};
      GPUDevice.prototype.createBindGroup=function(descriptor){
        if(descriptor.label==='fused_ffn_bind_group'){
          const buffer=descriptor.entries.find(e=>e.binding===4).resource.buffer;
          const label=buffer.label;attribution.fusedOutputs[label]=(attribution.fusedOutputs[label]??0)+1;
          if(label==='decode_ffn_intermediate'){scratchBuffers.add(buffer);attribution.scratchBuffers=scratchBuffers.size;}
        }
        return bind.call(this,descriptor);
      };
    }
    const {resetLinearAttentionRuntime}=await import('/src/inference/pipelines/text/linear-attention.js');
    let loader,handle,secondHandle;
    const originalLoad=DopplerLoader.prototype.load;
    DopplerLoader.prototype.load=async function(...args){loader=this;return originalLoad.apply(this,args);};
    const check=(condition,message)=>{if(!condition)throw new Error(message);};
    const hash=async values=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',values.buffer)),b=>b.toString(16).padStart(2,'0')).join('');
    const options={useChatTemplate:false,temperature:0,topK:1,topP:1,maxTokens:32,seed:0,repetitionPenalty:1};
    const prompt='The capital of France is';
    const rows=[];
    const generate=async(text=prompt)=>{
      handle.resetGenerationState();const start=performance.now(),tokens=[],chunks=[],delivery=[];let first=null;
      for await(const chunk of handle.generate(text,{...options,onToken:id=>tokens.push(id)})){
        first??=performance.now()-start;delivery.push(performance.now()-start);chunks.push(chunk);
      }
      return {text:chunks.join(''),tokens,delivery,firstTokenMs:first,completeMs:performance.now()-start,stats:handle.advanced.getStats(),pool:getBufferPool().getStats()};
    };
    const poolStats=()=>getDevice()?getBufferPool().getStats():{deviceAbsent:true};
    const memory=()=>({pool:poolStats(),loader:loader?.getStats(),
      ownedWeightBytes:[...(loader?.gpuBuffers??[])].reduce((n,b)=>n+b.size,0),
      plan:loader?.weightRepresentationPlan??null});
    try {
      const start=performance.now();handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      rows.push({stage:'open',openingMs:performance.now()-start,memory:memory()});await checkpoint(rows.at(-1));
      rows.push({stage:'first',...await generate()});await checkpoint(rows.at(-1));
      for(let i=0;i<5;i++)rows.push({stage:'warm',...await generate()});
      handle.resetGenerationState();const first=await handle.advanced.prefillWithLogits(prompt,options);
      check(first.logits.length>0 && first.logits.every(Number.isFinite),'invalid public logits');
      rows.push({stage:'numerical',logitsSha256:await hash(first.logits),vocab:first.logits.length});
      first.cache.destroy();if(first.linearAttention)resetLinearAttentionRuntime(first.linearAttention);
      if(mode!=='lifecycle'){
        rows.push({stage:'memory',memory:memory(),attribution});
        await handle.unload();handle=null;rows.push({stage:'unloaded',pool:poolStats()});
        return {passed:true,rows};
      }

      rows.push({stage:'long-prefill',...await generate('A film catalog contains titles, durations, languages and release years. '.repeat(64))});
      const expected=rows.find(r=>r.stage==='first');check(expected.text.length>0,'empty answer');
      const abort=new AbortController();let chunks=0;handle.resetGenerationState();
      try{for await(const chunk of handle.generate(prompt,{...options,signal:abort.signal})){chunks++;abort.abort();}}
      catch(error){check(error.name==='AbortError',String(error));}
      check(chunks>0,'no cancellation output');const afterCancel=await generate();
      check(afterCancel.text===expected.text && JSON.stringify(afterCancel.tokens)===JSON.stringify(expected.tokens),'post-cancel output changed');
      rows.push({stage:'cancel',chunks,exactOutput:true,memory:memory()});await checkpoint(rows.at(-1));
      const primaryLoader=loader;
      const overlapGenerate=async(target,text,signal=null,onChunk=null)=>{
        target.resetGenerationState();const start=performance.now(),tokens=[],parts=[],delivery=[];
        for await(const chunk of target.generate(text,{...options,signal,onToken:id=>tokens.push(id)})){
          parts.push(chunk);delivery.push(performance.now()-start);onChunk?.();
        }
        return {text:parts.join(''),tokens,completeMs:performance.now()-start,delivery,stats:target.advanced.getStats()};
      };
      const otherPrompt='Two plus two equals';
      const overlapOpen=performance.now();secondHandle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      const overlapOpeningMs=performance.now()-overlapOpen;
      const secondExpected=await overlapGenerate(secondHandle,otherPrompt);
      const pair=await Promise.all([overlapGenerate(handle,prompt),overlapGenerate(secondHandle,otherPrompt)]);
      check(JSON.stringify(pair[0].tokens)===JSON.stringify(expected.tokens),'concurrent primary changed');
      check(JSON.stringify(pair[1].tokens)===JSON.stringify(secondExpected.tokens),'concurrent secondary changed');
      rows.push({stage:'concurrent',openingMs:overlapOpeningMs,pair,exactOutput:true,pool:poolStats(),
        ownedWeightBytes:[...primaryLoader.gpuBuffers,...loader.gpuBuffers].reduce((n,b)=>n+b.size,0)});await checkpoint(rows.at(-1));
      const isolatedAbort=new AbortController();let isolatedChunks=0;
      const settled=await Promise.allSettled([
        overlapGenerate(secondHandle,otherPrompt,isolatedAbort.signal,()=>{isolatedChunks++;isolatedAbort.abort();}),
        overlapGenerate(handle,prompt),
      ]);
      check(isolatedChunks>0,'cancelled overlap produced no output');
      check(settled[0].status==='fulfilled'||settled[0].reason?.name==='AbortError','unexpected overlap cancellation failure');
      check(settled[1].status==='fulfilled'&&JSON.stringify(settled[1].value.tokens)===JSON.stringify(expected.tokens),'cancellation affected other request');
      const reused=await overlapGenerate(secondHandle,otherPrompt);
      check(JSON.stringify(reused.tokens)===JSON.stringify(secondExpected.tokens),'cancelled session not reusable');
      rows.push({stage:'isolated-cancel',chunks:isolatedChunks,otherExact:true,reuseExact:true,pool:poolStats()});await checkpoint(rows.at(-1));
      let pendingUnload=null,unloadSettled=false;
      const unloadPair=await Promise.all([
        overlapGenerate(secondHandle,otherPrompt,null,()=>{
          if(!pendingUnload)pendingUnload=secondHandle.unload().then(()=>{unloadSettled=true;});
          check(!unloadSettled,'unload freed a session before its stream completed');
        }),
        overlapGenerate(handle,prompt),
      ]);
      await pendingUnload;
      check(JSON.stringify(unloadPair[0].tokens)===JSON.stringify(secondExpected.tokens),'in-flight unload changed output');
      check(JSON.stringify(unloadPair[1].tokens)===JSON.stringify(expected.tokens),'in-flight unload changed unrelated output');
      rows.push({stage:'in-flight-unload',exactOutput:true,unrelatedExact:true,settled:unloadSettled});
      secondHandle=null;loader=primaryLoader;
      const afterOtherUnload=await generate();check(JSON.stringify(afterOtherUnload.tokens)===JSON.stringify(expected.tokens),'other unload affected primary');
      rows.push({stage:'other-unloaded',exactOutput:true,memory:memory()});await checkpoint(rows.at(-1));
      const old=getDevice(),queue=old.queue,submit=queue.submit;let submissions=0;
      queue.submit=function(commands){submit.call(this,commands);if(++submissions===1)queueMicrotask(()=>old.destroy());};
      let failure;
      try{await handle.advanced.prefillWithLogits(prompt,options);}catch(error){failure=String(error);}finally{queue.submit=submit;}
      check(submissions>0 && failure,'submitted device destruction must reject');await old.lost;
      await handle.unload();handle=null;destroyDevice();
      rows.push({stage:'lost-cleanup',failure,submissions,pool:poolStats()});await checkpoint(rows.at(-1));
      const reopen=performance.now();handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      const reopenMs=performance.now()-reopen;check(getDevice()!==old,'device not recreated');
      const recovered=await generate();check(recovered.text===expected.text && JSON.stringify(recovered.tokens)===JSON.stringify(expected.tokens),'recreated device output changed');
      rows.push({stage:'recreated',openingMs:reopenMs,exactOutput:true,memory:memory()});
      await handle.unload();handle=null;rows.push({stage:'unloaded',pool:poolStats()});
      return {passed:true,rows};
    }catch(error){await checkpoint({stage:'failure',error:String(error),memory:memory()});throw error;}
    finally{DopplerLoader.prototype.load=originalLoad;await secondHandle?.unload();await handle?.unload();destroyDevice();}
  },{runtimeConfig:config,mode});
  receipt.passed=receipt.result.passed;
}catch(error){receipt.failure=error.stack;process.exitCode=1;}
finally{
  clearInterval(sampler);await cdp?.detach().catch(()=>{});await browser?.close();await new Promise(r=>server.close(r));
  receipt.peakRssSumBytes=Math.max(0,...receipt.memorySamples.map(s=>s.rssSumBytes??0));
  await mkdir(resolve(output,'..'),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');
}
console.log(JSON.stringify({passed:receipt.passed,failure:receipt.failure,peakRssSumBytes:receipt.peakRssSumBytes}));
