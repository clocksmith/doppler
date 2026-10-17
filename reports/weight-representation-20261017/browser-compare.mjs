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
  scope: 'Physical browser public logits, cancellation, completed-work and post-submission device destruction and recreation; no spontaneous-loss claim',
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
      receipt.memorySamples.push({time:Date.now(),processes,rssSumBytes:processes.reduce((n,p)=>n+p.rssBytes,0)});
    }catch(error){receipt.memorySamples.push({time:Date.now(),error:String(error)});}finally{sampling=false;}
  },500);
  const page=await browser.newPage();page.on('console',m=>receipt.console.push(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.exposeFunction('checkpoint',async data=>{receipt.checkpoints??=[];receipt.checkpoints.push(data);await writeFile(output+'.partial',JSON.stringify(receipt,null,2)+'\n');});
  receipt.result=await page.evaluate(async(runtimeConfig)=>{
    globalThis.__DOPPLER_KERNEL_BASE_PATH__='/src/gpu/kernels/';
    const {load}=await import('/src/index.js');
    const {DopplerLoader}=await import('/src/loader/doppler-loader.js');
    const {getDevice,destroyDevice}=await import('/src/gpu/device.js');
    const {getBufferPool}=await import('/src/memory/buffer-pool.js');
    const {resetLinearAttentionRuntime}=await import('/src/inference/pipelines/text/linear-attention.js');
    let loader,handle;
    const originalLoad=DopplerLoader.prototype.load;
    DopplerLoader.prototype.load=async function(...args){loader=this;return originalLoad.apply(this,args);};
    const check=(condition,message)=>{if(!condition)throw new Error(message);};
    const hash=async values=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',values.buffer)),b=>b.toString(16).padStart(2,'0')).join('');
    const options={useChatTemplate:false,temperature:0,topK:1,topP:1,maxTokens:32,seed:0,repetitionPenalty:1};
    const prompt='The capital of France is';
    const rows=[];
    const generate=async(text=prompt)=>{
      handle.resetGenerationState();const start=performance.now(),tokens=[],chunks=[];let first=null;
      for await(const chunk of handle.generate(text,{...options,onToken:id=>tokens.push(id)})){
        first??=performance.now()-start;chunks.push(chunk);
      }
      return {text:chunks.join(''),tokens,firstTokenMs:first,completeMs:performance.now()-start,stats:handle.advanced.getStats()};
    };
    const poolStats=()=>getDevice()?getBufferPool().getStats():{deviceAbsent:true};
    const memory=()=>({pool:poolStats(),loader:loader?.getStats(),
      ownedWeightBytes:[...(loader?.gpuBuffers??[])].reduce((n,b)=>n+b.size,0),
      plan:loader?.weightRepresentationPlan??null});
    try {
      const start=performance.now();handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      rows.push({stage:'open',openingMs:performance.now()-start,memory:memory()});await checkpoint(rows.at(-1));
      rows.push({stage:'first',...await generate()});await checkpoint(rows.at(-1));
      for(let i=0;i<3;i++)rows.push({stage:'warm',...await generate()});
      handle.resetGenerationState();const first=await handle.advanced.prefillWithLogits(prompt,options);
      check(first.logits.length>0 && first.logits.every(Number.isFinite),'invalid public logits');
      rows.push({stage:'numerical',logitsSha256:await hash(first.logits),vocab:first.logits.length});
      first.cache.destroy();if(first.linearAttention)resetLinearAttentionRuntime(first.linearAttention);
      rows.push({stage:'long-prefill',...await generate('A film catalog contains titles, durations, languages and release years. '.repeat(64))});
      const expected=rows.find(r=>r.stage==='first');check(expected.text.length>0,'empty answer');
      const abort=new AbortController();let chunks=0;handle.resetGenerationState();
      try{for await(const chunk of handle.generate(prompt,{...options,signal:abort.signal})){chunks++;abort.abort();}}
      catch(error){check(error.name==='AbortError',String(error));}
      check(chunks>0,'no cancellation output');const afterCancel=await generate();
      check(afterCancel.text===expected.text && JSON.stringify(afterCancel.tokens)===JSON.stringify(expected.tokens),'post-cancel output changed');
      rows.push({stage:'cancel',chunks,exactOutput:true,memory:memory()});await checkpoint(rows.at(-1));
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
    finally{DopplerLoader.prototype.load=originalLoad;await handle?.unload();destroyDevice();}
  },config);
  receipt.passed=receipt.result.passed;
}catch(error){receipt.failure=error.stack;process.exitCode=1;}
finally{
  clearInterval(sampler);await cdp?.detach().catch(()=>{});await browser?.close();await new Promise(r=>server.close(r));
  receipt.peakRssSumBytes=Math.max(0,...receipt.memorySamples.map(s=>s.rssSumBytes??0));
  await mkdir(resolve(output,'..'),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');
}
console.log(JSON.stringify({passed:receipt.passed,failure:receipt.failure,peakRssSumBytes:receipt.peakRssSumBytes}));
