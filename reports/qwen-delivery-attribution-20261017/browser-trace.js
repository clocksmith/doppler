import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat, writeFile, mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
const [rootArg, modelArg, configPath, output, intervalArg] = process.argv.slice(2);
const root = resolve(rootArg), model = resolve(modelArg);
const baselineBytes=await readFile(configPath);
const baseline=JSON.parse(baselineBytes);
const config=baseline.sections.warm.doppler.request.runtimeConfig;
const interval=intervalArg==null?null:Number(intervalArg);
if(interval!==null){assert(Number.isInteger(interval)&&interval>0);config.inference.session.decodeLoop.readbackInterval=interval;}
const receipt = { delta:interval==null?null:{'inference.session.decodeLoop.readbackInterval':interval}, archiveSha256: '75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2',
  scope: 'Controlled observation of frozen-runtime token delivery; separate from primary comparison timings',
  baselineSha256:createHash('sha256').update(baselineBytes).digest('hex'),
  hypothesis:'The generator waits for a 16-token GPU readback before yielding; queueing or DOM delivery is not the source of these groups.',
  modelManifestSha256: createHash('sha256').update(await readFile(resolve(model, 'manifest.json'))).digest('hex'),
  files: {}, console: [], passed: false, driverSha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex') };
let browser;
const server = createServer(async (req,res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/') { res.setHeader('Content-Type','text/html');res.end('<!doctype html>');return; }
    const base = pathname.startsWith('/model/') ? model : root;
    const relative = pathname.startsWith('/model/') ? pathname.slice(7) : pathname.slice(1);
    const file = resolve(base, relative); assert(file.startsWith(base + sep));
    const info = await stat(file);
    if (base === root && /\.(js|json|wgsl)$/.test(file)) receipt.files[pathname] = createHash('sha256').update(await readFile(file)).digest('hex');
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
  const cdp=await browser.newBrowserCDPSession();receipt.hardware=await cdp.send('SystemInfo.getInfo');await cdp.detach();
  const page=await browser.newPage();page.on('console',m=>receipt.console.push(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  receipt.result=await page.evaluate(async({runtimeConfig,prompt,reference})=>{
    globalThis.__DOPPLER_KERNEL_BASE_PATH__='/src/gpu/kernels/';
    const {load}=await import('/src/index.js');
    const {getDevice,destroyDevice}=await import('/src/gpu/device.js');
    const {PipelineGenerator}=await import('/src/inference/pipelines/text/generator.js');
    const options={...reference.generationConfig,maxTokens:128,useChatTemplate:false,temperature:0,topK:1,topP:1,repetitionPenalty:1};
    if(options.seed==null)delete options.seed;
    const node=document.createElement('pre');document.body.append(node);
    let handle,active=null;const restores=[];
    const span=(type,start,extra={})=>{if(active)active.events.push({type,start,end:performance.now(),...extra});};
    const wrap=(owner,key,asyncMethod,type,extra=()=>({}))=>{
      const original=owner[key];if(typeof original!=='function')throw new Error(`Missing ${key}`);
      owner[key]=asyncMethod?async function(...args){const start=performance.now();try{return await original.apply(this,args);}finally{span(type,start,extra.call(this,args));}}
        :function(...args){const start=performance.now();try{return original.apply(this,args);}finally{span(type,start,extra.call(this,args));}};
      restores.push(()=>{owner[key]=original;});
    };
    const install=()=>{
      wrap(getDevice().queue,'submit',false,'queue-submit',args=>({commandBuffers:args[0].length}));
      wrap(getDevice().queue,'onSubmittedWorkDone',true,'queue-done-wait');
      wrap(GPUBuffer.prototype,'mapAsync',true,'map-wait',function(){return {label:this.label,size:this.size};});
      wrap(GPUBuffer.prototype,'getMappedRange',false,'mapped-range',function(){return {label:this.label};});
      wrap(PipelineGenerator.prototype,'_generateNTokensGPU',true,'decode-group',args=>({tokensRequested:args[1]}));
      wrap(PipelineGenerator.prototype,'_sampleNextTokenFromLogits',false,'javascript-sampling');
    };
    const run=async(instrumented)=>{
      handle.resetGenerationState();node.textContent='';
      const row={instrumented,start:performance.now(),events:[],chunks:[],tokens:[],frames:[]};active=instrumented?row:null;
      let controller;const stream=new ReadableStream({start(c){controller=c;}});let pendingFrames=[];
      const producer=(async()=>{
        try {let index=0;for await(const text of handle.generate(prompt,{...options,onToken:id=>row.tokens.push(id)})){
          const yielded=performance.now();const item={index:index++,text,yielded};row.chunks.push(item);
          item.enqueued=performance.now();controller.enqueue(item);item.enqueueReturned=performance.now();
        }controller.close();}catch(error){controller.error(error);throw error;}
      })();
      const consumer=(async()=>{const reader=stream.getReader();try {while(true){const {done,value}=await reader.read();if(done)break;
        value.dequeued=performance.now();node.append(document.createTextNode(value.text));value.domUpdated=performance.now();
        pendingFrames.push(new Promise(resolve=>requestAnimationFrame(t=>{row.frames.push({index:value.index,callback:performance.now(),frameTimestamp:t});resolve();})));
      }}finally{reader.releaseLock();}})();
      await Promise.all([producer,consumer]);row.end=performance.now();active=null;
      await Promise.all(pendingFrames);row.text=node.textContent;row.stats=handle.advanced.getStats();
      if(JSON.stringify(row.tokens)!==JSON.stringify(reference.tokens.ids))throw new Error(`Token mismatch: observed ${row.tokens.length}; reference ${JSON.stringify(Object.keys(reference.tokens))}`);
      return row;
    };
    try {
      handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      const warmups=[];for(let i=0;i<2;i++){const r=await run(false);warmups.push({tokens:r.tokens.length,totalMs:r.end-r.start});}
      const baseline=[];for(let i=0;i<3;i++)baseline.push(await run(false));
      install();const traced=[];for(let i=0;i<3;i++)traced.push(await run(true));
      for(const row of traced)for(const type of ['queue-submit','map-wait','decode-group'])if(!row.events.some(e=>e.type===type))throw new Error(`Missing observation: ${type}`);
      return {passed:true,warmups,baseline,traced,consumer:'ReadableStream then DOM append; rAF callback is an opportunity to present, not proof of screen pixels or Reploid rendering'};
    }finally{active=null;for(const restore of restores.reverse())restore();await handle?.unload();destroyDevice();}
  },{runtimeConfig:config,prompt:baseline.promptContract.promptRendered,reference:baseline.sections.warm.doppler.result.metrics.referenceTranscript});
  receipt.passed=receipt.result.passed;
} catch(error) {receipt.failure=error.stack;process.exitCode=1;}
finally {await browser?.close();await new Promise(r=>server.close(r));await mkdir(resolve(output,'..'),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');}
console.log(JSON.stringify({passed:receipt.passed,failure:receipt.failure,runs:receipt.result?.traced?.length}));
