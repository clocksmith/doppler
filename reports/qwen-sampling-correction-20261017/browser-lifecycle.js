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
const receipt = { archiveSha256: '75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2',
  scope: 'Physical browser public logits, cancellation, completed-work and post-submission device destruction and recreation; no spontaneous-loss claim',
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
  receipt.result=await page.evaluate(async(runtimeConfig)=>{
    globalThis.__DOPPLER_KERNEL_BASE_PATH__='/src/gpu/kernels/';
    const {load}=await import('/src/index.js');
    const {getDevice,destroyDevice}=await import('/src/gpu/device.js');
    const {resetLinearAttentionRuntime}=await import('/src/inference/pipelines/text/linear-attention.js');
    const check=(condition,message)=>{if(!condition)throw new Error(message);};
    const rows=[];let handle;
    const bounded=(promise)=>Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error('lifecycle operation timed out')),30000))]);
    const options={useChatTemplate:false,temperature:0,topK:1,topP:1,maxTokens:16,seed:0,repetitionPenalty:1};
    const prompt='The capital of France is';
    const generate=async()=>{handle.resetGenerationState();const chunks=[];for await(const chunk of handle.generate(prompt,options))chunks.push(chunk);return chunks.join('');};
    const inspect=values=>{check(values instanceof Float32Array && values.length>0,'missing CPU logits');let best=0;for(let i=0;i<values.length;i++){check(Number.isFinite(values[i]),'nonfinite logits');if(values[i]>values[best])best=i;}return best;};
    try {
      handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      const first=await handle.advanced.prefillWithLogits(prompt,options);
      const firstToken=inspect(first.logits);
      const second=await handle.advanced.decodeStepLogits([firstToken],options);
      const nextToken=inspect(second.logits);
      first.cache.destroy();if(first.linearAttention)resetLinearAttentionRuntime(first.linearAttention);
      rows.push({publicLogits:true,vocabSize:first.logits.length,firstToken,nextToken});
      const expected=await generate();check(expected.length>0,'empty answer');
      const abort=new AbortController();let chunks=0;
      handle.resetGenerationState();
      try {for await(const chunk of handle.generate(prompt,{...options,signal:abort.signal})){chunks++;abort.abort();}}
      catch(error){check(error.name==='AbortError',String(error));}
      check(chunks>0,'cancellation did not follow output');check(await generate()===expected,'post-cancel output changed');
      rows.push({cancelAfterChunks:chunks,nextRequestExactText:true});
      const oldDevice=getDevice();await oldDevice.queue.onSubmittedWorkDone();oldDevice.destroy();
      const loss=await bounded(oldDevice.lost);
      let rejected=false;try {await bounded(handle.advanced.prefillWithLogits(prompt,options));}catch(error){rejected=true;rows.push({lostRequestError:String(error)});}
      check(rejected,'stale session accepted execution after device destruction');
      await bounded(handle.unload());handle=null;destroyDevice();
      handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      check(getDevice()!==oldDevice,'device not replaced');check(await generate()===expected,'recreated device output changed');
      rows.push({deviceLostReason:loss.reason,explicitUnloadSettled:true,deviceRecreated:true,exactTextAfterRecreation:true});
      const inflightDevice=getDevice(), queue=inflightDevice.queue;
      const originalSubmit=queue.submit;let submissions=0;
      queue.submit=function(commands){
        originalSubmit.call(this,commands);
        submissions++;
        if(submissions===1)queueMicrotask(()=>inflightDevice.destroy());
      };
      let inflightError;
      try {await bounded(handle.advanced.prefillWithLogits(prompt,options));}
      catch(error){inflightError=String(error);}
      finally {queue.submit=originalSubmit;}
      check(submissions>0,'in-flight loss did not follow a real submission');
      check(inflightError && !inflightError.includes('timed out'),'in-flight operation did not reject promptly');
      const inflightLoss=await bounded(inflightDevice.lost);
      await bounded(handle.unload());handle=null;destroyDevice();
      handle=await load({url:new URL('/model/',location.href).href},{runtimeConfig});
      check(getDevice()!==inflightDevice,'in-flight loss did not replace device');
      check(await generate()===expected,'output changed after in-flight loss');
      rows.push({inflightSubmissions:submissions,inflightError,reason:inflightLoss.reason,
        unloadSettled:true,deviceRecreated:true,exactTextAfterInflightLoss:true});
      return {rows,passed:true};
    } finally {await handle?.unload();destroyDevice();}
  },config);
  receipt.passed=receipt.result.passed;
} catch(error) {receipt.failure=error.stack;process.exitCode=1;}
finally {await browser?.close();await new Promise(r=>server.close(r));await mkdir(resolve(output,'..'),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');}
console.log(JSON.stringify({passed:receipt.passed,failure:receipt.failure,result:receipt.result}));
