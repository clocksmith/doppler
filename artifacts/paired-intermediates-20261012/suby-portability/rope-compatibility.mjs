// Isolated immutable-package RoPE preparation; not a model acceptance oracle.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {resolve,sep,extname} from 'node:path';
import {chromium} from 'playwright';
const [packageDirectory,destination]=process.argv.slice(2),root=resolve(packageDirectory);
const receipt={scope:'Exact paired RoPE table preparation with independent F64 sin/cos checks; not whole-model qualification',controllerSha256:createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'),events:[],served:{},geometry:{theta:1e7,ropeScale:1,rotaryDim:64,frequencyBaseDim:64,maxSeqLen:262144,positionCount:8192,scalingType:null,intermediatePrecision:'two-f32'}};
let browser,timer;
const server=createServer(async(req,res)=>{try{const path=new URL(req.url,'http://localhost').pathname;if(path==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html>');return;}const file=resolve(root,'.'+path);assert(file.startsWith(root+sep));const bytes=await readFile(file);receipt.served[path]=createHash('sha256').update(bytes).digest('hex');res.setHeader('Content-Type',extname(file)==='.js'?'text/javascript':extname(file)==='.json'?'application/json':extname(file)==='.wasm'?'application/wasm':'application/octet-stream');res.end(bytes);}catch{res.writeHead(404).end();}});
try{
 await new Promise(done=>server.listen(0,'127.0.0.1',done));
 browser=await chromium.launch({headless:true,dumpio:true,args:['--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan','--disable-gpu-sandbox']});receipt.browser=browser.version();
 const page=await browser.newPage();await page.exposeFunction('recordEvent',row=>receipt.events.push(row));await page.goto('http://127.0.0.1:'+server.address().port);
 receipt.result=await Promise.race([page.evaluate(async geometry=>{
  const runtime=await import('/src/partitions.js');const {getDevice,getKernelCapabilities,destroyDevice}=await import('/src/gpu/device.js');const {runRoPEPrecompute}=await import('/src/gpu/kernels/rope-precompute.js');const {readBufferSlice,releaseBuffer,getBufferPool,destroyBufferPool}=await import('/src/memory/buffer-pool.js');
  await runtime.configureDeviceMemoryBudget({maxBytes:1420000000});const device=getDevice();await recordEvent({phase:'device',adapter:getKernelCapabilities().adapterInfo});device.lost.then(info=>recordEvent({phase:'device-lost',reason:info.reason,message:info.message}));device.addEventListener('uncapturederror',e=>recordEvent({phase:'validation',message:e.error.message}));let table;
  try{await recordEvent({phase:'precompute-start'});table=await runRoPEPrecompute(geometry);await device.queue.onSubmittedWorkDone();await recordEvent({phase:'precompute-complete',memory:runtime.inspectDeviceMemory()});const count=geometry.positionCount*geometry.rotaryDim/2;let maximum=0,values=0;
   for(const [name,fn] of [['sin',Math.sin],['cos',Math.cos]]){const data=new Float32Array(await readBufferSlice(table[name],0,count*8));for(let p=0;p<geometry.positionCount;p++)for(let d=0;d<geometry.rotaryDim/2;d++){const i=p*geometry.rotaryDim/2+d,expected=fn(p*geometry.theta**(-2*d/geometry.frequencyBaseDim));const error=Math.abs(data[i*2]+data[i*2+1]-expected);if(!Number.isFinite(error))throw Error('Nonfinite RoPE output');maximum=Math.max(maximum,error);values++;}}
   if(maximum>2e-10)throw Error('RoPE F64 control exceeds retained2e-10 bound: '+maximum);return {maximum,values,tolerance:2e-10,passed:true};
  }catch(error){await recordEvent({phase:'primary-failure',message:error.message});throw error;}
  finally{for(const [phase,step] of [['queue',()=>device.queue.onSubmittedWorkDone()],['release',()=>{if(table){releaseBuffer(table.cos);releaseBuffer(table.sin);}}],['pool-stats',()=>recordEvent({phase:'pool-stats',stats:getBufferPool().getStats(),memory:runtime.inspectDeviceMemory()})],['pool-destroy',()=>destroyBufferPool()],['device-destroy',()=>destroyDevice()]]){try{await step();}catch(error){await recordEvent({phase,message:error.message,cleanupFailure:true});}}}
 },receipt.geometry),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Bounded180000ms RoPE preparation timeout')),180000);})]);receipt.passed=true;
}catch(error){receipt.passed=false;receipt.failure=error.stack;process.exitCode=1;}
finally{clearTimeout(timer);await browser?.close();server.closeAllConnections();await new Promise(done=>server.close(done));await writeFile(destination,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify({passed:receipt.passed,result:receipt.result,failure:receipt.failure}));}
