/** Actual WebGPU checks. No CPU emulation is used to satisfy a GPU assertion. */
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { referencePixels } from './reference.mjs';
const ROOT=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const read=p=>fs.readFileSync(path.join(ROOT,p),'utf8');
const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
const helpers=new Function(read('lib/webgpu.js')+'\nreturn {withDevice,dispatchText};')();
const result={schema:'five-quines-gpu-verification/v1',node:process.version,checks:[],passed:false,status:'not-started'};
let ownedGPU;
try {
 if(!globalThis.navigator?.gpu) {
  const {create,globals}=await import('webgpu');
  Object.assign(globalThis,globals);ownedGPU=create([]);
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:ownedGPU}});
 }
 result.status='running';
 globalThis.__QUINE_LIBRARY__=true;
 const load=source=>{vm.runInThisContext(source);return globalThis.Quine;};
 const rounds=2;
 async function note(name,work) {
  const began=performance.now();const data=await work();
  result.checks.push({name,passed:true,elapsedMs:Math.round(performance.now()-began),...data});
  console.log('PASS',name);
 }
 for(const name of ['01-application','02-polyglot','03-multiquine','04-visual','05-compiler']) {
  await note(name+' actual WGSL rounds',async()=>{
   const original=read('quines/'+name+'.js');let current=original,identity=null;
   for(let n=0;n<rounds;n++) {
    const api=load(current);assert.equal(api.self(),current);
    const shader=api.wgsl();let out;
    if(name==='02-polyglot') {
     assert.equal(shader,current);
     out=await helpers.withDevice(device=>helpers.dispatchText(device,current));
    } else if(name==='03-multiquine') {
     const self=await helpers.withDevice(device=>helpers.dispatchText(device,shader));
     assert.equal(self.text,shader);
     out=await helpers.withDevice(device=>helpers.dispatchText(device,self.text,'javascript'));
    } else if(name==='05-compiler') {
     const self=await helpers.withDevice(device=>helpers.dispatchText(device,shader,'main',{TARGET:1,EXTERNAL:0},[]));
     assert.equal(self.text,shader);
     out=await helpers.withDevice(device=>helpers.dispatchText(device,self.text,'main',{TARGET:0,EXTERNAL:0},[]));
    } else out=await api.run();
    assert.equal(out.text,current);identity=out.adapter;
    if(name==='04-visual') {
     assert.equal(api.decodeImage(out.rgba,out.width,out.height),current);
     const reference=referencePixels(current,shader);
     assert.deepEqual(out.rgba,reference.rgba);
    }
    current=out.text; // The actual returned program becomes the next program.
   }
   return {rounds,adapter:identity,sourceSha256:digest(current)};
  });
 }
 const compiler=load(read('quines/05-compiler.js'));
 for(const {name,expected} of JSON.parse(read('compiler/examples.json'))) {
  await note('GPU compiler builds and executes '+name,async()=>{
   const program=JSON.parse(read('compiler/'+name+'.json'));
   const js=await compiler.run('js',program);assert.equal(js.text,compiler.compile(program,'js'));
   assert.equal(load(js.text).execute(),expected);
   const wg=await compiler.run('wgsl',program);assert.equal(wg.text,compiler.compile(program,'wgsl'));
   const execution=await helpers.withDevice(device=>helpers.dispatchText(device,wg.text,'main',{TARGET:1,EXTERNAL:0},[]));
   assert.equal(execution.text,expected);
   return {adapter:execution.adapter,output:execution.text};
  });
 }
 result.passed=true;result.status='passed';
} catch(error) {
 result.status=result.checks.length===0 && /Cannot find package 'webgpu'/.test(error.message)?'blocked':'failed';
 result.error={name:error.name,message:error.message};process.exitCode=1;
 console.error('GPU checks did not pass:',error.message);
} finally {
 fs.writeFileSync(path.join(ROOT,'verification/gpu-results.json'),JSON.stringify(result,null,2)+'\n');
 // Let the Node Dawn provider be released after all devices have been destroyed.
 if(ownedGPU){delete globalThis.navigator;ownedGPU=null;}
}
