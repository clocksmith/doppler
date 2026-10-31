import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const [root, output] = process.argv.slice(2);
const sha = x => createHash('sha256').update(x).digest('hex');
const reservation = JSON.parse(readFileSync('reservation.json'));
const receipt = { schema: 'doppler.paired-sigmoid-factoring-diagnostic/v1', pid: process.pid, startedAt: new Date().toISOString(), reservation, events: [], allocations: 0, dispatches: 0 };
const persist = () => writeFileSync(output, JSON.stringify(receipt, null, 2));
const mark = (phase, data = {}) => { receipt.events.push({ phase, ...data, at: new Date().toISOString() }); persist(); };
const bounded = async (task, ms, label) => { let timer; try { return await Promise.race([task, new Promise((_, reject) => { timer = setTimeout(() => reject(Error(label + ' deadline')), ms); })]); } finally { clearTimeout(timer); } };
const load = file => import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src', file)));
for (const entry of JSON.parse(readFileSync(resolve(root, 'native-delivery.json'))).entries) {
 const bytes = readFileSync(resolve(root, entry.path)); assert.equal(bytes.length, entry.bytes); assert.equal(sha(bytes), entry.sha256);
}
assert.equal(sha(readFileSync(resolve(root, 'native-delivery.json'))), reservation.nativeDescriptorSha256);
assert.equal(sha(readFileSync(resolve(root, 'deploy/artifacts/doppler-gpu-0.6.28.tgz'))), reservation.runtimeArchiveSha256);
for (const [path, hash] of Object.entries(reservation.runtimeClosure)) assert.equal(sha(readFileSync(resolve(root, 'node_modules/doppler-gpu', path))), hash);
const { createNativeGpuHost } = await import(pathToFileURL(resolve(root, 'server/native-participant.js')));
const host = createNativeGpuHost(() => import(pathToFileURL(resolve(root, 'node_modules/webgpu/index.js'))));
const navigatorBefore = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
let device, lostObserved = false;
const owned = new Set();
try {
 await bounded((async () => {
  mark('host-open-start'); await host.open(); mark('host-open-complete');
  const adapter = await navigator.gpu.requestAdapter(); assert(adapter && !adapter.info.isFallbackAdapter);
  receipt.adapter = Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k => [k, adapter.info[k]]));
  receipt.node = process.version; receipt.features = [...adapter.features];
  receipt.mappedProviderFiles = [...new Set(readFileSync('/proc/self/maps','utf8').split('\n').map(line => line.split(/\s+/).at(-1)).filter(path => path?.startsWith('/') && /dawn|webgpu|vulkan|intel|LLVM|dri\//i.test(path)))].map(path => ({ path, sha256: sha(readFileSync(path)) }));
  device = await adapter.requestDevice();
  const { setDevice } = await load('gpu/device.js'); setDevice(device, { platformConfig: null });
  const pkg = JSON.parse(readFileSync(resolve(root,'node_modules/doppler-gpu/package.json')));
  const tooling = await import(pathToFileURL(resolve(root,'node_modules/doppler-gpu',pkg.exports['./tooling'].import)));
  assert.equal(tooling.getDevice(), device);
  device.lost.then(info => { lostObserved = true; receipt.deviceLost = { reason: info.reason, message: info.message }; mark('device-lost'); });
  receipt.errors = []; device.addEventListener('uncapturederror', event => { receipt.errors.push(event.error.message); persist(); });
  const createBuffer = device.createBuffer.bind(device), createEncoder = device.createCommandEncoder.bind(device);
  device.createBuffer = () => { receipt.allocations++; throw Error('Compile-only control must not create buffers'); };
  device.createCommandEncoder = () => { receipt.dispatches++; throw Error('Compile-only control must not encode work'); };
  const moduleMethod = device.createShaderModule.bind(device);
  device.createShaderModule = descriptor => {
   mark('shader-module-start',{label:descriptor.label,sha256:sha(descriptor.code)});
   try { const module = moduleMethod(descriptor); mark('shader-module-complete',{label:descriptor.label});
    const infoMethod = module.getCompilationInfo.bind(module);
    module.getCompilationInfo = (...args) => { mark('compilation-info-start',{label:descriptor.label}); return infoMethod(...args).then(result => { mark('compilation-info-complete',{label:descriptor.label,messages:result.messages.map(x=>({type:x.type,message:x.message}))}); return result; }, error => {mark('compilation-info-rejected',{message:error.message});throw error;}); }; return module;
   } catch(error) { mark('shader-module-rejected',{message:error.message}); throw error; }
  };
  const pipelineMethod = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = descriptor => { mark('pipeline-start',{label:descriptor.label,entryPoint:descriptor.compute.entryPoint,constants:descriptor.compute.constants}); return pipelineMethod(descriptor).then(result=>{mark('pipeline-complete',{label:descriptor.label});return result;},error=>{mark('pipeline-rejected',{label:descriptor.label,message:error.message});throw error;}); };
  const { getPipelineFast } = await load('gpu/kernels/pipeline-cache.js');
  for (const variant of reservation.variants) { mark('variant-start',{variant}); await getPipelineFast('paired_pointwise',variant,null,null,device); mark('variant-complete',{variant}); }
  assert.equal(receipt.allocations,0); assert.equal(receipt.dispatches,0);
  receipt.compileOnly = { allocations: 0, dispatches: 0 };
  device.createBuffer = descriptor => { receipt.allocations++; const buffer = createBuffer(descriptor); owned.add(buffer); return buffer; };
  device.createCommandEncoder = (...args) => { receipt.dispatches++; return createEncoder(...args); };
  const { createTensor } = await load('gpu/tensor.js');
  const { runSiLU, recordSiLU } = await load('gpu/kernels/silu.js');
  const { CommandRecorder } = await load('gpu/command-recorder.js');
  const { readBufferSlice, getBufferPool } = await load('memory/buffer-pool.js');
  const high = [-1000,-104,-40,-16,-1,-.125,-(2**-149),-0,0,2**-149,.125,1,16,40,104,1000];
  const input = new Float32Array(high.flatMap((value,i) => [value, Math.abs(value)<1e-20 ? 0 : (i%2 ? -1 : 1) * 2**-30]));
  const representation = {precision:'two-f32',components:2,elementStrideBytes:8};
  const make = () => device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const inputBuffer=make(); device.queue.writeBuffer(inputBuffer,0,input);
  const tensor=createTensor(inputBuffer,'f32',[1,high.length],'sigmoid_factoring_control',representation);
  receipt.inputWords=Array.from(new Uint32Array(input.buffer));receipt.tolerance=6e-12;receipt.rows=[];
  for(const recorded of [false,true]) {
    const recorder=recorded?new CommandRecorder(device,'sigmoid_factoring_control'):null;
    const outputBuffer=make();
    try {
      mark('operator-start',{recorded});
      const options={size:high.length,gate:null,inputActivation:'silu',gateActivation:'silu',swigluLimit:null,outputBuffer};
      const result=recorded?await recordSiLU(recorder,tensor,options):await runSiLU(tensor,options);
      if(recorder)await recorder.submitAndWait();
      const actual=new Float32Array(await readBufferSlice(result.buffer,0,input.byteLength));
      let maxError=0;
      for(let i=0;i<high.length;i++) {
        const x=input[i*2]+input[i*2+1];
        const sigmoid=x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
        const expected=x*sigmoid, error=Math.abs(actual[i*2]+actual[i*2+1]-expected)/Math.max(1,Math.abs(expected));
        assert(Number.isFinite(error)&&error<=receipt.tolerance,JSON.stringify({i,x,actual:[actual[i*2],actual[i*2+1]],expected,error}));
        maxError=Math.max(maxError,error);
      }
      assert.equal(result.precision,'two-f32');assert.equal(result.components,2);assert.equal(result.elementStrideBytes,8);
      receipt.rows.push({recorded,maxError,outputWords:Array.from(new Uint32Array(actual.buffer))});mark('operator-complete',{recorded,maxError});
    } finally { if(recorder)await recorder.abort(); }
  }
  await device.queue.onSubmittedWorkDone();
  receipt.poolStats=getBufferPool().getStats();assert.equal(receipt.poolStats.activeBuffers,0);
  for(const buffer of owned)buffer.destroy();owned.clear();
  assert.equal(receipt.errors.length,0);receipt.operationsPassed=true;
 })(),300000,'compile-only operation');
} catch(error) { receipt.error={name:error.name,message:error.message,stack:error.stack}; process.exitCode=1; }
finally {
 mark('host-close-start');
 try { await host.close(); assert(!device || lostObserved); assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis,'navigator'),navigatorBefore); receipt.hostClosed=true; mark('host-close-complete'); }
 catch(error) {receipt.cleanupError={name:error.name,message:error.message};receipt.ownership='unknown';process.exitCode=1;}
 receipt.passed=receipt.operationsPassed===true && receipt.hostClosed===true && !receipt.error && !receipt.cleanupError;
 receipt.finishedAt=new Date().toISOString();persist();
}
