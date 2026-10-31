import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const [root, output] = process.argv.slice(2);
const sha = x => createHash('sha256').update(x).digest('hex');
const reservation = JSON.parse(readFileSync('reservation.json'));
const receipt = { schema: 'doppler.paired-compile-reachability/v1', pid: process.pid, startedAt: new Date().toISOString(), reservation, events: [], allocations: 0, dispatches: 0 };
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
  assert.equal(receipt.allocations,0); assert.equal(receipt.dispatches,0); assert.equal(receipt.errors.length,0); receipt.operationsPassed = true;
 })(),300000,'compile-only operation');
} catch(error) { receipt.error={name:error.name,message:error.message,stack:error.stack}; process.exitCode=1; }
finally {
 mark('host-close-start');
 try { await host.close(); assert(!device || lostObserved); assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis,'navigator'),navigatorBefore); receipt.hostClosed=true; mark('host-close-complete'); }
 catch(error) {receipt.cleanupError={name:error.name,message:error.message};receipt.ownership='unknown';process.exitCode=1;}
 receipt.passed=receipt.operationsPassed===true && receipt.hostClosed===true && !receipt.error && !receipt.cleanupError;
 receipt.finishedAt=new Date().toISOString();persist();
}
