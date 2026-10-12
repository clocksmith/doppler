// Contract doubles exercise both real factory owners; no WebGPU or qualification claim.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const manifestTarget = new URL('../../src/client/resident-partitions.js', import.meta.url).href;
const signedTarget = new URL('../../src/client/model-host/index.js', import.meta.url).href;
let finish;
globalThis.residentFactoryControl = {
  unloads: 0, storageCloses: 0, signal: null,
  manifest: { modelId: 'synthetic', architecture: {} },
  allocation: { model: { id: 'synthetic' }, plan: {}, index: 0 },
  handle: { advanced: {}, async unload() { residentFactoryControl.unloads++; } },
  source: { modelId: 'synthetic', manifest: { modelId: 'synthetic', architecture: {} } },
  open(_pipeline, _allocation, _tokens, _close, signal) {
    this.signal = signal;
    return new Promise(resolve => { finish = resolve; }).then(() => { signal.throwIfAborted(); return {}; });
  },
};
const shared = {
  pipeline: 'export class InferencePipeline {} export const createPipeline=async()=>new InferencePipeline();',
  handle: 'export const createModelHandle=()=>residentFactoryControl.handle; export const assertSupportedGenerationOptions=()=>{};',
  resident: 'export const createResidentPartitionSession=(...a)=>residentFactoryControl.open(...a);',
  allocation: 'export const resolveResidentPartitionAllocation=()=>residentFactoryControl.allocation;',
};
const manifestModules = {
  '../inference/pipelines/text.js': shared.pipeline,
  './model-host/model-session.js': shared.handle,
  '../inference/pipelines/text/resident-partition.js': shared.resident,
  '../inference/pipelines/text/resident-partition-contract.js': shared.allocation,
};
const signedModules = {
  '../../inference/pipelines/text.js': shared.pipeline,
  './model-session.js': shared.handle,
  '../../inference/pipelines/text/resident-partition.js': shared.resident,
  '../runtime/composition-root.js': `export const createDopplerRun=ports=>({openCapsule:async(capsule,options)=>ports.programFactory({capsule,targetPlan:{},artifactStore:{releaseBacking:async()=>{}},registries:null,options:{...options,residentPartition:residentFactoryControl.allocation}})});`,
  '../runtime/capsule-artifact-source.js': 'export const createCapsuleArtifactSource=async()=>residentFactoryControl.source;',
  '../runtime/capsule-program-adapter.js': 'export const createCapsuleProgramAdapter=()=>({});',
  '../../config/initial-execution-identity.js': 'export const resolveProgramLoadRuntimeConfig=()=>({});',
  '../../gpu/device.js': 'export const initDevice=async()=>({}); export const getKernelCapabilities=()=>({});',
  '../../loader/embedding-limit-preflight.js': 'export const resolveManifestGpuResidentEmbeddingLimitError=()=>null;',
  '../../loader/doppler-loader.js': 'export const createDopplerLoader=()=>({});',
  '../runtime/model-source.js': `export const resolveModelSource=async x=>x; export const sha256ManifestText=async()=> 'a'.repeat(64); export const resolveManifestArtifactSource=async x=>x; export const resolveLoadProgressHandlers=()=>({}); export const createDefaultNodeLoadProgressLogger=()=>{}; export const fetchManifestPayloadFromBaseUrl=()=>{throw Error('unexpected fetch')};`,
  '../runtime/node-source-cache.js': '',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  const modules = context.parentURL === manifestTarget ? manifestModules : context.parentURL === signedTarget ? signedModules : null;
  if (modules && Object.hasOwn(modules, specifier) && modules[specifier]) return {shortCircuit: true, url:`data:text/javascript,${encodeURIComponent(modules[specifier])}`};
  return next(specifier, context);
} });
async function flush() { for (let i=0;i<100;i++) await new Promise(resolve=>setImmediate(resolve)); }
try {
  const { createManifestResidentPartitionFactory } = await import(manifestTarget);
  const { createDopplerRuntimeService } = await import(signedTarget);
  for (const kind of ['manifest', 'signed']) {
    const controller = new AbortController(); const before = residentFactoryControl.unloads;
    residentFactoryControl.signal = null;
    const factory = kind === 'manifest' ? createManifestResidentPartitionFactory({
      manifest: residentFactoryControl.manifest, manifestIdentity:`sha256:${'a'.repeat(64)}`, runtimeConfig:{},
      createStorage:async()=>({close:async()=>{residentFactoryControl.storageCloses++;}}),
    }) : createDopplerRuntimeService({ ensureWebGPUAvailable:async()=>{}, resolveCapsuleInput:async()=>({capsule:{},artifactStore:{}}) });
    const opening = kind === 'manifest'
      ? factory.openResidentPartition({signal:controller.signal,model:{generation:{}}})
      : factory.openCapsule('synthetic',{signal:controller.signal});
    let settled = false; opening.then(()=>{settled=true;},()=>{settled=true;}); await flush();
    assert.equal(settled,false,`${kind} factory must wait for resident preparation`);
    assert(residentFactoryControl.signal,`${kind} factory must forward opening cancellation`);
    controller.abort(Error(`${kind} cancelled`)); await flush();
    assert.equal(residentFactoryControl.unloads,before,'no disposal while provider preparation is pending');
    finish(); await assert.rejects(opening,new RegExp(`${kind} cancelled`));
    assert.equal(residentFactoryControl.unloads,before+1,`${kind} failed preparation must unload its own program`);
  }
  assert.equal(residentFactoryControl.storageCloses,1,'manifest owner closes its storage after abort');
} finally { hooks.deregister(); delete globalThis.residentFactoryControl; }
console.log('manifest and signed factories: aborted preparation settles before owned unload');
