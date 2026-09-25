import fs from 'node:fs/promises';
import path from 'node:path';
import { syncBuiltinESMExports } from 'node:module';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = '/home/x/deco/doppler';
const role = process.argv[2];
assert(['embedding', 'reranker'].includes(role));
const source = JSON.parse(await fs.readFile(`${root}/examples/document-search/capsules/${role}/artifacts/evidence/qualification.json`));
globalThis.__DOPPLER_KERNEL_BASE_PATH__ = new URL(`./${role}-shaders/`, import.meta.url).href;
const packageRoot = new URL('./node_modules/doppler-gpu/', import.meta.url);
const installed = rel => import(new URL(rel, packageRoot));
const requests = [];
const readFile = fs.readFile;
fs.readFile = async (name, ...args) => {
  if (String(name).endsWith('.wgsl')) requests.push(String(name).startsWith('file:') ? String(name) : pathToFileURL(name).href);
  return readFile(name, ...args);
};
syncBuiltinESMExports();
const { load } = await installed('src/client/doppler-api.js');
const { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } = await installed('src/tooling/node-webgpu.js');
const { destroyDevice } = await installed('src/gpu/device.js');
const { observeInitialExecutionIdentity } = await installed('src/config/initial-execution-identity.js');
const { evaluateEmbeddingReference } = await installed('src/config/embedding-reference.js');
const { evaluateRerankReference } = await installed('src/config/rerank-reference.js');
const { computeCanonicalSha256 } = await installed('src/formats/canonical-hash.js');
const report = { schema: source.schema, passed: false, generatedAt: new Date().toISOString(),
  config: {...source.config, mode:'model', repeatRuns:1, packageBundlePath: path.dirname(new URL(import.meta.url).pathname)},
  reference: source.reference, referenceDigest: source.referenceDigest, model:source.model,
  installedPackage: {sha256:createHash('sha256').update(await readFile(new URL('./vendor/doppler-gpu-0.6.2.tgz',import.meta.url))).digest('hex')},
  nodeVersion:process.version, providerVersion:'0.4.0', requests,
  runtime: { surface:'node-webgpu',host:'node',executionGraphHash:source.runtime.executionGraphHash },
  boundary:{signedCapsuleExecution:false,externalAdoption:false,sourceComparison:true},stage:'provider'};
let session;
try {
 const provider=await bootstrapNodeWebGPUProvider('webgpu',{createArgs:[['enable-dawn-features=allow_unsafe_apis']]});
 assert(provider.ok,provider.detail);
 const info=provider.session.adapter.info;
 report.provider=provider.receipt;
 report.runtime.adapterInfo=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,info[k]]));
 assert.equal(info.vendor.toLowerCase(),'amd');
 assert.equal(info.isFallbackAdapter,false);
 globalThis.fetch=async input=>{throw new Error('Network disabled in model probe: '+input);};
 report.stage='load'; console.log(role,report.stage);
 const began=performance.now();
 session=await load({url:pathToFileURL(source.config.modelDir+'/').href},{runtimeConfig:source.config.runtimeConfig});
 report.loadMs=performance.now()-began;
 report.initialExecutionIdentity=observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
 report.stage='execute'; console.log(role,report.stage);
 if(role==='embedding'){
  const outputs=[]; const times=[];
  for(let repeat=0;repeat<=1;repeat++) for(const [index,text] of source.reference.input.texts.entries()){
   const start=performance.now(); const result=await session.embedWithEvidence(text);
   const observed={text,tokenIds:result.tokens,embedding:Array.from(result.embedding)};
   if(repeat===0) outputs.push(observed); else assert.deepEqual(observed,outputs[index]);
   times.push({repeat,index,elapsedMs:performance.now()-start});
  }
  report.observation={input:source.reference.input,embeddingContract:source.reference.embeddingContract,outputs};
  report.result=evaluateEmbeddingReference(source.reference,report.observation); report.timings=times;
 }else{
  const executions=[];
  for(let repeat=0;repeat<=1;repeat++){
   const start=performance.now(); const result=await session.rerankWithEvidence(source.reference.input.query,source.reference.input.documents);
   executions.push({repeat,elapsedMs:performance.now()-start,evidence:result});
  }
  report.observation={input:source.reference.input,scoringConfig:session.manifest.inference.rerank,outputs:executions[0].evidence.scores};
  report.comparisons=executions.map(run=>evaluateRerankReference(source.reference,{...report.observation,outputs:run.evidence.scores}));
  report.result=report.comparisons[0];report.executions=executions;
  assert(report.comparisons.every(r=>r.passed));
 }
 assert(report.result.passed);report.passed=true;report.stage='complete';
}catch(error){report.error={message:error.message,stack:error.stack};}
finally{
 const errors=[];for(const close of [()=>session?.unload(),()=>destroyDevice(),()=>releaseNodeWebGPU()])try{await close();}catch(error){errors.push(error.message);}
 report.cleanup={passed:errors.length===0,errors};report.passed&&=report.cleanup.passed;
 fs.readFile=readFile;
 await fs.writeFile(new URL(`./${role}-model-qualification.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify({passed:report.passed,stage:report.stage,error:report.error,shaderReads:requests.length}));
if(!report.passed)process.exitCode=1;
