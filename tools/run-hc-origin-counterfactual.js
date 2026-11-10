// Test-process-only first-HC intervention; never production fallback or qualification.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { registerHooks } from 'node:module';
const sha=b=>createHash('sha256').update(b).digest('hex');
const [protocolPath]=process.argv.slice(2);
const protocolBytes=await fs.readFile(protocolPath),p=JSON.parse(protocolBytes);
assert(['baseline','first-down','first-hc','prefix-state'].includes(p.mode));
assert.equal(p.qualificationAuthority,false);
const root=p.runtimeRoot,output=path.resolve(p.outputDirectory);
const targetLayer=p.mode==='prefix-state'?p.targetLayer:0;assert(Number.isSafeInteger(targetLayer)&&targetLayer>=0&&targetLayer<48);
const captureBytes=await fs.readFile(p.capturePath),capture=JSON.parse(captureBytes);
assert.equal(sha(captureBytes),p.captureSha256);
for(const [file,hash] of Object.entries(capture.runtimeSourceSha256))assert.equal(sha(await fs.readFile(path.join(root,file))),hash,`runtime: ${file}`);
const fixtureBytes=await fs.readFile(path.join(p.fixtureDirectory,'fixture.json')),fixture=JSON.parse(fixtureBytes);
assert.equal(sha(fixtureBytes),p.fixtureSha256);
assert.equal(fixture.manifestSha256,capture.manifestSha256);
assert.equal(fixture.inputsSha256,capture.inputsSha256);
assert.equal(fixture.gpuObservationSha256,p.captureSha256);
const tensors={};
for(const [name,desc]of Object.entries(fixture.payloads)){
 const bytes=await fs.readFile(path.join(p.fixtureDirectory,desc.file));assert.equal(bytes.length,desc.bytes);assert.equal(sha(bytes),desc.sha256);
 const values=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));assert(values.every(Number.isFinite));tensors[name]=bytes;
}
const baseBytes=await fs.readFile(p.baseRunPath);assert.equal(sha(baseBytes),p.baseRunSha256);
const run=JSON.parse(baseBytes);
assert.equal(run.expectedManifestSha256,capture.manifestSha256);
assert.equal(sha(await fs.readFile(run.inputsPath)),capture.inputsSha256);
assert.equal(sha(await fs.readFile(path.join(run.modelDir,'manifest.json'))),capture.manifestSha256);
run.sequences=[{id:'long',maxTokens:1}];run.captureRouterLayers=Array.from({length:26},(_,i)=>i);
run.captureLogitRows=true;run.captureQsaLayers=p.captureQsaLayers??[];run.captureLayers=false;run.lifecycle=false;
run.outputPath=path.join(output,'observation.json');
assert.equal(run.referenceRouterControl,undefined);
await fs.mkdir(output);
await fs.writeFile(path.join(output,'run.json'),JSON.stringify(run,null,2)+'\n',{flag:'wx'});
const target=pathToFileURL(path.join(root,'src/inference/pipelines/text/hyper-connection/decoder-layer.js')).href;
const original=await fs.readFile(new URL(target),'utf8');assert.equal(sha(original),capture.instrumentation.originalSha256);
const key=Symbol.for('doppler.test.first-hc-origin-control');assert.equal(globalThis[key],undefined);
const control={producerSha256:sha(await fs.readFile(new URL(import.meta.url))),kind:'counterfactual-first-hc-origin',mode:p.mode,scope:'Diagnostic arithmetic substitution; never normal-model qualification',normalModelQualified:false,protocolSha256:sha(protocolBytes),fixtureSha256:sha(fixtureBytes),baselineSourceSha256:sha(original),replaced:[],observed:[],boundaryCaptures:[]};
let layer=-1;
globalThis[key]=async ({role,stage,tensor,numTokens,target:recorder})=>{
 if(role==='attn_hc'&&stage==='input')layer++;
 if(role==='attn_hc'&&layer===p.captureBoundaryLayer&&p.captureBoundaryStages.includes(stage)){
 assert.equal(recorder,null);const memory=await import(pathToFileURL(path.join(root,'src/memory/buffer-pool.js')).href);
 const size=tensor.shape.reduce((a,b)=>a*b,1)*4,bytes=Buffer.from(await memory.readBuffer(tensor.buffer,size));
 assert(new Float32Array(bytes.buffer,bytes.byteOffset,size/4).every(Number.isFinite));
 const file=path.join(output,`boundary-L${layer}.${stage}.f32`);await fs.writeFile(file,bytes,{flag:'wx'});
 control.boundaryCaptures.push({layer,role,stage,dtype:'f32',shape:tensor.shape,file,sha256:sha(bytes)});
 await fs.writeFile(path.join(output,'control.json'),JSON.stringify(control,null,2)+'\n');
 }
 if(role!=='attn_hc'||layer!==targetLayer)return;
 assert.equal(recorder,null,'counterfactual requires immediate execution');assert.equal(numTokens,2304);assert.equal(tensor.dtype,'f32');
 const replace=(p.mode==='first-down'&&stage==='down')||(p.mode==='first-hc'&&['branchInput','gate'].includes(stage))||(p.mode==='prefix-state'&&stage==='input');
 if(stage!=='down'&&!replace)return;
 const memory=await import(pathToFileURL(path.join(root,'src/memory/buffer-pool.js')).href);
 const {getDevice}=await import(pathToFileURL(path.join(root,'src/gpu/device.js')).href);
 const size=tensor.shape.reduce((a,b)=>a*b,1)*4;
 const prior=Buffer.from(await memory.readBuffer(tensor.buffer,size));assert(new Float32Array(prior.buffer,prior.byteOffset,size/4).every(Number.isFinite));
 control.observed.push({layer,stage,shape:tensor.shape,sha256:sha(prior)});
 if(replace){const value=tensors[`full-${stage}`];assert.equal(value.length,size);getDevice().queue.writeBuffer(tensor.buffer,0,value);
 const written=Buffer.from(await memory.readBuffer(tensor.buffer,size));assert.equal(sha(written),sha(value));
 control.replaced.push({layer,role,stage,shape:tensor.shape,originalSha256:sha(prior),fixtureSha256:sha(value),writeVerified:true});}
 await fs.writeFile(path.join(output,'control.json'),JSON.stringify(control,null,2)+'\n');
};
const anchors=[
 ['const ops = operators(target);','input','streams'],
 ['output = await branch(branchInput);','branchOutput','output'],
 ['const down = keep(await project(`${role}_down`, normed, weights.down, projectionRank, width));','down','down'],
 ['const branchInput = await ops.branch(mix, normed, geometry);','branchInput','branchInput'],
 ["const gate = await ops.activate(logits, { mode: 'inject_gate', rows: numTokens, columns: streamCount, streamCount });",'gate','gate']
];
const routeRecords=[];
const {installMoeRoutingObservation}=await import(pathToFileURL(path.join(root,'tests/helpers/moe-routing-observation-loader.js')).href);
const routeUrl=pathToFileURL(path.join(root,'src/inference/pipelines/text/moe-gpu.js')).href;
const routeHook=installMoeRoutingObservation(routeUrl,sha(await fs.readFile(new URL(routeUrl))),async record=>{
 if(record.layer>(p.mode==='prefix-state'?47:25))return;
 const memory=await import(pathToFileURL(path.join(root,'src/memory/buffer-pool.js')).href);
 const row={layer:record.layer,numTokens:record.numTokens,topK:record.topK};
 for(const [name,buffer,type]of [['indices',record.indicesBuffer,'u32'],['weights',record.weightsBuffer,'f32']]){
  const bytes=Buffer.from(await memory.readBuffer(buffer,record.numTokens*record.topK*4));
  const file=path.join(output,`actual-routes-L${record.layer}.${name}.${type}`);await fs.writeFile(file,bytes,{flag:'wx'});row[name]={file,bytes:bytes.length,sha256:sha(bytes)};
 }
 routeRecords.push(row);await fs.writeFile(path.join(output,'actual-routes.json'),JSON.stringify({scope:'Original computed GPU route selections, no route replacement',records:routeRecords},null,2)+'\n');
});
const hooks=registerHooks({load(url,ctx,next){const result=next(url,ctx);if(url!==target)return result;
 let source=typeof result.source==='string'?result.source:Buffer.from(result.source).toString('utf8');assert.equal(sha(source),sha(original));
 for(const [anchor,stage,tensor]of anchors){assert.equal(source.split(anchor).length-1,1);source=source.replace(anchor,`${anchor}\n    await globalThis[Symbol.for('doppler.test.first-hc-origin-control')]({role,stage:'${stage}',tensor:${tensor},numTokens,target});`);}
 control.instrumentedSourceSha256=sha(source);return {...result,source};}});
process.env.DOPPLER_FLASH_NEXT_PROBE=path.join(output,'run.json');
try{
 await import(pathToFileURL(path.join(root,'tests/integration/flash-next-learned-physical.test.js')).href);
 assert.equal(control.replaced.length,p.mode==='baseline'?0:p.mode==='first-hc'?2:1);
 control.completed=true;
}catch(error){control.failure={name:error.name,message:error.message};throw error;}
finally{
 hooks.deregister();routeHook.restore();delete globalThis[key];
 control.actualRoutingObservation=routeHook.receipt;control.actualRoutingLayers=routeRecords.map(row=>row.layer);
 await fs.writeFile(path.join(output,'control.json'),JSON.stringify(control,null,2)+'\n');
 try{const observed=JSON.parse(await fs.readFile(run.outputPath));observed.scope=p.mode==='baseline'?'Frozen-runtime baseline with test-only readback; no arithmetic replacement':'Counterfactual HC-boundary substitution; NOT normal learned-model qualification';observed.counterfactual=control;observed.normalModelQualified=false;await fs.writeFile(run.outputPath,JSON.stringify(observed,null,2)+'\n');}catch(error){console.error('control report binding failed:',error.message);process.exitCode=1;}
}
