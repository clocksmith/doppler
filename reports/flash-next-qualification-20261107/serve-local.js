import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { once } from 'node:events';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { load } from 'doppler-gpu/compat';
import { createServeHandler, ServeRequestError } from '../../src/cli/doppler-serve.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

// Local operator composition, not a catalogue promotion or Reploid provider.
// The user-owned process keeps one loaded model and one bounded active request.
const [file] = process.argv.slice(2);
const raw=await fs.readFile(file),run=JSON.parse(raw);
assert.equal(run.schema,'doppler.flash-next-local-serve/v1');
assert.equal(run.host,'127.0.0.1');
assert(Number.isSafeInteger(run.port) && run.port>0 && run.port<65536);
assert(Number.isSafeInteger(run.maxTokens) && run.maxTokens>0 && run.maxTokens<=256);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=await fs.readFile(path.join(run.modelDir,'manifest.json'));
assert.equal(hash(bytes),run.expectedManifestSha256);
const manifest=JSON.parse(bytes);
assert.equal(run.runtimeConfig.inference.chatTemplate.enabled,true);
installNodeFileFetchShim();
const model=await load({url:pathToFileURL(run.modelDir).href},{runtimeConfig:run.runtimeConfig,isolatedLoader:true});
let active=null,closing=false;
const entry={modelId:manifest.modelId,modes:['text'],manifestVariantId:`sha256:${hash(bytes)}`,
  artifactCompleteness:'complete',runtimePromotionState:'experimental',weightsRefAllowed:false};
const options=({model:ignored,...rest})=>{
  const maxTokens=rest.maxTokens ?? run.maxTokens;
  if(maxTokens>run.maxTokens) throw new ServeRequestError(`This local operator caps requests at ${run.maxTokens} output tokens.`);
  return {...rest,maxTokens,useChatTemplate:true,signal:active.controller.signal};
};
const handler=createServeHandler({listModels:async()=>[entry],
  resolveModel:async id=>{assert.equal(id,entry.modelId,'Unknown local model');return entry;},
  resolveRuntimeModel:()=>({url:pathToFileURL(run.modelDir).href}),
  dopplerClient:{chatText:(messages,opts)=>model.chatText(messages,options(opts)),
    chat:(messages,opts)=>model.chat(messages,options(opts))}});
const server=http.createServer(async(req,res)=>{
  if(req.url==='/health') {
    res.setHeader('Content-Type','application/json');
    return res.end(JSON.stringify({ready:!closing,model:entry.modelId,manifestSha256:hash(bytes),
      qualification:'Short chat smoke passed; long-context numerical gate remains open',active:active!==null}));
  }
  if(closing || active) {res.writeHead(503);return res.end('Local operator busy; one request at a time.');}
  let done;
  const slot={controller:new AbortController(),done:new Promise(resolve=>{done=resolve;})};
  active=slot;
  const disconnect=()=>slot.controller.abort(new Error('Local HTTP client disconnected'));
  req.once('aborted',disconnect);res.once('close',disconnect);
  try {model.resetGenerationState();await handler(req,res);}
  finally {req.off('aborted',disconnect);res.off('close',disconnect);active=null;done();}
});
const stop=async()=>{
  if(closing)return;
  closing=true;
  server.close();
  const pending=active;
  pending?.controller.abort(new Error('Local operator stopping'));
  await pending?.done;
  await model.unload();destroyDevice();await releaseNodeWebGPU();
};
for(const signal of ['SIGTERM','SIGINT'])process.once(signal,()=>stop().catch(error=>{console.error(error);process.exitCode=1;}));
try {
  server.listen(run.port,run.host);
  await once(server,'listening');
  console.log(JSON.stringify({ready:true,pid:process.pid,host:run.host,port:run.port,model:entry.modelId,
    configSha256:hash(raw),scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),
    qualification:entry.runtimePromotionState,fullModelQualified:false}));
} catch(error) {await stop();throw error;}
