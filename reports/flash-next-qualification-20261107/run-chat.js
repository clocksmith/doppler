import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { load } from 'doppler-gpu/compat';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

// Explicit operator composition using the public API; no catalogue entry,
// participant service, precision default or acceptance policy is changed.
const [config] = process.argv.slice(2);
const bytes=await fs.readFile(config),run=JSON.parse(bytes);
assert.equal(run.schema,'doppler.flash-next-chat-probe/v1');
const hash=value=>createHash('sha256').update(value).digest('hex');
const manifestBytes=await fs.readFile(path.join(run.modelDir,'manifest.json'));
assert.equal(hash(manifestBytes),run.expectedManifestSha256);
assert(run.runtimeConfig.inference.chatTemplate.enabled);
assert(run.cases.length>0 && run.cases.length<=8);
assert(Number.isSafeInteger(run.maxTokens) && run.maxTokens>0 && run.maxTokens<=64);
installNodeFileFetchShim();
const report={scope:'Actual public-API short chat smoke, not long-context or full-model qualification',
  runSha256:hash(bytes),scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),
  manifestSha256:hash(manifestBytes),rows:[],gpuErrors:[],fullModelQualified:false};
let model;
try {
  model=await load({url:pathToFileURL(run.modelDir).href},{runtimeConfig:run.runtimeConfig,isolatedLoader:true});
  getDevice().addEventListener('uncapturederror',event=>report.gpuErrors.push(event.error.message));
  assert.equal(model.manifestHash,hash(manifestBytes));
  for (const entry of run.cases) {
    model.resetGenerationState();
    const started=performance.now();
    const result=await model.chatText(entry.messages,{maxTokens:run.maxTokens,temperature:0,topK:1,topP:1,
      seed:0,repetitionPenalty:1,useChatTemplate:true});
    const row={id:entry.id,content:result.content,usage:result.usage,elapsedMs:performance.now()-started,
      generationConfigHash:result.evidence.generationConfigHash,
      passed:new RegExp(entry.expectedPattern,'iu').test(result.content.trim())};
    report.rows.push(row);
    await fs.writeFile(run.outputPath,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(row));
    assert(row.passed,`Declared short-answer smoke failed: ${entry.id}`);
  }
  assert.deepEqual(report.gpuErrors,[]);
  report.completed=true;
} catch(error) {
  report.failure=String(error);
  throw error;
} finally {
  const failures=[];
  try {await model?.unload();} catch(error) {failures.push(String(error));}
  try {destroyDevice();} catch(error) {failures.push(String(error));}
  try {await releaseNodeWebGPU();} catch(error) {failures.push(String(error));}
  report.cleanupFailures=failures;
  await fs.writeFile(run.outputPath,JSON.stringify(report,null,2)+'\n');
  assert.deepEqual(failures,[]);
}
