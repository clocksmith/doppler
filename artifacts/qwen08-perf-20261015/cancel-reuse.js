import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { doppler } from '../../src/index.js';
import { getDevice } from '../../src/gpu/device.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
const [modelDir, inputPath, expectedPath, policyPath, outputPath] = process.argv.slice(2);
const policy=JSON.parse(await fs.readFile(policyPath));
const input=JSON.parse(await fs.readFile(inputPath));
const expected=JSON.parse(await fs.readFile(expectedPath)).runs[0].evidence.tokenIds.slice(0,16);
const result={complete:false,provider:process.env.DOPPLER_NODE_WEBGPU_MODULE ?? 'webgpu'};
let model;
try {
  model=await doppler.load({url:pathToFileURL(path.resolve(modelDir)+path.sep).href},{runtimeConfig:policy.engines.doppler.runtimeConfig});
  const options={inputIds:input.prompts[0].tokenIds,maxTokens:128,temperature:0,topK:1,topP:1,repetitionPenalty:1,useChatTemplate:false,suppressTokenIds:[]};
  const abort=new AbortController();let count=0;
  try {
    const cancelled=await model.generateWithEvidence('',{...options,signal:abort.signal,onToken(){if(++count===3)abort.abort();}});
    result.cancelled={tokenCount:cancelled.tokenIds.length,stopReason:cancelled.stats.stopReason};
    assert.equal(cancelled.stats.stopReason,'aborted');
  }catch(error){
    if(!abort.signal.aborted)throw error;
    if(error.code!=='aborted' && error.name!=='AbortError' && !/aborted/i.test(error.message))throw error;
    result.cancelled={errorName:error.name,errorCode:error.code,message:error.message};
  }
  assert.ok(count>=3&&count<128);
  model.resetGenerationState();
  const reused=await model.generateWithEvidence('',{...options,maxTokens:16});
  assert.deepEqual(reused.tokenIds,expected);
  result.reuseTokenIds=reused.tokenIds;result.complete=true;
}catch(error){result.error={message:error.message,stack:error.stack};process.exitCode=1;}
finally{
  await model?.unload();
  if(getDevice()){result.poolAfterUnload=getBufferPool().getStats();getDevice().destroy();}
  await fs.writeFile(outputPath,JSON.stringify(result,null,2)+'\n');
}
