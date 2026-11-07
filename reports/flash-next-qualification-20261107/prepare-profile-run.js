import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { applyRuntimeProfile } from '../../src/tooling/browser-harness/runtime-config.js';

// Exercise the named profile through its actual resolver, without the diagnostic
// HC/router graph. The artifact and its declared execution graph stay immutable.
const [parent, output, observation] = process.argv.slice(2);
const raw=await fs.readFile(parent);
const run=JSON.parse(raw);
installNodeFileFetchShim();
const id='profiles/flash-next-routed-f32kv-qualification';
run.runtimeConfig=await applyRuntimeProfile(id);
assert(run.runtimeConfig.inference.kernelPath == null, 'The profile must not supply an inline execution override');
assert.equal(run.runtimeConfig.inference.session.kvcache.kvDtype,'f32');
run.runtimeProfile=id;
run.candidate={scope:'Named profile and manifest program; no inline diagnostic graph',
  parentRunSha256:createHash('sha256').update(raw).digest('hex'),normalModelQualified:false};
run.sequences=[{id:'short',maxTokens:3},{id:'long',maxTokens:1}];
run.outputPath=observation;
run.captureRouterLayers=[];
run.lifecycle=true;
await fs.writeFile(output,JSON.stringify(run,null,2)+'\n');
console.log(JSON.stringify({output,profile:id,inlineKernelPath:false}));
