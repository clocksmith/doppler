import { observeStage } from './ownership-observer.js';
// Bounded numerical observation using the unchanged frozen physical fixture owners.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const [root, output, stageText, endpoint] = process.argv.slice(2), index = Number(stageText);
assert(root && output && Number.isInteger(index));
const load = path => import(pathToFileURL(root + '/' + path));
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const { chromium } = await load('node_modules/playwright/index.mjs');
const { physicalWebGpuBrowserOptions } = await load('tests/fixtures/physical-webgpu-browser.js');
const { preparePartitionMemory } = await load('tests/fixtures/prepare-partition-memory.js');
const { observeResidentOpening } = await load('tests/fixtures/resident-opening-observer.js');
const { connectNativePartitionEndpoint } = await load('tests/fixtures/native-partition-endpoint.js');
const { createConcurrentOperationObserver } = await load('tests/fixtures/concurrent-partition-memory.js');
const { assertPartitionMemorySettlement } = await load('tests/fixtures/partition-memory-settlement.js');
const { verifyNumericalTestDelivery } = await load('tests/fixtures/numerical-test-delivery.js');
const { bindPairedPartitionFixture } = await load('tests/fixtures/paired-partition-binding.js');
const config = JSON.parse(await readFile(root + '/config-preparation.json'));
const read = async key => { const bytes = await readFile(config[key].path); assert.equal(sha(bytes), config[key].hash); return bytes; };
const [manifestBytes, graphBytes, archiveBytes, referenceBytes, pairedBytes, policyBytes] = await Promise.all(['manifest','graph','archive','reference','pairedBinding','policy'].map(read));
const reference = JSON.parse(referenceBytes), manifest = JSON.parse(manifestBytes);
const pairedBinding = await bindPairedPartitionFixture({ bindingBytes: pairedBytes, graphBytes, modelBytes: manifestBytes, archiveBytes, referenceIdentity: reference.modelIdentity });
const delivery = await verifyNumericalTestDelivery({ root, bindingPath: config.delivery.path, bindingSha256: config.delivery.hash.slice(7), archivePath: config.archive.path, runtimeDirectory: config.runtimeDirectory });
const runtime = await load('vendor/doppler/0.6.28/src/partitions.js');
const plan = runtime.createOrderedLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture, splitLayers: [8,16], activationPrecision: 'two-f32' });
runtime.validateOrderedLayerPartitionPlan(manifest, plan);
const evidence = { schema: 'doppler.ordered-output-stage-diagnostic/v1', index, scope: 'Installed output-stage opening, original long-prefill execution and canonical tensor capture', startedAt: new Date().toISOString(), delivery, plan, operations: {} };
let writes=Promise.resolve(); const persist=()=>{ const bytes=JSON.stringify(evidence); writes=writes.then(()=>writeFile(output,bytes)); return writes; };
const evaluate=createConcurrentOperationObserver({evidence:evidence.operations,persist});
let browser,context,page,observer;
try {
 if(index===0) {
  browser=await chromium.launch(physicalWebGpuBrowserOptions('darwin')); context=await browser.newContext(); page=await context.newPage();
  const cdp=await browser.newBrowserCDPSession();
  const command=await cdp.send('Browser.getBrowserCommandLine'); const executable=command.arguments[0]; const bytes=await readFile(executable);
  const processes=await cdp.send('SystemInfo.getProcessInfo'); await cdp.detach();
  evidence.executableIdentity={path:executable,sha256:createHash('sha256').update(bytes).digest('hex'),sizeBytes:bytes.length};
  evidence.pid=processes.processInfo.find(row=>row.type==='browser').id; evidence.browserVersion=browser.version();
  await page.goto('http://127.0.0.1:29850/config/partition-policy.json');
 } else {
  browser=await connectNativePartitionEndpoint(endpoint);context=await browser.newContext();page=await context.newPage();
  evidence.nativeIdentity=browser.identity;evidence.pid=browser.identity.pid;
  evidence.executableIdentity=await page.evaluate(()=>{const fs=process.getBuiltinModule('fs'),crypto=process.getBuiltinModule('crypto');const b=fs.readFileSync(process.execPath);return{path:process.execPath,sha256:crypto.createHash('sha256').update(b).digest('hex'),sizeBytes:b.length};});
  await browser.open();
 }
 evidence.endpoint=endpoint||'owned Chromium launch';await persist(); console.log(JSON.stringify({phase:'preparation',index,pid:evidence.pid}));
 const runtimeBaseUrl=index===0?'http://127.0.0.1:29850/vendor/doppler/0.6.28/src/':browser.identity.fixture.runtimeBaseUrl;
 observer=await observeResidentOpening(page,{runtimeBaseUrl,factorySource:await readFile(config.runtimeDirectory+'/src/client/resident-partitions.js','utf8'),identitySource:await readFile(config.runtimeDirectory+'/src/config/initial-execution-identity.js','utf8')});
 evidence.prepared=await evaluate(page,`load-only-stage-${index}`,preparePartitionMemory,{model:{...config.model,generation:reference.generation},index,participantId:`ordered-stage-${index}`,orderedPlan:plan,pairedBinding,modelSource:'http://127.0.0.1:29851/',policy:JSON.parse(policyBytes),config:{DOPPLER_PACKAGE_VERSION:'0.6.28',DOPPLER_PARTITIONS_MODULE_URL:new URL('partitions.js',runtimeBaseUrl).href,DOPPLER_KERNEL_BASE_URL:new URL('gpu/kernels/',runtimeBaseUrl).href}});
 evidence.openingObservation=await observer.finish();
 await observer.close(); observer=null;
 if(index!==0)evidence.nativeProvider=await page.evaluate(()=>__nativePartitionFixture.observeProvider());
 evidence.observedAt=new Date().toISOString();evidence.preparedSuccessfully=true;await persist();
 await observeStage({ page, runtimeBaseUrl, runtimeDirectory: config.runtimeDirectory, evaluate, evidence, persist, inputsPath: new URL('./isolated-stage-input.json', import.meta.url), execute: (await load('tests/fixtures/ordered-partition-reference.js')).executeOrderedPhysicalStage });
} catch(error){evidence.failure={name:error.name,message:error.message,stack:error.stack};process.exitCode=1;}
finally{
 try{await observer?.close();}catch(error){evidence.observerCloseError=error.message;}
 if(page)try{evidence.settlement=await evaluate(page,`close-load-only-stage-${index}`,async()=>{await globalThis.resident?.close();return globalThis.settleMemory?.();},undefined,true);if(evidence.settlement)assertPartitionMemorySettlement(evidence.settlement);}catch(error){evidence.cleanupFailure=error.message;evidence.ownership='unknown';process.exitCode=1;}
 try{if(context)await evaluate({evaluate:()=>context.close()},'close-context',()=>{},undefined,true,false);}catch(error){evidence.contextError=error.message;process.exitCode=1;}
 try{if(browser)evidence.providerSettlement=await evaluate({evaluate:()=>browser.close()},'close-provider',()=>{},undefined,true,false);}catch(error){evidence.providerError=error.message;process.exitCode=1;}
 evidence.finishedAt=new Date().toISOString();await persist();console.log(JSON.stringify({index,prepared:evidence.preparedSuccessfully,failure:evidence.failure,cleanupFailure:evidence.cleanupFailure,providerError:evidence.providerError}));
}
