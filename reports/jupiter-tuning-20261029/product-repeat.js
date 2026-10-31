import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
const exec = promisify(execFile);
const root=path.dirname(fileURLToPath(import.meta.url));
const suite=JSON.parse(await fs.readFile(path.join(root,'product-repeat-contract.json')));
const pooledRoot='$HOME/deco/hatch/pooled-doppler-components-20261029/_work/pooled-runtime',dopplerRoot=path.join(root,'runtime-accepted');
const evidence=path.join(root,'product-repeat');await fs.mkdir(evidence);
const {chromium}=await import(pathToFileURL(path.join(pooledRoot,'node_modules/playwright/index.mjs')));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const shared=suite.shared;
const save=(name,value)=>fs.writeFile(path.join(evidence,name),typeof value==='string'?value:JSON.stringify(value,null,2)+'\n');
const rendered=prompt=>`<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
async function pooled(prompt,id){
 const port=8796, logs=[],errors=[];
 const server=spawn(process.execPath,[path.join(pooledRoot,'tests/bench/serve.mjs'),pooledRoot,String(port)],{cwd:pooledRoot,env:{...process.env,WEIGHT_CACHE:'$HOME/deco/hatch/pooled-doppler-components-20261029/_work/pooled-weight-cache'},stdio:['ignore','pipe','pipe']});
 server.stdout.on('data',b=>logs.push(b.toString()));server.stderr.on('data',b=>logs.push(b.toString()));
 let context;
 try{
  let ready=false;
  for(let i=0;i<100;i++){try{ready=(await fetch(`http://127.0.0.1:${port}/tests/bench/bench.html`)).ok;}catch{}if(ready)break;await new Promise(r=>setTimeout(r,100));}
  if(!ready)throw new Error('Pooled local benchmark server did not become ready');
  context=await chromium.launchPersistentContext(path.join(root,'warm-pooled'),{headless:true,executablePath:shared.browserExecutable,args:shared.browserArgs});
  const page=await context.newPage();page.on('console',m=>logs.push(`browser ${m.type()}: ${m.text()}`));page.on('pageerror',e=>errors.push(String(e)));
  const original=await fs.readFile(path.join(pooledRoot,'tests/bench/bench.html'),'utf8');
  const hook=(await fs.readFile(path.join(root,'pooled-hook.js'),'utf8')).replace('__COMPARISON_CONTRACT__',JSON.stringify({...shared,prompt}));
  if(original.split('  const out = [];').length!==2)throw new Error('Upstream benchmark hook boundary changed');
  let hookServed=false;
  await page.route(/\/tests\/bench\/bench\.html(?:\?|$)/,r=>{hookServed=true;logs.push('comparison-hook-served');return r.fulfill({status:200,contentType:'text/html',body:original.replace('  const out = [];',hook+'\n  const out = [];')});});
  const url=`http://127.0.0.1:${port}/tests/bench/bench.html?model=/models/compare-qwen35-08b/model.gguf&tokens=${shared.maxNewTokens}&opts=${encodeURIComponent(JSON.stringify({maxSeq:shared.maxSeqLen}))}&wcache=1`;
  await page.goto(url);if(!hookServed)throw new Error('Shared-contract page hook was not served');await page.waitForFunction(()=>window.RESULT!==undefined,null,{timeout:300000});
  const result=await page.evaluate(()=>window.RESULT);result.errors=errors;
  await save(`${id}.json`,result);if(result.error)throw new Error(result.error);if(result.engine!=='pooled'||!Array.isArray(result.results)||result.results.length!==shared.warmupRuns+shared.timedRunsPerSession)throw new Error('Pooled did not execute the shared comparison contract');if(errors.length||result.gpuErrors?.length)throw new Error('Pooled reported browser/GPU errors');
  return result;
 }finally{await context?.close();server.kill();await save(`${id}.txt`,logs.join('\n'));}
}
async function doppler(prompt,id){
 const config=JSON.parse(await fs.readFile(path.join(root,'product-repeat-config.json')));
 const runtime=config.request.runtimeConfig;
 runtime.shared.benchmark.run={...runtime.shared.benchmark.run,customPrompt:rendered(prompt),maxNewTokens:shared.maxNewTokens,warmupRuns:shared.warmupRuns,timedRuns:shared.timedRunsPerSession,seed:0,sampling:{...shared.sampling,seed:0,greedyThreshold:0}};
 if(process.argv.includes('--native')){delete runtime.inference.session.decodeLoop;runtime.inference.generation.disableMultiTokenDecode=false;}
 runtime.inference.prompt=rendered(prompt);runtime.inference.generation.maxTokens=shared.maxNewTokens;runtime.inference.sampling={...shared.sampling,seed:0,greedyThreshold:0};
 runtime.inference.kvcache={maxSeqLen:shared.maxSeqLen};runtime.inference.session.kvcache={maxSeqLen:shared.maxSeqLen};
 config.run.browser={...config.run.browser,executablePath:shared.browserExecutable,browserArgs:shared.browserArgs,userDataDir:path.join(root,'warm-candidate'),port:19932};
 await save(`${id}-config.json`,config);
 let stdout='',stderr='';
 try{({stdout,stderr}=await exec(process.execPath,[path.join(dopplerRoot,'src/cli/doppler-cli.js'),'bench','--config',path.join(evidence,`${id}-config.json`),'--json'],{cwd:dopplerRoot,timeout:360000,maxBuffer:32*1024*1024}));}
 catch(e){stdout=e.stdout||'';stderr=e.stderr||'';throw e;}
 finally{await save(`${id}.json`,stdout);await save(`${id}-stderr.txt`,stderr);}
 const result=JSON.parse(stdout);if(!result.ok)throw new Error(result.error?.message||'Doppler command failed');
 const vendor=String(result.result.deviceInfo?.adapterInfo?.vendor||'').toLowerCase();if(vendor!==shared.requiredVendor)throw new Error(`Expected physical ${shared.requiredVendor}, got ${vendor}`);
 return result;
}
const receipt={schema:suite.schema,scope:suite.scope,suiteSha256:sha(await fs.readFile(path.join(root,'product-repeat-contract.json'))),runnerSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),hookSha256:sha(await fs.readFile(path.join(root,'pooled-hook.js'))),startedAt:new Date().toISOString(),mode:process.argv.includes('--smoke')?'smoke':'full',dopplerCadence:'explicit-profile-sequential2x32',sessions:[]};
await save('paired-receipt.json',receipt);
try{
 for(let pair=0;pair<shared.pairsPerPrompt;pair++)for(const [promptIndex,prompt] of shared.prompts.entries()){
  const order=(pair+promptIndex)%2?['pooled','doppler']:['doppler','pooled'];
  for(const engine of order){const id=`${prompt.id}-pair${pair}-${engine}`;console.log(`START ${id}`);const begin=Date.now();const raw=await(engine==='pooled'?pooled:doppler)(prompt.text,id);receipt.sessions.push({id,engine,promptId:prompt.id,pair,order,elapsedMs:Date.now()-begin,rawFile:`${id}.json`});await save('paired-receipt.json',receipt);console.log(`DONE ${id}`);}
  const d=JSON.parse(await fs.readFile(path.join(evidence,`${prompt.id}-pair${pair}-doppler.json`))).result;
  const p=JSON.parse(await fs.readFile(path.join(evidence,`${prompt.id}-pair${pair}-pooled.json`)));
  const pr=p.results.find(r=>!r.warmup),dt=d.metrics.referenceTranscript;
  const checks={promptIdsMatch:JSON.stringify(pr.promptIds)===JSON.stringify(dt.prompt.ids),browserMatches:p.userAgent===d.env.browserUserAgent,pooledGenerated:pr.tokenIds.length,dopplerGenerated:dt.tokens.ids.length,outputIdsMatch:JSON.stringify(pr.tokenIds)===JSON.stringify(dt.tokens.ids)};
  receipt.pairChecks??=[];receipt.pairChecks.push({promptId:prompt.id,pair,...checks});await save('paired-receipt.json',receipt);
  if(!checks.promptIdsMatch||!checks.browserMatches||checks.pooledGenerated!==shared.maxNewTokens||checks.dopplerGenerated!==shared.maxNewTokens)throw new Error('Shared prompt/browser/work-count gate failed');
 }
 receipt.completedAt=new Date().toISOString();receipt.complete=true;
}catch(error){receipt.failure=String(error.stack||error);receipt.complete=false;process.exitCode=1;console.error(error);}
finally{await save('paired-receipt.json',receipt);}
