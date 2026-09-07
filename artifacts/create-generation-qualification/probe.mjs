import fs from 'node:fs';
import { openBrowserAudit } from '../../../simulatte/tools/simulatte/browser-session.mjs';
const out = new URL('./', import.meta.url);
const label = process.env.PROBE_LABEL || 'browser-probe';
const record = { startedAt: new Date().toISOString(), lane: 'browser-generation-diagnostic', qualified: false };
let browser;
try {
  browser = await openBrowserAudit({ publicRoot: process.cwd(), chromePath: '/snap/bin/chromium', webgpu: true, commandTimeoutMs: 60000 });
  record.launchArguments = browser.launchArguments;
  const { client } = browser;
  await client.send('Page.enable');
  await client.send('Runtime.enable');
  const loaded = client.waitForEvent('Page.loadEventFired');
  await client.send('Page.navigate', { url: new URL('artifacts/create-generation-qualification/probe.html', browser.host.baseUrl).href });
  await loaded;
  const evaluate = async expression => {
    const result = await client.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  record.device = await evaluate(`(async () => { const a = await navigator.gpu.requestAdapter(); if (!a) throw new Error('No WebGPU adapter'); return { userAgent: navigator.userAgent, info: {vendor:a.info.vendor, architecture:a.info.architecture, device:a.info.device, description:a.info.description}, features:[...a.features], maxBufferSize:a.limits.maxBufferSize }; })()`);
  console.log(JSON.stringify(record.device));
  await evaluate(`window.modelSourceUrl = ${JSON.stringify(process.env.PROBE_MODEL_URL || 'https://huggingface.co/clocksmith/rdrr/resolve/7c3d30e300bcb02cbd68fb0db3eee64fbf738f99/models/gemma-3-1b-it-q4k-ehf16-af32')}; window.probe = {status:'loading', events:[]}; void (async () => {
    try {
      const { doppler } = await import('/src/client/doppler-api.browser.js');
      const loadStarted = performance.now();
      window.probeModel = await doppler.load({url:new URL(window.modelSourceUrl,location.href).href}, {cache:'opfs', onProgress:e => {window.probe.progress = e;}});
      window.probe.loadMs = performance.now() - loadStarted;
      window.probe.status = 'generating';
      const { createJsonGrammarMask } = await import('/src/inference/pipelines/structured/json-grammar-mask.js');
      const options = {maxTokens:96,temperature:0};
      if (${process.env.PROBE_CONSTRAINT === '1'}) {
        const stopTokenIds = window.probeModel.advanced.getStopTokenIds();
        const maskSource = await (await fetch('/src/inference/pipelines/structured/json-grammar-mask.js')).text();
        const digest = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify({maskSource,stopTokenIds})));
        options.logitMaskIdentity = {id:'json-object-syntax-v1',contentDigest:'sha256:'+Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('')};
        options.logitMaskFn = createJsonGrammarMask({stopTokenIds, cacheBudget:262144});
        window.probe.constraint = 'json-object-syntax; application-schema-validation-separate';
      }
      window.probe.evidence = await window.probeModel.generateWithEvidence([{role:'user', content:'Return only JSON with one field named count whose value is the number of lanterns here: Three glass lanterns circle a wooden tree.'}], options);
      if (${process.env.PROBE_EXTENDED === '1'}) {
        const warmStarted = performance.now();
        const warm = await doppler.load({url:new URL(window.modelSourceUrl,location.href).href},{cache:'opfs'});
        window.probe.warmReuse = {sameHandle:warm === window.probeModel, durationMs:performance.now()-warmStarted};
        window.probe.repeat = await warm.generateWithEvidence([{role:'user',content:'Return only JSON with a count field. How many lanterns are there? Five paper lanterns circle a tree.'}],options);
        const abort = new AbortController();
        window.probe.cancelled = await warm.generateWithEvidence('Write a detailed story about space travel.',{maxTokens:96,temperature:0,signal:abort.signal,onToken:()=>abort.abort('qualification cancellation')});
        try {await warm.generateWithEvidence('Return an empty JSON object.',{...options,logitMaskFn:()=>{throw new Error('qualification malformed constraint');}});}
        catch(error) {window.probe.maskFailure={message:error.message};}
        window.probe.recovered = await warm.generateWithEvidence([{role:'user',content:'Return only JSON with a count field. How many lanterns are there? Two glass lanterns circle a tree.'}],options);
      }
      window.probe.status = 'finished';
    } catch(e) {window.probe.status = 'failed'; window.probe.error={message:e.message,stack:e.stack};}
  })(); true`);
  const deadline = Date.now() + 900000;
  while (Date.now() < deadline) {
    record.result = await evaluate('window.probe');
    fs.writeFileSync(new URL(label+'.json', out), JSON.stringify(record,null,2)+'\n');
    console.log(JSON.stringify({status:record.result.status,progress:record.result.progress?.phase,error:record.result.error?.message,output:record.result.evidence?.outputText}));
    if (['finished','failed'].includes(record.result.status)) break;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  if (!['finished','failed'].includes(record.result.status)) record.observationExpired = true;
  record.diagnostics = client.diagnostics();
} catch(error) {record.error={message:error.message,stack:error.stack,browserProcessLog:error.browserProcessLog}; console.error(error); process.exitCode=1;}
finally {
  record.finishedAt = new Date().toISOString();
  if(browser) record.browserLog = browser.processOutput.snapshot();
  fs.writeFileSync(new URL(label+'.json',out), JSON.stringify(record,null,2)+'\n');
  await browser?.close();
}
