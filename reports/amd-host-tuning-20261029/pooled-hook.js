// Executed inside the upstream benchmark page after its unchanged engine setup.
const contract = __COMPARISON_CONTRACT__;
const loadToHookMs=performance.now()-t0;
const results = [];
const vendor = String(ad.info?.vendor || '').toLowerCase();
if (vendor !== contract.requiredVendor) throw new Error(`Expected physical ${contract.requiredVendor} adapter, got ${vendor}`);
for (let run = 0; run < contract.warmupRuns + contract.timedRunsPerSession; run++) {
  eng.reset();
  if (eng.mtp) eng.mtpFill = false;
  await device.queue.onSubmittedWorkDone();
  const start = performance.now();
  const ids = chat(contract.prompt);
  await eng.prefillTokens(ids.slice(0, -1));
  let next = pick(await fwd(ids.at(-1)));
  const firstTokenAt = performance.now();
  const generated = [next], tokenMs = [];
  for (let j = 1; j < contract.maxNewTokens; j++) {
    const step = performance.now();
    next = pick(await fwd(next));
    tokenMs.push(performance.now() - step);
    generated.push(next);
  }
  const end = performance.now();
  results.push({warmup:run < contract.warmupRuns,promptIds:ids,tokenIds:generated,text:tok.decode(generated),firstTokenMs:firstTokenAt-start,decodeMs:end-firstTokenAt,totalRunMs:end-start,decodeTokens:generated.length-1,decodeTokensPerSec:(generated.length-1)*1000/(end-firstTokenAt),tokenMs});
}
window.RESULT = {loadToHookMs,engine:'pooled',adapterInfo:{vendor:ad.info?.vendor,architecture:ad.info?.architecture,device:ad.info?.device,description:ad.info?.description},userAgent:navigator.userAgent,gpuErrors,settings:{gpuSample:GS,mtp:!!eng.mtp,kv:eng.kvQ8?'q8':'f16',embeddingGgmlType:G.tensors['token_embd.weight']?.ggmlType,weightEntries:weights?Object.fromEntries(['embed','head'].map(k=>[k,weights[k]?.kind])):null},results};
throw 'done';
