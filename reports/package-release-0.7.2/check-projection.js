// Real-hardware check of layer projection across prompts and models: for every
// answer token, the last layer's projection must be the token the run recorded,
// at the probability it recorded. Also measures what a projection costs.
//
//   DOPPLER_PREVIEW_ROOT=<doppler with projectLayerOutputs> node reports/package-release-0.7.2/check-projection.js [--models a,b] [--out file]
//
// The projection is read at the position where each token was chosen. Sampling
// is deterministic with no penalties, so the recorded probability is the plain
// softmax the projection also computes.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from '../../../hatch/doppler-observatory/scripts/dev.js';

const root = path.resolve('../hatch/doppler-observatory');
const arg = (name) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : null);
const models = (arg('--models') ?? 'gemma-3-270m-it-q4k-ehf16-af32').split(',');
const PROMPTS = [
  'Explain in one short paragraph why the sky is blue.',
  'List three prime numbers greater than ten.',
  'Write one sentence about a cat who learns to swim.',
  'Translate "good morning" into French and Spanish.',
];
const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist'];
const given = process.argv.slice(2).find((value) => /^https?:\/\//.test(value)) ?? null;
const local = given ? null : await startServer();
const url = given ?? local.url;
const context = await chromium.launchPersistentContext(process.env.CHECK_PROFILE ?? path.join(root, '.build', 'chromium-profile'),
  { headless: true, args: gpuArgs, ...(process.env.CHECK_CHANNEL ? { channel: process.env.CHECK_CHANNEL } : {}) });
const report = { schema: 'doppler.observatory.projection-check/v1', startedAt: new Date().toISOString(), device: null, runtime: null, models: [] };
let failed = false;
try {
  const page = context.pages()[0] ?? await context.newPage();
  if (process.env.GEMMA_CANDIDATE) await context.route('**/__gemma_candidate/**', async route => {
    const file = path.basename(new URL(route.request().url()).pathname);
    await route.fulfill({ path: path.join(process.env.GEMMA_CANDIDATE, file), contentType: file.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
  });
  context.on('response', response => { if (response.status() >= 400) console.log('HTTP', response.status(), response.url()); });
  page.on('pageerror', (error) => console.log(`  pageerror: ${error.message.slice(0, 300)}`));
  await page.goto(new URL('explore.html?preview=views.experimental,runtime.preview&runtime=preview', url).href, { waitUntil: 'domcontentloaded' });
  for (const modelId of models) {
    const result = await page.evaluate(async ({ modelId, prompts, candidate }) => {
      const runtime = await import(new URL('./runtime-preview/src/index.js', location.href).href);
      const shared = await import(new URL('./shared-run.js', location.href).href);
      // The runtime's own account of GPU buffers it holds: the same module the runtime uses.
      const poolModule = await import(new URL('./runtime-preview/src/memory/buffer-pool.js', location.href).href).catch(() => null);
      // Asked for each time: the buffers belong to the device, which exists only once a model is loaded.
      const stats = () => { try { return poolModule?.getBufferPool()?.getStats() ?? null; } catch { return null; } };
      const allocated = () => stats()?.currentBytesAllocated ?? null;
      const adapter = await navigator.gpu?.requestAdapter();
      const out = { modelId, version: runtime.DOPPLER_VERSION, adapter: adapter ? { vendor: adapter.info?.vendor ?? null, architecture: adapter.info?.architecture ?? null } : null, userAgent: navigator.userAgent, prompts: [] };
      const loadStarted = performance.now();
      let handle;
      try { handle = await runtime.dr.load(candidate && modelId === 'gemma-3-1b-it-q4k-ehf16-af32' ? { url: new URL('./__gemma_candidate', location.href).href } : modelId, { cache: 'opfs' }); }
      catch (error) { out.error = `the model did not load: ${String(error.message).split('\n').slice(0, 3).join(' ').slice(0, 300)}`; return out; }
      out.loadMs = performance.now() - loadStarted;
      if (typeof handle.inspect.projectLayerOutputs !== 'function') { out.error = 'this runtime has no projectLayerOutputs'; return out; }
      try {
        for (const prompt of prompts) {
          const entry = { prompt };
          try {
            const receipt = await handle.inspect.generate(prompt, { policyId: 'demo/activation-grid', topKSize: 8,
              generation: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, presencePenalty: 0, maxTokens: 16, useChatTemplate: true } });
            entry.outputText = receipt.outputText;
            const run = await shared.createRun({ receipt, prompt, environment: {} });
            entry.tokens = run.tokens.length;
            entry.layers = run.layerOutputs?.layers ?? null;
            entry.width = run.layerOutputs?.width ?? null;
            entry.activationDtype = run.environment.activationDtype ?? receipt.fingerprint?.identity?.activationDtype ?? null;
            const rows = [];
            for (const token of run.tokens) { const chosen = shared.choiceVectors(run, token.index); if (chosen) rows.push({ token, last: Float32Array.from(chosen.vectors.at(-1)), stack: chosen.vectors }); }
            entry.projectable = rows.length;
            if (!rows.length) { entry.note = 'no answer token had every layer recorded at its choosing position'; out.prompts.push(entry); continue; }
            // Agreement: the last layer at the choosing position, for every token, in one call.
            const started = performance.now();
            const last = await handle.inspect.projectLayerOutputs({ vectors: rows.map((row) => row.last), topK: 3 });
            entry.lastLayerBatchMs = performance.now() - started;
            entry.vocabSize = last.vocabSize;
            entry.agree = 0; entry.maxProbabilityGap = 0; entry.disagreements = [];
            rows.forEach((row, i) => {
              const top = last.rows[i].candidates[0];
              if (top.tokenId === row.token.tokenId) entry.agree++;
              else entry.disagreements.push({ index: row.token.index, recorded: row.token.text, recordedProbability: row.token.probability, projected: top.text, projectedProbability: top.probability });
              const same = last.rows[i].candidates.find((candidate) => candidate.tokenId === row.token.tokenId);
              if (same && row.token.probability !== null) entry.maxProbabilityGap = Math.max(entry.maxProbabilityGap, Math.abs(same.probability - row.token.probability));
            });
            // Cost: one token's whole stack of layers, three times, and the heap around it.
            const heap = () => performance.memory?.usedJSHeapSize ?? null;
            const before = heap();
            const times = [];
            // GPU buffers: what the runtime holds before, the most seen while projecting
            // (sampled every millisecond, so a short spike can be missed), and after.
            const gpuBefore = allocated();
            const lifetimePeakBefore = stats()?.peakBytesAllocated ?? null;
            let gpuMost = gpuBefore;
            const sampler = setInterval(() => { const now = allocated(); if (now !== null && now > gpuMost) gpuMost = now; }, 1);
            for (let repeat = 0; repeat < 3; repeat++) { const t = performance.now(); await handle.inspect.projectLayerOutputs({ vectors: rows[0].stack.map((vector) => Float32Array.from(vector)), topK: 3 }); times.push(performance.now() - t); }
            clearInterval(sampler);
            entry.gpu = gpuBefore !== null ? { beforeBytes: gpuBefore, mostSampledBytes: gpuMost, afterBytes: allocated(), lifetimePeakBefore, lifetimePeakAfter: stats()?.peakBytesAllocated ?? null } : null;
            entry.stackProjectionMs = times;
            entry.jsHeapDeltaBytes = before === null ? null : heap() - before;
            // What one call holds at its peak on the JavaScript side: a full vocabulary row per vector.
            entry.transientLogitBytes = rows[0].stack.length * last.vocabSize * 4;
            entry.wallMs = receipt.wallTimingMs ?? null;
          } catch (error) { entry.error = String(error.message).slice(0, 400); }
          out.prompts.push(entry);
        }
      } finally { await handle.unload?.().catch(() => {}); }
      return out;
    }, { modelId, prompts: PROMPTS, candidate: Boolean(process.env.GEMMA_CANDIDATE) });
    report.models.push(result);
    report.device ??= { adapter: result.adapter, userAgent: result.userAgent };
    report.runtime ??= { version: result.version };
    console.log(`  ${modelId}: ${result.error ?? `loaded in ${(result.loadMs / 1000).toFixed(1)} s`}`);
    for (const entry of result.prompts) {
      if (entry.error) { console.log(`    "${entry.prompt.slice(0, 40)}": error: ${entry.error}`); failed = true; continue; }
      if (!entry.projectable) { console.log(`    "${entry.prompt.slice(0, 40)}": ${entry.note}`); failed = true; continue; }
      const median = [...entry.stackProjectionMs].sort((a, b) => a - b)[1];
      console.log(`    "${entry.prompt.slice(0, 40)}": last layer = chosen token for ${entry.agree}/${entry.projectable}, largest probability gap ${entry.maxProbabilityGap.toExponential(1)}; ${entry.layers} layers projected in ${median.toFixed(0)} ms (generation took ${entry.wallMs?.toFixed(0)} ms); ${(entry.transientLogitBytes / 2 ** 20).toFixed(1)} MiB of logits read back`
        + (entry.gpu ? `; GPU buffers ${(entry.gpu.beforeBytes / 2 ** 20).toFixed(0)} MiB before, largest observed ${((Math.max(entry.gpu.mostSampledBytes, entry.gpu.lifetimePeakAfter > entry.gpu.lifetimePeakBefore ? entry.gpu.lifetimePeakAfter : 0) - entry.gpu.beforeBytes) / 2 ** 20).toFixed(1)} MiB more while projecting, ${((entry.gpu.afterBytes - entry.gpu.beforeBytes) / 2 ** 20).toFixed(1)} MiB more after` : '; GPU buffers not readable'));
      for (const miss of entry.disagreements) console.log(`      differs at token ${miss.index + 1}: recorded "${miss.recorded}" ${miss.recordedProbability}, projected "${miss.projected}" ${miss.projectedProbability}`);
      if (entry.agree !== entry.projectable || entry.maxProbabilityGap > 1e-5) failed = true;
    }
    if (result.error) failed = true;
  }
  const release = await page.evaluate(() => fetch('./runtime-preview/release.json').then((response) => response.json()).catch(() => null));
  report.runtime = { ...report.runtime, sourceCommit: release?.runtime?.sourceCommit ?? null };
} finally {
  await context.close();
  local?.server.close();
}
const out = arg('--out');
if (out) { await fs.mkdir(path.dirname(path.resolve(out)), { recursive: true }); await fs.writeFile(path.resolve(out), `${JSON.stringify(report, null, 1)}\n`); }
console.log(failed ? 'projection: FAILED (see above)' : 'projection: passed');
process.exit(failed ? 1 : 0);
