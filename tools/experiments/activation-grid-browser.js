import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { resolveBrowserArgs } from '../demo-hardware-smoke.js';

// Explicit one-model browser proof. Never starts from the unit-test suite.
assert(process.argv.includes('--physical'), 'Pass --physical to authorize this real-model GPU check.');
const root = process.cwd(), out = path.join(root, '.cache/activation-grid');
await fs.mkdir(out, { recursive: true });
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css' };
const server = http.createServer(async (request, response) => {
  const target = path.resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
  if (!target.startsWith(root + path.sep)) return response.writeHead(403).end();
  try { response.setHeader('content-type', mime[path.extname(target)] ?? 'application/octet-stream'); response.end(await fs.readFile(target)); }
  catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ headless: true, args: resolveBrowserArgs() });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  page.setDefaultTimeout(600000);
  page.on('console', message => { if (['error', 'warning'].includes(message.type())) console.log(message.type(), message.text()); });
  page.on('pageerror', error => console.error('pageerror', error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/demo/index.html${process.argv.includes('--repaired') ? '?repaired=1' : ''}`);
  await page.waitForFunction(() => document.querySelector('#model-select')?.options.length > 1);
  await page.exposeFunction('retainActivationEvidence', async value => {
    await fs.writeFile(path.join(out, 'physical-api-receipt.json'), JSON.stringify(value));
  });
  const result = await page.evaluate(async () => {
    const { dr } = await import('doppler-gpu');
    const { onModelLoaded, runGeneration } = await import('/demo/core.js');
    const { state } = await import('/demo/ui/state.js');
    const grid = await import('/demo/ui/activation-grid/index.js');
    const modelId = 'qwen-3-5-0-8b-q4k-ehaf16';
    const source = location.search.includes('repaired')
      ? { url: new URL('/.cache/qwen08-sha256-repair/', location.href).href, logicalModelId: modelId }
      : modelId;
    const model = await dr.load(source, { cache: 'opfs' });
    console.log('Physical activation test: model loaded');
    const prompt = 'Say hello in one word.';
    const generation = { temperature: 0, topK: 1, topP: 1, maxTokens: 4, useChatTemplate: true };
    const baseline = await model.inspect.generate(prompt, { policyId: 'demo/guided-quality', generation });
    const observed = await model.inspect.generate(prompt, { policyId: 'demo/activation-grid', generation });
    if (JSON.stringify(baseline.generatedTokenIds) !== JSON.stringify(observed.generatedTokenIds)) throw new Error('Capture changed output token IDs');
    if (JSON.stringify(baseline.tokens) !== JSON.stringify(observed.tokens)) throw new Error('Capture changed observed logits/probabilities');
    const trace = observed.activationTrace;
    if (!trace?.frames.some(f => f.phase === 'prefill' && f.status === 'captured')
      || !trace.frames.some(f => f.phase === 'decode' && f.status === 'captured')) throw new Error('Missing physical prefill/decode captures');
    await globalThis.retainActivationEvidence({ trace, tokenIds: observed.generatedTokenIds, output: observed.outputText });
    onModelLoaded(model, modelId);
    state.modelId = modelId;
    document.querySelector('#activation-record').checked = true;
    document.querySelector('#set-max-tokens').add(new Option('4', '4'));
    document.querySelector('#set-max-tokens').value = '4';
    document.querySelector('#set-max-tokens').dispatchEvent(new Event('change', { bubbles: true }));
    document.querySelector('#prompt-input').value = prompt;
    document.querySelector('#prompt-input').dispatchEvent(new Event('input', { bubbles: true }));
    await runGeneration();
    if (!grid.getActivationRecording()?.trace.frames.length) throw new Error('Demo did not receive live frames: ' + document.querySelector('#output-phase').textContent);
    const recording = JSON.parse(JSON.stringify(grid.getActivationRecording()));
    const firstCache = model.persistentCache;
    const { runLivePrecision } = await import('/demo/ui/precision-replay/live.js');
    document.querySelector('#prompt-input').value = 'What is 2 plus 2? Answer briefly.';
    await runLivePrecision();
    const precision = { status: document.querySelector('#live-precision-status').textContent, lanes: [...document.querySelectorAll('#live-precision-results article')].map(x => x.textContent) };
    if (precision.lanes.length !== 2) throw new Error('Live comparison failed: ' + precision.status);
    await model.unload(); state.model = null;
    const reused = await dr.load(source, { cache: 'opfs' });
    const reusedCache = reused.persistentCache;
    if (reusedCache?.fromCache !== true) throw new Error('Persistent cache was not reused');
    const reuseOutput = await reused.inspect.generate(prompt, { policyId: 'demo/guided-quality', generation });
    if (JSON.stringify(reuseOutput.generatedTokenIds) !== JSON.stringify(baseline.generatedTokenIds)) throw new Error('Reuse changed output');
    await reused.unload();
    grid.importActivationRecording(recording);
    await new Promise(requestAnimationFrame);
    const canvas = document.querySelector('#activation-canvas');
    const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    return { firstCache, reusedCache, precision, identity: trace.identity, tokenIds: observed.generatedTokenIds, output: observed.outputText,
      frames: trace.frames.length, captured: trace.frames.filter(f => f.status === 'captured').length,
      dropped: trace.frames.filter(f => f.status === 'dropped').length,
      phases: [...new Set(trace.frames.map(f => f.phase))], geometry: trace.geometry,
      offlinePlayback: state.model === null && pixels.some((n, i) => i % 4 !== 3 && n !== 48), recording };
  });
  assert(result.offlinePlayback);
  await page.locator('#live-precision').evaluate(el => { el.open = true; });
  const downloadPending = page.waitForEvent('download');
  await page.click('#live-precision-export');
  const download = await downloadPending;
  await download.saveAs(path.join(out, 'precision-comparison.json'));
  await page.locator('#activation-view').screenshot({ path: path.join(out, 'physical-grid.png') });
  await fs.writeFile(path.join(out, 'physical-receipt.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ ...result, recording: 'retained in .cache/activation-grid/physical-receipt.json' }));
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
