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
    await fs.writeFile(path.join(out, 'activity-receipt.json'), JSON.stringify(value));
  });
  const result = await page.evaluate(async (policyId) => {
    const { dr } = await import('doppler-gpu');
    const modelId = 'qwen-3-5-0-8b-q4k-ehaf16';
    const model = await dr.load({ url: new URL('/.cache/qwen08-sha256-repair/', location.href).href, logicalModelId: modelId }, { cache: 'opfs' });
    const generation = { temperature: 0, topK: 1, topP: 1, maxTokens: 3, useChatTemplate: true };
    const prompt = 'Say hello in one word.';
    const baselinePolicy = policyId === 'demo/model-activity-timing' ? 'demo/deep-xray' : 'demo/guided-quality';
    const baseline = await model.inspect.generate(prompt, { policyId: baselinePolicy, generation });
    const captured = await model.inspect.generate(prompt, { policyId, generation });
    await globalThis.retainActivationEvidence(captured);
    const activity = captured.activationTrace?.activity;
    const sameTokens = JSON.stringify(baseline.generatedTokenIds) === JSON.stringify(captured.generatedTokenIds);
    const sameScores = JSON.stringify(baseline.tokens) === JSON.stringify(captured.tokens);
    const cancelled = new AbortController();
    let sawLayer = false, cancelledCleanly = false;
    try {
      await model.inspect.generate(prompt, { policyId, generation: { ...generation, signal: cancelled.signal },
        onEvent(event) { if (event.type === 'activation-frame' && event.frame.layer === 0 && event.frame.status === 'captured' && !sawLayer) { sawLayer = true; cancelled.abort('activity cancellation test'); } },
      });
    } catch (error) { cancelledCleanly = sawLayer && cancelled.signal.aborted; }
    const reused = await model.inspect.generate(prompt, { policyId: baselinePolicy, generation });
    const reuseMatches = JSON.stringify(reused.generatedTokenIds) === JSON.stringify(baseline.generatedTokenIds);
    await model.unload();
    return { policyId, baselinePolicy, sameTokens, sameScores, cancelledCleanly, reuseMatches, output: captured.outputText,
      frames: activity?.frames.length, roles: [...new Set(activity?.frames.map(f => f.role))],
      failed: activity?.frames.filter(f => f.status !== 'captured').map(f => ({role: f.role, reason:f.reason})),
      weights: activity?.weights.map(w => ({ id: w.id, status: w.status, reason:w.reason, raw:w.raw.length, summaries:w.rms.length })),
      bytes: activity?.recordedBytes };
  }, process.argv.includes('--timing') ? 'demo/model-activity-timing' : 'demo/model-activity');
  await fs.writeFile(path.join(out, 'activity-summary.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ ...result, weights: { total: result.weights?.length, observed: result.weights?.filter(w => w.status === 'captured').length }, failed: result.failed?.slice(0, 5), gaps: result.failed?.length }));
  assert(result.cancelledCleanly && result.reuseMatches, 'Capture cancellation or reuse failed.');
  assert.equal(result.failed.length, 0, 'Example has missing captures.');
  assert(result.sameTokens && result.sameScores, 'Diagnostic capture changed execution output.');
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
