// Real-hardware check of an interrupted first visit: the model download is cut
// part-way (network lost), the page must say so instead of hanging, and after
// the network returns a second attempt must finish and answer. It also reports
// how many model bytes the second attempt fetched, which shows whether the
// download resumed or started again.
//
//   node scripts/check-resume.js [served address]
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { startServer } from './dev.js';

const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist'];
const MODEL = 'gemma-3-270m-it-q4k-ehf16-af32';
const given = process.argv.slice(2).find((arg) => /^https?:\/\//.test(arg)) ?? null;
const local = given ? null : await startServer();
const url = given ?? local.url;
// A profile of its own, so nothing of the model is saved before the first attempt.
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'observatory-resume-'));
const context = await chromium.launchPersistentContext(profile, { headless: true, args: gpuArgs, ...(process.env.CHECK_CHANNEL ? { channel: process.env.CHECK_CHANNEL } : {}), viewport: { width: 1280, height: 1200 } });
const MiB = (bytes) => `${(bytes / 2 ** 20).toFixed(0)} MiB`;
try {
  const page = context.pages()[0] ?? await context.newPage();
  // Bytes of large responses (model files), counted per attempt from the browser's own network log.
  const cdp = await context.newCDPSession(page);
  await cdp.send('Network.enable');
  let fetched = 0;
  const sizes = new Map();
  cdp.on('Network.dataReceived', (event) => { sizes.set(event.requestId, (sizes.get(event.requestId) ?? 0) + event.encodedDataLength); fetched += event.encodedDataLength; });

  await page.goto(url, { waitUntil: 'domcontentloaded' });
  // Explore loads the model when a question is sent; the status line and the busy flag say how it went.
  await page.waitForFunction(() => !document.getElementById('ws-model').disabled, null, { timeout: 600_000 });
  await page.selectOption('#ws-model', MODEL);
  await page.fill('#ws-prompt', 'Explain in one short paragraph why the sky is blue.');
  const state = () => page.evaluate(() => { const ws = globalThis.__DOPPLER_WORKSPACE__; return { busy: ws.busy, loaded: ws.model?.modelId ?? null, status: document.getElementById('ws-status').textContent, answers: ws.runs.filter((run) => run.kind === 'answer').length }; });
  assert.equal((await state()).loaded, null, 'No model is loaded before the first attempt');

  // First attempt: cut the network once a good part of the model has arrived.
  const before = fetched;
  await page.click('#ws-send');
  const CUT_AFTER = 120 * 2 ** 20;
  const deadline = Date.now() + 600_000;
  while (fetched - before < CUT_AFTER) {
    if (Date.now() > deadline) throw new Error(`Only ${MiB(fetched - before)} of the model arrived in ten minutes; the download is too slow to interrupt part-way.`);
    if ((await state()).loaded) throw new Error('The model finished loading before it could be interrupted.');
    await page.waitForTimeout(250);
  }
  const firstBytes = fetched - before;
  await context.setOffline(true);
  const cutAt = Date.now();
  // The page must notice: a stated failure, and Send usable again.
  await page.waitForFunction(() => !globalThis.__DOPPLER_WORKSPACE__.busy && /Could not answer/.test(document.getElementById('ws-status').textContent), null, { timeout: 300_000 });
  const interrupted = await state();
  console.log(`  interrupted after ${MiB(firstBytes)}; the page said so after ${((Date.now() - cutAt) / 1000).toFixed(0)} s: "${interrupted.status}"`);
  assert.equal(await page.isDisabled('#ws-send'), false, 'Send can be pressed again');

  // Second attempt, with the network back.
  await context.setOffline(false);
  const again = fetched;
  await page.click('#ws-send');
  await page.waitForFunction(() => globalThis.__DOPPLER_WORKSPACE__.runs.some((run) => run.kind === 'answer'), null, { timeout: 900_000 });
  const secondBytes = fetched - again;
  console.log(`  second attempt fetched ${MiB(secondBytes)} and answered; first attempt had fetched ${MiB(firstBytes)}`);
  console.log(`  ${secondBytes < 0.8 * (firstBytes + secondBytes) && secondBytes + firstBytes < 1.2 * 420 * 2 ** 20 ? 'resumed: the second attempt fetched only what was missing' : 'started again: the second attempt fetched the whole model'}`);
  const answer = await page.evaluate(() => { const run = globalThis.__DOPPLER_WORKSPACE__.runs.find((item) => item.kind === 'answer'); return { tokens: run.tokens.length, text: run.outputText.slice(0, 50) }; });
  console.log(`  answered after the interrupted download: ${JSON.stringify(answer)}`);
  assert.ok(answer.tokens > 0);
  console.log('resume: passed');
} finally {
  await context.close();
  local?.server.close();
  await fs.rm(profile, { recursive: true, force: true });
}
