import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import { resolveBrowserArgs } from '../demo-hardware-smoke.js';

assert(process.argv.includes('--physical'), 'Pass --physical to run the real hosted model.');
const origin = process.argv.find(v => v.startsWith('--origin='))?.slice(9) || 'https://d4da.com';
const out = '.cache/activation-grid/hosted';
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, args: resolveBrowserArgs() });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 950 } });
  page.setDefaultTimeout(600000);
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(origin + '/doppler/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelector('#model-select')?.options.length > 1);
  await page.selectOption('#model-select', 'qwen-3-5-0-8b-q4k-ehaf16');
  await page.click('#model-select-action');
  await page.waitForFunction(() => document.querySelector('#model-select-action').textContent === 'Loaded'
    || document.querySelector('#status-text').textContent.startsWith('Load failed:'));
  assert.equal(await page.locator('#model-select-action').textContent(), 'Loaded', await page.locator('#status-text').textContent());
  await page.locator('#activation-view').evaluate(el => { el.open = true; });
  await page.check('#activation-record');
  await page.evaluate(() => { const select = document.querySelector('#set-max-tokens'); select.add(new Option('3', '3')); select.value = '3'; select.dispatchEvent(new Event('change', { bubbles: true })); });
  await page.fill('#prompt-input', 'Say hello in one word.');
  await page.click('#run-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Complete');
  const receipt = await page.evaluate(async () => {
    const base = document.querySelector('script[src$="/demo.js"]').src;
    const { state } = await import(new URL('ui/state.js', base));
    const { getActivationRecording } = await import(new URL('ui/activation-grid/index.js', base));
    const recording = getActivationRecording(), trace = recording.trace;
    const { activityModes } = await import(new URL('ui/activation-grid/views.js', base));
    return { identity: trace.identity, persistentCache: state.model.persistentCache,
      output: state.lastInspection.outputText, tokenIds: state.lastInspection.generatedTokenIds,
      phases: [...new Set(trace.frames.map(f => f.phase))], frames: trace.frames.length,
      captured: trace.frames.filter(f => f.status === 'captured').length,
      gaps: trace.frames.filter(f => f.status === 'dropped' || f.status === 'failed').length,
      activityFrames: trace.activity?.frames.length ?? 0,
      activityGaps: trace.activity?.frames.filter(f => f.status !== 'captured').length ?? null,
      modes: activityModes(recording).map(m => ({ id: m.id, available: !m.reason, reason: m.reason })) };
  });
  assert(receipt.phases.includes('prefill') && receipt.phases.includes('decode'));
  assert.equal(receipt.activityGaps, 0); assert(receipt.activityFrames > 0);
  assert.equal(receipt.modes.filter(m => m.available).length, 18);
  assert.equal(receipt.gaps, 0); assert.equal(receipt.captured, receipt.frames);
  const release = await (await page.request.get(origin + '/doppler/release.json')).json();
  const result = { origin, browser: browser.version(), release, receipt, errors };
  await page.locator('#activation-view').screenshot({ path: out + '/grid.png' });
  await fs.writeFile(out + '/receipt.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result)); assert.deepEqual(errors, []);
} finally { await browser.close(); }
