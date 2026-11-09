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
  await page.waitForFunction(() => document.querySelector('#prompt-input')?.value.trim());
  const defaultPrompt = await page.inputValue('#prompt-input');
  await page.selectOption('#model-select', 'qwen-3-5-0-8b-q4k-ehaf16');
  await page.click('#model-select-action');
  await page.waitForFunction(() => document.querySelector('#model-select-action').textContent === 'Loaded'
    || document.querySelector('#status-text').textContent.startsWith('Load failed:'));
  assert.equal(await page.locator('#model-select-action').textContent(), 'Loaded', await page.locator('#status-text').textContent());
  await page.locator('#activation-view').evaluate(el => { el.open = true; });
  assert.equal(await page.locator('#activation-record').isChecked(), true, 'Live activity defaults on');
  await page.evaluate(() => { const select = document.querySelector('#set-max-tokens'); select.add(new Option('16', '16')); select.value = '16'; select.dispatchEvent(new Event('change', { bubbles: true })); });
  assert.equal(await page.inputValue('#prompt-input'), defaultPrompt, 'Loading a model preserves the ready-to-run prompt');
  await page.evaluate(() => {
    globalThis.__liveGridCheck = { captions: [], duringGeneration: 0, colorChanges: 0, pathChanges: 0, activationChanges: 0, visibleActivations: 0, last: '', lastActivation: '' };
    const observe = () => {
      const caption = document.querySelector('#activation-caption').textContent;
      if (!caption.startsWith('Live')) return;
      const result = globalThis.__liveGridCheck, canvas = document.querySelector('#activation-canvas');
      const bytes = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let digest = 0; for (let i = 0; i < bytes.length; i++) digest = (Math.imul(digest, 31) + bytes[i]) | 0;
      const paths = document.querySelector('#activity-paths');
      if (paths && !paths.hidden) {
        const activationCanvas = paths.querySelector('canvas');
        if (activationCanvas) {
          let activationDigest = 0;
          for (const v of activationCanvas.getContext('2d').getImageData(0,0,activationCanvas.width,activationCanvas.height).data) activationDigest = (Math.imul(activationDigest,31) + v) | 0;
          if (result.lastActivation !== String(activationDigest)) result.activationChanges++;
          result.lastActivation = String(activationDigest);
          const box = activationCanvas.getBoundingClientRect();
          if (box.top >= 0 && box.bottom <= innerHeight) result.visibleActivations++;
        }
        const text = paths.textContent; digest = 0;
        for (let i = 0; i < text.length; i++) digest = (Math.imul(digest,31) + text.charCodeAt(i)) | 0;
        if (result.last !== String(digest)) result.pathChanges++;
      }
      if (result.last !== String(digest)) result.colorChanges++;
      result.last = String(digest);
      if (document.querySelector('#stop-btn').checkVisibility()) result.duringGeneration++;
      if (result.captions.length < 16) result.captions.push(caption);
    };
    new MutationObserver(observe).observe(document.querySelector('#activation-caption'), {childList: true});
  });
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
  assert.equal(receipt.modes.filter(m => m.available).length, 19);
  assert.equal(receipt.gaps, 0); assert.equal(receipt.captured, receipt.frames);
  const live = await page.evaluate(() => globalThis.__liveGridCheck);
  assert(live.duringGeneration > 1 && live.pathChanges > 1, 'Visible model paths change while generation is active');
  assert(live.activationChanges > 1 && live.visibleActivations > 1, 'The activation heatmap changes in the viewport during generation');
  const release = await (await page.request.get(origin + '/doppler/release.json')).json();
  const result = { origin, browser: browser.version(), release, defaultPrompt, receipt, live, errors };
  await page.locator('#activation-view').screenshot({ path: out + '/grid.png' });
  await fs.writeFile(out + '/receipt.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result)); assert.deepEqual(errors, []);
} finally { await browser.close(); }
