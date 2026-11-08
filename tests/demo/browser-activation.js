import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { validateRecording } from '../../demo/ui/activation-grid/data.js';

// Geometry protects the first-load gap and makes every preview an immediate entry.
export async function checkFirstLoadLayout(page, empty = false) {
  await fs.mkdir('.cache/demo-first-load', { recursive: true });
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => new Promise(requestAnimationFrame));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
    if (empty) {
      const surface = await page.locator('.chat-surface').boundingBox();
      const composer = await page.locator('#input-area').boundingBox();
      assert(surface.height < 80, `Empty conversation must not reserve a transcript at ${width}px`);
      assert(composer.y - surface.y - surface.height < 40, `Composer follows empty conversation at ${width}px`);
      await page.locator('#dialogue-section').screenshot({ path: `.cache/demo-first-load/empty-${width}.png` });
    } else {
      assert.equal(await page.locator('.activity-preview').count(), 6);
      const images = await page.locator('.activity-preview img').evaluateAll(nodes => nodes.map(n => ({ loaded: n.complete && n.naturalWidth > 1, width: n.getBoundingClientRect().width })));
      assert(images.every(i => i.loaded && i.width >= 90), 'Real grid previews are visible before any download or model run');
      assert.equal(await page.locator('#activity-selectors').isVisible(), false, 'No unusable selectors on first load');
      assert.equal(await page.locator('#activation-player').isVisible(), false);
      await page.locator('#inspection-workspace').screenshot({ path: `.cache/demo-first-load/activity-${width}.png` });
    }
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
}

// Synthetic data exercises only the UI. Physical captures are a separate opt-in check.
export async function checkActivationPlayback(page) {
  const frames = [0, 1, 0, 1].map((layer, index) => ({
    index, forward: Math.floor(index / 2), atMs: index * 5,
    phase: index < 2 ? 'prefill' : 'decode', role: 'layer-output', layer,
    tokenStart: index < 2 ? 0 : 1, tokenIds: [10 + Math.floor(index / 2)], shape: [1, 3],
    dtype: 'f32', precision: null, status: 'captured', values: [index, index ? -index : 0, .5], reason: null,
  }));
  const recording = { schema: 'doppler.activation-recording/v1', demoBuild: 'synthetic-ui-test',
    display: { layout: 'layer-major-rgb/v1', palette: 'signed-tanh-rgb/v1', scale: 1 },
    trace: { schema: 'doppler.activation-trace/v1', identity: { modelId: 'synthetic-ui-test', manifestHash: 'a'.repeat(64), runtimeVersion: 'fixture' },
      geometry: { layers: 2, width: 3 }, capturePolicy: { maxBytes: 1024, maxFrameBytes: 128, maxFrames: 8, maxGridValues: 64, maxInFlight: 2 },
      status: 'complete', stoppedReason: null, recordedBytes: 96, frames } };
  assert.equal(await page.locator('#activation-player').isVisible(), false, 'An instruction replaces the empty grid');
  await page.setInputFiles('#import-file', { name: 'trace.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(recording)) });
  await page.selectOption('#activity-playback', 'captures');
  await page.waitForFunction(() => document.querySelector('#activation-scrub').max === '3');
  assert.equal(await page.locator('#activation-canvas').isVisible(), true);
  const final = await page.locator('#activation-canvas').evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  const scale = await page.inputValue('#activation-scale');
  await page.locator('#activation-scrub').fill('0');
  await page.waitForFunction(() => document.querySelector('#activation-position').value === '0');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  const first = await page.locator('#activation-canvas').evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  assert.notDeepEqual(first, final, 'Scrubbing changes measured samples');
  assert.equal(await page.inputValue('#activation-scale'), scale, 'Playback never rescales as token position changes');
  if (!await page.locator('#activation-speed').isVisible()) await page.locator('#run-details > summary').click();
  await page.selectOption('#activation-speed', '0.25');
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 0);
  await page.click('#activation-play');
  assert.equal(await page.locator('#activation-play').textContent(), 'Play');
  if (!await page.locator('#activation-export').isVisible()) await page.locator('#activity-files > summary').click();
  const pending = page.waitForEvent('download'); await page.click('#activation-export');
  const download = await pending;
  const reopened = validateRecording(JSON.parse(await fs.readFile(await download.path(), 'utf8')));
  assert.deepEqual(reopened.trace, recording.trace, 'Offline export preserves every numerical value');
  assert(reopened.display.scales['layer-output:overview'] > 0, 'Export preserves the fixed view scale');
  await page.setInputFiles('#import-file', { name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"schema":"doppler.activation-recording/v1"}') });
  await page.waitForFunction(() => document.querySelector('#activation-status').textContent.includes('Invalid'));
  assert.equal(await page.evaluate(async () => (await import(new URL('ui/activation-grid/index.js', document.querySelector('script[src$="/demo.js"]').src))).getActivationRecording().demoBuild), 'synthetic-ui-test', 'Invalid imports preserve the accepted recording');
  assert.equal(await page.evaluate(async () => (await import(new URL('report.js', document.querySelector('script[src$="/demo.js"]').src))).buildReport()?.activationRecording ?? null), null, 'A recording is never attached to an unrelated answer receipt');
}

export async function checkTokenGeometry(page) {
  await fs.mkdir('.cache/activation-grid', { recursive: true });
  for (const width of [1200, 390, 320]) {
    await page.setViewportSize({ width, height: 950 });
    const bars = await page.locator('.token-candidate-bar-bg').evaluateAll(nodes => nodes.map(n => {
      const b = n.getBoundingClientRect(); return { left: b.left, right: b.right, width: b.width };
    }));
    assert(bars.length > 1 && bars.every(b => b.width > 0));
    assert(bars.every(b => Math.abs(b.left - bars[0].left) < 1 && Math.abs(b.right - bars[0].right) < 1), `Candidate bars align at ${width}px`);
    const gradients = await page.locator('.token-chip:not(.token-chip--unavailable)').evaluateAll(nodes => nodes.map(n => getComputedStyle(n).backgroundImage));
    assert(gradients.length && gradients.every(s => s.startsWith('linear-gradient')));
    assert(new Set(gradients).size > 3, 'Continuous probability gradients are not three categorical colors');
    await page.locator('#token-inspector-view').screenshot({ path: `.cache/activation-grid/token-inspector-${width}.png` });
    assert(await page.locator('#model-select').isVisible(), 'Model selector remains visible outside Settings');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No page overflow at ${width}px`);
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
}

// Uses the published physical recording; does not run a model or fixture generator.
export async function checkRealActivityPlayback(page) {
  // Reopen the initial gallery to exercise its actual entry point.
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.activity-preview[data-mode="contributions"]').click();
  await page.waitForFunction(() => !document.querySelector('#activity-example').disabled, null, { timeout: 120000 });
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.match(await page.locator('#activation-caption').textContent(), /token position 17/, await page.locator('#activity-availability').textContent() + ' ' + await page.locator('#activation-status').textContent());
  assert.equal(await page.inputValue('#activity-mode'), 'contributions', 'The selected preview opens its own interactive grid');
  assert.equal(await page.locator('#activity-previews').isVisible(), false);
  assert.match(await page.locator('#activity-source').textContent(), /Recorded example.*qwen/);
  const modes = await page.locator('#activity-mode option').evaluateAll(options => options.map(o => ({ id: o.value, disabled: o.disabled })));
  const results = [];
  for (const mode of modes) {
    if (mode.disabled) { results.push({ ...mode }); continue; }
    await page.selectOption('#activity-mode', mode.id);
    await page.evaluate(() => new Promise(requestAnimationFrame));
    const state = await page.evaluate(() => {
      const c = document.querySelector('#activation-canvas');
      const layout = document.querySelector('#activation-layout').textContent;
      return { caption: document.querySelector('#activation-caption').textContent, layout, width: c.width, height: c.height,
        pixelColors: new Set(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).size,
        position: document.querySelector('#activation-position').value };
    });
    assert.equal(state.position, '17', 'Switching grids preserves the shared token position');
    assert(state.width > 1 && state.pixelColors > 2, `${mode.id} renders recorded numerical values`);
    assert(!state.layout.startsWith('0/'), `${mode.id} has real capture coverage`);
    results.push({ id: mode.id, ...state });
  }
  assert.equal(modes.filter(m => !m.disabled).length, 18);
  assert.equal(modes.find(m => m.id === 'expert-routing').disabled, true, 'A dense model cannot pretend to route experts');
  await page.selectOption('#activity-mode', 'weights');
  await page.selectOption('#activity-detail', 'raw');
  await page.selectOption('#activity-weight', 'lm_head');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.match(await page.locator('#activation-layout').textContent(), /Other connections require/);
  await page.selectOption('#activity-mode', 'recurrent-memory');
  await page.selectOption('#activity-detail', 'overview');
  await page.locator('#activity-weight').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#activity-weight').isVisible(), false);
  await page.selectOption('#activity-mode', 'layer-output');
  if (!await page.locator('#activation-position').isVisible()) await page.locator('#run-details > summary').click();
  await fs.mkdir('.cache/activation-grid/playback', { recursive: true });
  for (const width of [1200, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
    for (const id of ['activity-mode', 'activation-play', 'activation-canvas']) {
      const rect = await page.locator(`#${id}`).boundingBox();
      assert(rect && rect.x >= 0 && rect.x + rect.width <= width + 1, `${id} fits at ${width}px`);
    }
    await page.locator('#inspection-workspace').screenshot({ path: `.cache/activation-grid/playback/activity-${width}.png` });
  }
  await page.selectOption('#activity-playback', 'tokens');
  await page.locator('#activation-scrub').fill('0');
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 1);
  await page.click('#activation-play');
  const position = await page.inputValue('#activation-position');
  await page.selectOption('#activity-mode', 'kv-cache');
  assert.equal(await page.inputValue('#activation-position'), position);
  if (!await page.locator('#activation-export').isVisible()) await page.locator('#activity-files > summary').click();
  const pending = page.waitForEvent('download');
  await page.click('#activation-export');
  const downloaded = await pending;
  const exported = await fs.readFile(await downloaded.path());
  // Reopen the saved numerical file with all networking blocked.
  await page.route('**/*', route => route.abort());
  await page.setInputFiles('#import-file', { name: 'saved-activity.json', mimeType: 'application/json', buffer: exported });
  await page.waitForFunction(() => document.querySelector('#activation-status').textContent.includes('0 missing captures'));
  assert.match(await page.locator('#activation-caption').textContent(), /stored keys and values/);
  await page.unroute('**/*');
  await fs.writeFile('.cache/activation-grid/playback/receipt.json', JSON.stringify({ schema: 'doppler.activity-playback-check/v1', origin: page.url(),
    viewports: [1200,390,320], modes: results, offlineImport: true, exportBytes: exported.length, modelExecutionRequired: false }, null, 2));
}
