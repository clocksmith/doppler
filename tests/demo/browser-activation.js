import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

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
  await page.waitForFunction(() => document.querySelector('#activation-caption').textContent.includes('capture 4/4'));
  assert.equal(await page.locator('#activation-canvas').isVisible(), true);
  const final = await page.locator('#activation-canvas').evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  await page.locator('#activation-scrub').fill('0');
  await page.waitForFunction(() => document.querySelector('#activation-caption').textContent.includes('capture 1/4'));
  const first = await page.locator('#activation-canvas').evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  assert.notDeepEqual(first, final, 'Scrubbing changes measured samples');
  if (!await page.locator('#activation-speed').isVisible()) await page.locator('#run-details > summary').click();
  await page.selectOption('#activation-speed', '0.25');
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 0);
  await page.click('#activation-play');
  assert.equal(await page.locator('#activation-play').textContent(), 'Play');
  const pending = page.waitForEvent('download'); await page.click('#activation-export');
  const download = await pending;
  assert.deepEqual(JSON.parse(await fs.readFile(await download.path(), 'utf8')), recording, 'Offline export is self-contained');
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
