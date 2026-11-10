import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

// Run presentation only: no model inference or GPU qualification.
export async function checkAutomaticReplay(page) {
  const sample = await page.evaluate(async () => (await import(new URL('data/sample-inspection.js', document.querySelector('script[src$="/demo.js"]').src))).SAMPLE_INSPECTION_RECEIPT);
  const firstPosition = String(sample.inputTokenIds.length - 1);
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text
    && document.querySelector('#activation-play').textContent === 'Play', sample.outputText, { timeout: 120000 });
  const finalPosition = await page.locator('#activation-scrub').getAttribute('max');
  assert.equal(await page.locator('#sample-run-btn').count(), 0);
  assert.equal(await page.locator('#inspect-recorded-run').count(), 0);
  assert.equal(await page.locator('#chat-thread .chat-message-text').first().textContent(), 'Explain why the sky is blue in one sentence.');
  // The label names the model of the saved run itself and says nothing is running here.
  const recordedModel = await page.evaluate(id => [...document.querySelectorAll('#model-select option')]
    .find(option => option.value === id)?.textContent.split(' · ')[0] ?? id, sample.recording.modelId);
  assert.equal(await page.locator('#recorded-run-note').textContent(), `Recorded replay · ${recordedModel} · not running on this device`);
  // A first visit is a still picture of the answer's least likely word, not a moving one.
  const leastSure = sample.tokens.reduce((low, token, index) => token.probability < sample.tokens[low].probability ? index : low, 0);
  assert.equal(await page.inputValue('#activation-position'), String(sample.inputTokenIds.length - 1 + leastSure), 'Opens paused on the least likely word');
  assert.equal(await page.locator('#activity-hint').isVisible(), true, 'The pause is explained');
  assert.match(await page.locator('#token-inspector-card-container').textContent(), /likely/, 'The paused word shows what else the model considered');
  assert.equal(await page.locator('#activity-unsure-prev').isEnabled(), true);
  await page.locator('#activity-unsure-prev').click();
  const previousUnsure = sample.tokens.findLastIndex((token, index) => index < leastSure && token.probability < 0.5);
  await page.waitForFunction(position => document.querySelector('#activation-position').value === position, String(sample.inputTokenIds.length - 1 + previousUnsure));
  assert.equal(await page.locator('#activity-hint').isVisible(), false, 'The opening note leaves once the reader moves');
  await page.locator('#activity-unsure-next').click();
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text, sample.outputText);
  assert.equal(await page.locator('#recording-details').isVisible(), false);
  assert.equal(await page.locator('#activation-player').isVisible(), true);
  await fs.mkdir('.cache/demo-first-replay', { recursive: true });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.screenshot({ path: `.cache/demo-first-replay/replay-${width}.png`, fullPage: true });
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
  assert.equal(await page.evaluate(() => __demoContract.calls.length), 0, 'Replay never runs inference');
  assert.equal(await page.inputValue('#prompt-input'), '', 'The submitted first prompt is not duplicated in the composer');
  await page.locator('#activation-scrub').fill('0');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === '');
  await page.locator('#activation-scrub').fill(firstPosition);
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text, sample.tokens[0].text);
  await page.locator('#activation-scrub').fill(finalPosition);
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text, sample.outputText);
  assert.equal(await page.locator('#token-inspector-view').isVisible(), true);
  const scale = await page.inputValue('#activation-scale');
  const before = await page.locator('#activation-canvas').evaluate(c => c.toDataURL());
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForFunction(before => document.querySelector('#activation-canvas').toDataURL() !== before, before);
  assert.equal(await page.inputValue('#activation-scale'), scale, 'Theme changes do not rescale numbers');
  assert.match(await page.locator('.activity-color-bar').evaluate(n => n.style.background), /38, 42, 50/);
  await page.screenshot({ path: '.cache/demo-first-replay/replay-dark.png', fullPage: true });
  await page.emulateMedia({ colorScheme: 'light' });
  await page.waitForFunction(before => document.querySelector('#activation-canvas').toDataURL() === before, before);

  await page.locator('#activation-play').click();
  await page.fill('#prompt-input', 'My next question');
  assert.equal(await page.locator('#activation-play').textContent(), 'Play', 'Typing pauses automatic playback');
  await page.fill('#prompt-input', '');
  await page.locator('#activation-scrub').fill(finalPosition);
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text, sample.outputText);
}

export async function checkBootLayout(page) {
  await fs.mkdir('.cache/demo-run-coherence', { recursive: true });
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 800 });
    const positions = await page.evaluate(async () => {
      const entry = document.querySelector('script[src$="/demo.js"]').src;
      const boot = await import(new URL('boot.js', entry));
      const overlay = document.querySelector('#boot-overlay');
      overlay.hidden = false;
      const measure = () => ['boot-d4', 'boot-status', 'boot-detail', 'boot-timing'].map(id => {
        const rect = document.getElementById(id).getBoundingClientRect();
        return { id, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
      });
      const results = [];
      for (const [status, detail] of [
        ['Loading Doppler…', ''],
        ['Loading saved model…', 'Reading weights'],
        ['Loading saved model…', 'Verifying a very long model path and loading its weights into memory. '.repeat(8)],
        ['Ready', ''],
      ]) {
        boot.setBootStatus(status, detail);
        await new Promise(requestAnimationFrame);
        results.push(measure());
      }
      boot.stopBootProgress();
      boot.setBootStatus('Loading model…', 'Checking weights');
      boot.stopBootProgress();
      return results;
    });
    for (const position of positions.slice(1)) assert.deepEqual(position, positions[0], `Loading anchors stay fixed at ${width}px`);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: `.cache/demo-run-coherence/loading-${width}.png` });
    await page.evaluate(() => { document.querySelector('#boot-overlay').hidden = true; });
  }
}

export async function checkPendingExample(page) {
  let release;
  const held = new Promise(resolve => { release = resolve; });
  let arrived;
  const requested = new Promise(resolve => { arrived = resolve; });
  await page.route('**/assets/activity-example.json', async route => {
    arrived();
    await held;
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }).catch(() => {});
  });
  await page.locator('#run-details > summary').click();
  await page.click('#activity-example');
  await requested;
  await page.locator('#activation-record').uncheck();
  await page.fill('#prompt-input', 'A new live answer replaces the pending example.');
  await page.click('#run-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Complete');
  release();
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.equal(await page.locator('#output-text').textContent(), 'Contract generation passed.');
  assert.equal(await page.locator('#activation-player').isVisible(), false);
  assert.equal(await page.locator('#activity-source').isVisible(), false);
  assert.equal(await page.locator('#activity-example').isEnabled(), true);
  await page.unroute('**/assets/activity-example.json');
}

// A delayed first-visit download must never steal the next user action.
export async function checkPendingInitialReplay(page) {
  let release, arrived;
  const held = new Promise(resolve => { release = resolve; });
  const requested = new Promise(resolve => { arrived = resolve; });
  await page.route('**/assets/activity-example.json', async route => {
    arrived(); await held;
    await route.fulfill({ status: 404, body: 'unavailable' }).catch(() => {});
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await requested;
  await page.waitForFunction(() => document.querySelector('#boot-overlay').hidden);
  await page.fill('#prompt-input', 'Keep my question');
  release();
  await page.waitForLoadState('networkidle');
  assert.equal(await page.inputValue('#prompt-input'), 'Keep my question');
  assert.equal(await page.locator('#output-text').textContent(), '');
  assert.equal(await page.locator('#activation-player').isVisible(), false);
  assert.equal(await page.locator('#activity-source').isVisible(), false);
  assert.equal(await page.evaluate(() => __demoContract.calls.length), 0);
  await page.unroute('**/assets/activity-example.json');
  await page.route('**/assets/activity-example.json', route => route.fulfill({ status: 404, body: 'unavailable' }));
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelector('#activity-availability').textContent.includes('unavailable'));
  assert.equal(await page.locator('#activity-availability').isVisible(), true, 'Failed replay opening stays actionable');
  assert.equal(await page.locator('#output-text').textContent(), '', 'Missing evidence never becomes an invented answer');
  assert.equal(await page.locator('#run-btn').isEnabled(), true, 'Live use remains available after replay download failure');
  await page.unroute('**/assets/activity-example.json');
}
