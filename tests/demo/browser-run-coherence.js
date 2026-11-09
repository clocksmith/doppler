import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

// Run presentation only: no model inference or GPU qualification.
export async function checkAutomaticReplay(page) {
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === 'Hello!'
    && document.querySelector('#activation-play').textContent === 'Play', null, { timeout: 120000 });
  assert.equal(await page.locator('#sample-run-btn').count(), 0);
  assert.equal(await page.locator('#inspect-recorded-run').count(), 0);
  assert.equal(await page.locator('#chat-thread .chat-message-text').first().textContent(), 'Say hello in one word.');
  assert.equal(await page.locator('#recorded-run-note').textContent(), 'Recorded replay · Qwen 0.8B');
  assert.equal(await page.locator('#recording-details').isVisible(), false);
  assert.equal(await page.locator('#activation-player').isVisible(), true);
  await fs.mkdir('.cache/demo-first-replay', { recursive: true });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.screenshot({ path: `.cache/demo-first-replay/replay-${width}.png`, fullPage: true });
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
  assert.equal(await page.evaluate(() => __demoContract.calls.length), 0, 'Replay never runs inference');
  const defaultPrompt = await page.evaluate(async () => {
    const entry = document.querySelector('script[src$="/demo.js"]').src;
    return (await (await fetch(new URL('examples.json', entry))).json()).illustrative.prompt;
  });
  assert.equal(await page.inputValue('#prompt-input'), defaultPrompt, 'Replay preserves the next-request draft');
  await page.locator('#activation-scrub').fill('0');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === '');
  await page.locator('#activation-scrub').fill('17');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === 'Hello');
  await page.locator('#activation-scrub').fill('19');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === 'Hello!');
  assert.equal(await page.locator('#token-inspector-view').isVisible(), true);
  await page.locator('#activation-play').click();
  await page.fill('#prompt-input', 'My next question');
  assert.equal(await page.locator('#activation-play').textContent(), 'Play', 'Typing pauses automatic playback');
  await page.fill('#prompt-input', defaultPrompt);
  await page.locator('#activation-scrub').fill('19');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent === 'Hello!');
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
