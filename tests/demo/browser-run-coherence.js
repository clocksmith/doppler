import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

// Run presentation only: no model inference or GPU qualification.
export async function checkEmptyRun(page) {
  const defaultPrompt = await page.evaluate(async () => {
    const entry = document.querySelector('script[src$="/demo.js"]').src;
    return (await (await fetch(new URL('examples.json', entry))).json()).illustrative.prompt;
  });
  assert.equal(await page.inputValue('#prompt-input'), defaultPrompt);
  assert.equal(await page.locator('#activation-player').isVisible(), false);
  assert.equal(await page.locator('#output-text').textContent(), '');
  assert.equal(await page.locator('#recorded-run-note').isVisible(), false);
  assert.equal(await page.evaluate(() => performance.getEntriesByType('resource').some(r => r.name.endsWith('activity-example.json.gz'))), false,
    'Neither fresh nor cached-model visits fetch an unrelated recording');
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
