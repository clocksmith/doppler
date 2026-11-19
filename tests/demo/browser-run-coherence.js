import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

// Count attempted downloads as well as completed requests, including cached responses.
export function watchFullReplay(page) {
  let count = 0;
  page.on('request', request => {
    if (new URL(request.url()).pathname.endsWith('/assets/activity-example.json.gz')) count++;
  });
  return () => count;
}

export async function checkPreviewAndReplay(page, fullReplayRequests) {
  const sample = await page.evaluate(async () => (await import(new URL('data/sample-inspection.js', document.querySelector('script[src$="/demo.js"]').src))).SAMPLE_INSPECTION_RECEIPT);
  const firstPosition = String(sample.inputTokenIds.length - 1);
  await page.locator('.token-chip').first().waitFor();
  assert.equal(await page.locator('.token-chip[aria-pressed="true"]').count(), 0);
  assert.equal(await page.locator('#preview-word').textContent(), '“appears”');
  assert.equal(await page.locator('#preview-bars span[style]').count(), 24);
  assert.equal(await page.locator('#token-inspector-card-container').isVisible(), false);
  assert.equal(await page.inputValue('#prompt-input'), '', 'The example prompt appears once in the conversation');
  assert.equal(await page.locator('#chat-thread .chat-message-text').first().textContent(), sample.prompt);
  assert.match(await page.locator('#recorded-run-note').textContent(), /recorded/i);
  await page.locator('.token-chip').first().click();
  assert.equal(await page.locator('.token-inspector-badge').textContent(), '"The"');
  assert.equal(await page.locator('#preview-word').textContent(), '“The”', 'The visualization follows the chosen token');
  const firstToken = page.locator('.token-chip').first();
  assert.equal(await firstToken.getAttribute('title'), null);
  await page.getByRole('button', { name: 'Close alternatives' }).click();
  assert.equal(await firstToken.getAttribute('aria-expanded'), 'false');
  await firstToken.press('Enter');
  assert.equal(await firstToken.getAttribute('aria-expanded'), 'true');
  assert.equal(fullReplayRequests(), 0, 'Opening the preview and exploring tokens both avoid the full recording download');
  assert.equal(await page.locator('#activation-player').isVisible(), false);
  assert.match(await page.locator('#activity-example').textContent(), /23 MB/);

  await fs.mkdir('.cache/demo-first-replay', { recursive: true });
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: `.cache/demo-first-replay/preview-${width}.png`, fullPage: true });
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.locator('#activity-example').click();
  await page.locator('#activation-player').waitFor({state:'visible'});
  assert.equal(fullReplayRequests(), 1, 'The full recording loads on explicit request');
  assert.equal(await page.inputValue('#activation-position'), firstPosition, 'The full replay opens at the token selected in the preview');
  assert.equal(await page.evaluate(() => __demoContract.calls.length), 0, 'Replay never runs inference');
  assert.equal(await page.inputValue('#prompt-input'), '', 'Opening a replay leaves the composer free for a new question');
  const finalPosition = await page.locator('#activation-scrub').getAttribute('max');
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

  assert.equal(await page.getByRole('button', { name: 'Replay answer', exact: true }).count(), 1);
  assert.match(await page.locator('#activation-caption').textContent(), /Answer token \d+ \/ \d+ · Activity capture \d+ \/ \d+/);
  await page.locator('#replay-link').click();
  await page.locator('#activation-play').click();
  await page.waitForFunction(() => document.querySelector('#activation-play').textContent === 'Pause');
  await page.fill('#prompt-input', 'My next question');
  assert.equal(await page.locator('#activation-play').textContent(), 'Replay answer', 'Typing pauses automatic playback');
  await page.fill('#prompt-input', '');
  await page.locator('#activation-scrub').fill(finalPosition);
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text, sample.outputText);
}

// A browser without WebGPU cannot run a model, but the recorded example needs none.
export async function checkWithoutWebGpu(page, fullReplayRequests, { hasApi = false } = {}) {
  await page.waitForFunction(() => document.querySelector('#boot-overlay').hidden, null, { timeout: 60000 });
  assert.equal(await page.evaluate(() => 'gpu' in navigator && Boolean(navigator.gpu)), hasApi, 'The page has the WebGPU state under test');
  assert.equal(await page.locator('#boot-error').isVisible(), false, 'Missing WebGPU is not a startup failure');
  assert.match(await page.locator('#model-select-detail').textContent(), /can’t run models here: WebGPU isn’t available\. You can explore the recorded answer\./);
  assert.equal(await page.locator('#model-select').isVisible(), false, 'No picker names a model that cannot run');
  assert.equal(await page.locator('#model-select-action').isDisabled(), true);
  assert.equal(await page.locator('#webgpu-retry').isVisible(), true, 'The reader can ask the browser again');
  await page.waitForFunction(() => document.querySelector('#output-text').textContent.startsWith('The sky appears blue'), null, { timeout: 120000 });
  assert.match(await page.locator('#recorded-run-note').textContent(), /recorded/i);
  assert.equal(fullReplayRequests(), 0);
  await page.locator('.token-chip').first().click();
  assert(await page.locator('.token-candidates-list').isVisible(), 'The preview works without WebGPU');
  await page.locator('#activity-example').click();
  await page.locator('#activation-player').waitFor();
  assert.equal(fullReplayRequests(), 1, 'Full playback also works without WebGPU');
  await page.fill('#prompt-input', 'Can this browser answer?');
  assert.equal(await page.locator('#run-btn').isDisabled(), true, 'Send cannot start a model here');
  assert.match(await page.locator('#send-hint').textContent(), /Live inference is not available in this browser/, 'A disabled Send says why in words');
  await page.locator('#live-precision-run').evaluate(button => { const details = button.closest('details'); if (details) details.open = true; });
  await page.locator('#live-precision-run').click();
  assert.match(await page.locator('#live-precision-status').textContent(), /nothing to compare here/, 'Compare says why it cannot run');
  assert.equal(await page.evaluate(() => __demoContract.loads.length), 0);
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
  await page.reload({ waitUntil: 'networkidle' });
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

// A failed download preserves the working preview and a question the visitor drafted.
export async function checkReplayDownloadFailure(page) {
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.token-chip').first().waitFor();
  const preview = await page.locator('#output-text').textContent();
  await page.fill('#prompt-input', 'Keep my question');
  await page.route('**/assets/activity-example.json', route => route.fulfill({ status: 404, body: 'unavailable' }));
  await page.click('#activity-example');
  await page.waitForFunction(() => document.querySelector('#activity-availability').textContent.includes('unavailable'));
  assert.equal(await page.locator('#activity-availability').isVisible(), true);
  assert.equal(await page.locator('#output-text').textContent(), preview);
  assert.equal(await page.inputValue('#prompt-input'), 'Keep my question');
  assert.equal(await page.locator('#run-btn').isEnabled(), true);
  assert.equal(await page.locator('#activity-example').isEnabled(), true);
  await page.unroute('**/assets/activity-example.json');
}
