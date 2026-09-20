import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { checkDemoStreaming } from './browser-streaming.js';

async function completeRun(page, prompt) {
  await page.fill('#prompt-input', prompt);
  await page.click('#run-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Complete'));
}

async function exportedReport(page) {
  const download = page.waitForEvent('download');
  await page.click('#export-btn');
  return JSON.parse(await readFile(await (await download).path(), 'utf8'));
}

async function importReport(page, value) {
  const chooser = page.waitForEvent('filechooser');
  await page.click('#import-btn');
  await (await chooser).setFiles({
    name: 'receipt.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(value)),
  });
}

async function assertControlContrast(page) {
  const failures = await page.evaluate(() => {
    const rgba = (color) => color.match(/[\d.]+/g).map(Number);
    const luminance = (rgb) => rgb.slice(0, 3).map((value) => {
      const channel = value / 255;
      return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    const blend = (foreground, background, alpha = foreground[3] ?? 1) => foreground.slice(0, 3)
      .map((value, index) => value * alpha + background[index] * (1 - alpha));
    const backdrop = (element) => {
      if (!element) return [255, 255, 255];
      return blend(rgba(getComputedStyle(element).backgroundColor), backdrop(element.parentElement));
    };
    const errors = [];
    for (const element of document.querySelectorAll('button, summary, input, select, textarea, a')) {
      if (!element.checkVisibility({ checkVisibilityCSS: true })) continue;
      const style = getComputedStyle(element);
      let opacity = 1;
      for (let parent = element; parent; parent = parent.parentElement) opacity *= Number(getComputedStyle(parent).opacity);
      const bg = backdrop(element);
      const foreground = blend(rgba(style.color), bg, opacity);
      const [lo, hi] = [luminance(foreground), luminance(bg)].sort((a, b) => a - b);
      const ratio = (hi + 0.05) / (lo + 0.05);
      const threshold = element.type === 'checkbox' ? 3 : 4.5;
      if (ratio < threshold) errors.push({ id: element.id || element.textContent.trim().slice(0, 35), ratio });
    }
    return errors;
  });
  assert.deepEqual(failures, [], 'Visible controls must have readable foreground/background pairs');
}

export async function checkDemoControls(page) {
  await page.locator('#chat-controls > summary').click();
  assert.equal(await page.locator('#token-inspector-view').isVisible(), true);
  assert.equal(await page.locator('.token-chip').count(), 3);
  assert.equal(await page.locator('#output-text').isVisible(), false, 'Token evidence replaces the duplicate plain answer');
  assert.equal(await page.locator('.token-chip').first().evaluate((element) => (
    getComputedStyle(element).textDecorationLine.includes('underline')
  )), true, 'Generated tokens use confidence underlines instead of boxes');
  const desktopGeometry = await page.evaluate(() => {
    const box = (selector) => document.querySelector(selector).getBoundingClientRect();
    const select = box('#set-max-tokens');
    const actions = ['#sample-run-btn', '#shuffle-btn', '#run-btn'].map(box);
    const toolbar = box('.chat-toolbar');
    const optionItems = [
      '#token-inspector-toggle',
      '#chat-controls > summary',
      '#xray-toggle-all',
      '#set-word-quality',
      '#settings-toggle',
    ].map((selector) => box(selector));
    const statusItems = ['#output-phase', '#output-toks', '#clear-history-btn'].map((selector) => box(selector));
    return {
      composerBottoms: [select.bottom, ...actions.map((rect) => rect.bottom)],
      toolbarHeight: toolbar.height,
      optionCenters: optionItems.map((rect) => rect.top + rect.height / 2),
      statusCenters: statusItems.map((rect) => rect.top + rect.height / 2),
      optionsBottom: Math.max(...optionItems.map((rect) => rect.bottom)),
      statusTop: Math.min(...statusItems.map((rect) => rect.top)),
    };
  });
  assert.ok(
    Math.max(...desktopGeometry.composerBottoms) - Math.min(...desktopGeometry.composerBottoms) <= 2,
    'Composer controls share one bottom edge'
  );
  assert.ok(desktopGeometry.toolbarHeight <= 116, 'Expanded run options stay compact');
  assert.ok(
    Math.max(...desktopGeometry.optionCenters) - Math.min(...desktopGeometry.optionCenters) <= 2,
    'Expanded run options share one center line'
  );
  assert.ok(
    Math.max(...desktopGeometry.statusCenters) - Math.min(...desktopGeometry.statusCenters) <= 2,
    'Completion status and clear action share one center line'
  );
  assert.ok(desktopGeometry.optionsBottom <= desktopGeometry.statusTop, 'Status rail cannot overlap run options');
  await page.locator('#xray-toggle-all').focus();
  await page.keyboard.press('Space');
  assert.equal(await page.locator('#xray-toggle-all').isChecked(), true);
  assert.equal(await page.locator('#runtime-notice').isVisible(), true);
  assert.equal(await page.locator('#settings-panel').isVisible(), false);
  await page.locator('#xray-shell > summary').click();
  assert.equal(await page.locator('#xray-container .xray-section').count(), 5, 'X-Ray can inspect an existing receipt');
  await assertControlContrast(page);

  await page.locator('#set-word-quality').check();
  assert.deepEqual(await page.evaluate(() => [
    localStorage.getItem('doppler.demo.xray-enabled'),
    localStorage.getItem('doppler.demo.word-quality-enabled'),
  ]), ['true', 'true']);
  await completeRun(page, 'Inspect both options.');
  assert.equal(await page.evaluate(() => __demoContract.calls.at(-1).policyId), 'demo/deep-xray');
  await page.locator('#xray-toggle-all').uncheck();
  assert.equal(await page.locator('#xray-shell').isVisible(), false);
  await completeRun(page, 'Inspect word quality.');
  assert.equal(await page.evaluate(() => __demoContract.calls.at(-1).policyId), 'demo/guided-quality');
  assert.equal(await page.locator('#word-quality-output .word-quality').count(), 1);
  await page.locator('#set-word-quality').uncheck();
  assert.equal(await page.locator('#runtime-notice').isVisible(), false);

  await page.locator('#settings-toggle').click();
  assert.equal(await page.locator('#settings-panel').isVisible(), true);
  await page.locator('.profile-details > summary').click();
  for (const id of ['set-trace', 'set-batch-max-tokens', 'set-readback', 'set-kv-dtype', 'set-kv-max-seq', 'set-log-level']) {
    assert.equal(await page.locator(`#${id}`).isDisabled(), true, `${id} is explicitly profile-owned`);
  }
  await assertControlContrast(page);
  const recordedSettings = (await exportedReport(page)).settings;
  for (const profile of ['profiles/throughput', 'profiles/verbose-trace', 'profiles/production', 'profiles/low-memory', 'profiles/trace-layers', 'profiles/default']) {
    await page.selectOption('#set-profile', profile);
    await page.waitForFunction(() => !document.querySelector('#set-profile').disabled);
    assert.equal(await page.inputValue('#set-profile'), profile);
    assert.equal(await page.evaluate(() => __demoContract.loads.at(-1).runtimeProfile), profile, 'Profile changes reload the active model');
    assert.deepEqual((await exportedReport(page)).settings, recordedSettings, 'A profile change cannot rewrite a completed run');
  }
  await page.evaluate(() => { __demoContract.failLoad = true; });
  await page.selectOption('#set-profile', 'profiles/production');
  await page.waitForFunction(() => !document.querySelector('#set-profile').disabled);
  assert.equal(await page.inputValue('#set-profile'), 'profiles/default');
  assert.equal(await page.locator('#settings-error').isVisible(), true, 'Profile failure is visible');
  await page.evaluate(() => { __demoContract.failLoad = false; });

  await page.fill('#set-top-k', '0');
  await page.fill('#prompt-input', 'Preserve invalid input.');
  const beforeInvalid = await page.evaluate(() => __demoContract.calls.length);
  await page.click('#run-btn');
  assert.equal(await page.evaluate(() => __demoContract.calls.length), beforeInvalid);
  assert.equal(await page.inputValue('#prompt-input'), 'Preserve invalid input.');
  await page.fill('#set-top-k', '7');
  await page.selectOption('#set-max-tokens', '128');
  await page.fill('#set-temperature', '0.4');
  await page.fill('#set-top-p', '0.8');
  await completeRun(page, 'Apply sampling.');
  assert.deepEqual(await page.evaluate(() => {
    const { temperature, topK, topP, maxTokens } = __demoContract.calls.at(-1).generation;
    return { temperature, topK, topP, maxTokens };
  }), { temperature: 0.4, topK: 7, topP: 0.8, maxTokens: 128 });
  await page.locator('#settings-toggle').click();
  assert.equal(await page.locator('#settings-panel').isVisible(), false);

  const previousPrompt = await page.inputValue('#prompt-input');
  await page.click('#shuffle-btn');
  assert.notEqual(await page.inputValue('#prompt-input'), previousPrompt);
  await page.fill('#prompt-input', 'Keyboard');
  await page.locator('#prompt-input').press('Shift+Enter');
  assert.equal(await page.inputValue('#prompt-input'), 'Keyboard\n');
  await page.locator('#prompt-input').press('Enter');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Complete'));

  const report = await exportedReport(page);
  assert.equal(report.output, 'Contract generation passed.');
  await importReport(page, { schema: 'invalid', output: 'reject' });
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Import failed:'));
  await importReport(page, { ...report, output: 'Imported receipt output.' });
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Imported report');
  assert.equal((await exportedReport(page)).output, 'Imported receipt output.');
  await completeRun(page, 'Fresh receipt.');
  assert.equal((await exportedReport(page)).output, 'Contract generation passed.', 'New generation replaces imported evidence');

  await page.evaluate(() => { __demoContract.blockNext = true; });
  await page.fill('#prompt-input', 'Cancel this generation.');
  await page.click('#run-btn');
  for (const id of ['model-select', 'model-select-action', 'model-select-remove', 'clear-history-btn', 'import-btn', 'set-profile', 'xray-toggle-all', 'set-word-quality']) {
    assert.equal(await page.locator(`#${id}`).isDisabled(), true, `${id} is locked during execution`);
  }
  await page.click('#stop-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Stopped');
  assert.equal(await page.locator('#stop-btn').isVisible(), false);
  assert.equal(await page.locator('#run-btn').isEnabled(), true);
  assert.equal(await page.inputValue('#prompt-input'), 'Cancel this generation.');
  assert.equal(await page.locator('#export-btn').isDisabled(), true);
  await page.evaluate(() => { __demoContract.blockNext = true; __demoContract.resolveOnAbort = true; });
  await page.click('#run-btn');
  await page.click('#stop-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Stopped');
  assert.equal(await page.locator('#export-btn').isDisabled(), true, 'Cancellation cannot publish a late receipt');
  await page.click('#clear-history-btn');
  assert.equal(await page.locator('.chat-message-text').count(), 0);
  assert.equal(await page.locator('#clear-history-btn').isEnabled(), true, 'Loaded model state remains resettable');

  await checkDemoStreaming(page);

  await page.evaluate(() => {
    const originalFetch = window.fetch;
    window.fetch = (...args) => {
      if (String(args[0]).endsWith('/f16-precision-collapse/manifest.json')) {
        window.fetch = originalFetch;
        return Promise.reject(new Error('Contract evidence failure'));
      }
      return originalFetch(...args);
    };
  });
  await page.click('#precision-replay-toggle');
  await page.waitForFunction(() => document.querySelector('#precision-replay-status').textContent.includes('Contract evidence failure'));
  assert.equal(await page.locator('#precision-replay-toggle').isEnabled(), true);
  await page.click('#precision-replay-toggle');
  assert.equal(await page.locator('#precision-replay-panel').isVisible(), false);
  await page.click('#precision-replay-toggle');
  await page.waitForSelector('#precision-replay-table-body tr');
  for (const button of await page.locator('[data-precision-mode]').all()) {
    await button.focus();
    await page.keyboard.press('Enter');
    assert.equal(await button.getAttribute('aria-pressed'), 'true');
    assert.equal(await button.evaluate((element) => element === document.activeElement), true);
  }
  const promptOptions = await page.locator('#precision-replay-prompt-select option').evaluateAll((options) => options.map((option) => option.value));
  const initialPrompt = await page.inputValue('#precision-replay-prompt-select');
  await page.evaluate(() => {
    const originalFetch = window.fetch;
    globalThis.__releaseSlice = null;
    window.fetch = (...args) => {
      if (String(args[0]).includes('/curated/slices/')) {
        window.fetch = originalFetch;
        return new Promise((resolve) => {
          globalThis.__releaseSlice = () => resolve(originalFetch(...args));
        });
      }
      return originalFetch(...args);
    };
  });
  await page.selectOption('#precision-replay-prompt-select', promptOptions.find((value) => value !== initialPrompt));
  await page.waitForFunction(() => typeof __releaseSlice === 'function');
  await page.selectOption('#precision-replay-prompt-select', initialPrompt);
  await page.evaluate(() => __releaseSlice());
  await page.waitForLoadState('networkidle');
  assert.equal(await page.locator('#precision-replay-table-body').getAttribute('data-prompt-id'), initialPrompt, 'A late response cannot replace the selected prompt');
  for (const value of promptOptions) {
    await page.selectOption('#precision-replay-prompt-select', value);
    await page.waitForFunction((id) => document.querySelector('#precision-replay-table-body').dataset.promptId === id, value);
  }
  await page.click('#precision-replay-use-prompt');
  assert.equal(await page.inputValue('#prompt-input'), await page.locator('#precision-replay-prompt').textContent());
  await assertControlContrast(page);
  await page.click('#precision-replay-toggle');
  assert.equal(await page.locator('#precision-replay-panel').isVisible(), false);

  await page.locator('#model-browser > summary').click();
  await page.locator('[data-model-id="second-model"]').click();
  assert.equal(await page.inputValue('#model-select'), 'second-model');
  assert.equal(await page.locator('[data-model-id="second-model"]').getAttribute('aria-pressed'), 'true');
  await page.evaluate(() => { __demoContract.failLoad = true; });
  await page.click('#model-select-action');
  await page.waitForFunction(() => document.querySelector('#status-text').textContent.startsWith('Load failed:'));
  assert.equal(await page.locator('#model-select-action').isEnabled(), true);
  await page.evaluate(() => { __demoContract.failLoad = false; });
  await page.click('#model-select-action');
  await page.waitForFunction(() => document.querySelector('#model-select-action').textContent === 'Loaded');
  assert.equal(await page.locator('#model-select-action').isDisabled(), true);
  await page.click('#model-select-remove');
  await page.locator('#remove-model-dialog button[value="cancel"]').click();
  assert.equal(await page.evaluate(() => __demoContract.removals), 0);
  await page.click('#model-select-remove');
  await page.keyboard.press('Escape');
  assert.equal(await page.evaluate(() => __demoContract.removals), 0);
  await page.click('#model-select-remove');
  await page.click('#remove-model-confirm-btn');
  await page.waitForFunction(() => __demoContract.removals === 1);
  assert.equal(await page.locator('#run-btn').isDisabled(), true);
  assert.equal(await page.locator('#model-select-remove').isVisible(), false);
  assert.equal(await page.locator('#status-text').textContent(), 'Select model');

  await page.evaluate(() => {
    const event = new Event('beforeinstallprompt', { cancelable: true });
    event.prompt = () => { globalThis.__installPromptUsed = true; };
    event.userChoice = Promise.resolve({ outcome: 'dismissed' });
    window.dispatchEvent(event);
  });
  await page.locator('#install-btn').hover();
  await assertControlContrast(page);
  await page.click('#install-btn');
  assert.equal(await page.evaluate(() => __installPromptUsed), true);
  assert.equal(await page.locator('#install-btn').isVisible(), false);

  for (const width of [360, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.emulateMedia({ colorScheme: width === 390 ? 'dark' : 'light' });
    await assertControlContrast(page);
    const horizontalLayout = await page.evaluate(() => ({
      fits: document.documentElement.scrollWidth <= innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      offenders: [...document.querySelectorAll('body *')]
        .filter((element) => element.getBoundingClientRect().right > innerWidth + 1)
        .slice(0, 8)
        .map((element) => element.id || element.className || element.tagName),
    }));
    assert.equal(
      horizontalLayout.fits,
      true,
      `No page overflow at ${width}px: ${JSON.stringify(horizontalLayout)}`
    );
  }
  for (const button of await page.locator('button:visible:not(:disabled)').all()) {
    await button.hover();
    await assertControlContrast(page);
  }
  process.stderr.write('Demo controls: sampling, profiles, diagnostics, keyboard, stop, receipts, removal, install handoff, contrast and responsive layouts passed (mocked execution).\n');
}
