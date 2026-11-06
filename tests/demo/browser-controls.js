import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { checkDemoStreaming } from './browser-streaming.js';

async function completeRun(page, prompt) {
  await page.fill('#prompt-input', prompt);
  await page.click('#run-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Complete'));
}

async function waitForPhase(page, phase) {
  await page.waitForFunction((expected) => document.querySelector('#output-phase').textContent === expected, phase);
}

async function branchAt(page, index) {
  await page.locator(`.token-chip[data-token-index="${index}"]`).click();
  await page.locator('.token-candidate-branch').first().click();
}

// Branch replays exact token IDs, keeps the original behind a switcher,
// reuses an existing alternative, and keeps a cancelled nested branch.
async function checkBranching(page) {
  // The answer as read, without the fork switchers' own labels.
  const streamText = () => page.locator('#token-stream-container').evaluate((element) => {
    const copy = element.cloneNode(true);
    copy.querySelectorAll('.branch-switch').forEach((node) => node.remove());
    return copy.textContent;
  });
  await branchAt(page, 1);
  await waitForPhase(page, 'Branch complete');
  let call = await page.evaluate(() => __demoContract.calls.at(-1));
  assert.deepEqual(call.inputTokenIds, [100, 101, 1, 5], 'A branch replays the prompt IDs, the prefix IDs and the chosen ID');
  assert.equal(call.generation.maxTokens, 254, 'A branch keeps the answer within the output limit');
  assert.equal(call.policyId, 'demo/guided-quality');
  assert.match(await streamText(), /Contract 🙂\nAnother line\./);
  assert.equal(await page.locator('.branch-switch-count').textContent(), '2/2');
  await page.locator('.branch-switch-btn').first().click();
  assert.equal(await page.locator('.branch-switch-count').textContent(), '1/2');
  assert.match(await streamText(), /Contract generation passed\./);
  assert.match(await page.locator('#output-text').textContent(), /Contract generation passed\./);
  const calls = await page.evaluate(() => __demoContract.calls.length);
  await branchAt(page, 1);
  await waitForPhase(page, 'Showing existing branch');
  assert.equal(await page.evaluate(() => __demoContract.calls.length), calls, 'An existing alternative is reopened, not regenerated');
  await page.evaluate(() => { __demoContract.streamNext = true; });
  await branchAt(page, 2);
  await page.waitForFunction((count) => __demoContract.calls.length === count + 1, calls);
  await page.evaluate(() => __demoContract.emitTokens([4]));
  await page.locator('#stop-btn').click();
  await waitForPhase(page, 'Branch stopped');
  call = await page.evaluate(() => __demoContract.calls.at(-1));
  assert.deepEqual(call.inputTokenIds, [100, 101, 1, 5, 5], 'A nested branch replays its parent branch exactly');
  assert.equal(await page.locator('.branch-switch').count(), 2, 'Each fork on the shown path keeps a switcher');
  assert.match(await streamText(), /🙂 🙂\nAnother line\./, 'A stopped branch keeps its partial answer');
  assert.equal(await page.locator('#export-btn').isDisabled(), true, 'A stopped branch publishes no receipt');
  await page.locator('.branch-switch').first().locator('.branch-switch-btn').first().click();
  assert.match(await streamText(), /Contract generation passed\./);
  assert.equal(await page.locator('#export-btn').isDisabled(), false, 'The original answer republishes its receipt');
}

async function exportedReport(page) {
  if (!await page.locator('#export-btn').isVisible()) {
    await page.locator('#run-details > summary').click();
  }
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
      if (ratio < threshold) errors.push({
        id: element.id || element.textContent.trim().slice(0, 35), ratio,
        color: style.color, background: style.background, opacity, disabled: element.disabled === true,
      });
    }
    return errors;
  });
  assert.deepEqual(failures, [], 'Visible controls must have readable foreground/background pairs');
}

export async function checkDemoControls(page) {
  assert.equal(await page.evaluate(() => typeof globalThis.DOPPLER?.log?.info), 'function');
  assert.equal(await page.locator('#settings-panel').isVisible(), false);
  assert.equal(await page.locator('#status').isVisible(), false, 'Model readiness stays in setup, without repeating run progress');
  await page.locator('#settings-toggle').click();
  assert.equal(await page.locator('#settings-panel').isVisible(), true);
  assert.equal(await page.locator('#xray-toggle-all').isChecked(), false);
  assert.equal(await page.locator('#set-word-quality').count(), 0, 'Certainty has one opt-in control');
  assert.equal(await page.locator('#token-inspector-toggle, #branch-all-tokens, #branch-threshold').count(), 0, 'Branching and underlines need no opt-in controls');
  assert.equal(await page.locator('#output-prefill-bar').isVisible(), false);
  assert.equal(await page.inputValue('#set-max-tokens'), '256');
  assert.equal(await page.evaluate(() => __demoContract.calls.at(-1).policyId), 'demo/guided-quality',
    'Every live answer records alternatives');
  const firstChat = await page.evaluate(() => __demoContract.calls.at(-1).prompt);
  assert.deepEqual(firstChat, { messages: [{role:'user',content:'Run the demo contract.'}] });

  assert.match(await page.locator('#output-toks').textContent(), /Diagnostic timing/);
  assert.match(await page.locator('#runtime-notice').textContent(), /Recording token alternatives changes execution/);
  assert.equal(await page.locator('#runtime-notice').isVisible(), true);
  // Even high-likelihood tokens are underlined and branchable immediately.
  assert.equal(await page.locator('.token-chip').count(), 3, 'Every token is underlined');
  await page.locator('.token-chip[data-token-index="0"]').click();
  assert.match(await page.locator('.token-inspector-metrics').textContent(), /95.0%/);
  assert.equal(await page.locator('.token-candidate-branch').first().isEnabled(), true,
    'Clicking a high-likelihood word offers branching without a toggle');
  await page.locator('#xray-toggle-all').focus();
  await page.keyboard.press('Space');
  assert.equal(await page.locator('#xray-toggle-all').isChecked(), true);
  await page.keyboard.press('Space');
  assert.equal(await page.locator('#xray-toggle-all').isChecked(), false);
  await page.locator('#xray-toggle-all').check();
  assert.equal(await page.locator('#inspection-workspace').isVisible(), true);
  assert.equal(await page.locator('#xray-shell').isVisible(), true);
  assert.equal(await page.locator('#xray-container h3').first().textContent(), 'Where time went');
  await page.locator('#xray-container .xray-section > summary').click();
  await page.waitForFunction(() => document.querySelector('#xray-container pre').textContent.length > 0);
  assert.equal(JSON.parse(await page.locator('#xray-container pre').textContent()).outputText, 'Contract generation passed.');
  await assertControlContrast(page);
  await completeRun(page, 'Inspect GPU work.');
  const nextChat = await page.evaluate(() => __demoContract.calls.at(-1).prompt);
  assert.deepEqual(nextChat.messages.slice(0, 3), [
    {role:'user',content:'Run the demo contract.'}, {role:'assistant',content:'Contract generation passed.'},
    {role:'user',content:'Inspect GPU work.'},
  ], 'The next generation receives the previous user and assistant turns');

  assert.equal(await page.evaluate(() => __demoContract.calls.at(-1).policyId), 'demo/deep-xray');
  assert.match(await page.locator('#output-toks').textContent(), /Diagnostic timing/);
  assert.equal(await page.locator('.xray-waterfall-segment').count(), 2);
  assert.match(await page.locator('.xray-kernel-table').textContent(), /fused_ffn/);
  assert.match(await page.locator('.xray-kernel-table').textContent(), /1.96 ms/);
  assert.match(await page.locator('#xray-container').textContent(), /prefill.*not captured/);
  assert.equal(await page.locator('.xray-probability-chart').count(), 0);
  await page.locator('#xray-toggle-all').uncheck();
  await completeRun(page, 'Inspect token probability.');
  assert.equal(await page.evaluate(() => __demoContract.calls.at(-1).policyId), 'demo/guided-quality');
  assert.match(await page.locator('#output-toks').textContent(), /Diagnostic timing/);
  assert.equal(await page.locator('#token-inspector-view').isVisible(), true);
  assert.equal(await page.locator('.token-chip').count(), 3, 'All tokens keep their probability underlines');
  const colors = await page.locator('.token-chip').evaluateAll(nodes => nodes.map(n => n.style.getPropertyValue('--token-surprisal')));
  assert.ok(Number(colors[1]) > Number(colors[0]), 'A low-probability token is redder than a high-probability token');
  await checkBranching(page);
  assert.equal(await page.locator('#runtime-notice').isVisible(), true, 'Recording stays disclosed');

  assert.equal(await page.locator('#settings-panel').isVisible(), true);
  for (const id of ['set-trace', 'set-batch-max-tokens', 'set-readback', 'set-kv-dtype', 'set-kv-max-seq', 'set-log-level']) {
    assert.equal(await page.locator(`#${id}`).isDisabled(), true, `${id} is explicitly profile-owned`);
  }
  assert.equal(await page.locator('#set-kv-dtype').isVisible(), false, 'Profile-owned values unfold on demand');
  await page.locator('.profile-details > summary').click();
  assert.equal(await page.locator('#set-kv-dtype').isVisible(), true);
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

  // A tampered form still fails closed; zero is not an offered output limit.
  await page.evaluate(() => document.querySelector('#set-max-tokens').add(new Option('Invalid', '0')));
  await page.selectOption('#set-max-tokens', '0');
  await page.fill('#prompt-input', 'Preserve invalid input.');
  const beforeInvalid = await page.evaluate(() => __demoContract.calls.length);
  await page.click('#run-btn');
  assert.equal(await page.evaluate(() => __demoContract.calls.length), beforeInvalid);
  assert.equal(await page.inputValue('#prompt-input'), 'Preserve invalid input.');
  await page.selectOption('#set-max-tokens', '128');
  await page.fill('#set-temperature', '0.4');
  await page.fill('#set-top-k', '7');
  await page.fill('#set-top-p', '0.8');
  await completeRun(page, 'Apply sampling.');
  assert.deepEqual(await page.evaluate(() => {
    const { temperature, topK, topP, maxTokens } = __demoContract.calls.at(-1).generation;
    return { temperature, topK, topP, maxTokens };
  }), { temperature: 0.4, topK: 7, topP: 0.8, maxTokens: 128 });
  await page.locator('#settings-toggle').click();
  assert.equal(await page.locator('#settings-panel').isVisible(), false);

  // Each category deals its whole bank before repeating; Surprise draws from every bank.
  const { categories } = await page.evaluate(async () => {
    const entrypoint = document.querySelector('script[src$="/demo.js"]').src;
    return (await fetch(new URL('examples.json', entrypoint))).json();
  });
  for (const { label, prompts } of categories) {
    const dealt = new Set();
    for (let draw = 0; draw < prompts.length; draw++) {
      await page.locator('#prompt-suggestions button', { hasText: label }).click();
      dealt.add(await page.inputValue('#prompt-input'));
    }
    assert.deepEqual([...dealt].sort(), [...prompts].sort(), `${label} deals every prompt in its bank`);
  }
  const bank = new Set(categories.flatMap((entry) => entry.prompts));
  for (let draw = 0; draw < 3; draw++) {
    const previousPrompt = await page.inputValue('#prompt-input');
    await page.click('#shuffle-btn');
    const surprise = await page.inputValue('#prompt-input');
    assert.notEqual(surprise, previousPrompt, 'Surprise never repeats the prompt already shown');
    assert.ok(bank.has(surprise), 'Surprise draws from the category banks');
  }
  await page.fill('#prompt-input', 'Keyboard');
  await page.locator('#prompt-input').press('Shift+Enter');
  assert.equal(await page.inputValue('#prompt-input'), 'Keyboard\n');
  await page.locator('#prompt-input').press('Enter');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Complete');

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
  for (const id of ['model-select', 'model-select-action', 'model-select-remove', 'clear-history-btn', 'import-btn', 'set-profile', 'xray-toggle-all']) {
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
  const historyCount = await page.locator('.chat-message-text').count();
  await page.evaluate(() => { __demoContract.failReset = true; });
  await page.click('#clear-history-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent.includes('Contract reset failure'));
  assert.equal(await page.locator('.chat-message-text').count(), historyCount, 'Failed reset preserves conversation');
  await page.evaluate(() => { __demoContract.failReset = false; });
  await page.click('#clear-history-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Conversation and model state cleared');
  assert.equal(await page.locator('.chat-message-text').count(), 0);
  assert.equal(await page.evaluate(() => __demoContract.resets), 1);
  assert.equal(await page.locator('#clear-history-btn').isEnabled(), true, 'Loaded model remains resettable');

  await page.locator('#sample-run-btn').click();
  const sampleSettingsOpen = await page.locator('#settings-panel').isVisible();
  if (!sampleSettingsOpen) await page.click('#settings-toggle');
  assert.equal(await page.locator('#output-text').isVisible(), false, 'The underlined view replaces the duplicate formatted answer');
  assert.equal(await page.locator('#token-inspector-view').isVisible(), true, 'Every recorded answer is underlined');
  await page.locator('.token-chip').first().click();
  assert.match(await page.locator('.token-inspector-note').textContent(), /Only a live answer can branch/);
  assert.equal(await page.locator('.token-candidate-branch').first().isDisabled(), true, 'Recorded evidence cannot pretend to be a live branch');
  assert.equal(await page.locator('#runtime-notice').isVisible(), true, 'Recording stays disclosed');
  if (!sampleSettingsOpen) await page.click('#settings-toggle');

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
  assert.equal(await page.locator('#precision-replay-toggle').isVisible(), true, 'Closed replay keeps its opening control visible');
  await page.click('#precision-replay-toggle');
  await page.waitForFunction(() => document.querySelector('#precision-replay-status').textContent.includes('Contract evidence failure'));
  assert.equal(await page.locator('#precision-replay-toggle').isEnabled(), true);
  await page.click('#precision-replay-toggle');
  assert.equal(await page.locator('#precision-replay-panel').isVisible(), false);
  assert.equal(await page.locator('#precision-replay-toggle').isVisible(), true, 'Failed evidence loading remains retryable after closing');
  await page.click('#precision-replay-toggle');
  await page.locator('.precision-evidence-details > summary').click();
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

  await page.locator('#settings-toggle').click();
  if (!await page.locator('#model-select').isVisible()) {
    await page.locator('#model-picker > summary').click();
  }
  await page.locator('#model-browser > summary').click();
  await page.locator('[data-model-id="second-model"]').click();
  assert.equal(await page.inputValue('#model-select'), 'second-model');
  assert.equal(await page.locator('[data-model-id="second-model"]').getAttribute('aria-pressed'), 'true');
  await page.evaluate(() => { __demoContract.failLoad = true; });
  await page.click('#model-select-action');
  await page.waitForFunction(() => document.querySelector('#status-text').textContent.startsWith('Load failed:'));
  assert.equal(await page.locator('#model-select-action').isEnabled(), true);
  await page.locator('#settings-toggle').click();
  assert.equal(await page.locator('#status').isVisible(), false);
  assert.equal(await page.locator('#output-phase').isVisible(), true);
  assert.equal(await page.locator('#output-phase').textContent(), 'Load failed: Contract load failure', 'Load failures remain visible when setup is collapsed');
  await page.locator('#settings-toggle').click();
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
  assert.equal(await page.locator('#run-btn').textContent(), 'Send');
  assert.equal(await page.locator('#run-btn').isEnabled(), true);
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
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `No page overflow at ${width}px`);
  }
  for (const button of await page.locator('button:visible:not(:disabled)').all()) {
    await button.hover();
    await assertControlContrast(page);
  }
  process.stderr.write('Demo controls: sampling, profiles, diagnostics, keyboard, stop, receipts, removal, install handoff, contrast and responsive layouts passed (mocked execution).\n');
}
