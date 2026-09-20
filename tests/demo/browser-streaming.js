import assert from 'node:assert/strict';

export async function checkDemoStreaming(page) {
  await page.locator('#set-word-quality').uncheck();
  await page.locator('#xray-toggle-all').uncheck();
  const url = page.url();
  let navigations = 0;
  const onNavigation = () => { navigations++; };
  page.on('framenavigated', onNavigation);

  for (const outcome of ['complete', 'stop', 'error']) {
    await page.evaluate(() => { __demoContract.streamNext = true; });
    await page.fill('#prompt-input', `Streaming ${outcome}`);
    await page.click('#run-btn');
    await page.evaluate(() => {
      const output = document.querySelector('#output-text');
      globalThis.__streamNodes = { output, text: output.firstChild, history: document.querySelector('#chat-thread').firstChild };
      __demoContract.emitTokens([1]);
    });
    await page.waitForFunction(() => document.querySelector('#output-text').textContent === 'Contract');
    await page.waitForFunction(() => document.querySelectorAll('.token-chip').length === 1);
    assert.equal(await page.locator('#token-inspector-view').isVisible(), true);
    assert.equal(await page.locator('.token-chip').first().textContent(), 'Contract');
    assert.equal(await page.textContent('#output-phase'), 'Generating');
    assert.equal(await page.locator('#export-btn').isDisabled(), true);
    assert.equal(await page.locator('#stop-btn').isVisible(), true);
    assert.equal(await page.locator('#live-assistant-message').getAttribute('aria-busy'), 'true');

    await page.evaluate(() => __demoContract.emitTokens([2, 3, ...Array(80).fill(4)]));
    await page.waitForFunction(() => document.querySelector('#output-text').textContent.includes('Another line.'));
    await page.waitForFunction(() => document.querySelectorAll('.token-chip').length === 83);
    await page.evaluate(() => { document.querySelector('.chat-surface').scrollTop = 0; });
    await page.evaluate(() => __demoContract.emitTokens([5]));
    await page.waitForFunction(() => document.querySelector('#output-text').textContent.endsWith('🙂'));
    assert.equal(await page.locator('.chat-surface').evaluate((el) => el.scrollTop), 0, 'Streaming respects a reader scrolling up');

    if (outcome === 'complete') {
      await page.evaluate(() => __demoContract.completeStream());
      await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Complete'));
      assert.equal(await page.locator('#export-btn').isEnabled(), true);
    } else if (outcome === 'stop') {
      await page.click('#stop-btn');
      await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Stopped');
    } else {
      await page.evaluate(() => __demoContract.failStream());
      await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Error: Contract stream failure');
    }
    const text = await page.textContent('#output-text');
    assert.ok(text.startsWith('Contract generation passed.'));
    assert.ok(text.endsWith('🙂'));
    assert.equal(await page.locator('#live-assistant-message').getAttribute('aria-busy'), 'false');
    assert.equal(await page.evaluate(() => {
      const nodes = globalThis.__streamNodes;
      return nodes.output === document.querySelector('#output-text')
        && nodes.history === document.querySelector('#chat-thread').firstChild;
    }), true, 'Streaming and completion preserve the output and conversation containers');
    await page.evaluate(() => __demoContract.emitTokens([1, 2, 3]));
    await page.evaluate(() => new Promise(requestAnimationFrame));
    assert.equal(await page.textContent('#output-text'), text, 'Late events cannot overwrite a settled answer');
    if (outcome !== 'complete') assert.equal(await page.locator('#export-btn').isDisabled(), true);
    await page.fill('#prompt-input', 'Next turn');
    await page.click('#run-btn');
    await page.waitForFunction(() => document.querySelector('#output-phase').textContent.startsWith('Complete'));
    assert.ok((await page.textContent('#chat-thread')).includes(text.trim()), 'The next run preserves partial and complete answers');
    await page.click('#clear-history-btn');
  }

  // Deterministically drive animation frames to check batching, Unicode repair,
  // final flush, and cancellation without relying on wall-clock timing.
  const renderer = await page.evaluate(async () => {
    const base = location.pathname.startsWith('/doppler/') ? '/doppler' : '';
    const { beginChatTurn, createOutputStream, clearOutput } = await import(`${base}/demo/output.js`);
    const requestFrame = window.requestAnimationFrame;
    const cancelFrame = window.cancelAnimationFrame;
    const frames = new Map();
    let nextFrame = 0;
    let decodes = 0;
    window.requestAnimationFrame = (callback) => { frames.set(++nextFrame, callback); return nextFrame; };
    window.cancelAnimationFrame = (id) => frames.delete(id);
    try {
      beginChatTurn([{ role: 'user', content: 'Unicode' }]);
      const signal = new AbortController();
      const stream = createOutputStream((ids) => {
        decodes++;
        return ids.length < 4 ? 'Hi \uFFFD' : 'Hi 🙂';
      }, signal.signal);
      stream.push(1);
      stream.push(2);
      stream.push(3);
      const scheduled = frames.size;
      const paint = [...frames.values()][0];
      frames.clear();
      paint();
      const partial = document.querySelector('#output-text').textContent;
      stream.push(4);
      signal.abort();
      stream.push(5);
      const stopped = stream.finish('late result');
      const pending = frames.size;
      const final = createOutputStream(() => 'unused');
      final.push(1);
      final.finish('Authoritative final text');
      return { scheduled, partial, stopped, pending, decodes, finalPending: frames.size, final: document.querySelector('#output-text').textContent };
    } finally {
      window.requestAnimationFrame = requestFrame;
      window.cancelAnimationFrame = cancelFrame;
      clearOutput();
    }
  });
  assert.deepEqual(renderer, {
    scheduled: 1, partial: 'Hi ', stopped: 'Hi 🙂', pending: 0, decodes: 2,
    finalPending: 0, final: 'Authoritative final text',
  });
  assert.equal(page.url(), url);
  assert.equal(navigations, 0, 'No page reload or navigation during streaming');
  page.off('framenavigated', onNavigation);
}
