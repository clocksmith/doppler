// Browser check of Explore, the front page. Without a GPU it opens the recorded
// answer and its activity, walks every view, keeps one selection across them,
// saves a run and opens it again, and compares two runs. With a GPU (on by
// default when the browser gets an adapter) it also answers a question, stops an
// answer part-way, forks from a recorded alternative and compares the fork.
//
//   node scripts/check-workspace.js [page url] [--no-gpu] [--model <id>]
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from './dev.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, '.build', 'workspace');
const given = process.argv.slice(2).find((arg) => /^https?:/.test(arg));
const useGpu = !process.argv.includes('--no-gpu');
const model = process.argv.includes('--model') ? process.argv[process.argv.indexOf('--model') + 1] : 'gemma-3-270m-it-q4k-ehf16-af32';
const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
await fs.mkdir(out, { recursive: true });
const local = given ? null : await startServer();
const url = given ?? local.url;
const problems = [];
const context = await chromium.launchPersistentContext(path.join(root, '.build', 'chromium-profile'), { headless: true, args: useGpu ? gpuArgs : [], viewport: { width: 1440, height: 1000 } });
const ws = (page) => page.evaluate(() => { const s = globalThis.__DOPPLER_WORKSPACE__; return { busy: s.busy, runs: s.runs.map((run) => ({ id: run.id, kind: run.kind, tokens: run.tokens.length, parent: run.parent?.runId ?? null, layers: Boolean(run.layerOutputs) })), status: document.getElementById('ws-status').textContent }; });
try {
  const page = context.pages()[0] ?? await context.newPage();
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message.slice(0, 300)}`));
  page.on('console', (message) => { if (message.type() === 'error') problems.push(`error: ${message.text().slice(0, 300)}`); });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => globalThis.__DOPPLER_WORKSPACE__?.runs.length >= 1 && document.querySelectorAll('#ws-tokens .token-chip').length > 0, null, { timeout: 60_000 });

  // One navigation, two destinations.
  assert.deepEqual(await page.locator('nav.app-links a[data-site-page]').allTextContents(), ['Explore', 'Benchmark']);
  assert.equal(await page.locator('nav.app-links a[aria-current="page"]').textContent(), 'Explore');

  // The recorded answer, its signals and one shared selection.
  const columns = await page.locator('#ws-tokens .token-chip').count();
  assert.equal(await page.locator('.ws-bar').count(), columns * 3, 'Three signal rows, one column per token');
  assert.match(await page.textContent('#ws-perplexity'), /perplexity over (all \d+|the \d+ of \d+) answer tokens/);
  await page.locator('.ws-bar').nth(3).click();
  assert.match(await page.textContent('#ws-selection'), /^Word 4:/, 'A signal column selects its word everywhere');
  assert.match(await page.textContent('#token-title'), /answer token 4/);
  assert.equal(await page.locator('#ws-tokens .token-chip').nth(3).getAttribute('aria-pressed'), 'true');

  // The recorded activity and every inside-the-model view.
  await page.click('#ws-tab-journey');
  await page.click('.ws-load-activity button');
  await page.waitForFunction(() => !document.querySelector('.ws-load-activity') && globalThis.__DOPPLER_WORKSPACE__.runs.some((run) => run.layerOutputs), null, { timeout: 120_000 });
  assert.match(await page.textContent('#ws-selection'), /^Word 4:/, 'Opening the activity keeps the selected word');
  await page.selectOption('#ws-layer', '7');
  for (const id of ['journey', 'attention', 'memory', 'gearbox', 'forest', 'atlas', 'time']) {
    await page.click(`#ws-tab-${id}`);
    const text = (await page.locator(`#ws-view-${id}`).innerText()).trim();
    console.log(`  view ${id}: ${text.slice(0, 90).replace(/\s+/g, ' ')}`);
    assert.ok(text.length > 0, `${id} shows something, if only what was not recorded`);
    assert.doesNotMatch(text, /different model files/, `${id} accepts the activity recorded with this answer`);
  }
  // A sparse capture offers the nearest recorded token, and following it shows the view.
  for (const [id, drawn] of [['attention', '#attention svg path'], ['gearbox', '#gearbox svg rect']]) {
    await page.selectOption('#ws-layer', '');
    await page.locator('.ws-bar').nth(3).click();
    await page.click(`#ws-tab-${id}`);
    const offer = page.locator(`#ws-view-${id} button`, { hasText: 'Go to the nearest recorded token' });
    if (await offer.count()) {
      console.log(`  ${id}: ${(await offer.textContent()).trim()}`);
      await offer.click();
    }
    await page.waitForFunction((selector) => document.querySelectorAll(selector).length > 0, drawn, { timeout: 30_000 });
    console.log(`  ${id}: drawn at ${await page.textContent('#ws-selection')}`);
  }
  await page.screenshot({ path: path.join(out, 'recorded.png'), fullPage: true });

  // Save a run and open it again: the saved-file format is unchanged.
  const download = page.waitForEvent('download');
  await page.locator('#ws-runs li.is-current button', { hasText: 'Save' }).click();
  const saved = path.join(out, 'saved-run.json');
  await (await download).saveAs(saved);
  const savedId = JSON.parse(await fs.readFile(saved, 'utf8')).id;
  await page.setInputFiles('#ws-open', [saved, path.join(root, 'package.json')]);
  await page.waitForFunction(() => /package\.json/.test(document.getElementById('ws-status').textContent));
  assert.match(await page.textContent('#ws-status'), /package\.json: .*not a run this page can open/);
  assert.ok((await ws(page)).runs.some((run) => run.id === savedId), 'The saved run opens again');

  if (useGpu && await page.evaluate(async () => Boolean(await navigator.gpu?.requestAdapter()))) {
    await page.waitForFunction(() => !document.getElementById('ws-model').disabled, null, { timeout: 120_000 });
    // Model and recording settings live in the header menu.
    await page.evaluate(() => { document.getElementById('ws-menu').open = true; });
    await page.selectOption('#ws-model', model);
    await page.setChecked('#ws-layers', true);
    // A real answer, streamed.
    await page.fill('#ws-prompt', 'Explain in one short paragraph why the sky is blue.');
    await page.click('#ws-send');
    await page.waitForFunction(() => globalThis.__DOPPLER_WORKSPACE__.runs.some((run) => run.kind === 'answer'), null, { timeout: 900_000 });
    const answer = (await ws(page)).runs.find((run) => run.kind === 'answer');
    console.log(`  answer: ${JSON.stringify(answer)}`);
    assert.ok(answer.tokens > 0 && answer.layers, 'A live answer with layer outputs');
    const labels = await page.locator('.ws-row-label > span:first-child').allTextContents();
    const entropy = await page.evaluate(() => globalThis.__DOPPLER_WORKSPACE__.runs.find((run) => run.kind === 'answer').tokens.slice(0, 3).map((token) => token.entropy));
    console.log(`  signals: ${labels.join(', ')} · first entropies ${JSON.stringify(entropy)}`);
    // Runtimes from Doppler eacf6d79 record each token's full-distribution entropy.
    if (entropy.some((value) => value !== null && value !== undefined)) assert.ok(labels.includes('Confusion'), 'Recorded entropy is shown as confusion');
    assert.equal(await page.isDisabled('.ws-chip >> nth=1'), false, 'Example chips are usable again after the answer');

    // Stop part-way: Send becomes Stop, and what streamed stays without becoming a run.
    const before = (await ws(page)).runs.length;
    await page.fill('#ws-length', '200');
    await page.fill('#ws-prompt', 'Write a long story about a lighthouse keeper.');
    await page.click('#ws-send');
    await page.waitForFunction(() => document.querySelectorAll('#ws-tokens .token-chip').length >= 3, null, { timeout: 300_000 });
    assert.equal(await page.textContent('#ws-send'), 'Stop');
    await page.click('#ws-send');
    await page.waitForFunction(() => !globalThis.__DOPPLER_WORKSPACE__.busy, null, { timeout: 120_000 });
    const stopped = await ws(page);
    console.log(`  stop: ${stopped.status}`);
    assert.match(stopped.status, /^Stopped after \d+ tokens/);
    assert.equal(stopped.runs.length, before, 'A stopped answer is not kept as a run');
    assert.equal(await page.textContent('#ws-send'), 'Send');
    await page.fill('#ws-length', '48');

    // Fork from a recorded alternative of the live answer.
    await page.locator('#ws-runs li button.ws-run').filter({ hasText: 'why the sky is blue' }).first().click();
    const answerRun = (await ws(page)).runs.find((run) => run.kind === 'answer');
    await page.locator('#ws-runs li', { hasText: answerRun.id.slice(4, 10) }).locator('button.ws-run').click();
    await page.locator('#ws-tokens .token-chip').nth(1).click();
    await page.locator('#candidates button', { hasText: 'Branch here' }).first().click();
    await page.waitForFunction((id) => globalThis.__DOPPLER_WORKSPACE__.runs.some((run) => run.parent?.runId === id), answerRun.id, { timeout: 300_000 });
    const fork = (await page.evaluate((id) => globalThis.__DOPPLER_WORKSPACE__.runs.find((run) => run.parent?.runId === id)?.id ?? null, answerRun.id));
    console.log(`  fork: ${fork} · ${(await ws(page)).status}`);
    assert.ok(fork, 'The fork became a run');
    // A branch reads as one answer: inherited words with their underlines, the swapped-in word outlined, then its own.
    await page.locator('#ws-runs li', { hasText: fork.slice(4, 10) }).locator('button.ws-run').click();
    const shape = await page.evaluate(() => ({ chips: document.querySelectorAll('#ws-tokens .token-chip').length, inherited: document.querySelectorAll('#ws-tokens .ws-inherited').length, forked: document.querySelectorAll('#ws-tokens .ws-forked').length, note: document.getElementById('ws-answer-note').textContent }));
    console.log(`  branch display: ${JSON.stringify(shape)}`);
    assert.equal(shape.forked, 1, 'The swapped-in word is marked');
    assert.equal(shape.inherited, 1, 'The fork was at word 2, so one word is inherited');
    await page.locator('#ws-tokens .ws-inherited').first().click();
    assert.equal(await page.isVisible('#token-panel'), true, 'An inherited word opens its candidates');
    assert.match(await page.textContent('#token-title'), /Inherited from run/);
    await page.locator('#ws-tokens .ws-forked').click();
    assert.match(await page.textContent('#token-title'), /This branch put/);
    assert.ok(await page.locator('#candidates button', { hasText: 'Show this branch' }).count() >= 1, 'The fork word lists this branch among its choices');
    // The selected word's panel never covers the sections below it.
    const overlap = await page.evaluate(() => { window.scrollTo(0, document.querySelector('.ws-runs').offsetTop - 200); const a = document.querySelector('.ws-token').getBoundingClientRect(), b = document.querySelector('.ws-runs').getBoundingClientRect(); return a.bottom - b.top; });
    assert.ok(overlap <= 0, `The word panel ends above Runs (overlap ${overlap}px)`);
    // A fork asked a different question (its input carries the replaced token), so it is not comparable.
    // The fork is shown now; compare it with the run it branched from.
    await page.selectOption('#ws-compare-with', answerRun.id);
    await page.waitForFunction(() => !document.getElementById('compare-view').hidden);
    assert.match(await page.textContent('#compare-verdict'), /not the same question/);

    // The same question asked again is: tokens, then layer outputs, side by side.
    await page.waitForFunction(() => !globalThis.__DOPPLER_WORKSPACE__.busy);
    await page.fill('#ws-prompt', 'Explain in one short paragraph why the sky is blue.');
    await page.click('#ws-send');
    await page.waitForFunction(() => globalThis.__DOPPLER_WORKSPACE__.runs.filter((run) => run.kind === 'answer').length >= 2 || !globalThis.__DOPPLER_WORKSPACE__.busy, null, { timeout: 600_000 });
    const answers = (await ws(page)).runs.filter((run) => run.kind === 'answer' && run.parent === null);
    // Deterministic settings: the same prompt gives the same run, which is then not added twice.
    const again = answers.find((run) => run.id !== answerRun.id) ?? null;
    console.log(`  asked again: ${again ? again.id : 'identical run (same id), so it was not added twice'}`);
    if (again) {
      // The new answer is the one shown; compare it with the first.
      await page.selectOption('#ws-compare-with', answerRun.id);
      await page.waitForFunction(() => !document.getElementById('compare-view').hidden);
      console.log(`  compare: ${(await page.textContent('#compare-verdict')).trim()} · ${(await page.textContent('#compare-first')).trim().slice(0, 140)}`);
      assert.equal(await page.locator('#compare-blockers li').count(), 0, 'Two answers to one question are comparable');
    }
    await page.screenshot({ path: path.join(out, 'live.png'), fullPage: true });
  } else {
    console.log('  live: skipped (no GPU adapter or --no-gpu)');
  }

  // Phone width: no sideways scroll.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelectorAll('#ws-tokens .token-chip').length > 0, null, { timeout: 60_000 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'No sideways scroll at phone width');
  // Old addresses still lead here, keeping the visitor's query.
  for (const [from, to] of [['explore.html?preview=views.experimental', /\/\?preview=views\.experimental$/], ['workspace.html', /\/$/], ['index-preview.html', /\/\?preview=runtime\.preview&runtime=preview$/], ['compare.html', /benchmark\.html#compare$/]]) {
    await page.goto(new URL(from, url).href);
    await page.waitForURL(to, { timeout: 10_000 });
  }
} finally {
  await context.close();
  local?.server.close();
}
console.log(`problems: ${problems.length}`);
for (const problem of problems) console.log(`  ${problem}`);
assert.deepEqual(problems, [], 'No browser errors');
console.log('workspace: passed');
