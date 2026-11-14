// Real-hardware check that Benchmark and Explore work as one site: a benchmark
// result is compared on the Benchmark page and carries one run identity into Explore.
//
//   node scripts/check-site.js [page root url]
//
// It also opens any results saved earlier under .build/results/ (by
// check-benchmark.js, one per runtime) to compare them.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from './dev.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, '.build', 'site');
const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
await fs.mkdir(out, { recursive: true });
const local = process.argv[2] ? null : await startServer();
const url = process.argv[2] ?? local.url;
const problems = [];
const context = await chromium.launchPersistentContext(path.join(root, '.build', 'chromium-profile'), { headless: true, args: gpuArgs, viewport: { width: 1200, height: 1300 } });
try {
  const page = context.pages()[0] ?? await context.newPage();
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message.slice(0, 300)}`));
  page.on('console', (message) => { if (message.type() === 'error') problems.push(`error: ${message.text().slice(0, 300)}`); });
  page.on('response', (response) => { if (response.status() >= 400) problems.push(`HTTP ${response.status()}: ${response.url().slice(0, 200)}`); });

  // Benchmark
  await page.goto(new URL('benchmark.html', url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelector('nav.app-links a[data-site-page]') && !document.getElementById('run-button').disabled, null, { timeout: 60_000 });
  assert.deepEqual(await page.locator('nav.app-links a[data-site-page]').allTextContents(), ['Explore', 'Benchmark']);
  assert.equal(await page.locator('nav.app-links a[aria-current="page"]').textContent(), 'Benchmark');
  await page.click('#run-button');
  await page.waitForFunction(() => globalThis.__DOPPLER_BENCHMARK_EVIDENCE__, null, { timeout: 900_000 });
  const result = await page.evaluate(() => { const e = globalThis.__DOPPLER_BENCHMARK_EVIDENCE__; return { state: e.verdict.state, reasons: e.verdict.reasons, runId: e.run?.id ?? null, tokens: e.run?.tokens.length ?? null, representative: e.run?.timing.representative ?? null }; });
  console.log(`  run: ${JSON.stringify(result)}`);
  assert.match(result.runId ?? '', /^run-[0-9a-f]{16}$/, 'The benchmark answer has a run identity');
  assert.equal(result.representative, true, 'A benchmark-tier run is marked as a real timing');
  assert.match(await page.textContent('#identity'), new RegExp(`run ${result.runId.slice(4, 10)}`));
  await page.screenshot({ path: path.join(out, 'run.png'), fullPage: true });

  // The comparison on the same page, carrying this result
  await page.click('#to-compare');
  await page.waitForFunction(() => globalThis.__DOPPLER_COMPARE__?.results.length === 1);
  assert.equal(await page.locator('nav.app-links a[aria-current="page"]').textContent(), 'Benchmark');
  assert.match(await page.locator('#cmp-results tbody tr').first().innerText(), new RegExp(`run ${result.runId.slice(4, 10)}`), 'Compare names the same run');
  const saved = (await fs.readdir(path.join(root, '.build', 'results')).catch(() => [])).filter((name) => name.endsWith('.json')).map((name) => path.join(root, '.build', 'results', name));
  const older = await fs.stat(path.join(root, '.build', 'evidence-hosted-0.6.36.json')).then(() => [path.join(root, '.build', 'evidence-hosted-0.6.36.json')], () => []);
  await page.setInputFiles('#cmp-open', [...saved, ...older, path.join(root, 'package.json')]);
  await page.waitForFunction(() => /package\.json/.test(document.getElementById('cmp-open-status').textContent));
  const rows = await page.evaluate(() => [...document.querySelectorAll('#cmp-results tbody tr')].map((row) => ({ state: row.dataset.state, cells: [...row.cells].slice(0, 5).map((cell) => cell.innerText.replace(/\s+/g, ' ').trim()) })));
  for (const row of rows) console.log(`  compare: ${row.state} | ${row.cells.join(' | ')}`);
  console.log(`  compare: ${await page.textContent('#cmp-summary')} ${await page.textContent('#cmp-thin')}`);
  console.log(`  compare: ${await page.textContent('#cmp-open-status')}`);
  for (const reason of await page.locator('#cmp-reasons li').allTextContents()) console.log(`  not compared: ${reason}`);
  assert.equal(rows[0].state, result.state);
  assert.match(await page.textContent('#cmp-open-status'), /package\.json: This is not a benchmark result this page can open/);
  if (older.length) assert.ok(rows.some((row) => row.state === 'other-protocol'), 'A result under an older protocol is shown but set apart');
  await page.screenshot({ path: path.join(out, 'compare.png'), fullPage: true });

  // Explore, carrying the run of the first row
  await page.locator('#cmp-results tbody tr').first().locator('button', { hasText: 'Explore' }).click();
  await page.waitForFunction((id) => globalThis.__DOPPLER_WORKSPACE__?.runs.some((run) => run.id === id), result.runId, { timeout: 60_000 });
  assert.equal(await page.locator('nav.app-links a[aria-current="page"]').textContent(), 'Explore');
  const explored = await page.evaluate(() => ({ status: document.getElementById('ws-status').textContent, shown: document.querySelector('#ws-runs li.is-current code')?.textContent ?? null }));
  console.log(`  explore: ${explored.status}`);
  assert.equal(explored.shown, result.runId.slice(4, 10), 'Explore shows the very run the benchmark produced');
  assert.match(explored.status, /only tokens and time/);
  await page.screenshot({ path: path.join(out, 'explore.png'), fullPage: true });

  // The preview runtime: not offered by default, not reachable by its address alone,
  // and clearly marked when a visitor turns it on and picks it.
  await page.goto(new URL('benchmark.html?runtime=preview', url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.getElementById('runtime-name').textContent !== 'checking…' && !document.getElementById('run-button').disabled, null, { timeout: 60_000 });
  const hostedName = await page.textContent('#runtime-name');
  assert.doesNotMatch(hostedName, /preview/, 'Asking for the preview without its switch still gives the hosted runtime');
  assert.equal(await page.locator('#runtime-switch').count(), 0, 'No runtime choice is shown by default');
  assert.equal(await page.locator('#runtime-notice').isHidden(), true);
  await page.goto(new URL('benchmark.html?preview=runtime.preview&runtime=preview', url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.getElementById('runtime-name').textContent !== 'checking…' && !document.getElementById('run-button').disabled, null, { timeout: 60_000 });
  const previewName = await page.textContent('#runtime-name');
  console.log(`  runtimes: hosted "${hostedName}" · preview "${previewName}"`);
  assert.match(previewName, /preview, not qualified/);
  assert.match(await page.textContent('#runtime-notice'), /^Preview runtime\. .*not qualified/);
  assert.equal(await page.inputValue('#runtime-switch'), 'preview');
  assert.match(await page.locator('nav.app-links a[data-site-page]', { hasText: 'Explore' }).getAttribute('href'), /runtime=preview/, 'The choice follows the visitor from page to page');
  await page.click('#run-button');
  await page.waitForFunction(() => globalThis.__DOPPLER_BENCHMARK_EVIDENCE__, null, { timeout: 900_000 });
  const preview = await page.evaluate(() => { const e = globalThis.__DOPPLER_BENCHMARK_EVIDENCE__; return { state: e.verdict.state, reasons: e.verdict.reasons, notes: e.verdict.notes, runtime: e.runtime, channel: e.run?.environment.runtimeChannel ?? null }; });
  console.log(`  preview run: ${JSON.stringify({ state: preview.state, reasons: preview.reasons, runtime: preview.runtime })}`);
  assert.deepEqual([preview.runtime.channel, preview.runtime.preview, preview.channel], ['preview', true, 'preview']);
  assert.ok(preview.notes.some((note) => /preview runtime/.test(note)), 'A preview result carries its warning');
  assert.match(await page.locator('#notes').innerText(), /preview runtime/);
  await page.click('#to-compare');
  await page.waitForFunction(() => globalThis.__DOPPLER_COMPARE__?.results.length === 1);
  assert.equal(await page.locator('#cmp-results tbody tr .tag', { hasText: 'preview' }).count(), 1, 'Compare marks the preview result');
  await page.screenshot({ path: path.join(out, 'preview-compare.png'), fullPage: true });

  // No sideways scrolling at phone width on either page.
  await page.setViewportSize({ width: 390, height: 844 });
  for (const name of ['benchmark.html', '']) {
    await page.goto(new URL(name, url).href, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('nav.app-links a[data-site-page]'));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, `No sideways page scroll at phone width on "${name || 'Explore'}"`);
  }
} finally {
  await context.close();
  local?.server.close();
}
assert.deepEqual(problems, [], 'No browser errors or failed requests');
console.log('site: passed');
