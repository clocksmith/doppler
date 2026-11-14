// Real-hardware check that the front page works with no network after one
// online visit: every page is closed, the network is cut, the page is reopened,
// the saved model loads and a question is answered. It is the demo's own
// offline journey, run against this site's page and worker.
//
//   node scripts/check-offline.js [served address]
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { startServer } from './dev.js';

const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
const MODEL = 'gemma-3-270m-it-q4k-ehf16-af32';
const PROMPT = 'Explain in one short paragraph why the sky is blue.';
const given = process.argv.slice(2).find((arg) => /^https?:\/\//.test(arg)) ?? null;
const local = given ? null : await startServer();
const url = given ?? local.url;
// A profile of its own: the worker and the saved model must come from this visit.
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'observatory-offline-'));
const context = await chromium.launchPersistentContext(profile, { headless: true, args: gpuArgs, viewport: { width: 1280, height: 1200 } });
const problems = [];
const answerOnce = async (page) => {
  await page.waitForFunction(() => !document.getElementById('ws-model').disabled, null, { timeout: 600_000 });
  await page.selectOption('#ws-model', MODEL);
  await page.fill('#ws-prompt', PROMPT);
  await page.click('#ws-send');
  await page.waitForFunction(() => globalThis.__DOPPLER_WORKSPACE__.runs.some((run) => run.kind === 'answer'), null, { timeout: 900_000 });
  return page.evaluate(() => { const run = globalThis.__DOPPLER_WORKSPACE__.runs.find((item) => item.kind === 'answer'); return { id: run.id, tokens: run.tokens.length, text: run.outputText.slice(0, 60), commit: run.environment.runtimeCommit }; });
};
try {
  let page = context.pages()[0] ?? await context.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const online = await answerOnce(page);
  const kept = await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
    const names = (await caches.keys()).filter((name) => name.startsWith('doppler-observatory-shell-') && !name.endsWith('-filling'));
    return { names, files: names.length ? (await (await caches.open(names[0])).keys()).length : 0 };
  });
  console.log(`  online: ${JSON.stringify(online)}; kept for offline: ${kept.files} files in ${kept.names.join(', ')}`);
  assert.equal(kept.names.length, 1, 'One complete copy of the page\'s files is kept');
  assert.ok(kept.files > 100);

  // Installable: the browser's own verdict on the manifest, icons and worker.
  const cdp = await context.newCDPSession(page);
  const { installabilityErrors } = await cdp.send('Page.getInstallabilityErrors');
  const manifest = await cdp.send('Page.getAppManifest');
  console.log(`  installable: ${installabilityErrors.length === 0 ? 'yes' : JSON.stringify(installabilityErrors)}; manifest ${manifest.url}`);
  assert.equal(installabilityErrors.length, 0, 'The browser finds nothing that prevents installing the page as an app');
  assert.match(manifest.url, /app-manifest\.json$/);

  for (const open of context.pages()) await open.close();
  await context.setOffline(true);
  page = await context.newPage();
  page.on('pageerror', (error) => problems.push(`offline pageerror: ${error.message.slice(0, 300)}`));
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const offline = await answerOnce(page);
  console.log(`  offline, all pages closed first: ${JSON.stringify(offline)}`);
  assert.ok(offline.tokens > 0 && offline.commit === online.commit, 'The reopened page answered with the same runtime, with no network');
  // The site's other pages are kept too.
  await page.goto(new URL('benchmark.html', url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelector('nav.app-links a[data-site-page]'), null, { timeout: 30_000 });
  console.log(`problems: ${problems.length}`);
  for (const problem of problems) console.log(`  ${problem}`);
  assert.equal(problems.length, 0);
  console.log('offline: passed');
} finally {
  await context.close();
  local?.server.close();
  await fs.rm(profile, { recursive: true, force: true });
}
