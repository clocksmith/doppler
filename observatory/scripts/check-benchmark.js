// Real-hardware check: runs the benchmark page in a GPU-enabled Chromium and
// writes the evidence it produced. This downloads the model on first use
// (about 420 MB) and needs a machine with a working WebGPU adapter.
//
//   node scripts/check-benchmark.js            # serve locally and run
//   node scripts/check-benchmark.js <page url>  # run against a served page
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from './dev.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildDir = path.join(root, '.build');
const timeoutMs = Number(process.env.BENCHMARK_TIMEOUT_MS ?? 900_000);
// macOS needs the Metal backend named; other platforms use Chromium's default.
const gpuArgs = process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];

await fs.mkdir(buildDir, { recursive: true });
const local = process.argv[2] ? null : await startServer();
const url = process.argv[2] ?? local.url;
// The profile is kept so the saved model is reused by the next run on this origin.
const context = await chromium.launchPersistentContext(path.join(buildDir, 'chromium-profile'), { headless: true, args: gpuArgs, viewport: { width: 900, height: 1200 } });
const page = context.pages()[0] ?? await context.newPage();
const problems = [];
page.on('pageerror', (error) => problems.push(`pageerror: ${error.message.slice(0, 300)}`));
page.on('response', (response) => { if (response.status() >= 400) problems.push(`HTTP ${response.status()}: ${response.url().slice(0, 200)}`); });
page.on('console', (message) => { if (['error', 'warning'].includes(message.type())) problems.push(`${message.type()}: ${message.text().slice(0, 300)}`); });

let evidence = null;
let stopped = null;
try {
  await page.goto(new URL('benchmark.html', url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.getElementById('run-note')?.textContent, null, { timeout: 30_000 });
  if (await page.isDisabled('#run-button')) {
    stopped = `The page did not offer a run: ${await page.textContent('#run-note')}`;
  } else {
    await page.click('#run-button');
    const deadline = Date.now() + timeoutMs;
    let lastStage = '';
    while (Date.now() < deadline) {
      evidence = await page.evaluate(() => globalThis.__DOPPLER_BENCHMARK_EVIDENCE__ ?? null);
      if (evidence) break;
      const stage = await page.evaluate(() => document.querySelector('#stages [data-state="running"]')?.textContent ?? '');
      if (stage && stage !== lastStage) { console.log(`  ${stage}`); lastStage = stage; }
      await page.waitForTimeout(1000);
    }
    if (!evidence) stopped = `No result after ${timeoutMs / 1000} s; last stage: ${lastStage || 'none'}`;
  }
  await page.screenshot({ path: path.join(buildDir, 'device-benchmark.png'), fullPage: true });
} finally {
  await context.close();
  local?.server.close();
}

const out = path.join(buildDir, 'device-benchmark-evidence.json');
if (evidence) {
  await fs.writeFile(out, JSON.stringify({ ...evidence, harness: { url, browserProblems: problems.slice(0, 40) } }, null, 2));
  // The result exactly as the page would download it, kept per runtime for the Compare page.
  await fs.mkdir(path.join(buildDir, 'results'), { recursive: true });
  // Every result is kept, refused and failed ones included: the file name carries the verdict and the time.
  const when = evidence.startedAt.slice(0, 19).replace(/[:T]/g, '-');
  await fs.writeFile(path.join(buildDir, 'results', `result-${evidence.runtime?.version ?? 'unknown'}-${(evidence.runtime?.release?.sourceCommit ?? 'nocommit').slice(0, 8)}-${evidence.verdict.state}-${when}.json`), JSON.stringify(evidence, null, 2));
}
if (stopped) { console.error(stopped); process.exit(2); }
const { verdict } = evidence;
console.log(`${verdict.state}: ${evidence.device.adapter?.description ?? evidence.device.adapter?.vendor ?? 'unknown adapter'}, runtime ${evidence.runtime?.version}`);
for (const reason of verdict.reasons) console.log(`  reason: ${reason}`);
for (const note of verdict.notes) console.log(`  note: ${note}`);
if (verdict.summary) console.log(`  ${JSON.stringify(verdict.summary)}`);
console.log(`  browser warnings and errors: ${problems.length}`);
console.log(`  evidence: ${out}`);
process.exit(verdict.state === 'failed' ? 1 : 0);
