import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpus, platform, release } from 'node:os';
import { chromium } from 'playwright';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';
import { resolveBrowserArgs } from '../../tools/demo-hardware-smoke.js';

const [assets, output, mode = 'all', workload = fileURLToPath(new URL('workload.json', import.meta.url))] = process.argv.slice(2);
if (!assets || !output || !['oracle', 'parity', 'stock', 'stock-mixed', 'doppler', 'all'].includes(mode)) {
  throw new Error('Usage: node benchmarks/tabgpt/run.js <prepared-assets> <result.json> [all|oracle|parity|stock|stock-mixed|doppler] [workload.json]');
}
const spec = JSON.parse(await readFile(workload, 'utf8'));
const rootDir = fileURLToPath(new URL('../../', import.meta.url));
const sourceFiles = execFileSync('git', ['ls-files', '-co', '--exclude-standard', '--',
  'src/training-browser.js', 'src/experimental/training/trainer.js', 'src/experimental/training/tabgpt*', 'src/gpu/kernels/training', 'src/gpu/kernels/matmul_f32.wgsl',
  'src/gpu/kernels/backward/matmul_transpose_a.wgsl', 'src/gpu/kernels/shader-cache.js',
  'benchmarks/tabgpt/*.js', 'benchmarks/tabgpt/tiny-pytorch*-oracle.json', 'benchmarks/tabgpt/workload.json',
], { cwd: rootDir, encoding: 'utf8' }).trim().split('\n').sort();
const sourceSha256 = {};
for (const file of new Set(sourceFiles)) {
  sourceSha256[file] = createHash('sha256').update(await readFile(resolve(rootDir, file))).digest('hex');
}
const server = await createStaticFileServer({ rootDir, host: '127.0.0.1', staticMounts: [{ urlPrefix: '/reference-assets', rootDir: resolve(assets) }] });
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: resolveBrowserArgs() });
  const report = {
    developmentBenchmark: true, browser: browser.version(),
    host: { platform: platform(), osRelease: release(), chip: cpus()[0]?.model },
    revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: rootDir, encoding: 'utf8' }).trim(),
    sourceSha256, spec, runs: {},
  };
  const schedule = [];
  if (mode === 'all') {
    schedule.push({ arm: 'oracle' }, { arm: 'parity' });
    for (let trial = 1; trial <= spec.pairedRepetitions; trial++) {
      const order = trial % 2 === 1 ? ['stock', 'doppler'] : ['doppler', 'stock'];
      for (const arm of order) schedule.push({ arm, trial });
    }
  } else {
    schedule.push({ arm: mode });
  }
  for (const { arm, trial } of schedule) {
    const page = await browser.newPage();
    await page.route('**/favicon.ico', route => route.fulfill({ status: 204 }));
    page.on('pageerror', error => console.error(error.message));
    page.on('console', msg => { if (msg.type() === 'error') console.error(msg.text()); });
    try {
      await page.goto(server.baseUrl + '/package.json');
      const run = await page.evaluate(async ({ arm, spec }) => {
        const { runBrowser } = await import('/benchmarks/tabgpt/browser.js');
        return runBrowser(arm, spec);
      }, { arm, spec });
      const key = trial ? arm + '-' + trial : arm;
      report.runs[key] = run;
      console.log(key, JSON.stringify({ numParams: run.result.numParams, pass: run.result.pass, tokensPerSecond: run.result.tokensPerSecond, validationAfter: run.result.validationAfter }));
      await writeFile(output, JSON.stringify(report, null, 2) + '\n');
      if ((arm === 'oracle' || arm === 'parity') && !run.result.pass) throw new Error(arm + ' failed; see ' + output);
    } finally { await page.close(); }
  }
  if (mode === 'all') {
    function median(values) {
      const sorted = [...values].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    }
    const trials = [];
    for (let trial = 1; trial <= spec.pairedRepetitions; trial++) {
      const stock = report.runs['stock-' + trial].result;
      const native = report.runs['doppler-' + trial].result;
      const maxTrainLossDifference = Math.max(...stock.steps.map(({ loss }, i) => Math.abs(loss - native.steps[i].loss)));
      const maxEvalLossDifference = Math.max(Math.abs(stock.validationBefore - native.validationBefore), Math.abs(stock.validationAfter - native.validationAfter));
      trials.push({ maxTrainLossDifference, maxEvalLossDifference,
        pass: Math.max(maxTrainLossDifference, maxEvalLossDifference) <= spec.lossAbsTolerance });
    }
    report.agreement = { trials, pass: trials.every(row => row.pass) };
    report.medianTokensPerSecond = {};
    for (const arm of ['stock', 'doppler']) {
      report.medianTokensPerSecond[arm] = median(trials.map((_, i) => report.runs[arm + '-' + (i + 1)].result.tokensPerSecond));
    }
    await writeFile(output, JSON.stringify(report, null, 2) + '\n');
    if (!report.agreement.pass) throw new Error('Matched training trajectory diverged; see ' + output);
  }
} finally {
  try { await browser?.close(); } finally { await server.close(); }
}
