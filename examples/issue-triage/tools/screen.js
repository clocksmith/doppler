// Development screen: triage a dataset sample in Chrome through either Doppler's
// source compat API or Transformers.js on the same source checkpoint.
// Not qualification; the release evaluation uses installed packages.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { loadSplit, stratifiedSample } from '../dataset.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
const [sourceRoot, modelDirectory, csvDirectory, split, perCellArg, outputPath, policyOverridePath] = process.argv.slice(2);
// ENGINE=transformersjs MODEL_ID=<path under modelDirectory> selects the comparator.
const engine = process.env.ENGINE ?? 'doppler';
const policyOverride = policyOverridePath ? JSON.parse(await fs.readFile(policyOverridePath, 'utf8')) : null;
if (!outputPath) throw new Error('usage: screen.js <dopplerSourceRoot> <modelDir> <csvDir> <train|test> <perCell> <out.json>');
const { createStaticFileServer } = await import(path.join(sourceRoot, 'src/tooling/node-browser-command-runner.js'));

const rows = await loadSplit(split, { readLocal: (s) => fs.readFile(path.join(csvDirectory, `issues_${s}.csv`), 'utf8') });
const sample = stratifiedSample(rows, { split, perCell: Number(perCellArg), seed: 'issue-triage-20261021' });
const server = await createStaticFileServer({
  rootDir: sourceRoot, host: '127.0.0.1', port: 0,
  staticMounts: [{ urlPrefix: '/model', rootDir: modelDirectory }, { urlPrefix: '/app', rootDir: appRoot }],
});
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: [
  '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
  ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']),
] });
try {
  const page = await browser.newPage();
  await page.goto(engine === 'doppler' ? server.baseUrl : `${server.baseUrl}/app/tools/tjs-page.html`);
  const result = await page.evaluate(async ({ issues, policyOverride, engineName, modelId }) => {
    const { suggestRoute, TRIAGE_POLICY } = await import('/app/triage.js');
    const policy = { ...TRIAGE_POLICY, ...(policyOverride ?? {}) };
    const started = performance.now();
    let model;
    if (engineName === 'doppler') {
      globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
      const { load } = await import('/src/index.js');
      model = await load({ url: new URL('/model/', location.href).href }, {});
    } else {
      const transformers = await import('/node_modules/@huggingface/transformers/dist/transformers.web.js');
      transformers.env.allowRemoteModels = false;
      transformers.env.allowLocalModels = true;
      transformers.env.localModelPath = '/model/';
      const { openTransformersEngine } = await import('/app/engines.js');
      const opened = await openTransformersEngine({ transformers, modelId });
      model = { scoreChoices: opened.scoreChoices, unload: opened.close, version: transformers.env.version };
    }
    const openMs = performance.now() - started;
    const rows = [];
    try {
      for (const issue of issues) {
        const t = performance.now();
        const s = await suggestRoute((request, options) => model.scoreChoices(request, options), issue, { policy });
        rows.push({ index: issue.index, repo: issue.repo, label: issue.label, ...s, elapsedMs: performance.now() - t });
      }
    } finally {
      await model.unload();
    }
    return { openMs, rows, engineVersion: model.version ?? null };
  }, { issues: sample.map(({ index, repo, label, title, body }) => ({ index, repo, label, title, body })), policyOverride, engineName: engine, modelId: process.env.MODEL_ID ?? null });
  const correct = result.rows.filter((r) => r.suggestedRoute === r.label).length;
  const summary = { engine, engineVersion: result.engineVersion, split, issues: result.rows.length, accuracy: correct / result.rows.length, openMs: result.openMs };
  for (const label of ['bug', 'feature', 'question']) {
    const tp = result.rows.filter((r) => r.label === label && r.suggestedRoute === label).length;
    const fp = result.rows.filter((r) => r.label !== label && r.suggestedRoute === label).length;
    const fn = result.rows.filter((r) => r.label === label && r.suggestedRoute !== label).length;
    summary[`f1_${label}`] = tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
  }
  summary.macroF1 = (summary.f1_bug + summary.f1_feature + summary.f1_question) / 3;
  await fs.writeFile(outputPath, `${JSON.stringify({ summary, rows: result.rows }, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await browser.close();
  await server.close();
}
