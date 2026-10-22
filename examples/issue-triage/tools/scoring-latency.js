// Development probe: scoreChoices latency versus prompt length, with dispatched
// pipeline counts for one call at each length. Source compat API in Chrome.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
const [sourceRoot, modelDirectory, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error('usage: scoring-latency.js <dopplerSourceRoot> <modelDir> <out.json>');
const { createStaticFileServer } = await import(path.join(sourceRoot, 'src/tooling/node-browser-command-runner.js'));
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
  await page.goto(server.baseUrl);
  const result = await page.evaluate(async () => {
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
    const { load } = await import('/src/index.js');
    const { TRIAGE_POLICY, buildRouteRequest } = await import('/app/triage.js');
    const counts = {};
    let counting = false;
    const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      if (counting) counts[pipeline.label || '(unlabeled)'] = (counts[pipeline.label || '(unlabeled)'] ?? 0) + 1;
      return setPipeline.call(this, pipeline);
    };
    const submit = GPUQueue.prototype.submit;
    let submits = 0;
    GPUQueue.prototype.submit = function (buffers) {
      if (counting) submits += 1;
      return submit.call(this, buffers);
    };
    const model = await load({ url: new URL('/model/', location.href).href }, {});
    const words = 'The application crashes with a null pointer exception when the configuration file is missing. '.repeat(80);
    const rows = [];
    try {
      for (const chars of [0, 200, 800, 1600, 3200]) {
        const request = buildRouteRequest(TRIAGE_POLICY, TRIAGE_POLICY.routes[0],
          { title: 'Crash on startup', body: words.slice(0, chars) }, 100000);
        request.maxSeqLen = 2048;
        await model.scoreChoices(request); // warm this length
        const times = [];
        for (let i = 0; i < 5; i += 1) {
          if (i === 4) { counting = true; submits = 0; for (const k of Object.keys(counts)) delete counts[k]; }
          const t = performance.now();
          const r = await model.scoreChoices(request);
          times.push(performance.now() - t);
          if (i === 4) counting = false;
          if (i === 0) rows.push({ chars, promptTokenCount: r.promptTokenCount });
        }
        Object.assign(rows.at(-1), { times, submits, pipelines: { ...counts } });
      }
    } finally {
      await model.unload();
    }
    return rows;
  });
  await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  for (const row of result) {
    const sorted = [...row.times].sort((a, b) => a - b);
    console.log(`tokens=${row.promptTokenCount} median=${sorted[2].toFixed(1)}ms submits=${row.submits} dispatches=${Object.values(row.pipelines).reduce((a, b) => a + b, 0)}`);
  }
} finally {
  await browser.close();
  await server.close();
}
