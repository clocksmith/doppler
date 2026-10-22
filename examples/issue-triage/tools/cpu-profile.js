// Development probe: CPU profile of repeated short scoreChoices calls.
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
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 100 });
  await page.exposeFunction('__profile', async (on) => (on ? cdp.send('Profiler.start') : (globalThis.__cpuProfile = (await cdp.send('Profiler.stop')).profile, null)));
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
    const request = buildRouteRequest(TRIAGE_POLICY, TRIAGE_POLICY.routes[0], { title: 'Crash on startup', body: 'The application crashes.' }, 100000);
    for (let i = 0; i < 3; i += 1) await model.scoreChoices(request);
    const rows = [];
    await window.__profile(true);
    const t = performance.now();
    for (let i = 0; i < 20; i += 1) await model.scoreChoices(request);
    rows.push({ perCallMs: (performance.now() - t) / 20 });
    await window.__profile(false);
    await model.unload();
    return rows;
  });
  const profile = globalThis.__cpuProfile;
  await fs.writeFile(outputPath, JSON.stringify(profile));
  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  const self = new Map();
  const dt = profile.timeDeltas;
  profile.samples.forEach((id, i) => self.set(id, (self.get(id) ?? 0) + (dt[i] ?? 0)));
  const agg = {};
  let total = 0;
  for (const [id, us] of self) {
    const f = byId.get(id).callFrame;
    const key = `${f.functionName || '(anon)'} ${f.url.split('/').slice(-2).join('/')}:${f.lineNumber + 1}`;
    agg[key] = (agg[key] ?? 0) + us; total += us;
  }
  console.log(`perCallMs=${result[0].perCallMs.toFixed(1)} profiledMs=${(total / 1000).toFixed(0)}`);
  for (const [k, us] of Object.entries(agg).sort((a, b) => b[1] - a[1]).slice(0, 25)) console.log(`${(us / 1000).toFixed(1).padStart(8)} ms  ${k}`);

} finally {
  await browser.close();
  await server.close();
}
