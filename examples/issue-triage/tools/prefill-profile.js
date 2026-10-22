// Development probe: GPU timestamp breakdown of one ~600-token prefill.
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
  await page.evaluate((c) => { globalThis.__chars = c; }, process.env.CHARS ?? '3200');
  const result = await page.evaluate(async () => {
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
    const { load } = await import('/src/index.js');
    const { TRIAGE_POLICY, buildRouteRequest } = await import('/app/triage.js');
    const model = await load({ url: new URL('/model/', location.href).href }, {});
    const words = 'The application crashes with a null pointer exception when the configuration file is missing. '.repeat(80);
    const request = buildRouteRequest(TRIAGE_POLICY, TRIAGE_POLICY.routes[0], { title: 'Crash on startup', body: words.slice(0, Number(globalThis.__chars ?? 3200)) }, 100000);
    const rows = [];
    try {
      for (const profile of [false, true, true]) {
        model.resetGenerationState?.();
        for await (const _ of model.generate(request.prompt, { useChatTemplate: false, maxTokens: 1, temperature: 0, topK: 1, profile, executionObserver: profile })) { /* one token */ }
        if (!profile) continue;
        const stats = model.advanced.getStats();
        const groups = {};
        let measured = 0;
        for (const step of stats.prefillProfileSteps ?? []) {
          for (const interval of step.recorderStats?.profile?.intervals ?? []) {
            const key = interval.dispatch?.pipelineLabel ?? interval.label;
            groups[key] ??= { count: 0, ms: 0 };
            groups[key].count += 1; groups[key].ms += interval.durationMs; measured += interval.durationMs;
          }
        }
        rows.push({ prefillTokens: stats.prefillTokens, prefillTimeMs: stats.prefillTimeMs, measuredMs: measured, groups });
      }
    } finally {
      await model.unload();
    }
    return rows;
  });
  await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  const row = result.at(-1);
  console.log(`prefillTokens=${row.prefillTokens} prefillTimeMs=${row.prefillTimeMs?.toFixed?.(1)} measuredGpuMs=${row.measuredMs.toFixed(1)}`);
  for (const [k, v] of Object.entries(row.groups).sort((a, b) => b[1].ms - a[1].ms).slice(0, 12)) console.log(`${v.ms.toFixed(2).padStart(8)} ms  ${String(v.count).padStart(4)}x  ${k}`);
} finally {
  await browser.close();
  await server.close();
}
