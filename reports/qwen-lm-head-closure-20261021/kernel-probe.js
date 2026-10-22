// Kernel-timing sample: fresh Chrome, one warm request, then one timestamp-profiled
// request. Profiled submission shape differs; use only for operator timing.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const [sourceRoot, modelDirectory, contractPath, outputPath] = process.argv.slice(2);
if (!sourceRoot || !modelDirectory || !contractPath || !outputPath) {
  throw new Error('usage: probe.js <sourceRoot> <modelDirectory> <contract.json> <output.json>');
}
const contract = JSON.parse(await fs.readFile(contractPath, 'utf8'));
const { createStaticFileServer } = await import(
  pathToFileURL(path.join(sourceRoot, 'src/tooling/node-browser-command-runner.js')).href
);

const BROWSER_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--ignore-gpu-blocklist',
  ...(process.platform === 'darwin'
    ? ['--use-angle=metal']
    : ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']),
];

const result = { sourceRoot, modelDirectory, startedAt: new Date().toISOString(), lifecycle: {} };
let server;
let browser;
try {
  server = await createStaticFileServer({
    rootDir: sourceRoot,
    host: '127.0.0.1',
    port: 0,
    staticMounts: [{ urlPrefix: '/model', rootDir: modelDirectory }],
  });
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: BROWSER_ARGS });
  result.browserVersion = browser.version();
  const page = await browser.newPage();
  page.on('console', (msg) => {
    if (msg.type() === 'error') (result.consoleErrors ??= []).push(msg.text().slice(0, 500));
  });
  page.on('response', (response) => {
    if (response.status() >= 400) (result.failedResponses ??= []).push(`${response.status()} ${response.url()}`);
  });
  await page.goto(server.baseUrl);
  const evaluation = await page.evaluate(async (config) => {
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
    const { load } = await import('/src/index.js');
    // Count dispatched pipelines by label on the unprofiled warm request only.
    const pipelineCounts = {};
    let countPipelines = false;
    const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      if (countPipelines) pipelineCounts[pipeline.label || '(unlabeled)'] = (pipelineCounts[pipeline.label || '(unlabeled)'] ?? 0) + 1;
      return setPipeline.call(this, pipeline);
    };
    let runtimeConfig;
    if (config.runtimeProfile) {
      const { applyRuntimeProfile } = await import('/src/inference/browser-harness/runtime-config.js');
      runtimeConfig = await applyRuntimeProfile(config.runtimeProfile, { baseUrl: '/src/config/runtime' });
    }
    const { destroyDevice } = await import('/src/gpu/device.js');
    const out = { requests: [], lifecycle: {} };
    let handle;
    try {
      const openStart = performance.now();
      handle = await load({ url: new URL('/model/', location.href).href }, runtimeConfig ? { runtimeConfig } : {});
      out.runtimeProfile = config.runtimeProfile ?? null;
      out.lmHeadFusion = runtimeConfig?.inference?.session?.useGreedyLmHeadArgmaxFusion ?? null;
      out.openMs = performance.now() - openStart;
      const adapterInfo = handle.advanced?.getStats?.()?.adapterInfo ?? null;
      out.adapterInfo = adapterInfo;
      for (const [requestIndex, profile] of [false, false, true].entries()) {
        countPipelines = requestIndex === 1;
        handle.resetGenerationState();
        const tokens = [];
        const pieces = [];
        const start = performance.now();
        for await (const text of handle.generate(config.prompt, {
          ...config.generation,
          profile,
          executionObserver: profile,
          signal: AbortSignal.timeout(60000),
          onToken: (token) => tokens.push(token),
        })) {
          pieces.push(text);
        }
        const elapsedMs = performance.now() - start;
        const stats = handle.advanced.getStats() ?? {};
        const numericStats = {};
        for (const [key, value] of Object.entries(stats)) {
          if (typeof value === 'number') numericStats[key] = value;
        }
        const groups = {};
        if (profile) {
          for (const phase of ['prefill', 'decode']) {
            for (const step of stats[`${phase}ProfileSteps`] ?? []) {
              const p = step.recorderStats?.profile;
              if (!p) continue;
              for (const interval of p.intervals) {
                const key = `${phase}:${interval.dispatch?.pipelineLabel ?? interval.dispatch?.pipeline ?? interval.label}`;
                groups[key] ??= { count: 0, durationMs: 0 };
                groups[key].count += 1;
                groups[key].durationMs += interval.durationMs;
              }
              for (const k of ['computePasses', 'measuredPasses', 'unmeasuredPasses']) {
                groups[`${phase}:#${k}`] ??= { count: 0, durationMs: 0 };
                groups[`${phase}:#${k}`].count += p[k];
              }
            }
          }
        }
        out.requests.push({
          kind: profile ? 'profiled' : 'warm',
          groups,
          pipelineCounts: requestIndex === 1 ? { ...pipelineCounts } : null,
          elapsedMs,
          tokens,
          text: pieces.join(''),
          stats: numericStats,
        });
      }
    } catch (error) {
      out.error = String(error?.stack || error);
    } finally {
      try {
        await handle?.unload();
        out.lifecycle.unload = true;
      } catch (error) {
        out.lifecycle.unload = false;
        out.lifecycle.unloadError = String(error);
      }
      try {
        destroyDevice();
        out.lifecycle.destroyDevice = true;
      } catch (error) {
        out.lifecycle.destroyDevice = false;
        out.lifecycle.destroyDeviceError = String(error);
      }
    }
    return out;
  }, {
    prompt: contract.prompt,
    runtimeProfile: contract.runtimeProfile ?? null,
    generation: contract.generation,
    requests: contract.design.requestsPerProcess,
  });
  Object.assign(result, evaluation);
} catch (error) {
  result.error = String(error?.stack || error);
} finally {
  try {
    await browser?.close();
    result.lifecycle.browserClosed = true;
  } catch (error) {
    result.lifecycle.browserClosed = false;
    result.lifecycle.browserCloseError = String(error);
  }
  await server?.close();
  result.finishedAt = new Date().toISOString();
  await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
}
process.exitCode = result.error ? 1 : 0;
