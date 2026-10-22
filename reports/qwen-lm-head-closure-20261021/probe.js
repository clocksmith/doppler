// One independent sample: fresh Chrome, one model opening, one cold and N warm
// complete requests. Contracts with `lifecycleChecks` then cancel one request,
// reuse the session, unload, destroy the device and reopen on a new device; those
// checks run after the timed requests. Invoked by compare.js; never run two GPU
// probes at once.
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
      const runRequest = async (target, abortAfterTokens = null) => {
        target.resetGenerationState();
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(new Error('request timeout')), 60000);
        const tokens = [];
        const pieces = [];
        const start = performance.now();
        let abortedAtMs = null;
        try {
          for await (const text of target.generate(config.prompt, {
            ...config.generation,
            signal: controller.signal,
            onToken: (token) => {
              tokens.push(token);
              if (abortAfterTokens != null && tokens.length === abortAfterTokens && abortedAtMs == null) {
                abortedAtMs = performance.now();
                controller.abort();
              }
            },
          })) {
            pieces.push(text);
          }
        } catch (error) {
          if (abortedAtMs == null) throw error;
          return {
            settledAfterAbortMs: performance.now() - abortedAtMs,
            tokens,
            errorName: error?.name ?? null,
            errorCode: error?.code ?? null,
            errorMessage: String(error?.message ?? error).slice(0, 300),
          };
        } finally {
          clearTimeout(timeout);
        }
        return { elapsedMs: performance.now() - start, abortedAtMs, tokens, text: pieces.join('') };
      };
      const total = config.requests.cold + config.requests.warm;
      for (let i = 0; i < total; i += 1) {
        const request = await runRequest(handle);
        const stats = handle.advanced.getStats() ?? {};
        const numericStats = {};
        for (const [key, value] of Object.entries(stats)) {
          if (typeof value === 'number') numericStats[key] = value;
        }
        out.requests.push({
          kind: i < config.requests.cold ? 'cold' : 'warm',
          elapsedMs: request.elapsedMs,
          tokens: request.tokens,
          text: request.text,
          stats: numericStats,
        });
      }
      if (config.lifecycleChecks) {
        const checks = config.lifecycleChecks;
        out.lifecycle.cancel = {
          abortAfterTokens: checks.abortAfterTokens,
          ...(await runRequest(handle, checks.abortAfterTokens)),
        };
        const reuse = await runRequest(handle);
        out.lifecycle.reuseAfterCancel = { tokens: reuse.tokens, text: reuse.text };
        await handle.unload();
        handle = null;
        destroyDevice();
        const reopenStart = performance.now();
        handle = await load({ url: new URL('/model/', location.href).href }, runtimeConfig ? { runtimeConfig } : {});
        const reopenMs = performance.now() - reopenStart;
        const recreated = await runRequest(handle);
        out.lifecycle.deviceRecreation = { reopenMs, tokens: recreated.tokens, text: recreated.text };
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
    lifecycleChecks: contract.lifecycleChecks ?? null,
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
