#!/usr/bin/env node
// Diagnostic public-API probe; instrumented observations are not throughput claims.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { installGpuObservation } from './lib/installed-gpu-observation.js';

const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const sha256 = data => createHash('sha256').update(data).digest('hex');
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.installed-capsule-profile/v1');
assert(config.runs.length >= 2 && config.runs.every(run => typeof run.instrumented === 'boolean'));
assert(Number.isSafeInteger(config.cpuSamplingIntervalUs) && config.cpuSamplingIntervalUs > 0);
assert(Number.isSafeInteger(config.timeoutMs) && config.timeoutMs > 0);
assert.equal(config.model.descriptor.request.schema, 'doppler.capsule-operation-request/v2');
assert.equal(config.model.descriptor.request.operation.name, 'generate');
const bundle = await read(path.join(config.bundleRoot, 'receipt.json'));
assert(bundle.passed);
assert.equal(sha256(await fs.readFile(path.join(config.bundleRoot, bundle.package.filename))), bundle.package.sha256);
const consumer = path.join(config.bundleRoot, 'consumer');
const metadata = await read(path.join(consumer, 'node_modules/doppler-gpu/package.json'));
const entry = metadata.exports['./host'];
const hostUrl = `/node_modules/doppler-gpu/${(entry.browser ?? entry.import).replace(/^\.\//, '')}`;
await fs.writeFile(path.join(consumer, 'capsule-profile.html'), '<!doctype html><title>Installed Capsule profile</title>');
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.installed-capsule-profile-result/v1', passed: false,
  package: bundle.package, config, runs: [], logs: [], startedAtUtc: new Date().toISOString(),
  sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  probeSha256: sha256(await fs.readFile(new URL(import.meta.url))),
  observerSha256: sha256(await fs.readFile(new URL('./lib/installed-gpu-observation.js', import.meta.url))),
  scope: 'Public installed Capsule path. Instrumented CPU samples and existing WebGPU calls; no extra GPU work or fence. Waits overlap and must not be summed as independent GPU time.' };
let browser, server, page, timeout;
try {
  server = await createStaticFileServer({ rootDir: consumer, host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/model', rootDir: config.model.capsuleRoot }] });
  browser = await chromium.launch({ channel: config.channel, headless: true, args: config.launchArgs,
    env: { ...process.env, TMPDIR: config.temporaryDirectory } });
  report.browserVersion = browser.version();
  page = await browser.newPage();
  timeout = setTimeout(() => { void browser.close().catch(() => {}); }, config.timeoutMs);
  page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
  await page.goto(`${server.baseUrl}/capsule-profile.html`);
  report.hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('Physical adapter unavailable.');
    const { vendor, architecture, device, description, isFallbackAdapter } = adapter.info;
    return { vendor, architecture, device, description, isFallbackAdapter };
  });
  assert.equal(report.hardware.vendor, config.requiredVendor);
  assert.equal(report.hardware.isFallbackAdapter, false);
  await page.exposeFunction('profileProgress', progress => console.log(JSON.stringify(progress)));
  // Keep observer ownership in the page, outside the installed runtime.
  await page.evaluate(`globalThis.profileGpu = (${installGpuObservation.toString()})();`);
  const descriptor = structuredClone(config.model.descriptor);
  descriptor.capsuleUrl = `${server.baseUrl}/model/${config.model.capsuleFile}`;
  report.load = await page.evaluate(async ({ hostUrl, descriptor }) => {
    const host = await import(hostUrl);
    const start = performance.now();
    const session = await host.openCapsule(descriptor.capsuleUrl, { ...descriptor.openOptions,
      observer: { observe: globalThis.profileProgress } });
    globalThis.profileState = { host, session };
    return { elapsedMs: performance.now() - start, modelId: session.modelId,
      selectedTargetPlanDigest: session.selectedTargetPlanDigest, semanticRoot: session.semanticRoot };
  }, { hostUrl, descriptor });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: config.cpuSamplingIntervalUs });
  for (const [index, run] of config.runs.entries()) {
    if (run.instrumented) await cdp.send('Profiler.start');
    const result = await page.evaluate(async ({ descriptor, instrumented }) => {
      const { host, session } = globalThis.profileState;
      const request = { ...descriptor.request, limits: { ...descriptor.request.limits,
        deadlineAt: Date.now() + descriptor.maxDurationMs } };
      const accumulator = host.createCapsuleStreamAccumulator(request);
      const tokens = [];
      let text = '';
      if (instrumented) globalThis.profileGpu.start();
      const start = performance.now();
      for await (const event of session.executeOperation(request)) {
        accumulator.accept(event);
        if (event.status === 'partial') {
          text += event.delta.text;
          if (event.delta.tokenIds.length) tokens.push({ elapsedMs: performance.now() - start,
            tokenIds: event.delta.tokenIds, gpu: instrumented ? globalThis.profileGpu.snapshot() : null });
        }
      }
      const completed = accumulator.finish();
      if (text !== completed.output.text) throw new Error('Displayed additions disagree with verified completion.');
      return { elapsedMs: performance.now() - start, tokens, completed,
        gpu: instrumented ? globalThis.profileGpu.stop() : null };
    }, { descriptor, instrumented: run.instrumented });
    if (run.instrumented) {
      const { profile } = await cdp.send('Profiler.stop');
      const filename = `run-${index}.cpuprofile`;
      await fs.writeFile(path.join(config.outputDir, filename), JSON.stringify(profile));
      result.cpuProfile = filename;
    }
    assert.deepEqual(result.completed.output.tokenIds.slice(0, config.model.expectedTokenIds.length), config.model.expectedTokenIds);
    if (index > 0) assert.deepEqual(result.completed.output, report.runs[0].completed.output);
    report.runs.push({ ...run, ...result });
    await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ run: index, label: run.label, tokens: result.completed.output.tokenIds.length, elapsedMs: result.elapsedMs }));
  }
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearTimeout(timeout);
  report.cleanupErrors = [];
  if (page && !page.isClosed()) {
    try {
      await page.evaluate(async () => {
        try { await globalThis.profileState?.session.close(); }
        finally { globalThis.profileGpu?.restore(); }
      });
    } catch (error) { report.cleanupErrors.push(error.message); }
  }
  const cleanup = await Promise.allSettled([browser?.close(), server?.close()]);
  report.cleanupErrors.push(...cleanup.filter(row => row.status === 'rejected').map(row => row.reason.message));
  report.passed &&= report.cleanupErrors.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'receipt.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, error: report.error }));
if (!report.passed) process.exitCode = 1;
