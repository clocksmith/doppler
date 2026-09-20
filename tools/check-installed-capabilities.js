#!/usr/bin/env node
// Physical consumer acceptance; models/descriptors are explicit inputs, never source defaults.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { evaluateEmbeddingReference, assertEmbeddingSourceIdentity } from '../src/config/embedding-reference.js';
import { evaluateRerankReference, assertRerankSourceIdentity } from '../src/config/rerank-reference.js';
import { resolveCapsuleEmbeddingContract } from '../src/config/embedding-contract.js';
import { installCapabilityMemoryProbe } from '../tests/fixtures/installed-capability-memory.js';

const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.installed-capabilities-acceptance/v1');
assert(config.requiredVendor && Number.isSafeInteger(config.timeoutMs) && config.timeoutMs > 0);
assert(config.consumer === undefined || ['standalone', 'reploid', 'reploid-library'].includes(config.consumer));
if (config.consumer === 'reploid') assert(typeof config.reploidRoot === 'string');
if (config.consumer === 'reploid-library') assert(typeof config.libraryArchive === 'string');
if (config.sharedSessionOperation !== undefined) assert.equal(config.consumer, 'reploid');
if (config.lifecycle) assert(!config.consumer || config.consumer === 'standalone');
if (config.adapterProbe) {
  assert(Number.isSafeInteger(config.adapterProbe.attempts) && config.adapterProbe.attempts > 0);
  assert(Number.isSafeInteger(config.adapterProbe.retryDelayMs) && config.adapterProbe.retryDelayMs >= 0);
}
const operations = config.models.map(row => row.descriptor.request.operation.name);
assert(operations.length > 0 && new Set(operations).size === operations.length);
assert(operations.every(name => ['embed', 'generate', 'rerank'].includes(name)));
if (config.consumer === 'reploid-library') assert.deepEqual(operations, ['generate'], 'The Reploid provider exposes generation only');
const bundle = await read(path.join(config.bundleRoot, 'receipt.json'));
assert(bundle.passed);
assert.equal(createHash('sha256').update(await fs.readFile(path.join(config.bundleRoot, bundle.package.filename))).digest('hex'), bundle.package.sha256);
const consumer = path.join(config.bundleRoot, 'consumer');
const metadata = await read(path.join(consumer, 'node_modules/doppler-gpu/package.json'));
const publicExports = ['doppler-gpu/host', 'doppler-gpu'];
const imports = Object.fromEntries(publicExports.map(specifier => {
  const entry = metadata.exports[specifier === 'doppler-gpu' ? '.' : './host'];
  return [specifier, `/node_modules/doppler-gpu/${(entry.browser ?? entry.import).replace(/^\.\//, '')}`];
}));
if (config.consumer === 'reploid-library') {
  const library = await read(path.join(consumer, 'node_modules/reploid/package.json'));
  for (const entry of ['./doppler', './config']) imports[`reploid/${entry.slice(2)}`]
    = `/node_modules/reploid/${library.exports[entry].import.replace(/^\.\//, '')}`;
}
await fs.copyFile(new URL('../examples/capsule-capabilities/app.js', import.meta.url), path.join(consumer, 'capabilities-app.js'));
if (config.lifecycle) await fs.copyFile(new URL('../tests/fixtures/installed-capability-lifecycle.js', import.meta.url),
  path.join(consumer, 'capability-lifecycle.js'));
await fs.writeFile(path.join(consumer, 'capabilities.html'), `<!doctype html><title>Installed Capsule capabilities</title><script type="importmap">${JSON.stringify({ imports })}</script>`);
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.installed-capabilities-acceptance-result/v1', passed: false,
  startedAtUtc: new Date().toISOString(), package: bundle.package, config, results: [], logs: [], progress: [],
  fixtureSource: await read(path.join(config.bundleRoot, 'source-state.json')),
  runnerRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: new URL('..', import.meta.url), encoding: 'utf8' }).trim(),
  runnerSha256: createHash('sha256').update(await fs.readFile(new URL(import.meta.url))).digest('hex'),
  lifecycleFixtureSha256: config.lifecycle ? createHash('sha256').update(await fs.readFile(
    new URL('../tests/fixtures/installed-capability-lifecycle.js', import.meta.url))).digest('hex') : null,
  memoryProbeSha256: config.measureMemory === true ? createHash('sha256').update(await fs.readFile(
    new URL('../tests/fixtures/installed-capability-memory.js', import.meta.url))).digest('hex') : null,
  scope: 'Physical local browser execution of retained models; no new model or fleet qualification.' };
if (config.reploidRoot) {
  report.reploidRevision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: config.reploidRoot, encoding: 'utf8' }).trim();
}
if (config.consumer === 'reploid-library') report.libraryArchive = {
  path: config.libraryArchive,
  sha256: createHash('sha256').update(await fs.readFile(config.libraryArchive)).digest('hex'),
};
let server, browser;
try {
  server = await createStaticFileServer({ rootDir: consumer, host: '127.0.0.1', port: 0,
    staticMounts: [...config.models.map((row, index) => ({ urlPrefix: `/models/${index}`, rootDir: row.capsuleRoot })),
      ...(config.reploidRoot ? [{ urlPrefix: '/reploid', rootDir: config.reploidRoot }] : [])] });
  browser = await chromium.launch({ channel: config.channel, headless: true, args: config.launchArgs,
    env: { ...process.env, TMPDIR: config.temporaryDirectory } });
  report.browserVersion = browser.version();
  for (const [index, row] of config.models.entries()) {
    const page = await browser.newPage();
    if (config.measureMemory === true) await page.addInitScript(installCapabilityMemoryProbe);
    let timedOut = false;
    let hardware = null;
    let result = null;
    const lifecycleProgress = [];
    const timeout = setTimeout(() => {
      timedOut = true;
      void page.close().catch(error => report.logs.push({ type: 'cleanup', text: error.message }));
    }, config.timeoutMs);
    try {
      page.setDefaultTimeout(config.timeoutMs);
      page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
      page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
      await page.goto(`${server.baseUrl}/capabilities.html`);
      hardware = await page.evaluate(async probe => {
        const observations = [];
        for (let index = 0; index < (probe?.attempts ?? 1); index++) {
          const adapter = await navigator.gpu.requestAdapter();
          observations.push({ available: adapter !== null });
          if (adapter) return { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
            isFallbackAdapter: adapter.info.isFallbackAdapter, probe: observations };
          if (probe && index + 1 < probe.attempts) await new Promise(resolve => setTimeout(resolve, probe.retryDelayMs));
        }
        throw new Error(`Physical WebGPU adapter unavailable after ${observations.length} recorded attempts; no fallback is permitted.`);
      }, config.adapterProbe);
      assert.equal(hardware.vendor, config.requiredVendor);
      assert.equal(hardware.isFallbackAdapter, false);
      const descriptor = structuredClone(row.descriptor);
      descriptor.capsuleUrl = `${server.baseUrl}/models/${index}/${row.capsuleFile}`;
      if (descriptor.openOptions.releasePolicy) descriptor.openOptions.releasePolicy.now = new Date().toISOString();
      console.log(JSON.stringify({ stage: 'open-and-execute', operation: descriptor.request.operation.name, hardware }));
      await page.exposeFunction('reportCapabilityProgress', async progress => {
        report.progress.push({ operation: descriptor.request.operation.name, progress,
          ...(config.measureMemory === true ? { memory: await page.evaluate(() => readCapabilityMemory()) } : {}) });
        await fs.writeFile(path.join(config.outputDir, 'loading-progress.json'), JSON.stringify(report.progress, null, 2));
        console.log(JSON.stringify({ operation: descriptor.request.operation.name, progress }));
      });
      await page.exposeFunction('persistCapabilityCheckpoint', async checkpoint => {
        await fs.writeFile(path.join(config.outputDir, `${descriptor.request.operation.name}-checkpoint.json`), JSON.stringify(checkpoint, null, 2));
      });
      await page.exposeFunction('retainCapabilityObservation', async observation => {
        lifecycleProgress.push(observation);
        await fs.writeFile(path.join(config.outputDir, `${descriptor.request.operation.name}-lifecycle-progress.json`),
          JSON.stringify(lifecycleProgress, null, 2));
        if (row.expectedTokenIds) assert.deepEqual(observation.completed.output.tokenIds, row.expectedTokenIds,
          `${observation.phase}: frozen generation reference mismatch`);
      });
      result = await page.evaluate(async ({ descriptor, consumer, shared, lifecycle }) => {
        const { runCapability } = await import('/capabilities-app.js');
        const { DOPPLER_VERSION } = await import('doppler-gpu');
        let partials = 0;
        if (lifecycle) {
          const host = await import('doppler-gpu/host');
          const { runInstalledCapabilityLifecycle } = await import('/capability-lifecycle.js');
          return { ...await runInstalledCapabilityLifecycle(host, descriptor, {
            onProgress: globalThis.reportCapabilityProgress,
            persistReleaseCheckpoint: globalThis.persistCapabilityCheckpoint,
            onObservation: globalThis.retainCapabilityObservation,
          }, lifecycle), runtimeVersion: DOPPLER_VERSION };
        }
        if (consumer === 'reploid-library') {
          const runtime = { ...await import('doppler-gpu'), ...await import('doppler-gpu/host') };
          const { createDopplerProvider } = await import('reploid/doppler');
          const { resolveConfig } = await import('reploid/config');
          const session = await runtime.openCapsule(descriptor.capsuleUrl, {
            ...descriptor.openOptions, persistReleaseCheckpoint: globalThis.persistCapabilityCheckpoint,
            observer: { observe: globalThis.reportCapabilityProgress },
          });
          let provider;
          try {
            const contract = Object.fromEntries(['modelId', 'capsuleId', 'semanticRoot', 'selectedTargetPlanDigest']
              .map(key => [key, session[key]]));
            contract.runtimeVersion = DOPPLER_VERSION;
            const config = resolveConfig({ overrides: { models: { providerId: 'doppler', contract } } });
            provider = createDopplerProvider({ config, session, runtime, ownership: 'borrowed',
              toOperationRequest: () => ({ ...descriptor.request, limits: { ...descriptor.request.limits,
                deadlineAt: Date.now() + descriptor.maxDurationMs } }) });
            const additions = [];
            const result = await provider.generate([{ role: 'user', content: 'Execute the retained descriptor request.' }],
              addition => { partials++; additions.push(addition); });
            if (additions.join('') !== result.content) throw new Error('Displayed additions differ from accepted completion');
            return { completed: result.evidence, partials, runtimeVersion: DOPPLER_VERSION,
              library: { entry: 'reploid/doppler', displayMatchesCompletion: true } };
          } finally { await provider?.close(); await session.close(); }
        }
        if (consumer === 'reploid') {
          const host = await import('doppler-gpu/host');
          const { createReploidDopplerRuntimeService } = await import('/reploid/self/infrastructure/doppler-runtime-service.js');
          const { runPackOperation } = await import('/reploid/self/pool/pack-operation.js');
          const api = { ...host, DOPPLER_VERSION };
          const service = createReploidDopplerRuntimeService({ loadModule: async () => api, expectedVersion: DOPPLER_VERSION });
          const devices = [];
          const observedAdapters = new WeakSet();
          const observedDevices = new WeakMap();
          let executionScope = null;
          const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
          navigator.gpu.requestAdapter = async (...args) => {
            const adapter = await originalRequestAdapter(...args);
            if (!adapter || observedAdapters.has(adapter)) return adapter;
            observedAdapters.add(adapter);
            const originalRequestDevice = adapter.requestDevice.bind(adapter);
            adapter.requestDevice = async (...args) => {
              const device = await originalRequestDevice(...args);
              if (!observedDevices.has(device)) {
                const observation = { id: devices.length, executions: [] };
                devices.push(observation); observedDevices.set(device, observation);
                const submit = device.queue.submit.bind(device.queue);
                device.queue.submit = (...args) => {
                  if (executionScope && !observation.executions.includes(executionScope)) observation.executions.push(executionScope);
                  return submit(...args);
                };
              }
              return device;
            };
            return adapter;
          };
          try {
            const openOptions = { ...descriptor.openOptions, persistReleaseCheckpoint: globalThis.persistCapabilityCheckpoint,
              observer: { observe: globalThis.reportCapabilityProgress } };
            const first = await service.openCapsule({ scope: 'physical-first', source: descriptor.capsuleUrl, options: openOptions });
            const second = shared ? await service.openCapsule({ scope: 'physical-second', source: descriptor.capsuleUrl, options: openOptions }) : null;
            const capsule = await (await fetch(descriptor.capsuleUrl)).json();
            const binding = { ...first.capsuleIdentity, artifacts: capsule.artifacts, requiredOperation: descriptor.request.operation.name,
              acceptedTargetPlanDigests: [first.selectedTargetPlanDigest] };
            const execute = session => {
              executionScope = session === first ? 'first' : 'second';
              return runPackOperation({ binding, session, runtimeVersion: DOPPLER_VERSION, runtimeService: service,
              request: { ...descriptor.request, limits: { ...descriptor.request.limits, deadlineAt: Date.now() + descriptor.maxDurationMs } },
              onPartial: () => { partials++; } });
            };
            const execution = await execute(first);
            await service.close('physical-first');
            let sharedSessions = null;
            if (second) {
              const afterClose = await execute(second);
              const firstDevices = devices.filter(device => device.executions.includes('first'));
              const secondDevices = devices.filter(device => device.executions.includes('second'));
              sharedSessions = { deviceRequests: devices.length, devices, firstClosed: first.closed,
                sameExecutionDevice: firstDevices.length === 1 && secondDevices.length === 1 && firstDevices[0].id === secondDevices[0].id,
                secondOutputIdentical: JSON.stringify(afterClose.output) === JSON.stringify(execution.output),
                secondReceipt: afterClose.receipt };
              if (!sharedSessions.sameExecutionDevice || !sharedSessions.firstClosed || !sharedSessions.secondOutputIdentical) {
                throw new Error(`Shared physical device session isolation failed: ${JSON.stringify({
                  ...sharedSessions, firstOutput: execution.output, secondOutput: afterClose.output,
                })}`);
              }
            }
            return { completed: { status: 'completed', output: execution.output, receipt: execution.receipt,
              eventDigest: execution.finalEventDigest }, partials, runtimeVersion: DOPPLER_VERSION, sharedSessions };
          } finally { await service.closeAll(); navigator.gpu.requestAdapter = originalRequestAdapter; }
        }
        const completed = await runCapability(descriptor, {
          onProgress: globalThis.reportCapabilityProgress,
          persistReleaseCheckpoint: globalThis.persistCapabilityCheckpoint,
          onEvent: event => { if (event.status === 'partial') partials++; },
        });
        return { completed, partials, runtimeVersion: DOPPLER_VERSION };
      }, { descriptor, consumer: config.consumer, shared: config.sharedSessionOperation === descriptor.request.operation.name,
        lifecycle: config.lifecycle ? { repeatRuns: config.lifecycle.repeatRuns, reopenCycles: config.lifecycle.reopenCycles, cancellation: row.cancellation,
          ...(row.adapter ? { adapter: row.adapter } : {}) } : null });
      assert.equal(result.runtimeVersion, bundle.package.version);
      assert.equal(result.completed.status, 'completed');
      if (row.expectedTokenIds) assert.deepEqual(result.completed.output.tokenIds, row.expectedTokenIds);
      if (descriptor.request.operation.name === 'embed') {
        assert.equal(result.completed.output.embeddings.length, descriptor.request.input.texts.length);
        assert(result.completed.output.embeddings.every(item => item.embedding.length === row.expectedDimension && item.embedding.every(Number.isFinite)));
      }
      if (row.reference) {
        assert(result.lifecycle, 'Numerical reference checks require the observed public session contract.');
        const bytes = await fs.readFile(row.reference.path);
        assert.equal(`sha256:${createHash('sha256').update(bytes).digest('hex')}`, row.reference.digest);
        const reference = JSON.parse(bytes);
        const manifest = result.lifecycle.modelManifest;
        const embedding = descriptor.request.operation.name === 'embed';
        (embedding ? assertEmbeddingSourceIdentity : assertRerankSourceIdentity)(manifest.artifactIdentity, reference);
        result.comparisons = result.lifecycle.observations.map(({ phase, completed }) => ({ phase,
          ...(embedding ? evaluateEmbeddingReference(reference, {
            input: { texts: descriptor.request.input.texts }, embeddingContract: resolveCapsuleEmbeddingContract(manifest),
            outputs: completed.output.embeddings.map((item, i) => ({ text: descriptor.request.input.texts[i],
              tokenIds: item.tokens, embedding: item.embedding })),
          }) : evaluateRerankReference(reference, { input: { query: descriptor.request.input.query,
            documents: descriptor.request.input.documents }, scoringConfig: manifest.inference.rerank,
            outputs: completed.output.evidence.scores })),
        }));
        assert(result.comparisons.every(comparison => comparison.passed), 'Frozen numerical reference comparison failed.');
      }
      if (row.expectedTokenIds && result.lifecycle) {
        for (const observation of result.lifecycle.observations) assert.deepEqual(observation.completed.output.tokenIds, row.expectedTokenIds);
      }
      report.results.push({ passed: true, operation: descriptor.request.operation.name, hardware, ...result });
    } catch (error) {
      report.results.push({ passed: false, operation: row.descriptor.request.operation.name, hardware,
        observation: result,
        lifecycleProgress,
        error: { message: timedOut ? `Consumer exceeded ${config.timeoutMs}ms: ${error.message}` : error.message, stack: error.stack } });
    } finally {
      clearTimeout(timeout);
      if (config.measureMemory === true && !page.isClosed()) {
        try {
          report.progress.push({ operation: row.descriptor.request.operation.name, stage: 'after-session-cleanup',
            memory: await page.evaluate(() => readCapabilityMemory()) });
        } catch (error) {
          report.logs.push({ type: 'memory-observation-failed', text: error.message });
        }
      }
      await page.close();
      await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report.results, null, 2));
    }
  }
  report.passed = report.results.length === config.models.length && report.results.every(row => row.passed);
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  const cleanup = await Promise.allSettled([browser?.close(), server?.close()]);
  report.cleanupErrors = cleanup.filter(row => row.status === 'rejected').map(row => row.reason.message);
  report.passed &&= report.cleanupErrors.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'receipt.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, error: report.error }));
if (!report.passed) process.exitCode = 1;
