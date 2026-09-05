#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { _electron as electron } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { assertRerankReference, evaluateRerankReference, assertRerankSourceIdentity } from '../src/config/rerank-reference.js';
import { assertPhysicalAdapter } from './probe-electron-reranker.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';
import { hashStableJson } from '../src/tooling/program-bundle/materialize.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);

export async function qualifyRerankerElectron(config) {
  const faultKind = config?.fault?.kind ?? null;
  if (![null, 'artifact-corruption', 'artifact-interruption', 'device-loss'].includes(faultKind)
    || (faultKind && config.mode !== 'pack')) throw new Error('Unsupported qualification fault or mode.');
  for (const field of ['policyPath', 'referencePath', 'modelDir', 'packageRoot', 'outputDir']) {
    if (typeof config?.[field] !== 'string' || !config[field].trim()) throw new Error(`Qualification requires ${field}.`);
  }
  if (!['model', 'pack'].includes(config.mode)) throw new Error('Qualification mode must be model or pack.');
  if (config.mode === 'pack' && (!config.packPath || !config.openOptions?.trustedSigners
    || !config.openOptions?.acceptedTargetPlanDigests?.length || !config.application
    || !config.authorizedPack?.packId || !config.authorizedPack?.semanticRoot || !config.packageBundlePath)) {
    throw new Error('Pack mode requires a retained packageBundlePath, packPath, application, authorizedPack, trustedSigners and acceptedTargetPlanDigests.');
  }
  let installedPackage = null;
  if (config.packageBundlePath) {
    const bundle = path.resolve(config.packageBundlePath);
    const receipt = JSON.parse(await fs.readFile(path.join(bundle, 'receipt.json'), 'utf8'));
    const tarball = await fs.readFile(path.join(bundle, receipt.package.filename));
    if (!receipt.passed || hashBytesSha256(tarball) !== `sha256:${receipt.package.sha256}`
      || path.resolve(config.packageRoot) !== path.join(bundle, 'consumer/node_modules/doppler-gpu')) {
      throw new Error('Installed runtime must match a passing retained package bundle.');
    }
    installedPackage = { ...receipt.package,
      source: JSON.parse(await fs.readFile(path.join(bundle, 'source-state.json'), 'utf8')) };
  }
  const policy = JSON.parse(await fs.readFile(config.policyPath, 'utf8'));
  if (require('electron/package.json').version !== policy.electronVersion) throw new Error('Pinned Electron required.');
  const reference = assertRerankReference(JSON.parse(await fs.readFile(config.referencePath, 'utf8')));
  const manifestBytes = await fs.readFile(path.join(config.modelDir, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  assertRerankSourceIdentity(manifest.artifactIdentity, reference);
  if (manifest.modelId !== policy.modelId || manifest.artifactIdentity?.sourceCheckpointId !== reference.source.checkpointId) {
    throw new Error('Frozen source and model identity differ.');
  }
  let faultArtifact = null;
  let faultArtifactPath = null;
  if (faultKind === 'artifact-corruption' || faultKind === 'artifact-interruption') {
    const pack = JSON.parse(await fs.readFile(config.packPath, 'utf8'));
    faultArtifact = pack.artifacts.find((artifact) => artifact.artifactId === config.fault.artifactId);
    if (!faultArtifact || faultArtifact.role !== 'weight-shard') throw new Error('Artifact fault requires a declared weight-shard artifactId.');
    const root = await fs.realpath(path.dirname(config.packPath));
    faultArtifactPath = await fs.realpath(path.resolve(root, faultArtifact.path));
    const relative = path.relative(root, faultArtifactPath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Fault artifact escapes Pack distribution directory.');
  }
  await fs.mkdir(path.dirname(config.outputDir), { recursive: true });
  await fs.mkdir(config.outputDir);
  const report = {
    schema: config.mode === 'model' ? 'doppler.rerankModelQualification.v1' : 'doppler.rerankPackQualification.v1',
    passed: false, generatedAt: new Date().toISOString(), config, policy, installedPackage,
    model: { modelId: manifest.modelId, manifestHash: hashBytesSha256(manifestBytes), artifactIdentity: manifest.artifactIdentity },
    reference, referenceDigest: computeCanonicalSha256(reference),
    runtime: { surface: 'browser-webgpu', host: 'electron', electronVersion: policy.electronVersion,
      executionGraphHash: hashStableJson(manifest.inference.execution),
      sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim() },
    sourceStatus: execFileSync('git', ['status', '--porcelain=v1'], { cwd: ROOT, encoding: 'utf8' }),
    boundary: { externalAdoption: false, sourceComparison: true, signedPackExecution: false,
      applicationAuthorization: 'pinned-internal-evaluation-resolver', productionIpc: false },
    logs: [], requests: [], faultInjected: false, stage: 'launch',
  };
  let server;
  let application;
  let timer;
  try {
    server = await createStaticFileServer({ rootDir: path.resolve(config.packageRoot), host: '127.0.0.1',
      staticMounts: [{ urlPrefix: '/model', rootDir: path.resolve(config.modelDir) },
        ...(config.packPath ? [{ urlPrefix: '/pack', rootDir: path.dirname(path.resolve(config.packPath)) }] : [])] });
    application = await electron.launch({ executablePath: require('electron'), timeout: policy.timeoutMs,
      args: [...policy.launchArgs, `--doppler-probe-user-data=${path.resolve(config.outputDir, 'user-data')}`,
        path.join(ROOT, 'tools/fixtures/electron-webgpu-main.js')] });
    timer = setTimeout(() => { application.close().catch(() => {}); }, policy.timeoutMs);
    const page = await application.firstWindow();
    page.on('console', (message) => report.logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', (error) => report.logs.push({ type: 'pageerror', text: error.message }));
    await page.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      report.requests.push(url.href);
      if (url.origin !== server.baseUrl || (config.mode === 'pack' && url.pathname.startsWith('/model/'))) return route.abort();
      if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Reranker qualification</title>' });
      if (faultArtifact && url.pathname === `/pack/${faultArtifact.path}`) {
        report.faultInjected = true;
        if (faultKind === 'artifact-interruption') return route.abort('connectionreset');
        const bytes = await fs.readFile(faultArtifactPath);
        bytes[0] ^= 1;
        return route.fulfill({ contentType: 'application/octet-stream', body: bytes });
      }
      return route.continue();
    });
    await page.goto(`${server.baseUrl}/qualification`);
    report.runtime.adapterInfo = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter.');
      return { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
        device: adapter.info.device, description: adapter.info.description, isFallbackAdapter: adapter.isFallbackAdapter };
    });
    assertPhysicalAdapter(report.runtime.adapterInfo, policy.requiredVendor);
    report.stage = config.mode === 'pack' ? 'pack-execution' : 'model-execution';
    const result = await page.evaluate(async ({ config, input, runtimeConfig }) => {
      const api = config.mode === 'pack' ? await import('/src/client/pack-host.browser.js')
        : await import('/src/client/doppler-api.browser.js');
      const { observeInitialExecutionIdentity } = await import('/src/config/initial-execution-identity.js');
      const started = performance.now();
      let session;
      let loaded;
      let initialExecutionIdentity;
      try {
        let receipt;
        if (config.mode === 'pack') {
          const { createElectronRendererRuntime } = await import('/src/client/electron/renderer-runtime.js');
          const renderer = createElectronRendererRuntime({
            releaseState: { resolveCurrent: async () => ({ ...config.authorizedPack,
              path: `${location.origin}/pack/${config.packFilename}` }) },
            openPack: async (packPath, options) => {
              session = await api.openPack(packPath, { ...config.openOptions, ...options });
              loaded = performance.now();
              initialExecutionIdentity = session.observedInitialExecutionIdentity;
              if (config.fault?.kind === 'device-loss') {
                const { getDevice } = await import('/src/gpu/device.js');
                const device = getDevice();
                globalThis.__rerankQualificationFaultInjected = true;
                device.destroy();
                await device.lost;
              }
              return session;
            },
          });
          receipt = await renderer.rerank({ application: config.application, ...input, options: {} });
        } else {
          session = await api.load({ url: `${location.origin}/model/` }, { runtimeConfig });
          loaded = performance.now();
          initialExecutionIdentity = observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
          receipt = await session.rerankWithEvidence(input.query, input.documents);
          const after = observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
          if (after.digest !== initialExecutionIdentity.digest) throw new Error('Model execution changed its initial execution identity.');
        }
        const evidence = config.mode === 'pack' ? receipt.evidence : receipt;
        return { evidence, receipt: config.mode === 'pack' ? receipt : null, initialExecutionIdentity,
          packIdentity: session.packIdentity ?? null, selectedTargetPlanDigest: session.selectedTargetPlanDigest ?? null,
          adapterClosedSession: config.mode === 'pack' ? session.closed : null,
          manifest: session.manifest, loadMs: loaded - started, executionMs: performance.now() - loaded };
      } catch (error) {
        globalThis.__rerankQualificationFailure = { name: error.name, code: error.code ?? null,
          message: error.message, causeCode: error.cause?.code ?? null, sessionClosed: session?.closed ?? null };
        throw error;
      } finally { if (config.mode === 'pack') await session?.close(); else await session?.unload(); }
    }, { config: { ...config, packFilename: config.packPath ? path.basename(config.packPath) : null },
      input: reference.input, runtimeConfig: policy.runtimeConfig });
    report.raw = result;
    if (config.mode === 'pack') {
      if (result.adapterClosedSession !== true) throw new Error('Electron adapter did not close its Pack session.');
      report.boundary.signedPackExecution = true;
    }
    if (computeCanonicalSha256(parseManifest(JSON.stringify(result.manifest))) !== computeCanonicalSha256(parseManifest(JSON.stringify(manifest)))) throw new Error('Loaded manifest differs.');
    report.observation = { input: { query: result.evidence.query, documents: result.evidence.documents },
      scoringConfig: result.manifest.inference.rerank, outputs: result.evidence.scores };
    report.result = evaluateRerankReference(reference, report.observation);
    report.initialExecutionIdentity = result.initialExecutionIdentity;
    report.passed = report.result.passed;
    report.stage = 'complete';
  } catch (error) {
    report.error = { name: error.name, message: error.message, stack: error.stack };
    if (application) {
      try { report.rendererFailure = await (await application.firstWindow()).evaluate(() => globalThis.__rerankQualificationFailure ?? null); }
      catch (observationError) { report.failureObservationError = observationError.message; }
    }
    if (faultKind === 'device-loss' && application) {
      try { report.faultInjected = await (await application.firstWindow()).evaluate(() => globalThis.__rerankQualificationFaultInjected === true); }
      catch (observationError) { report.faultObservationError = observationError.message; }
    }
  }
  finally {
    clearTimeout(timer);
    const output = path.join(config.outputDir, 'qualification.json');
    await fs.writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
    for (const resource of [application, server]) {
      try { await resource?.close(); } catch (error) { report.passed = false; (report.cleanupErrors ??= []).push(error.message); }
    }
    await fs.writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
  const report = await qualifyRerankerElectron(config);
  console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error ?? null,
    failedChecks: report.result?.checks.filter((check) => !check.passed) ?? [] }));
  if (!report.passed) process.exitCode = 1;
}
