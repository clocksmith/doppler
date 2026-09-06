#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { _electron as electron } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { assertRerankReference, evaluateRerankReference, assertRerankSourceIdentity } from '../src/config/rerank-reference.js';
import { assertPhysicalAdapter } from './probe-electron-reranker.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';
import { hashStableJson } from '../src/tooling/program-bundle/materialize.js';
import { validateCaptureConfig } from '../src/debug/capture-policy.js';
import ts from 'typescript';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);

export function compileElectronReleasePreload(source, channel) {
  if (typeof channel !== 'string' || !channel) throw new Error('Electron preload requires its installed channel import.');
  let imports = 0;
  const parsed = ts.createSourceFile('preload.js', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.JS);
  const transformed = ts.transform(parsed, [context => node => ts.visitEachChild(node, function visit(child) {
      if (!ts.isImportDeclaration(child)) return ts.visitEachChild(child, visit, context);
      const bindings = child.importClause?.namedBindings;
      if (child.moduleSpecifier.text !== 'doppler-gpu/electron' || !bindings || !ts.isNamedImports(bindings)
        || bindings.elements.length !== 1 || bindings.elements[0].name.text !== 'ELECTRON_RELEASE_IPC_CHANNEL'
        || bindings.elements[0].propertyName || child.importClause.name) {
        throw new Error('Unsupported installed Electron preload import; update the explicit compilation boundary.');
      }
      imports += 1;
      return ts.factory.createVariableStatement(undefined, ts.factory.createVariableDeclarationList([
        ts.factory.createVariableDeclaration('ELECTRON_RELEASE_IPC_CHANNEL', undefined, undefined, ts.factory.createStringLiteral(channel)),
      ], ts.NodeFlags.Const));
    }, context)]);
  let detachedSource;
  try { detachedSource = ts.createPrinter().printFile(transformed.transformed[0]); }
  finally { transformed.dispose(); }
  if (imports !== 1) throw new Error('Electron preload requires its installed channel import.');
  // Rebind the printed source: CommonJS emission must not retain the old import's symbol.
  const compiled = ts.transpileModule(detachedSource, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  return `const { contextBridge, ipcRenderer } = require('electron');\n${compiled.outputText}\nexposeDocumentSearchReleaseBridge(contextBridge, ipcRenderer);\n`;
}

export async function resolveRerankerPackDistribution(packPath, distributionRoot) {
  const root = await fs.realpath(distributionRoot ?? path.dirname(packPath));
  const manifestPath = await fs.realpath(packPath);
  const relative = path.relative(root, manifestPath);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error('Qualification Pack must reside inside its declared distribution root.');
  }
  return { root, manifestPath, urlPath: `/pack/${relative.split(path.sep).map(encodeURIComponent).join('/')}` };
}

export async function qualifyRerankerElectron(config) {
  if (config?.packDistributionRoot !== undefined && (config.mode !== 'pack'
    || typeof config.packDistributionRoot !== 'string' || !path.isAbsolute(config.packDistributionRoot))) {
    throw new Error('Pack distribution root requires Pack mode and an absolute directory.');
  }
  if (config?.releaseCoordinator !== undefined && (config.mode !== 'pack'
    || typeof config.releaseCoordinator?.statePath !== 'string' || !path.isAbsolute(config.releaseCoordinator.statePath)
    || !config.releaseCoordinator.trustedSigners || !Array.isArray(config.releaseCoordinator.actions)
    || !Array.isArray(config.releaseCoordinator.allowedRendererActions)
    || config.releaseCoordinator.allowedRendererActions.some(action => !['status', 'resolve-current'].includes(action))
    || typeof config.releaseCoordinator.now !== 'string')) {
    throw new Error('Release coordinator qualification requires Pack mode, durable state, trust, explicit actions, clock and read-only renderer permissions.');
  }
  if (config?.releaseCheckpointPath !== undefined && (config.mode !== 'pack'
    || typeof config.releaseCheckpointPath !== 'string' || !path.isAbsolute(config.releaseCheckpointPath)
    || !config.openOptions?.releaseEvents || !config.openOptions?.releaseTrustedSigners
    || !config.openOptions?.releasePolicy
    || Object.keys(config.openOptions.releasePolicy).some(key => !['now', 'minimumSequence'].includes(key)))) {
    throw new Error('Durable release qualification requires Pack mode, an absolute releaseCheckpointPath, signed history and policy without an injected checkpoint.');
  }
  const diagnosticCapture = config?.diagnosticCapture ?? null;
  if (diagnosticCapture) {
    if (config.mode !== 'model' || !Number.isInteger(diagnosticCapture.documentIndex)
      || diagnosticCapture.documentIndex < 0 || !diagnosticCapture.captureConfig) {
      throw new Error('diagnosticCapture requires model mode, a non-negative documentIndex and captureConfig.');
    }
    validateCaptureConfig(diagnosticCapture.captureConfig);
  }
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
  let applicationFiles = null;
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
    applicationFiles = receipt.applicationFiles;
  }
  const policy = JSON.parse(await fs.readFile(config.policyPath, 'utf8'));
  if (require('electron/package.json').version !== policy.electronVersion) throw new Error('Pinned Electron required.');
  const reference = assertRerankReference(JSON.parse(await fs.readFile(config.referencePath, 'utf8')));
  if (diagnosticCapture && diagnosticCapture.documentIndex >= reference.input.documents.length) {
    throw new Error('diagnosticCapture.documentIndex exceeds the frozen reference documents.');
  }
  const manifestBytes = await fs.readFile(path.join(config.modelDir, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  assertRerankSourceIdentity(manifest.artifactIdentity, reference);
  if (manifest.modelId !== policy.modelId || manifest.artifactIdentity?.sourceCheckpointId !== reference.source.checkpointId) {
    throw new Error('Frozen source and model identity differ.');
  }
  const packDistribution = config.packPath
    ? await resolveRerankerPackDistribution(config.packPath, config.packDistributionRoot) : null;
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
  let checkpointStore;
  let releaseOptions;
  let coordinatorReceiptPath;
  try {
    if (config.releaseCheckpointPath) {
      report.stage = 'release-history-verification';
      const stateDirectory = await fs.realpath(path.dirname(config.releaseCheckpointPath));
      for (const servedRoot of [config.packageRoot, config.modelDir, packDistribution.root]) {
        const relative = path.relative(await fs.realpath(servedRoot), stateDirectory);
        if (relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))) {
          throw new Error('Release checkpoint must remain outside every served root.');
        }
      }
      const helperPath = path.resolve(config.packageBundlePath, 'consumer/release-storage.js');
      const helperDigest = hashBytesSha256(await fs.readFile(helperPath));
      if (helperDigest !== `sha256:${applicationFiles?.['release-storage.js']?.sha256}`) {
        throw new Error('Release storage helper does not match the retained installed-package test.');
      }
      const helper = await import(pathToFileURL(helperPath).href);
      checkpointStore = helper.createDocumentSearchCheckpointStore(config.releaseCheckpointPath);
      report.releaseHistory = { helperDigest, before: await checkpointStore.load(), persistence: [],
        verificationTime: config.openOptions.releasePolicy.now, clockAuthority: 'explicit-evaluation-policy' };
      releaseOptions = await helper.prepareDocumentSearchReleaseOptions({
        pack: JSON.parse(await fs.readFile(config.packPath, 'utf8')),
        releaseEvents: config.openOptions.releaseEvents, releaseTrustedSigners: config.openOptions.releaseTrustedSigners,
        checkpointStore, now: config.openOptions.releasePolicy.now,
        minimumSequence: config.openOptions.releasePolicy.minimumSequence,
      });
    }
    report.stage = 'launch';
    server = await createStaticFileServer({ rootDir: path.resolve(config.packageRoot), host: '127.0.0.1',
      staticMounts: [{ urlPrefix: '/model', rootDir: path.resolve(config.modelDir) },
        ...(packDistribution ? [{ urlPrefix: '/pack', rootDir: packDistribution.root }] : [])] });
    let mainConfigPath;
    if (config.releaseCoordinator) {
      const stateDirectory = await fs.realpath(path.dirname(config.releaseCoordinator.statePath));
      for (const root of [config.packageRoot, config.modelDir, packDistribution.root]) {
        const relative = path.relative(await fs.realpath(root), stateDirectory);
        if (!relative || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))) {
          throw new Error('Coordinator state must remain outside every served root.');
        }
      }
      const consumerDir = path.resolve(config.packageBundlePath, 'consumer');
      for (const file of ['main.js', 'preload.js', 'release-storage.js']) {
        if (hashBytesSha256(await fs.readFile(path.join(consumerDir, file))) !== `sha256:${applicationFiles?.[file]?.sha256}`) {
          throw new Error(`Installed application helper differs from retained package: ${file}.`);
        }
      }
      const packageJson = JSON.parse(await fs.readFile(path.join(config.packageRoot, 'package.json'), 'utf8'));
      const electronEntry = path.resolve(config.packageRoot, packageJson.exports['./electron'].import);
      const exported = await import(pathToFileURL(electronEntry).href);
      const preload = compileElectronReleasePreload(await fs.readFile(path.join(consumerDir, 'preload.js'), 'utf8'), exported.ELECTRON_RELEASE_IPC_CHANNEL);
      const preloadPath = path.resolve(config.outputDir, 'preload.js');
      await fs.writeFile(preloadPath, preload, { flag: 'wx' });
      coordinatorReceiptPath = path.resolve(config.outputDir, 'coordinator.json');
      mainConfigPath = path.resolve(config.outputDir, 'coordinator-config.json');
      await fs.writeFile(mainConfigPath, JSON.stringify({ ...config.releaseCoordinator, consumerDir, electronEntry,
        preloadPath, receiptPath: coordinatorReceiptPath, allowedOrigin: server.baseUrl }), { flag: 'wx' });
      report.coordinatorPreload = { sha256: hashBytesSha256(new TextEncoder().encode(preload)), transpilerVersion: ts.version,
        sourceDigest: applicationFiles['preload.js'].sha256, format: 'generated-sandbox-commonjs' };
      report.boundary.applicationAuthorization = 'installed-main-coordinator-with-frame-and-action-policy';
      report.boundary.referenceIpc = true;
    }
    application = await electron.launch({ executablePath: require('electron'), timeout: policy.timeoutMs,
      args: [...policy.launchArgs, `--doppler-probe-user-data=${path.resolve(config.outputDir, 'user-data')}`,
        ...(mainConfigPath ? [`--doppler-release-main=${mainConfigPath}`] : []),
        path.join(ROOT, 'tools/fixtures/electron-webgpu-main.js')] });
    timer = setTimeout(() => { application.close().catch(() => {}); }, policy.timeoutMs);
    const page = await application.firstWindow();
    if (releaseOptions) await page.exposeFunction('__persistRerankerReleaseCheckpoint', async (value) => {
      const observation = { checkpoint: structuredClone(value), persisted: false };
      report.releaseHistory.persistence.push(observation);
      try {
        await releaseOptions.persistReleaseCheckpoint(value);
        observation.persisted = true;
      } catch (error) { observation.error = error.message; throw error; }
    });
    page.on('console', (message) => report.logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', (error) => report.logs.push({ type: 'pageerror', text: error.message }));
    await page.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      report.requests.push(url.href);
      if (url.origin !== server.baseUrl || (config.mode === 'pack' && url.pathname.startsWith('/model/'))) return route.abort();
      if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Reranker qualification</title>' });
      if (faultArtifact && url.pathname === `${path.posix.dirname(packDistribution.urlPath)}/${faultArtifact.path}`) {
        report.faultInjected = true;
        if (faultKind === 'artifact-interruption') return route.abort('connectionreset');
        const bytes = await fs.readFile(faultArtifactPath);
        bytes[0] ^= 1;
        return route.fulfill({ contentType: 'application/octet-stream', body: bytes });
      }
      return route.continue();
    });
    await page.goto(`${server.baseUrl}/qualification`);
    if (config.releaseCoordinator) {
      report.deniedRendererMutation = await page.evaluate(async () => {
        try { await globalThis.dopplerRelease.rollback(`sha256:${'0'.repeat(64)}`); return { denied: false }; }
        catch (error) { return { denied: /not authorized/.test(error.message), message: error.message }; }
      });
      if (report.deniedRendererMutation.denied !== true) throw new Error('Renderer mutation was not rejected by the application policy.');
    }
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
        let diagnostic = null;
        let executed;
        if (config.mode === 'pack') {
          const { createElectronRendererRuntime } = await import('/src/client/electron/renderer-runtime.js');
          const renderer = createElectronRendererRuntime({
            releaseState: config.releaseCoordinator ? globalThis.dopplerRelease : { resolveCurrent: async () => ({ ...config.authorizedPack,
              path: `${location.origin}${config.packUrlPath}` }) },
            openPack: async (packPath, options) => {
              session = await api.openPack(packPath, { ...config.openOptions, ...options,
                ...(config.releaseCheckpointPath ? { persistReleaseCheckpoint: globalThis.__persistRerankerReleaseCheckpoint } : {}) });
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
          executed = performance.now();
        } else {
          session = await api.load({ url: `${location.origin}/model/` }, { runtimeConfig });
          loaded = performance.now();
          initialExecutionIdentity = observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
          receipt = await session.rerankWithEvidence(input.query, input.documents);
          executed = performance.now();
          if (config.diagnosticCapture) {
            const { formatRerankPrompt } = await import('/src/inference/rerank.js');
            const { documentIndex, captureConfig } = config.diagnosticCapture;
            const scoring = session.manifest.inference.rerank;
            const prompt = formatRerankPrompt(input.query, input.documents[documentIndex], scoring);
            session.resetGenerationState();
            const output = await session.advanced.prefillWithTokenLogits(prompt,
              [scoring.trueTokenId, scoring.falseTokenId], { useChatTemplate: false,
                diagnostics: { enabled: true, captureConfig } });
            const operatorDiagnostics = session.advanced.getStats().operatorDiagnostics;
            if (!operatorDiagnostics?.recordCount) throw new Error('Requested operator captures were not retained.');
            const ordinary = receipt.scores[documentIndex];
            const matchesOrdinary = JSON.stringify(output.tokens) === JSON.stringify(ordinary.tokenIds)
              && output.logitsByTokenId[scoring.trueTokenId] === ordinary.trueLogit
              && output.logitsByTokenId[scoring.falseTokenId] === ordinary.falseLogit;
            diagnostic = { documentIndex, prompt, tokens: output.tokens, logitsByTokenId: output.logitsByTokenId,
              operatorDiagnostics, matchesOrdinary, elapsedMs: performance.now() - executed, performanceClaim: false };
          }
          const after = observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
          if (after.digest !== initialExecutionIdentity.digest) throw new Error('Model execution changed its initial execution identity.');
        }
        const evidence = config.mode === 'pack' ? receipt.evidence : receipt;
        return { evidence, receipt: config.mode === 'pack' ? receipt : null, initialExecutionIdentity,
          packIdentity: session.packIdentity ?? null, selectedTargetPlanDigest: session.selectedTargetPlanDigest ?? null,
          adapterClosedSession: config.mode === 'pack' ? session.closed : null,
          manifest: session.manifest, loadMs: loaded - started, executionMs: executed - loaded, diagnostic };
      } catch (error) {
        globalThis.__rerankQualificationFailure = { name: error.name, code: error.code ?? null,
          message: error.message, causeCode: error.cause?.code ?? null, sessionClosed: session?.closed ?? null };
        throw error;
      } finally { if (config.mode === 'pack') await session?.close(); else await session?.unload(); }
    }, { config: { ...config,
      openOptions: releaseOptions ? { ...config.openOptions, releasePolicy: releaseOptions.releasePolicy } : config.openOptions,
      packUrlPath: packDistribution?.urlPath ?? null },
      input: reference.input, runtimeConfig: policy.runtimeConfig });
    report.raw = result;
    if (result.diagnostic?.matchesOrdinary === false) throw new Error('Diagnostic execution differs from ordinary selected-token reranking.');
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
    if (checkpointStore) {
      try { (report.releaseHistory ??= {}).after = await checkpointStore.load(); }
      catch (error) { report.passed = false; report.checkpointReadError = error.message; }
    }
    const output = path.join(config.outputDir, 'qualification.json');
    await fs.writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
    for (const resource of [application, server]) {
      try { await resource?.close(); } catch (error) { report.passed = false; (report.cleanupErrors ??= []).push(error.message); }
    }
    if (coordinatorReceiptPath) {
      try { report.coordinator = JSON.parse(await fs.readFile(coordinatorReceiptPath, 'utf8')); }
      catch (error) { report.passed = false; report.coordinatorReceiptError = error.message; }
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
