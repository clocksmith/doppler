#!/usr/bin/env node
// Pre-sealing qualification. Uses installed advanced execution with explicit
// artifact/shader storage; this is not a signed-Capsule acceptance receipt.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';

const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
const baseline = await read(config.baselineReceipt);
const qualification = config.qualificationReportPath ? await read(config.qualificationReportPath) : null;
const reference = config.sourceTranscriptPath ? await read(config.sourceTranscriptPath) : null;
const bundle = await read(path.join(config.bundleRoot, 'receipt.json'));
assert(bundle.passed && baseline.passed);
assert.equal(createHash('sha256').update(await fs.readFile(path.join(config.bundleRoot, bundle.package.filename))).digest('hex'), bundle.package.sha256);
const installedRoot = path.join(config.bundleRoot, 'consumer/node_modules/doppler-gpu');
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.gpu-token-selection-probe/v1', passed: false, signedCapsule: false,
  package: bundle.package, config, rows: [], logs: [],
  probeSha256: createHash('sha256').update(await fs.readFile(new URL(import.meta.url))).digest('hex'),
  sourceSha256: createHash('sha256').update(await fs.readFile(path.join(config.sourceRoot, config.sourceFile))).digest('hex') };
let browser, server, page, timer;
try {
  server = await createStaticFileServer({ rootDir: installedRoot, host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/candidate', rootDir: config.sourceRoot }] });
  browser = await chromium.launch({ channel: config.channel, headless: true, args: config.launchArgs,
    env: { ...process.env, TMPDIR: config.temporaryDirectory } });
  report.browserVersion = browser.version();
  timer = setTimeout(() => { void browser.close(); }, config.timeoutMs);
  page = await browser.newPage();
  page.on('console', message => report.logs.push(message.text()));
  page.on('pageerror', error => report.logs.push(error.message));
  await page.route('**/probe', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>GPU selection qualification</title>' }));
  await page.goto(server.baseUrl + '/probe');
  await page.exposeFunction('probeProgress', row => console.log(JSON.stringify(row)));
  const result = await page.evaluate(async ({ request, expected, sourceFile, runtimeConfig, qualification, reference }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
    const [{ load }, { createCapsuleArtifactSource }, { createFetchCapsuleArtifactStore }, { observeInitialExecutionIdentity }, { createVerifiedCapsuleArtifactStore }] = await Promise.all([
      import('/src/client/doppler-api.browser.js'), import('/src/client/runtime/capsule-artifact-source.js'),
      import('/src/client/runtime/fetch-capsule-artifact-store.js'), import('/src/config/initial-execution-identity.js'),
      import('/src/client/runtime/verified-capsule-artifact-store.js'),
    ]);
    const sourceUrl = location.origin + '/candidate/' + sourceFile;
    const candidate = await (await fetch(sourceUrl)).json();
    const store = createVerifiedCapsuleArtifactStore(candidate, createFetchCapsuleArtifactStore(sourceUrl));
    globalThis.probeStore = store;
    const source = await createCapsuleArtifactSource(candidate, store);
    const started = performance.now();
    const model = await load(source, { runtimeConfig, isolatedLoader: true });
    globalThis.probeModel = model;
    await globalThis.probeProgress({ stage: 'loaded', elapsedMs: performance.now() - started });
    const identity = observeInitialExecutionIdentity(model.advanced.getResolvedRuntimeSession());
    const promptTokens = model.advanced.tokenizePrompt(request.input.prompt, { useChatTemplate: request.options.useChatTemplate });
    const special = model.advanced.getSpecialTokens();
    const contract = { padTokenId: Number.isInteger(special.pad) ? special.pad : null };
    const options = { ...request.options, useChatTemplate: false };
    const rows = [];
    for (let repeat = 0; repeat < 2; repeat++) {
      model.resetGenerationState();
      const context = [...promptTokens], tokens = [], start = performance.now();
      for (let index = 0; index < expected.length; index++) {
        const result = index === 0
          ? await model.advanced.prefillWithToken('', { ...options, inputIds: promptTokens }, contract)
          : await model.advanced.decodeStepWithToken(context, options, contract);
        tokens.push(result.tokenId); context.push(result.tokenId);
        if (result.tokenId !== expected[index]) throw new Error(`First selected-token divergence at ${index}: expected ${expected[index]}, received ${result.tokenId}.`);
      }
      rows.push({ repeat, tokenIds: tokens, elapsedMs: performance.now() - start });
      await globalThis.probeProgress({ stage: 'generated', repeat, tokens: tokens.length, elapsedMs: rows.at(-1).elapsedMs });
    }
    let sourceParity = null;
    if (reference) {
      const referencePrompt = qualification.metrics.prompt;
      const inputIds = model.advanced.tokenizePrompt(referencePrompt, { useChatTemplate: true });
      if (JSON.stringify(inputIds) !== JSON.stringify(reference.promptTokenIds)) throw new Error('Source prompt token mismatch.');
      model.resetGenerationState();
      const context = [...inputIds], tokenIds = [];
      const sourceOptions = { ...options, ...qualification.metrics.referenceTranscript.generationConfig, useChatTemplate: false };
      for (let index = 0; index < reference.generatedTokenIds.length; index++) {
        const result = index === 0
          ? await model.advanced.prefillWithToken('', { ...sourceOptions, inputIds }, contract)
          : await model.advanced.decodeStepWithToken(context, sourceOptions, contract);
        tokenIds.push(result.tokenId); context.push(result.tokenId);
        if (result.tokenId !== reference.generatedTokenIds[index]) throw new Error(`Source token divergence at ${index}: expected ${reference.generatedTokenIds[index]}, received ${result.tokenId}.`);
      }
      sourceParity = { passed: true, promptTokenIds: inputIds, tokenIds };
      await globalThis.probeProgress({ stage: 'source-parity', tokens: tokenIds.length });
    }
    return { hardware, initialExecutionIdentity: identity, promptTokens, rows, sourceParity };
  }, { request: baseline.config.model.descriptor.request, expected: baseline.runs[0].completed.output.tokenIds,
    sourceFile: config.sourceFile, runtimeConfig: config.runtimeConfig, qualification, reference });
  Object.assign(report, result);
  assert.equal(result.hardware.vendor, config.requiredVendor);
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearTimeout(timer);
  report.cleanupErrors = [];
  if (page && !page.isClosed()) {
    try { await page.evaluate(async () => { try { await globalThis.probeModel?.unload(); } finally { globalThis.probeStore?.close(); } }); }
    catch (error) { report.cleanupErrors.push(error.message); }
  }
  for (const resource of [browser, server]) {
    try { await resource?.close(); } catch (error) { report.cleanupErrors.push(error.message); }
  }
  report.passed &&= report.cleanupErrors.length === 0;
  await fs.writeFile(path.join(config.outputDir, 'receipt.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ passed: report.passed, error: report.error, outputDir: config.outputDir }));
if (!report.passed) process.exitCode = 1;
