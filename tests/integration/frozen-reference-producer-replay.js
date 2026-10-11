// Forensic replay of a retained archive, not qualification or a new reference.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { tmpdir } from 'node:os';
import { resolve, join, sep, extname } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { compareReferenceModel } from '../../../reploid/tests/fixtures/distributed-reference-model.js';

const [archivePath, modelDirectory, destination, manifestMode = 'frozen'] = process.argv.slice(2);
const observationTarget = process.argv[6] ? JSON.parse(await readFile(process.argv[6], 'utf8')) : { index: 0, step: 0 };
assert(Number.isSafeInteger(observationTarget.index) && observationTarget.index >= 0
  && Number.isSafeInteger(observationTarget.step) && observationTarget.step >= 0);
const captureDecode = Boolean(process.argv[6]);
const fullPrefixes = process.env.DOPPLER_FORENSIC_FULL_PREFIXES === '1';
assert(archivePath && modelDirectory && destination, 'Supply retained archive, unchanged model directory and receipt');
assert(['frozen', 'current'].includes(manifestMode), 'Manifest mode must be frozen or current');
assert(['darwin', 'linux'].includes(process.platform));
const hash = data => createHash('sha256').update(data).digest('hex');
const fixtureRoot = new URL('../../../reploid/tests/fixtures/', import.meta.url);
const referenceBytes = gunzipSync(await readFile(new URL('distributed-reference.json.gz', fixtureRoot)));
assert.equal(hash(referenceBytes), '9444f0d632de4b51624752a8c3d05a1e7cd7aea4b4ebaef71d96663bb650b6bd');
const reference = JSON.parse(referenceBytes);
const frozenManifestBytes = gunzipSync(await readFile(new URL('distributed-reference-manifest.json.gz', fixtureRoot)));
assert.equal('sha256:' + hash(frozenManifestBytes), reference.modelIdentity);
const manifestBytes = manifestMode === 'current'
  ? await readFile(resolve(modelDirectory, 'manifest.json')) : frozenManifestBytes;
const manifest = JSON.parse(manifestBytes);
const changedKernelPins = compareReferenceModel(manifest, JSON.parse(frozenManifestBytes));
const policy = JSON.parse(await readFile(new URL('../../../reploid/self/config/partition-policy.json', import.meta.url)));
const directory = await mkdtemp(join(tmpdir(), 'doppler-reference-producer-'));
const receipt = { scope: 'Retained archive forensic fixed-prefix replay; producing ownership is unproven, no reference replacement or release acceptance',
  host: process.platform, producerSha256: hash(await readFile(new URL(import.meta.url))), archiveSha256: hash(await readFile(archivePath)),
  referenceSha256: hash(referenceBytes), manifestIdentity: 'sha256:' + hash(manifestBytes), results: [],
  frozenManifestIdentity: reference.modelIdentity, manifestMode, changedKernelPins,
  servedShaders: {}, modelDirectory: resolve(modelDirectory), fullPrefixes, observationTarget };
async function recordPrefill(row) {
  const retained = { ...row };
  receipt.results.push(retained);
  const decode = text => {
    const bytes = Buffer.from(text, 'base64');
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  };
  const actual = decode(row.logits), expected = decode(reference.expected[row.index].steps[row.step].logits);
  assert.equal(actual.length, expected.length);
  let maxDifference = 0, maxDifferenceIndex = null;
  for (let i = 0; i < actual.length; i++) {
    assert(Number.isFinite(actual[i]) && Number.isFinite(expected[i]));
    const error = Math.abs(actual[i] - expected[i]);
    if (error > maxDifference) { maxDifference = error; maxDifferenceIndex = i; }
  }
  Object.assign(retained, { maxDifference, maxDifferenceIndex, tolerance: 0.001, matches: maxDifference <= 0.001 });
}

let browser, server;
try {
  await promisify(execFile)('tar', ['-xzf', resolve(archivePath), '-C', directory]);
  const runtimeRoot = join(directory, 'package');
  const metadata = JSON.parse(await readFile(join(runtimeRoot, 'package.json')));
  receipt.package = { name: metadata.name, version: metadata.version };
  assert.equal(metadata.name, 'doppler-gpu');
  const compatEntry = metadata.exports['./compat'].import;
  const partitionsEntry = metadata.exports['./partitions'].import;
  const shaderMismatches = [];
  for (const [id, kernel] of Object.entries(manifest.inference.execution.kernels)) {
    const shader = (await readFile(join(runtimeRoot, 'src/gpu/kernels', kernel.kernel), 'utf8')).replaceAll('\r\n', '\n');
    const actual = 'sha256:' + hash(shader + '\n@@entry:' + kernel.entry);
    if (actual !== kernel.digest) shaderMismatches.push({ id, file: kernel.kernel, entry: kernel.entry,
      declaredDigest: kernel.digest, actualDigest: actual });
  }
  receipt.declaredShaderMismatches = shaderMismatches;
  if (manifestMode === 'current') assert.equal(shaderMismatches.length, 0,
    'Current-manifest replay requires exact archive shader pins');
  server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return; }
      if (pathname === '/model/manifest.json') { response.setHeader('Content-Type', 'application/json'); response.end(manifestBytes); return; }
      const root = pathname.startsWith('/runtime/') ? runtimeRoot
        : pathname.startsWith('/model/') ? resolve(modelDirectory) : null;
      if (!root) { response.writeHead(404); response.end(); return; }
      const file = resolve(root, pathname.slice(pathname.startsWith('/runtime/') ? 9 : 7));
      assert(file.startsWith(root + sep), 'Path escapes the owned fixture root');
      const bytes = await readFile(file);
      if (extname(file) === '.wgsl') receipt.servedShaders[pathname] = hash(bytes);
      response.setHeader('Content-Type', extname(file) === '.js' ? 'text/javascript'
        : extname(file) === '.json' ? 'application/json' : extname(file) === '.wasm' ? 'application/wasm' : 'application/octet-stream');
      response.end(bytes);
    } catch (error) { response.writeHead(404); response.end(error.message); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal']
      : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.exposeFunction('recordProducerPrefill', recordPrefill);
  page.on('console', event => { if (event.text().startsWith('producer-progress:')) console.log(event.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const observedTimeline = [];
  await page.exposeFunction('recordProducerObservation', row => { observedTimeline.push(row); });
  const result = await page.evaluate(async ({ compatEntry, partitionsEntry, generation, prompts, policy, fullPrefixes, expected, observationTarget, captureDecode }) => {
    // Observe selected native pipelines without rewriting shader arithmetic.
    const moduleSources = new WeakMap(), pipelines = new WeakMap(), normalizationDispatch = [];
    const createModule = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (descriptor) {
      const module = createModule.call(this, descriptor);
      moduleSources.set(module, descriptor.code); return module;
    };
    const recordPipeline = (pipeline, descriptor) => {
      const source = moduleSources.get(descriptor.compute.module);
      if (source?.includes('RMSNorm Kernel with Fused Residual Add')) {
        const record = { entry: descriptor.compute.entryPoint, constants: descriptor.compute.constants,
          source, selections: 0 };
        normalizationDispatch.push(record); pipelines.set(pipeline, record);
      }
      return pipeline;
    };
    const createPipeline = GPUDevice.prototype.createComputePipeline;
    GPUDevice.prototype.createComputePipeline = function (descriptor) {
      return recordPipeline(createPipeline.call(this, descriptor), descriptor);
    };
    const createPipelineAsync = GPUDevice.prototype.createComputePipelineAsync;
    GPUDevice.prototype.createComputePipelineAsync = function (descriptor) {
      return createPipelineAsync.call(this, descriptor).then(pipeline => recordPipeline(pipeline, descriptor));
    };
    const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      const record = pipelines.get(pipeline); if (record) record.selections++;
      return setPipeline.call(this, pipeline);
    };
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/runtime/src/gpu/kernels/';
    const { load, DOPPLER_VERSION } = await import('/runtime/' + compatEntry.replace(/^\.\//, ''));
    const runtime = await import('/runtime/' + partitionsEntry.replace(/^\.\//, ''));
    const { createHttpArtifactStorageContext } = await import('/runtime/src/storage/artifact-storage-context.js');
    const { getDevice } = await import('/runtime/src/gpu/device.js');
    await runtime.configureDeviceMemoryBudget({ maxBytes: 6000000000 });
    const source = new URL('/model/', location.href).href;
    const manifestText = await (await fetch(source + 'manifest.json')).text();
    const manifest = JSON.parse(manifestText);
    const manifestHash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
      new TextEncoder().encode(manifestText))), b => b.toString(16).padStart(2, '0')).join('');
    const rows = [];
    let handle;
    try {
      handle = await load({ manifest, manifestText, manifestHash, baseUrl: source,
        storage: createHttpArtifactStorageContext(source, manifest, { verifyHashes: true }) }, {
        onProgress: progress => console.log('producer-progress:' + JSON.stringify(progress)),
        runtimeConfig: { shared: { bufferPool: policy.bufferPool }, inference: { session: {
          kvcache: { maxSeqLen: generation.maxSeqLen }, prefillChunkLayers: policy.prefillChunkLayers,
          prefillTokenChunkSize: policy.prefillTokenChunkSize } } } });
      const { maxSeqLen: _contextLength, ...executionOptions } = generation;
      for (const index of [...prompts.keys(), 0]) {
        const messages = prompts[index];
        await handle.resetGenerationState();
        const inputIds = handle.advanced.tokenizePrompt(messages, generation);
        const record = async (value, step) => {
          const bytes = new Uint8Array(value.logits.buffer, value.logits.byteOffset, value.logits.byteLength);
          let text = ''; for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
          rows.push({ index, step, ordinal: rows.length, inputIds, logits: btoa(text), stats: handle.advanced.getStats(),
            resolvedRuntimeSession: typeof handle.advanced.getResolvedRuntimeSession === 'function'
              ? handle.advanced.getResolvedRuntimeSession() : null });
          await globalThis.recordProducerPrefill(rows.at(-1));
        };
        const value = await handle.advanced.prefillWithLogits(messages, { ...executionOptions, inputIds });
        try { await record(value, 0); } finally { value.cache?.destroy(); }
        if (fullPrefixes) {
          for (let step = 1; step < expected[index].steps.length; step++) {
            const value = await handle.advanced.decodeStepLogits([expected[index].steps[step - 1].tokenId], executionOptions);
            await record(value, step);
          }
        }
      }
      if (fullPrefixes && !captureDecode) return { runtimeVersion: DOPPLER_VERSION, rows, observation: null, normalizationDispatch };
      // Observe the same first prefix after reset; quantify instrumentation effects.
      await handle.resetGenerationState();
      let observedLogits = null; const observedSteps = [];
      const targetOpIds = ['embed.out', 'final_norm.pre', 'final_norm.out',
        ...['qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj',
          'linear_core_out', 'out', 'post_attn'].map(op => 'layer.0.attn.' + op),
        ...['in', 'gate', 'up', 'act', 'out'].map(op => 'layer.0.ffn.' + op),
        ...manifest.inference.layerPattern.layerTypes.flatMap((type, layer) =>
          type === 'full_attention'
            ? ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'q_rope', 'k_rope', 'core_out', 'out']
              .map(op => `layer.${layer}.attn.${op}`)
            : []),
        ...Array.from({ length: manifest.architecture.numLayers }, (_, layer) =>
          [`layer.${layer}.attn.post_input_norm`, `layer.${layer}.layer.out`]).flat()];
      for await (const _chunk of handle.generate(prompts[observationTarget.index], { ...executionOptions,
        disableCommandBatching: true, diagnostics: { enabled: true,
          captureConfig: { enabled: true, defaultLevel: 'none', targetOpIds, targetLevel: 'full' } },
        onLogits: values => { observedLogits = Array.from(values); observedSteps.push(observedLogits); },
      })) { if (observedSteps.length > observationTarget.step) break; }
      const { operatorDiagnostics } = handle.advanced.getStats();
      // Transfer one bounded operator observation at a time. Decimal arrays for
      // all attention histories can exceed the browser automation IPC limit.
      for (const row of operatorDiagnostics?.timeline || []) {
        let capture = row.capture;
        if (capture?.data) {
          const values = new Float32Array(capture.data);
          const bytes = new Uint8Array(values.buffer);
          let binary = '';
          for (let offset = 0; offset < bytes.length; offset += 32768) {
            binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
          }
          const { data: _data, ...metadata } = capture;
          capture = { ...metadata, encoding: 'base64-f32le', dataBase64: btoa(binary) };
        }
        await window.recordProducerObservation({ ...row, capture });
      }
      return { runtimeVersion: DOPPLER_VERSION, rows, normalizationDispatch, observation: { logits: observedLogits, observedSteps, target: observationTarget,
        samplingExcludedTokenIds: [handle.advanced.getSpecialTokens().pad, ...executionOptions.suppressTokenIds]
          .filter(Number.isInteger), timeline: [] } };
    } finally {
      await getDevice().queue.onSubmittedWorkDone();
      if (handle) await handle.unload();
      await getDevice().queue.onSubmittedWorkDone();
    }
  }, { compatEntry, partitionsEntry, generation: reference.generation, prompts: reference.prompts, policy,
    fullPrefixes, expected: reference.expected, observationTarget, captureDecode });
  receipt.physicalExecutionCompleted = true;
  receipt.runtimeVersion = result.runtimeVersion;
  receipt.normalizationDispatch = result.normalizationDispatch.map(({ source, ...record }) => ({
    ...record, shaderSha256: hash(source) }));
  receipt.observation = result.observation;
  if (receipt.observation) receipt.observation.timeline = observedTimeline;
  assert.equal(result.runtimeVersion, metadata.version);
  receipt.executionCompleted = true;
  receipt.referenceProducingOwnershipEstablished = false;
  if (fullPrefixes) {
    const expectedCount = reference.expected.reduce((count, row) => count + row.steps.length, 0)
      + reference.expected[0].steps.length;
    assert.equal(receipt.results.length, expectedCount, 'Every fixed prefix and the first request reuse must complete');
  }
  if (result.observation) {
    const first = receipt.results.findLast(row => row.index === observationTarget.index && row.step === observationTarget.step);
    assert(first, 'Requested observation prefix must also have an uninstrumented control');
    assert.equal(result.observation.observedSteps.length, observationTarget.step + 1);
    const bytes = Buffer.from(first.logits, 'base64');
    const plain = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const excluded = new Set(result.observation.samplingExcludedTokenIds);
    assert.equal(result.observation.logits.length, plain.length);
    let maximumUnmaskedDifference = 0;
    for (let i = 0; i < plain.length; i++) {
      if (!excluded.has(i)) {
        assert(Number.isFinite(result.observation.logits[i]), 'Unmasked observed logits must be finite');
        maximumUnmaskedDifference = Math.max(maximumUnmaskedDifference,
          Math.abs(result.observation.logits[i] - plain[i]));
      }
    }
    receipt.observation.maximumUnmaskedDifference = maximumUnmaskedDifference;
    assert.equal(maximumUnmaskedDifference, 0, 'Observation must preserve the controlled output logits');
  }
  console.log(JSON.stringify({ package: receipt.package, shaderPinMismatches: shaderMismatches.length,
    comparisons: receipt.results.map(({ index, inputIds, maxDifference, matches }) => ({ index, inputTokens: inputIds.length, maxDifference, matches })) }));
} catch (error) { receipt.executionCompleted = false; receipt.failure = error.message; throw error; }
finally {
  await writeFile(destination, JSON.stringify(receipt)); await browser?.close();
  if (server) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  await rm(directory, { recursive: true, force: true });
}
