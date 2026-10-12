// Forensic replay of a retained archive, not qualification or a new reference.
import assert from 'node:assert/strict';
import { readFile, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { tmpdir } from 'node:os';
import { resolve, join, sep, extname } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { compareReferenceModel } from '../../../reploid/tests/fixtures/distributed-reference-model.js';
import { validatePrefixOnlyTarget } from '../helpers/frozen-prefix-scope.js';
import { finishFrozenPrefixReplay } from '../helpers/frozen-prefix-receipt.js';

const [archivePath, modelDirectory, destination, manifestMode = 'frozen'] = process.argv.slice(2);
const observationTarget = process.argv[6] ? JSON.parse(await readFile(process.argv[6], 'utf8')) : { index: 0, step: 0 };
assert(Number.isSafeInteger(observationTarget.index) && observationTarget.index >= 0
  && Number.isSafeInteger(observationTarget.step) && observationTarget.step >= 0);
if (observationTarget.opIds !== undefined) {
  assert(Array.isArray(observationTarget.opIds) && observationTarget.opIds.length > 0
    && observationTarget.opIds.length <= 256);
  assert(observationTarget.opIds.every(id => typeof id === 'string' && /^[a-z0-9_.]+$/.test(id)));
  assert(observationTarget.opIds.includes('embed.out'));
}
const captureDecode = Boolean(process.argv[6]);
const physicalCache = observationTarget.physicalCache === true;
const fullPrefixes = process.env.DOPPLER_FORENSIC_FULL_PREFIXES === '1';
const recurrentStatePrecision = process.env.DOPPLER_RECURRENT_STATE_PRECISION ?? 'f32';
assert(['f32', 'two-f32'].includes(recurrentStatePrecision));
assert(archivePath && modelDirectory && destination, 'Supply retained archive, unchanged model directory and receipt');
assert(['frozen', 'current'].includes(manifestMode), 'Manifest mode must be frozen or current');
assert(['darwin', 'linux'].includes(process.platform));
const hash = data => createHash('sha256').update(data).digest('hex');
const fixtureRoot = new URL('../../../reploid/tests/fixtures/', import.meta.url);
const referenceBytes = gunzipSync(await readFile(new URL('distributed-reference.json.gz', fixtureRoot)));
assert.equal(hash(referenceBytes), '9444f0d632de4b51624752a8c3d05a1e7cd7aea4b4ebaef71d96663bb650b6bd');
const reference = JSON.parse(referenceBytes);
const prefixesOnly = validatePrefixOnlyTarget(observationTarget, reference);
assert(observationTarget.ropeTable === undefined || typeof observationTarget.ropeTable === 'boolean');
if (observationTarget.ropeTable === true) {
  assert(!prefixesOnly && !physicalCache && !fullPrefixes && !observationTarget.opIds,
    'Table-only observation cannot claim a prefix or generation replay');
}
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
  recurrentStatePrecision, servedShaders: {}, modelDirectory: resolve(modelDirectory), fullPrefixes, prefixesOnly, observationTarget };
const cacheObserverSource = await readFile(new URL('../helpers/physical-kv-cache-observer.js', import.meta.url));
if (physicalCache) receipt.cacheObserverSha256 = hash(cacheObserverSource);
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
      if (pathname === '/diagnostic/physical-kv-cache-observer.js') {
        response.setHeader('Content-Type', 'text/javascript'); response.end(cacheObserverSource); return;
      }
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
  const result = await page.evaluate(async ({ compatEntry, partitionsEntry, generation, prompts, policy, fullPrefixes, prefixesOnly, expected, observationTarget, captureDecode, recurrentStatePrecision }) => {
    const ropeBindings = [], uniformWrites = new WeakMap();
    if (observationTarget.ropeTable === true) {
      const writeBuffer = GPUQueue.prototype.writeBuffer;
      GPUQueue.prototype.writeBuffer = function (buffer, offset, data, dataOffset = 0, size) {
        if ((buffer.usage & GPUBufferUsage.UNIFORM) && offset === 0) {
          const unit = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT ?? 1 : 1;
          const bytes = ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
          uniformWrites.set(buffer, Array.from(bytes.slice(dataOffset * unit,
            size === undefined ? undefined : (dataOffset + size) * unit)));
        }
        return writeBuffer.call(this, buffer, offset, data, dataOffset, size);
      };
      const createBindGroup = GPUDevice.prototype.createBindGroup;
      GPUDevice.prototype.createBindGroup = function (descriptor) {
        const entry = binding => descriptor.entries.find(row => row.binding === binding)?.resource?.buffer;
        if (entry(2)?.label.startsWith('rope_cos_') && entry(3)?.label.startsWith('rope_sin_')) {
          ropeBindings.push({ device: this, uniform: uniformWrites.get(entry(0)), cos: entry(2), sin: entry(3) });
        }
        return createBindGroup.call(this, descriptor);
      };
    }
    // Observe selected native pipelines without rewriting shader arithmetic.
    const moduleSources = new WeakMap(), pipelines = new WeakMap(), normalizationDispatch = [], qkNormalizationDispatch = [], projectionDispatch = [], quantizedDispatch = [], passPipelines = new WeakMap();
    let dispatchOrdinal = 0;
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
      if (source?.includes('q_rows: u32,') && source.includes('q_input: array<f32>')) {
        const record = { entry: descriptor.compute.entryPoint, constants: descriptor.compute.constants,
          source, selections: 0 };
        qkNormalizationDispatch.push(record); pipelines.set(pipeline, record);
      }
      if (source?.includes('Register-Tiled Matrix Multiplication Kernel - f16 weights, f32 activations')) {
        const record = { entry: descriptor.compute.entryPoint, constants: descriptor.compute.constants,
          source, selections: 0 };
        projectionDispatch.push(record); pipelines.set(pipeline, record);
      }
      if (source?.includes('struct Q4KBlock')) {
        const record = { entry: descriptor.compute.entryPoint, constants: descriptor.compute.constants,
          source, selections: 0, dispatches: [] };
        quantizedDispatch.push(record); pipelines.set(pipeline, record);
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
      passPipelines.set(this, record);
      return setPipeline.call(this, pipeline);
    };
    const dispatchWorkgroups = GPUComputePassEncoder.prototype.dispatchWorkgroups;
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...dimensions) {
      const record = passPipelines.get(this);
      if (record?.dispatches && record.dispatches.length < 32) {
        record.dispatches.push({ ordinal: dispatchOrdinal, dimensions });
      }
      dispatchOrdinal++;
      return dispatchWorkgroups.call(this, ...dimensions);
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
    const physicalCache = observationTarget.physicalCache === true;
    let handle;
      let cacheObserver = null;
      if (physicalCache) {
        const { KVCache } = await import('/runtime/src/inference/kv-cache/base.js');
        const { observePhysicalKVCache } = await import('/diagnostic/physical-kv-cache-observer.js');
        const layers = manifest.inference.layerPattern.layerTypes.flatMap((type, layer) => type === 'full_attention' ? [layer] : []);
        cacheObserver = observePhysicalKVCache({ KVCache, getDevice, layers,
          allocationLayerIndices: observationTarget.sparseAttentionCache === true ? layers : null });
      }
    try {
      handle = await load({ manifest, manifestText, manifestHash, baseUrl: source,
        storage: createHttpArtifactStorageContext(source, manifest, { verifyHashes: true }) }, {
        onProgress: progress => console.log('producer-progress:' + JSON.stringify(progress)),
        runtimeConfig: { shared: { bufferPool: policy.bufferPool }, inference: { session: {
          linearRecurrentStatePrecision: recurrentStatePrecision,
          kvcache: { maxSeqLen: generation.maxSeqLen }, prefillChunkLayers: policy.prefillChunkLayers,
          prefillTokenChunkSize: policy.prefillTokenChunkSize } } } });
      if (observationTarget.ropeTable === true) {
        if (!ropeBindings.length) throw Error('No actual RoPE precompute binding observed');
        const ropeTables = [];
        for (const { device, uniform, cos, sin } of ropeBindings) {
          if (!uniform || uniform.length < 64) throw Error('Missing actual RoPE uniforms');
          const bytes = Uint8Array.from(uniform), view = new DataView(bytes.buffer);
          const rows = Math.min(view.getUint32(0, true), 256), width = view.getUint32(4, true) / 2;
          const size = rows * width * 4;
          const staging = device.createBuffer({ size: size * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          try {
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(cos, 0, staging, 0, size);
            encoder.copyBufferToBuffer(sin, 0, staging, size, size);
            device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
            ropeTables.push({ uniform, rows, width, values: Array.from(new Float32Array(staging.getMappedRange().slice(0))) });
          } finally {
            if (staging.mapState === 'mapped') staging.unmap();
            await device.queue.onSubmittedWorkDone(); staging.destroy();
          }
        }
        return { runtimeVersion: DOPPLER_VERSION, rows, ropeTables, observation: null,
          normalizationDispatch, qkNormalizationDispatch, projectionDispatch, quantizedDispatch };
      }
      const { maxSeqLen: _contextLength, ...executionOptions } = generation;
      for (const index of physicalCache || prefixesOnly ? [observationTarget.index] : [...prompts.keys(), 0]) {
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
        if (fullPrefixes || physicalCache || prefixesOnly) {
          for (let step = 1; step < (physicalCache || prefixesOnly ? observationTarget.step + 1 : expected[index].steps.length); step++) {
            const value = await handle.advanced.decodeStepLogits([expected[index].steps[step - 1].tokenId], executionOptions);
            await record(value, step);
          }
        }
      }
      if (prefixesOnly || (fullPrefixes && !captureDecode)) return { runtimeVersion: DOPPLER_VERSION, rows, observation: null, normalizationDispatch, qkNormalizationDispatch, projectionDispatch, quantizedDispatch };
      // Observe the same first prefix after reset; quantify instrumentation effects.
      await handle.resetGenerationState();
      let observedLogits = null; const observedSteps = [];
      const cacheReadbacks = [];
      const targetOpIds = observationTarget.opIds ?? ['embed.out', 'final_norm.pre', 'final_norm.out',
        ...manifest.inference.layerPattern.layerTypes.flatMap((type, layer) =>
          type === 'linear_attention'
            ? ['qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj',
              'linear_core_out', 'out', 'post_attn'].map(op => `layer.${layer}.attn.${op}`)
            : []),
        ...['in', 'gate', 'up', 'act', 'out'].map(op => 'layer.0.ffn.' + op),
        ...manifest.inference.layerPattern.layerTypes.flatMap((type, layer) =>
          type === 'full_attention'
            ? ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'q_rope', 'k_rope', 'core_out', 'out']
              .map(op => `layer.${layer}.attn.${op}`)
            : []),
        ...Array.from({ length: manifest.architecture.numLayers }, (_, layer) =>
          [`layer.${layer}.attn.post_input_norm`, `layer.${layer}.layer.out`]).flat()];
      let physicalRows;
      try { for await (const _chunk of handle.generate(prompts[observationTarget.index], { ...executionOptions,
        disableCommandBatching: true, diagnostics: { enabled: true,
          captureConfig: { enabled: true, defaultLevel: 'none', targetOpIds, targetLevel: 'full' } },
        onLogits: values => {
          if (cacheObserver) {
            const step = observedSteps.length;
            const pending = cacheObserver.capture(step, rows[0].inputIds.length + step);
            // Attach rejection immediately; all readbacks settle before cleanup.
            cacheReadbacks.push(pending.then(rows => ({ rows }), error => ({ error })));
          }
          observedLogits = Array.from(values); observedSteps.push(observedLogits);
        },
      })) { if (observedSteps.length > observationTarget.step) break; }
      } finally {
        cacheObserver?.restore();
        physicalRows = await Promise.all(cacheReadbacks);
      }
      const cacheFailure = physicalRows.find(value => value.error);
      if (cacheFailure) throw cacheFailure.error;
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
      return { runtimeVersion: DOPPLER_VERSION, rows, normalizationDispatch, qkNormalizationDispatch, projectionDispatch, quantizedDispatch, observation: { logits: observedLogits, observedSteps, target: observationTarget,
        physicalCache: cacheObserver ? physicalRows.flatMap(value => value.rows) : null,
        samplingExcludedTokenIds: [handle.advanced.getSpecialTokens().pad, ...executionOptions.suppressTokenIds]
          .filter(Number.isInteger), timeline: [] } };
    } finally {
      cacheObserver?.restore();
      await getDevice().queue.onSubmittedWorkDone();
      if (handle) await handle.unload();
      await getDevice().queue.onSubmittedWorkDone();
    }
  }, { compatEntry, partitionsEntry, generation: reference.generation, prompts: reference.prompts, policy,
    fullPrefixes, prefixesOnly, expected: reference.expected, observationTarget, captureDecode, recurrentStatePrecision });
  receipt.physicalExecutionCompleted = true;
  receipt.runtimeVersion = result.runtimeVersion;
  receipt.normalizationDispatch = result.normalizationDispatch.map(({ source, ...record }) => ({
    ...record, shaderSha256: hash(source) }));
  receipt.qkNormalizationDispatch = result.qkNormalizationDispatch.map(({ source, ...record }) => ({
    ...record, shaderSha256: hash(source) }));
  receipt.projectionDispatch = result.projectionDispatch.map(({ source, ...record }) => ({
    ...record, shaderSha256: hash(source) }));
  receipt.quantizedDispatch = result.quantizedDispatch.map(({ source, ...record }) => ({
    ...record, shaderSha256: hash(source) }));
  receipt.observation = result.observation;
  if (result.ropeTables) receipt.ropeTables = result.ropeTables;
  if (receipt.observation) receipt.observation.timeline = observedTimeline;
  assert.equal(result.runtimeVersion, metadata.version);
  receipt.executionCompleted = true;
  receipt.referenceProducingOwnershipEstablished = false;
  if (physicalCache || prefixesOnly) {
    assert.equal(receipt.results.length, observationTarget.step + 1);
  } else if (fullPrefixes) {
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
    if (physicalCache) {
      receipt.observation.prefixDifferences = result.observation.observedSteps.map((values, step) => {
        const control = receipt.results.find(row => row.index === observationTarget.index && row.step === step);
        const bytes = Buffer.from(control.logits, 'base64');
        const baseline = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        assert.equal(values.length, baseline.length);
        let difference = 0;
        for (let i = 0; i < baseline.length; i++) {
          if (!excluded.has(i)) difference = Math.max(difference, Math.abs(values[i] - baseline[i]));
        }
        assert.equal(difference, 0, `Physical cache readback changed prefix ${step}`);
        return { step, maximumUnmaskedDifference: difference };
      });
    }
  }
  console.log(JSON.stringify({ package: receipt.package, shaderPinMismatches: shaderMismatches.length,
    comparisons: receipt.results.map(({ index, inputIds, maxDifference, matches }) => ({ index, inputTokens: inputIds.length, maxDifference, matches })) }));
} catch (error) { receipt.executionCompleted = false; receipt.failure = error.message; throw error; }
finally {
  await finishFrozenPrefixReplay({ destination, receipt, browser, server, directory });
}
