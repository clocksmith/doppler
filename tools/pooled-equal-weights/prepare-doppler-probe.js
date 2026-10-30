import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const root = path.resolve(config.sourceRoot), destination = path.resolve(config.snapshotDir);
const output = destination + `.preparing-${process.pid}`;
const harnessDir = config.harnessRelativeDir;
if (!['src/tooling/browser-harness', 'src/inference/browser-harness'].includes(harnessDir)) throw new Error('Declare harnessRelativeDir for the pinned source layout');
if (destination === root || destination.startsWith(root + path.sep)) throw new Error('Snapshot must be outside the source checkout');
try { await fs.stat(destination); throw new Error('Snapshot destination already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
await fs.mkdir(output, { recursive: false });
try {
for (const name of ['src', 'demo', 'tests', 'tools', 'benchmarks', 'models']) {
  await fs.cp(path.join(root, name), path.join(output, name), { recursive: true, dereference: false,
    filter: source => source !== path.join(root, 'models/local') && !source.startsWith(path.join(root, 'models/local') + path.sep) });
}
await fs.copyFile(path.join(root, 'package.json'), path.join(output, 'package.json'));
await fs.symlink(path.join(root, 'node_modules'), path.join(output, 'node_modules'));
const file = path.join(output, harnessDir, 'benchmark-execution.js');
const original = await fs.readFile(file, 'utf8');
const anchor = '  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);';
if (original.split(anchor).length !== 2) throw new Error('Doppler untimed observation boundary changed');
const hook = `
  // Diagnostic snapshot only: observation after the measured request is complete.
  // No timings from this instrumented snapshot are performance evidence.
  const canonicalProbe = harness.pipeline.generator;
  globalThis.__canonicalLogitCaptureActive = false;
  metrics.canonicalLogitRows = globalThis.__canonicalFullLogitRows ?? [];
  for (const copy of globalThis.__canonicalGpuLogitCopies ?? []) {
    try {
      await copy.buffer.mapAsync(GPUMapMode.READ);
      metrics.canonicalLogitRows.push({ index: copy.index, stage: 'pre-sampling',
        values: Array.from(new Float32Array(copy.buffer.getMappedRange())) });
      copy.buffer.unmap();
    } finally { copy.buffer.destroy(); }
  }
  metrics.canonicalLogitRows.sort((a, b) => a.index - b.index);
  const requestedLogitPositions = ${JSON.stringify(config.captureLogitPositions ?? null)};
  if (requestedLogitPositions) {
    const expected = requestedLogitPositions.filter(index => index < metrics.referenceTranscript.tokens.ids.length).sort((a, b) => a - b);
    const observed = metrics.canonicalLogitRows.map(row => row.index);
    if (JSON.stringify(expected) !== JSON.stringify(observed)) throw new Error('Raw logit coverage is incomplete or duplicated: ' + JSON.stringify({ expected, observed }));
    for (const row of metrics.canonicalLogitRows) {
      if (row.values.length !== canonicalProbe._state.modelConfig.vocabSize || !row.values.every(Number.isFinite)) throw new Error('Raw logit observation is incomplete or nonfinite');
    }
  }
  const canonicalOptions = canonicalProbe._resolveStepOptions({ temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, useChatTemplate: false, benchmark: false });
  harness.pipeline.reset();
  metrics.canonicalFirstLogits = Array.from(await canonicalProbe._prefillInputIdsToLogits(
    metrics.referenceTranscript.prompt.ids, canonicalOptions));
  harness.pipeline.reset();
  globalThis.__canonicalBoundaryCopies = [];
  globalThis.__canonicalBoundaryActive = true;
  metrics.canonicalSingleTokenLogits = Array.from(await canonicalProbe._prefillInputIdsToLogits(
    metrics.referenceTranscript.prompt.ids.slice(0, 1), canonicalOptions));
  globalThis.__canonicalBoundaryActive = false;
  metrics.canonicalSingleTokenBoundaries = [];
  for (const copy of globalThis.__canonicalBoundaryCopies) {
    try {
      await copy.buffer.mapAsync(GPUMapMode.READ);
      metrics.canonicalSingleTokenBoundaries.push({ stage: copy.stage, layer: copy.layer,
        values: Array.from(new Float32Array(copy.buffer.getMappedRange())) });
      copy.buffer.unmap();
    } finally { copy.buffer.destroy(); }
  }
`;
let weightHook = '';
let weightVerifierIdentity = null;
if (config.canonicalInventoryPath && config.nativeWeightInventoryPath) throw new Error('Declare one weight observation lane.');
if (config.nativeWeightInventoryPath) {
  const inventory = JSON.parse(await fs.readFile(config.nativeWeightInventoryPath, 'utf8'));
  const { validateQ40Inventory } = await import('./verify-q4_0.js');
  validateQ40Inventory(inventory);
  const verifier = await fs.readFile(new URL('./verify-q4_0.js', import.meta.url));
  weightVerifierIdentity = { path: 'tools/pooled-equal-weights/verify-q4_0.js', sha256: createHash('sha256').update(verifier).digest('hex') };
  weightHook = `
  const { verifyQ40Resources } = await import('../../../tools/pooled-equal-weights/verify-q4_0.js');
  const { resolveDopplerQ40Resources } = await import('../../../tools/pooled-equal-weights/q4_0-doppler-resources.js');
  const { readBufferSlice } = await import('../../memory/buffer-pool.js');
  const nativeInventory = ${JSON.stringify(inventory)};
  metrics.nativeGgufWeights = await verifyQ40Resources(nativeInventory, await resolveDopplerQ40Resources(harness.pipeline, nativeInventory, readBufferSlice));
`;
}
if (config.canonicalInventoryPath) {
  const inventory = JSON.parse(await fs.readFile(config.canonicalInventoryPath, 'utf8'));
  if (inventory.schema !== 'doppler.equal-weight-canonical.v1' || inventory.completed !== true) throw new Error('Incomplete canonical GPU proof inventory');
  const verifier = await fs.readFile(new URL('./verify-doppler-gpu.js', import.meta.url));
  await fs.mkdir(path.join(output, 'tools/pooled-equal-weights'), { recursive: true });
  await fs.writeFile(path.join(output, 'tools/pooled-equal-weights/verify-doppler-gpu.js'), verifier);
  weightVerifierIdentity = { path: 'tools/pooled-equal-weights/verify-doppler-gpu.js',
    sha256: createHash('sha256').update(verifier).digest('hex') };
  weightHook = `
  const { verifyDopplerGpu } = await import('../../../tools/pooled-equal-weights/verify-doppler-gpu.js');
  metrics.canonicalGpuWeights = await verifyDopplerGpu(harness.pipeline,
    ${JSON.stringify(inventory)}, readCanonicalGpuBuffer);
`;
}
let instrumented = original.replace(anchor, hook + weightHook + '\n' + anchor);
if (config.canonicalInventoryPath) instrumented = "import { readBuffer as readCanonicalGpuBuffer } from '../../memory/buffer-pool.js';\n" + instrumented;
await fs.writeFile(file, instrumented);
const probeFile = path.join(output, 'src/inference/pipelines/text/probes.js');
const probes = await fs.readFile(probeFile, 'utf8');
const probeAnchor = 'export async function runProbes(stage, buffer, options) {';
if (probes.split(probeAnchor).length !== 2) throw new Error('Doppler boundary observation owner changed');
const stages = JSON.stringify(config.captureStages ?? []);
const probeHook = `
  if (globalThis.__canonicalBoundaryActive && ${stages}.includes(stage)) {
    if (options.dtype && options.dtype !== 'f32') throw new Error('Canonical diagnostic requires F32 boundaries');
    const device = getDevice(), bytes = options.numTokens * options.hiddenSize * 4;
    const copy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (options.recorder) options.recorder.getEncoder().copyBufferToBuffer(buffer, 0, copy, 0, bytes);
    else { const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, copy, 0, bytes); device.queue.submit([encoder.finish()]); }
    globalThis.__canonicalBoundaryCopies.push({ stage, layer: options.layerIdx, buffer: copy });
  }
`;
const capturedProbes = probes.replace(probeAnchor, probeAnchor + probeHook);
await fs.writeFile(probeFile, capturedProbes);
let logitObservation = null;
if (config.captureLogitPositions) {
  const positions = config.captureLogitPositions;
  if (!Array.isArray(positions) || positions.some(x => !Number.isSafeInteger(x) || x < 0) || new Set(positions).size !== positions.length) throw new Error('Invalid logit observation positions');
  const owner = path.join(output, 'src/inference/pipelines/text/generator.js');
  const source = await fs.readFile(owner, 'utf8');
  const marker = '  _sampleNextTokenFromLogits(logits, generatedIds, opts) {';
  if (source.split(marker).length !== 2) throw new Error('Full-logit observation boundary changed');
  const observation = `
  globalThis.__canonicalPromptLength ??= generatedIds.length;
  const canonicalObservedIndex = generatedIds.length - globalThis.__canonicalPromptLength;
  if (${JSON.stringify(positions)}.includes(canonicalObservedIndex)) {
    globalThis.__canonicalFullLogitRows ??= [];
    globalThis.__canonicalFullLogitRows.push({ index: canonicalObservedIndex, stage: 'pre-sampling', values: Array.from(logits) });
  }
`;
  const observed = source.replace(marker, marker + observation);
  await fs.writeFile(owner, observed);
  const stepsOwner = path.join(output, 'src/inference/pipelines/text/generator-steps.js');
  const stepsSource = await fs.readFile(stepsOwner, 'utf8');
  const stepsMarker = '        logitsBuffer = logits.logitsBuffer;';
  if (stepsSource.split(stepsMarker).length !== 2) throw new Error('Recorded pre-sampling logit boundary changed');
  const stepsHook = `
        if (globalThis.__canonicalLogitCaptureActive) {
          const index = state.currentSeqLen + i - globalThis.__canonicalPromptLength + 1;
          if (${JSON.stringify(positions)}.includes(index)) {
            const copy = device.createBuffer({ size: vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try { recorder.getEncoder().copyBufferToBuffer(logitsBuffer, 0, copy, 0, vocabSize * 4); }
            catch (error) { copy.destroy(); throw error; }
            globalThis.__canonicalGpuLogitCopies.push({ index, buffer: copy });
          }
        }
`;
  const observedSteps = stepsSource.replace(stepsMarker, stepsMarker + stepsHook);
  await fs.writeFile(stepsOwner, observedSteps);
  const decodeOwner = path.join(output, 'src/inference/pipelines/text/generator/decode.js');
  const decodeSource = await fs.readFile(decodeOwner, 'utf8');
  const decodeMarker = "\n      resources.register(logitsBuffer, 'decode logits');";
  if (decodeSource.split(decodeMarker).length !== 2) throw new Error('Single-step recorded pre-sampling boundary changed');
  const decodeHook = `
      if (globalThis.__canonicalLogitCaptureActive) {
        const index = currentIds.length - globalThis.__canonicalPromptLength;
        if (${JSON.stringify(positions)}.includes(index)) {
          if (logitsDtype !== 'f32') throw new Error('Raw canonical logits require F32');
          const copy = recorder.device.createBuffer({ size: vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          try { recorder.getEncoder().copyBufferToBuffer(logitsBuffer, 0, copy, 0, vocabSize * 4); }
          catch (error) { copy.destroy(); throw error; }
          globalThis.__canonicalGpuLogitCopies.push({ index, buffer: copy });
        }
      }
`;
  const immediateMarker = '        const sampledLogits = extractLastPositionLogits(rawLogits, numTokens, config.vocabSize);';
  if (decodeSource.split(immediateMarker).length !== 2) throw new Error('Immediate pre-sampling boundary changed');
  const immediateHook = `
        if (globalThis.__canonicalLogitCaptureActive) {
          const index = currentIds.length - globalThis.__canonicalPromptLength;
          if (${JSON.stringify(positions)}.includes(index)) {
            globalThis.__canonicalFullLogitRows.push({ index, stage: 'pre-sampling', values: Array.from(sampledLogits) });
          }
        }
`;
  const observedDecode = decodeSource.replace(decodeMarker, decodeMarker + decodeHook)
    .replace(immediateMarker, immediateMarker + immediateHook);
  await fs.writeFile(decodeOwner, observedDecode);
  logitObservation = { originalSha256: createHash('sha256').update(source).digest('hex'),
    instrumentedSha256: createHash('sha256').update(observed).digest('hex'), positions, stage: 'pre-sampling',
    originalStepsSha256: createHash('sha256').update(stepsSource).digest('hex'), instrumentedStepsSha256: createHash('sha256').update(observedSteps).digest('hex'),
    originalDecodeSha256: createHash('sha256').update(decodeSource).digest('hex'), instrumentedDecodeSha256: createHash('sha256').update(observedDecode).digest('hex') };
  const runMarker = '      const run = await withHarnessPhase(\n        `bench.runTextInference[${i}]`,';
  const benchmarkOwner = await fs.readFile(file, 'utf8');
  if (benchmarkOwner.split(runMarker).length !== 2) throw new Error('Logit observation run-reset boundary changed');
  await fs.writeFile(file, benchmarkOwner.replace(runMarker, '      globalThis.__canonicalPromptLength = null;\n      globalThis.__canonicalLogitCaptureActive = true;\n      globalThis.__canonicalGpuLogitCopies = [];\n      globalThis.__canonicalFullLogitRows = [];\n' + runMarker));
}
const sha = text => createHash('sha256').update(text).digest('hex');
let numericPrototype = null;
if (config.kvRoundingPrototypeShader) {
  const source = await fs.readFile(config.kvRoundingPrototypeShader, 'utf8');
  const target = path.join(output, 'src/gpu/kernels/kv_cache_write_f32_to_f16.wgsl');
  numericPrototype = { path: 'src/gpu/kernels/kv_cache_write_f32_to_f16.wgsl',
    originalSha256: sha(await fs.readFile(target)), prototypeSha256: sha(source),
    scope: 'Untimed rounding isolation only; source mutation is not a registered or promoted runtime capability.' };
  await fs.writeFile(target, source);
}
await fs.writeFile(path.join(output, 'observation-identity.json'), JSON.stringify({
  schema: 'doppler.untimed-observation-snapshot.v1', sourceRoot: root,
  completed: true,
  file: harnessDir + '/benchmark-execution.js',
  originalSha256: sha(original), instrumentedSha256: sha(await fs.readFile(file)),
  probeOriginalSha256: sha(probes), probeInstrumentedSha256: sha(capturedProbes), captureStages: config.captureStages ?? [],
  numericPrototype,
  weightVerifierIdentity,
  logitObservation,
  claim: 'Untimed pre-sampling and boundary observations. All timings are diagnostic and inadmissible for speed claims.' }, null, 2));
await fs.rename(output, destination);
} catch (error) {
  await fs.writeFile(path.join(output, 'preparation-failure.json'), JSON.stringify({ completed: false, error: String(error.stack || error) }, null, 2));
  throw error;
}
