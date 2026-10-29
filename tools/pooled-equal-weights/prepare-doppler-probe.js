import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const root = path.resolve(config.sourceRoot), output = path.resolve(config.snapshotDir);
if (output === root || output.startsWith(root + path.sep)) throw new Error('Snapshot must be outside the source checkout');
await fs.mkdir(output, { recursive: false });
for (const name of ['src', 'demo', 'tests', 'tools', 'benchmarks', 'models']) {
  await fs.cp(path.join(root, name), path.join(output, name), { recursive: true, dereference: false,
    filter: source => source !== path.join(root, 'models/local') && !source.startsWith(path.join(root, 'models/local') + path.sep) });
}
await fs.copyFile(path.join(root, 'package.json'), path.join(output, 'package.json'));
await fs.symlink(path.join(root, 'node_modules'), path.join(output, 'node_modules'));
const file = path.join(output, 'src/tooling/browser-harness/benchmark-execution.js');
const original = await fs.readFile(file, 'utf8');
const anchor = '  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);';
if (original.split(anchor).length !== 2) throw new Error('Doppler untimed observation boundary changed');
const hook = `
  // Diagnostic snapshot only: observation after the measured request is complete.
  // No timings from this instrumented snapshot are performance evidence.
  const canonicalProbe = harness.pipeline.generator;
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
const instrumented = original.replace(anchor, hook + '\n' + anchor);
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
  file: 'src/tooling/browser-harness/benchmark-execution.js',
  originalSha256: sha(original), instrumentedSha256: sha(instrumented),
  probeOriginalSha256: sha(probes), probeInstrumentedSha256: sha(capturedProbes), captureStages: config.captureStages ?? [],
  numericPrototype,
  claim: 'Only post-run logit observation. All timings are diagnostic and inadmissible for speed claims.' }, null, 2));
