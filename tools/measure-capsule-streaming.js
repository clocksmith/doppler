import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { Session } from 'node:inspector/promises';
import { execFileSync } from 'node:child_process';
import { createCapsuleOperationAdapters } from '../src/client/runtime/capsule-operation-adapters.js';
import { createCapsuleOperationExecutor } from '../src/client/runtime/capsule-operation-executor.js';
import { createCapsuleStreamAccumulator } from '../src/client/runtime/capsule-operation-stream.js';
import { hashCapsuleObservation } from '../src/config/capsule-operation.js';
import { BundledTokenizer } from '../src/inference/tokenizers/bundled.js';

// CPU output-processing experiment. Synthetic IDs replace inference entirely.
// No model, kernel, throughput, or physical-GPU qualification is implied.
const [configPath, outputPath] = process.argv.slice(2);
if (!configPath || !outputPath) throw new Error('Usage: node tools/measure-capsule-streaming.js <config.json> <report.json>');
const config = JSON.parse(await readFile(configPath, 'utf8'));
assert(config.lengths.every(value => Number.isSafeInteger(value) && value > 0));
assert(Number.isSafeInteger(config.repetitions) && config.repetitions > 0);
assert(Number.isSafeInteger(config.samplingInterval) && config.samplingInterval > 0);
assert(Number.isSafeInteger(config.deadlineMs) && config.deadlineMs > 0);
const tokenizer = new BundledTokenizer({ vocabSize: 0, deferSpecialTokens: true, addBosToken: false, addEosToken: false });
tokenizer.load({ model: { type: 'BPE', vocab: { x: 0, '<eos>': 1 }, merges: [] },
  pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false }, added_tokens: [{ id: 1, content: '<eos>', special: true }] });
const bytes = value => Buffer.byteLength(value);
const cpuMs = start => { const elapsed = process.cpuUsage(start); return (elapsed.user + elapsed.system) / 1000; };
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function producer(version, length) {
  let canonicalDecodeIds = 0, incrementalDecodeIds = 0;
  const program = {
    decodeTokens(ids) { canonicalDecodeIds += ids.length; return tokenizer.decode(ids, true, false); },
    createIncrementalDecoder() {
      const decoder = tokenizer.createIncrementalDecoder();
      return { push(id) { incrementalDecodeIds++; return decoder.push(id); }, finish: () => decoder.finish() };
    },
  };
  const request = { schema: `doppler.capsule-operation-request/v${version}`, operation: { name: 'generate', version: 1 },
    input: { promptTokens: [0] }, options: { maxTokens: length, maxSeqLen: length + 1, temperature: 0,
      topP: 1, topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false, stopSequences: [] },
    assignment: null, limits: { maxInputBytes: config.maxInputBytes, maxOutputBytes: config.maxOutputBytes, deadlineAt: Date.now() + config.deadlineMs } };
  const adapters = createCapsuleOperationAdapters({ program, async *generate() {
    for (let i = 0; i < length; i++) yield 0;
    return { generatedTokens: length, finishReason: 'max-tokens', resolvedOptions: request.options };
  } });
  return { request, stream: createCapsuleOperationExecutor({ adapters, identity: { modelId: 'synthetic-transport-only' }, assertCurrent: async () => {} })(request),
    counters: () => ({ canonicalDecodeIds, incrementalDecodeIds }) };
}

function consumer(request, version) {
  if (version === 2) return createCapsuleStreamAccumulator(request);
  const requestHash = hashCapsuleObservation(request);
  let index = 0, previous = null, complete = null;
  return {
    accept(event) {
      const { eventDigest, ...payload } = event;
      assert.equal(eventDigest, hashCapsuleObservation(payload));
      assert.equal(event.eventIndex, index++); assert.equal(event.previousEventDigest, previous);
      assert.equal(event.requestHash, requestHash); previous = eventDigest;
      if (event.status === 'completed') {
        const { receiptDigest, ...receipt } = event.receipt;
        assert.equal(receiptDigest, hashCapsuleObservation(receipt));
        assert.equal(receipt.outputHash, hashCapsuleObservation(event.output));
        complete = event;
      }
    },
    finish() { assert(complete); return complete; },
  };
}

async function measure(version, length) {
  const run = producer(version, length), reconstruction = consumer(run.request, version);
  const metrics = { producerCpuMs: 0, serializationCpuMs: 0, consumerCpuMs: 0, emittedBytes: 0,
    partialTokenIdElements: 0, partialTextUtf8Bytes: 0, eventCount: 0 };
  const start = performance.now();
  while (true) {
    let cpu = process.cpuUsage(); const next = await run.stream.next(); metrics.producerCpuMs += cpuMs(cpu);
    if (next.done) break;
    const event = next.value;
    if (event.status === 'partial') {
      metrics.partialTokenIdElements += version === 2 ? event.delta.tokenIds.length : event.output.tokenIds.length;
      metrics.partialTextUtf8Bytes += bytes(version === 2 ? event.delta.text : event.output.text);
    }
    cpu = process.cpuUsage(); const wire = JSON.stringify(event); metrics.serializationCpuMs += cpuMs(cpu);
    metrics.emittedBytes += bytes(wire); metrics.eventCount++;
    cpu = process.cpuUsage(); reconstruction.accept(JSON.parse(wire)); metrics.consumerCpuMs += cpuMs(cpu);
  }
  const final = reconstruction.finish();
  assert.equal(final.output.text, 'x'.repeat(length)); assert.deepEqual(final.output.tokenIds, Array(length).fill(0));
  return { ...metrics, elapsedMs: performance.now() - start, ...run.counters() };
}

async function sampledAllocations(version, length, side) {
  const run = producer(version, length), wires = [];
  if (side === 'consumer') for await (const event of run.stream) wires.push(JSON.stringify(event));
  const session = new Session(); session.connect();
  try {
    await session.post('HeapProfiler.startSampling', { samplingInterval: config.samplingInterval,
      includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    if (side === 'producer') {
      for await (const event of run.stream) JSON.stringify(event);
    } else {
      const reconstruction = consumer(run.request, version);
      for (const wire of wires) reconstruction.accept(JSON.parse(wire));
      reconstruction.finish();
    }
    const { profile } = await session.post('HeapProfiler.stopSampling');
    const total = node => node.selfSize + node.children.reduce((sum, child) => sum + total(child), 0);
    return total(profile.head);
  } finally { session.disconnect(); }
}

// Warm the same implementations before timing; allocation profiling is separate.
for (const version of [1, 2]) await measure(version, config.lengths[0]);
const results = [];
for (const version of [1, 2]) for (const length of config.lengths) {
  const runs = [];
  for (let i = 0; i < config.repetitions; i++) runs.push(await measure(version, length));
  const medians = Object.fromEntries(Object.keys(runs[0]).map(key => [key, median(runs.map(run => run[key]))]));
  results.push({ version, length, runs, medians, sampledAllocationBytes: {
    producerIncludingSerialization: await sampledAllocations(version, length, 'producer'),
    consumerIncludingParsing: await sampledAllocations(version, length, 'consumer'),
  } });
  console.log(JSON.stringify(results.at(-1)));
}
const v2 = results.filter(row => row.version === 2);
for (let i = 1; i < v2.length; i++) {
  const growth = v2[i].length / v2[i - 1].length;
  assert(v2[i].medians.emittedBytes / v2[i - 1].medians.emittedBytes < growth * config.byteGrowthTolerance);
  assert.equal(v2[i].medians.canonicalDecodeIds, 0);
  assert.equal(v2[i].medians.partialTokenIdElements, v2[i].length);
}
await writeFile(outputPath, JSON.stringify({ schema: 'doppler.stream-transport-experiment/v1', createdAt: new Date().toISOString(),
  sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  sourceStatus: execFileSync('git', ['status', '--short'], { encoding: 'utf8' }).trim(),
  environment: { node: process.version, platform: process.platform, arch: process.arch }, config,
  semantics: { workload: 'Synthetic ASCII token IDs; actual bundled decoding, operation adapters/executor, JSON wire, integrity and reconstruction helper. No GPU work.',
    copiedVolume: 'Partial token-ID elements and UTF-8 text bytes carried by output snapshots or deltas; logical payload counts, not total engine memory copies.',
    allocations: 'Separate V8 sampling estimates include temporary and collected allocations. Producer includes serialization; consumer includes parsing. Noisy estimates, not exact allocated bytes.',
    cpu: 'Process CPU medians; producer execution, JSON serialization and parsing plus consumer verification measured separately. v1 verifier checks chain/final hashes; v2 additionally reconstructs deltas.',
    transport: 'In-process JSON transfer, no HTTP, WebRTC, signing or durable storage timing.' }, results }, null, 2) + '\n');
