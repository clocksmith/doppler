import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { runPipelineOperation } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { beginGenerationContext, finishGenerationContext } from '../../src/inference/pipelines/text/sequence-state.js';
import { resolveGenerateOptions } from '../../src/inference/pipelines/text/generator-runtime.js';
import { createKVCache } from '../../src/inference/pipelines/text/init.js';

const directory = process.env.DOPPLER_CONTEXT_MODEL_DIR;
if (!directory) throw new Error('Set DOPPLER_CONTEXT_MODEL_DIR to the local Qwen hybrid model.');
const bytes = await fs.readFile(path.join(directory, 'manifest.json'));
const manifest = JSON.parse(bytes);
const report = { modelId: manifest.modelId, manifestHash: createHash('sha256').update(bytes).digest('hex'), cases: [] };
let pipeline;
try {
  const gpu = await probeNodeGPU({ installFileFetchShim: true });
  assert.ok(gpu.ready, gpu.reason);
  report.adapter = getKernelCapabilities().adapterInfo;
  report.provider = gpu.providerReceipt;
  pipeline = await createPipeline(manifest, {
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 256 } } } },
    storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(directory)).href, manifest),
  });
  const contexts = [pipeline.createGenerationContext({ maxBytes: 134217728 }), pipeline.createGenerationContext({ maxBytes: 134217728 })];
  const run = async (tokens, context, signal) => runPipelineOperation(pipeline, async p => {
    const rows = [], started = performance.now();
    const result = await p.generateTokenIds('', { inputIds: tokens, useChatTemplate: false,
      maxTokens: 2, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1,
      generationContext: context, signal, onLogits: logits => rows.push(Array.from(logits)) });
    return { tokens: result.tokenIds, rows, stats: structuredClone(p.getStats()), wallMs: performance.now() - started };
  });
  const a = Array.from(pipeline.tokenizer.encode('The capital of France is'));
  const b = Array.from(pipeline.tokenizer.encode('The result of two plus two is'));
  const firstA = await run(a, contexts[0]);
  const firstFresh = await run(a);
  assert.ok(firstA.rows.length > 0 && firstA.rows[0].length === 248320, 'Full Qwen vocabulary observation required');
  assert.deepEqual(firstA.tokens, firstFresh.tokens);
  let coldMaxError = 0;
  for (let r = 0; r < firstA.rows.length; r++) for (let i = 0; i < firstA.rows[r].length; i++) {
    const error = Math.abs(firstA.rows[r][i] - firstFresh.rows[r][i]);
    assert.ok(Number.isFinite(error) && error <= 0.001, 'Cold context must agree with ordinary prefill');
    coldMaxError = Math.max(coldMaxError, error);
  }
  report.cases.push({ kind: 'cold-checkpoint', maxError: coldMaxError, rows: firstA.rows.length });
  await run(b, contexts[1]);
  const extended = [...a, ...firstA.tokens, ...pipeline.tokenizer.encode('. Explain briefly.')];
  const cached = await run(extended, contexts[0]);
  const fresh = await run(extended);
  assert.ok(cached.stats.reusedPrefixTokens > 0, 'Follow-up must reuse the complete previous state');
  assert.deepEqual(cached.tokens, fresh.tokens);
  assert.equal(cached.rows.length, fresh.rows.length);
  let maxError = 0, violations = 0;
  for (let r = 0; r < cached.rows.length; r++) for (let i = 0; i < cached.rows[r].length; i++) {
    const error = Math.abs(cached.rows[r][i] - fresh.rows[r][i]);
    maxError = Math.max(maxError, error);
    if (!Number.isFinite(error) || error > 0.001) violations++;
  }
  report.cases.push({ kind: 'alternating-context-followup', maxError, violations,
    cached: cached.stats, fresh: fresh.stats, cachedWallMs: cached.wallMs, freshWallMs: fresh.wallMs,
    context: contexts[0].getStats() });
  assert.equal(violations, 0, 'Cached/fresh full-vocabulary comparison exceeds 0.001');
  const edited = await run(b, contexts[0]);
  assert.equal(edited.stats.reusedPrefixTokens, 0, 'Changed earlier text must rebuild');
  report.cases.push({ kind: 'edited-prefix', reused: edited.stats.reusedPrefixTokens });
  const chat = async (messages, context, controller) => runPipelineOperation(pipeline, async p => {
    let output = '';
    for await (const delta of p.generate(messages, { maxTokens: 4, temperature: 0, topK: 1, topP: 1,
      repetitionPenalty: 1, generationContext: context, signal: controller?.signal })) {
      output += delta;
      controller?.abort();
    }
    return { output, stats: structuredClone(p.getStats()) };
  });
  const messages = [{ role: 'user', content: 'Say hello briefly.' }];
  const firstChat = await chat(messages, contexts[1]);
  messages.push({ role: 'assistant', content: firstChat.output }, { role: 'user', content: 'Now say goodbye.' });
  const nextChat = await chat(messages, contexts[1]);
  const freshChat = await chat(messages);
  assert.equal(nextChat.output, freshChat.output);
  assert.ok(nextChat.stats.reusedPrefixTokens > 0, 'Templated chat must reuse its exact shared prefix');
  report.cases.push({ kind: 'templated-chat', output: nextChat.output, reused: nextChat.stats.reusedPrefixTokens });
  const controller = new AbortController();
  await chat([{ role: 'user', content: 'Count to ten.' }], contexts[1], controller).catch(error => {
    if (!controller.signal.aborted) throw error;
  });
  assert.equal(contexts[1].getStats().cachedTokens, 0, 'Cancelled work cannot publish reusable state');
  const recovered = await chat(messages, contexts[1]);
  assert.equal(recovered.stats.reusedPrefixTokens, 0);
  assert.equal(recovered.output, freshChat.output);
  report.cases.push({ kind: 'cancel-and-reuse', rebuilt: true });
  const longIds = Array.from(pipeline.tokenizer.encode('Remember these colors: ' + Array(24).fill('red green blue').join(' ') + '.'));
  await run(longIds, contexts[1]);
  const longFollowup = [...longIds, ...pipeline.tokenizer.encode(' Name the first color:')];
  const longCached = await run(longFollowup, contexts[1]), longFresh = await run(longFollowup);
  assert.ok(longCached.stats.reusedPrefixTokens >= 64);
  assert.deepEqual(longCached.tokens, longFresh.tokens);
  let longMaxError = 0;
  for (let r = 0; r < longCached.rows.length; r++) for (let i = 0; i < longCached.rows[r].length; i++) {
    const error = Math.abs(longCached.rows[r][i] - longFresh.rows[r][i]);
    assert.ok(Number.isFinite(error) && error <= 0.001, 'Longer cached prefix must agree with ordinary prefill');
    longMaxError = Math.max(longMaxError, error);
  }
  report.cases.push({ kind: 'longer-prefix', inputTokens: longFollowup.length,
    reused: longCached.stats.reusedPrefixTokens, maxError: longMaxError });
  const tiny = pipeline.createGenerationContext({ maxBytes: 1 });
  await run(a, tiny);
  assert.equal(tiny.getStats().status, 'budget');
  assert.equal(tiny.getStats().allocatedBytes, 0);
  tiny.close();
  report.cases.push({ kind: 'budget-fallback', allocatedBytes: 0 });
  await runPipelineOperation(pipeline, async p => {
    p.resetGenerationState();
    const context = p.createGenerationContext({ maxBytes: 134217728 });
    const options = resolveGenerateOptions(p, { inputIds: a, useChatTemplate: false, maxTokens: 2, generationContext: context });
    beginGenerationContext(p, context, options);
    await p.generator._prefillPromptToLogits('', options, 'replacement-test');
    const previous = p.kvCache;
    const replacement = createKVCache(p.modelConfig, p.useGPU, p.debug, p.runtimeConfig.inference);
    previous.destroy(); p.kvCache = replacement;
    await finishGenerationContext(p, a, true);
    assert.equal(p.kvCache, replacement, 'Finishing a parked context must preserve a replaced working cache');
    context.close(); p.resetGenerationState();
  });
  report.cases.push({ kind: 'working-cache-replacement', scope: 'Injected replacement after a real GPU checkpoint; no precision-recovery numerical claim', preserved: true });
  contexts[0].close();
  assert.equal(contexts[0].getStats().allocatedBytes, 0);
  report.cases.push({ kind: 'eviction', allocatedBytes: 0 });
} finally {
  await pipeline?.unload();
  destroyDevice();
  await releaseNodeWebGPU();
  if (process.env.DOPPLER_CONTEXT_RECEIPT) await fs.writeFile(process.env.DOPPLER_CONTEXT_RECEIPT, JSON.stringify(report, null, 2) + '\n');
}
console.log('generation-context-physical: passed');
