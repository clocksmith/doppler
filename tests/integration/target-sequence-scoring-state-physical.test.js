// Real-model target scoring state isolation on a physical GPU. Opt-in: set DOPPLER_TARGET_SCORE_MODEL_DIR
// to a local RDRR text model directory (manifest, tokenizer and shards). Covers hybrid linear-attention
// models (e.g. qwen-3-5-0-8b-q4k-ef16-af32-vf16), whose recurrent state a batched decode can advance past
// a stop token, as well as attention-only models (e.g. gemma-3-270m-it-q4k-ehf16-af32).
//
// No resetGenerationState() runs between operations on purpose: scoring must not depend on what an earlier
// generation or score left behind.
//  (a) the same request scores bit-identically twice, and again after an unrelated generation;
//  (b) every target log-probability equals log_softmax of the last-position logits that prefillWithLogits
//      produces for a fresh context of prompt + earlier target tokens;
//  (c) each greedy token is the argmax of that row and its log-probability is the row's maximum;
//  (d) greedy generation after scoring is unchanged.
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
import { scoreTargetSequence } from '../../src/inference/target-sequence-scoring.js';
import { validateTargetSequenceScoreResult } from '../../src/config/target-sequence-scoring.js';
import { formatChatMessages } from '../../src/inference/pipelines/text/generator/request/chat-format.js';
import { selectedTokenLogSoftmax } from '../../src/inference/pipelines/text/logits/selected-log-softmax.js';

// Cross-path tolerance (teacher-forced multi-row prefill vs a shorter single-row prefill), as in
// target-sequence-scoring-physical.test.js. Same-path repeats must be bit-identical.
const CROSS_PATH_TOLERANCE = 0.05;
const SYSTEM = "Answer the user's prompt directly. Return only the requested answer, without explanation.";
const PROMPTS = [
  'Name the capital of France in one word with a period.',
  'Say that you agree, in one word, with a period.',
];

const directory = process.env.DOPPLER_TARGET_SCORE_MODEL_DIR;
if (!directory) {
  console.log('target-sequence-scoring-state-physical: skipped (set DOPPLER_TARGET_SCORE_MODEL_DIR)');
} else {
  const bytes = await fs.readFile(path.join(directory, 'manifest.json'));
  const manifest = JSON.parse(bytes);
  const report = { modelId: manifest.modelId, manifestSha256: createHash('sha256').update(bytes).digest('hex'), cases: [] };
  const mismatchWarnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => {
    if (args.some(arg => typeof arg === 'string' && arg.includes('linear_attention state mismatch'))) {
      mismatchWarnings.push(String(args[0]));
    }
    originalWarn(...args);
  };
  let pipeline;
  try {
    const gpu = await probeNodeGPU({ installFileFetchShim: true });
    assert.ok(gpu.ready, gpu.reason);
    report.adapter = getKernelCapabilities().adapterInfo;
    pipeline = await createPipeline(manifest, {
      runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 256 } } } },
      storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(directory)).href, manifest),
    });
    const template = pipeline.modelConfig.chatTemplateType;
    const stopIds = new Set([...(pipeline.modelConfig.stopTokenIds ?? []), pipeline.tokenizer.getSpecialTokens?.()?.eos]);
    const requestFor = user => {
      const messages = [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }];
      const prompt = formatChatMessages(messages, template, { thinking: false });
      return { messages, prompt, promptIds: Array.from(pipeline.tokenizer.encode(prompt)) };
    };
    // Generation leaves its sequence state in place (no reset afterwards), as an application's would.
    const greedy = async ({ promptIds }) => runPipelineOperation(pipeline, async p => {
      const result = await p.generateTokenIds('', { inputIds: promptIds, useChatTemplate: false,
        maxTokens: 16, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1 });
      const ids = [...result.tokenIds];
      while (ids.length && stopIds.has(ids.at(-1))) ids.pop();
      return { ids, decodeMode: p.stats?.decodeMode ?? null };
    });
    const score = request => runPipelineOperation(pipeline, p => scoreTargetSequence(p, request));
    // Fresh-context last-position logits for prompt + prefix (prefillWithLogits appends, so reset around it).
    const freshRow = ids => runPipelineOperation(pipeline, async p => {
      p.resetGenerationState();
      try { return Float32Array.from((await p.prefillWithLogits('', { inputIds: ids, useChatTemplate: false })).logits); }
      finally { p.resetGenerationState(); }
    });

    const cases = PROMPTS.map(requestFor);
    for (const [index, item] of cases.entries()) {
      const other = cases[(index + 1) % cases.length];
      const { ids: greedyIds, decodeMode } = await greedy(item);
      assert.ok(greedyIds.length > 0, 'greedy decoding produced tokens');
      const target = pipeline.tokenizer.decode(greedyIds, true, false);
      const request = { messages: item.messages, target, maxSeqLen: 256 };

      // (a) Right after the generation, again, and after an unrelated generation: bit-identical.
      const first = validateTargetSequenceScoreResult(request, await score(request));
      assert.deepEqual(first.targetTokenIds, greedyIds, 'canonical target tokenization equals the greedy tokens');
      const repeat = await score(request);
      const unrelated = await greedy(other);
      const afterGeneration = await score(request);
      assert.deepEqual(repeat.tokenLogProbs, first.tokenLogProbs, 'repeat score is bit-identical');
      assert.deepEqual(afterGeneration.tokenLogProbs, first.tokenLogProbs, 'score after an unrelated generation is bit-identical');

      // (b) and (c) per target position against a fresh-context full row.
      const positions = [];
      for (let i = 0; i < greedyIds.length; i++) {
        const row = await freshRow([...item.promptIds, ...greedyIds.slice(0, i)]);
        let argmax = 0;
        for (let v = 1; v < row.length; v++) if (row[v] > row[argmax]) argmax = v;
        const independent = [greedyIds[i], argmax].map(id => selectedTokenLogSoftmax(row, row.length, [id]).logProbs[0]);
        positions.push({ position: i, tokenId: greedyIds[i], argmax, teacherForced: first.tokenLogProbs[i],
          freshLogProb: independent[0], maxLogProb: independent[1],
          error: Math.abs(first.tokenLogProbs[i] - independent[0]),
          maxError: Math.abs(first.tokenLogProbs[i] - independent[1]) });
      }
      // (d) Greedy decoding after all of the above is unchanged.
      const again = await greedy(item);
      report.cases.push({ prompt: item.messages[1].content, target, greedyIds, decodeMode, unrelatedTarget:
        pipeline.tokenizer.decode(unrelated.ids, true, false), totalLogProb: first.totalLogProb,
        repeatTotal: repeat.totalLogProb, afterGenerationTotal: afterGeneration.totalLogProb, positions,
        greedyAfterScoring: again.ids });
      for (const position of positions) {
        assert.equal(position.argmax, position.tokenId, `greedy token ${position.position} is the fresh-row argmax`);
        assert.ok(position.error <= CROSS_PATH_TOLERANCE,
          `target token ${position.position} log-probability differs from the fresh row by ${position.error}`);
        assert.ok(position.maxError <= CROSS_PATH_TOLERANCE,
          `greedy token ${position.position} log-probability differs from the row maximum by ${position.maxError}`);
      }
      assert.deepEqual(again.ids, greedyIds, 'greedy generation after scoring is unchanged');
    }
    report.linearAttentionStateMismatchWarnings = mismatchWarnings.length;
    assert.equal(mismatchWarnings.length, 0, `scoring met stale linear-attention state: ${mismatchWarnings[0]}`);
    console.log(JSON.stringify({ status: 'passed', modelId: report.modelId, cases: report.cases.map(c => ({
      target: c.target, total: c.totalLogProb, maxError: Math.max(...c.positions.map(p => p.error)) })) }));
  } finally {
    console.warn = originalWarn;
    await pipeline?.unload();
    destroyDevice();
    await releaseNodeWebGPU();
    if (process.env.DOPPLER_TARGET_SCORE_RECEIPT) {
      await fs.writeFile(process.env.DOPPLER_TARGET_SCORE_RECEIPT, JSON.stringify(report, null, 2) + '\n');
    }
  }
}
