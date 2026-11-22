// Real-model target scoring on a physical GPU. Opt-in: set DOPPLER_TARGET_SCORE_MODEL_DIR to a
// local RDRR text model directory (manifest, tokenizer and shards), e.g. gemma-3-270m-it-q4k-ehf16-af32.
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

const directory = process.env.DOPPLER_TARGET_SCORE_MODEL_DIR;
if (!directory) {
  console.log('target-sequence-scoring-physical: skipped (set DOPPLER_TARGET_SCORE_MODEL_DIR)');
} else {
  const bytes = await fs.readFile(path.join(directory, 'manifest.json'));
  const manifest = JSON.parse(bytes);
  const report = { modelId: manifest.modelId, manifestSha256: createHash('sha256').update(bytes).digest('hex'), cases: [] };
  let pipeline;
  try {
    const gpu = await probeNodeGPU({ installFileFetchShim: true });
    assert.ok(gpu.ready, gpu.reason);
    report.adapter = getKernelCapabilities().adapterInfo;
    pipeline = await createPipeline(manifest, {
      runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 256 } } } },
      storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(directory)).href, manifest),
    });
    const messages = [
      { role: 'system', content: 'Answer with a single word.' },
      { role: 'user', content: 'What is the capital of France?' },
    ];
    const prompt = formatChatMessages(messages, pipeline.modelConfig.chatTemplateType, { thinking: false });
    const promptIds = Array.from(pipeline.tokenizer.encode(prompt));
    const stopIds = new Set([...(pipeline.modelConfig.stopTokenIds ?? []), pipeline.tokenizer.getSpecialTokens?.()?.eos]);
    const greedy = async () => runPipelineOperation(pipeline, async p => {
      try {
        const result = await p.generateTokenIds('', { inputIds: promptIds, useChatTemplate: false,
          maxTokens: 6, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1 });
        return result.tokenIds.filter(id => !stopIds.has(id));
      } finally { p.resetGenerationState(); }
    });
    const score = request => runPipelineOperation(pipeline, p => scoreTargetSequence(p, request));
    const greedyIds = await greedy();
    const target = pipeline.tokenizer.decode(greedyIds, true, false);
    assert.ok(target.length > 0, 'greedy decoding produced text');
    const request = { prompt, target, maxSeqLen: 256 };
    const started = performance.now();
    const greedyScore = validateTargetSequenceScoreResult(request, await score(request));
    report.cases.push({ kind: 'greedy-target', target, greedyIds, wallMs: performance.now() - started, result: greedyScore });
    assert.deepEqual(greedyScore.targetTokenIds, greedyIds, 'canonical target tokenization equals the greedy tokens');

    // Same request, same runtime and GPU: bit-identical values.
    const repeat = await score(request);
    assert.deepEqual(repeat.tokenLogProbs, greedyScore.tokenLogProbs);
    assert.equal(repeat.totalLogProb, greedyScore.totalLogProb);
    // messages and the equivalent formatted prompt are the same request.
    const fromMessages = await score({ messages, target, maxSeqLen: 256 });
    assert.deepEqual(fromMessages.tokenLogProbs, greedyScore.tokenLogProbs);

    // Independent path: full last-position logits from prefillWithLogits give the first target token.
    const firstRow = await runPipelineOperation(pipeline, async p => {
      try { return (await p.prefillWithLogits(prompt, { inputIds: promptIds, useChatTemplate: false })).logits; }
      finally { p.resetGenerationState(); }
    });
    const independent = selectedTokenLogSoftmax(Float32Array.from(firstRow), firstRow.length, [greedyIds[0]]).logProbs[0];
    const firstTokenError = Math.abs(independent - greedyScore.tokenLogProbs[0]);
    report.cases.push({ kind: 'first-token-cross-check', independent, teacherForced: greedyScore.tokenLogProbs[0], firstTokenError });
    assert.ok(firstTokenError <= 0.05, `first-token log-probability differs from the last-position path by ${firstTokenError}`);

    // A perturbed target is less likely than what greedy decoding produces.
    const perturbed = await score({ prompt, target: `Banana ${target}`, maxSeqLen: 256 });
    report.cases.push({ kind: 'perturbed-target', target: `Banana ${target}`, result: perturbed });
    assert.ok(greedyScore.totalLogProb > perturbed.totalLogProb,
      `greedy ${greedyScore.totalLogProb} must exceed perturbed ${perturbed.totalLogProb}`);

    // Scoring resets the continuation: greedy decoding afterwards is unchanged.
    assert.deepEqual(await greedy(), greedyIds);
    console.log(JSON.stringify({ status: 'passed', modelId: report.modelId, target,
      roundedTotalLogProb: greedyScore.roundedTotalLogProb, perturbed: perturbed.roundedTotalLogProb, firstTokenError }));
  } finally {
    await pipeline?.unload();
    destroyDevice();
    await releaseNodeWebGPU();
    if (process.env.DOPPLER_TARGET_SCORE_RECEIPT) {
      await fs.writeFile(process.env.DOPPLER_TARGET_SCORE_RECEIPT, JSON.stringify(report, null, 2) + '\n');
    }
  }
}
