import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { doppler } from '../../src/client/doppler-api.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { loadRuntimeProfile } from '../../src/tooling/browser-harness/runtime-config.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const [modelId, directory, profile, output, consent] = process.argv.slice(2);
assert(modelId && directory && profile && output && consent === '--physical',
  'Usage: qwen-installed-smoke.js MODEL DIRECTORY PROFILE OUTPUT --physical');
installNodeFileFetchShim();
// Test-owned admission cap, not a runtime default or recipe promotion.
const runtime = profile === 'default'
  ? { shared: { bufferPool: { budget: { maxTotalBytes: 25769803776,
    highWatermarkRatio: 0.9, emergencyTrimTargetRatio: 0.8, hardFailOnBudgetExceeded: true } } } }
  : (await loadRuntimeProfile(profile, { baseUrl: new URL('../../src/config/runtime/', import.meta.url).href })).runtime;
const receipt = { schema: 'doppler.qwen-installed-smoke/v1', modelId, profile,
  scope: 'Functional smoke; not an isolated benchmark or full numerical/quality qualification',
  runtime, passed: false };
let model;
try {
  model = await doppler.load({ url: pathToFileURL(path.resolve(directory)).href },
    { runtimeConfig: runtime, isolatedLoader: true });
  receipt.manifestHash = model.manifestHash;
  if (modelId.includes('embedding')) {
    const result = await model.embedWithEvidence('The sky is blue.');
    assert(result.embedding.length > 0 && [...result.embedding].every(Number.isFinite));
    receipt.embedding = result;
  } else if (modelId.includes('reranker')) {
    receipt.reranking = await model.rerankWithEvidence('What color is the sky?',
      ['The sky is blue.', 'Apples grow on trees.']);
    assert(receipt.reranking.scores.every(row => Number.isFinite(row.score) && Number.isFinite(row.probability)));
    assert(receipt.reranking.scores[0].score > receipt.reranking.scores[1].score);
  } else {
    const options = { maxTokens: 4, temperature: 0, topK: 1, topP: 1,
      repetitionPenalty: 1, useChatTemplate: true };
    const first = await model.generateWithEvidence('Say hello in one word.', options);
    model.resetGenerationState();
    const reused = await model.generateWithEvidence('Say hello in one word.', options);
    assert.deepEqual(first.tokenIds, reused.tokenIds, 'Resident reuse changed output');
    assert(first.outputText.trim());
    receipt.generations = [first, reused];
  }
  receipt.passed = true;
} catch (error) {
  receipt.failure = String(error);
  process.exitCode = 1;
} finally {
  try {
    await model?.unload();
    destroyDevice();
    await releaseNodeWebGPU();
  } catch (error) {
    receipt.cleanupFailure = String(error);
    process.exitCode = 1;
  }
  await fs.writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ modelId, passed: receipt.passed, failure: receipt.failure }));
}
