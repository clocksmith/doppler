import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const modelId = 'qwen-3-embedding-0-6b-q4k-ehf16-af32';
const config = readJson(`src/config/conversion/qwen3/${modelId}.json`);
const catalog = readJson('models/catalog.json');
const catalogEntry = catalog.models.find((entry) => entry.modelId === modelId);

assert.ok(catalogEntry, `${modelId} must be cataloged as an onboarding target`);
assert.equal(catalogEntry.lifecycle?.status?.conversion, 'ready');
assert.equal(catalogEntry.lifecycle?.status?.runtime, 'active');
assert.equal(catalogEntry.lifecycle?.status?.tested, 'verified');
assert.equal(catalogEntry.lifecycle?.tested?.result, 'pass');
assert.equal(catalogEntry.quickstart, false);
assert.equal(catalogEntry.demoVisible, false);
assert.equal(catalogEntry.hf?.repoId, 'Clocksmith/rdrr');
assert.equal(catalogEntry.hf?.path, `models/${modelId}`);
assert.equal(catalogEntry.hf?.revision, null);
assert.equal(catalogEntry.runtimePromotionState, 'manifest-owned');
assert.equal(catalogEntry.artifactCompleteness, 'complete');
assert.equal(catalogEntry.verify?.workload, 'embedding');

assert.equal(config.modelType, 'embedding');
assert.equal(config.output?.modelBaseId, modelId);
assert.equal(config.manifest?.artifactIdentity?.sourceCheckpointId, 'Qwen/Qwen3-Embedding-0.6B');
assert.equal(config.manifest?.artifactIdentity?.artifactCompleteness, 'complete');

assert.deepEqual(config.inference?.output?.embeddingPostprocessor, {
  poolingMode: 'last',
  includePrompt: true,
  projections: [],
  normalize: 'l2',
});

assert.equal(config.inference?.attention?.queryPreAttnScalar, 128);
assert.equal(config.inference?.attention?.queryKeyNorm, true);
assert.equal(config.inference?.attention?.causal, true);
assert.equal(config.inference?.attention?.slidingWindow, null);
assert.equal(config.inference?.attention?.attentionOutputGate, false);
assert.equal(config.inference?.normalization?.rmsNormWeightOffset, false);
assert.equal(config.inference?.ffn?.activation, 'silu');
assert.equal(config.inference?.ffn?.branchMode, 'dense');
assert.equal(config.inference?.rope?.ropeTheta, 1000000);
assert.equal(config.inference?.rope?.mropeInterleaved, false);
assert.equal(config.inference?.rope?.partialRotaryFactor, null);
assert.equal(config.inference?.layerPattern?.type, 'uniform');
assert.equal(config.inference?.chatTemplate?.enabled, false);

assert.deepEqual(
  config.execution?.postLayer,
  [['final_norm', 'rmsnorm']],
  `${modelId} must expose embeddings from hidden states instead of logits`
);
assert.ok(!Object.hasOwn(config.execution?.kernels ?? {}, 'lm_head_q4'));
assert.ok(!Object.hasOwn(config.execution?.kernels ?? {}, 'lm_head_gemv'));
assert.ok(!Object.hasOwn(config.execution?.kernels ?? {}, 'sample'));
assert.equal(config.execution?.kernels?.attn_stream?.kernel, 'attention_streaming_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_stream?.precision?.kvDtype, 'f16');

assert.doesNotThrow(
  () => validateRequiredInferenceFields(config.inference, modelId),
  `${modelId} conversion config must satisfy required inference-field validation`
);

console.log('qwen3-embedding-onboarding-contract.test: ok');
