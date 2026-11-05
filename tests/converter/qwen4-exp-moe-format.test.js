import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveManifestMoEConfig, resolveMoEExpertFormat } from '../../src/converter/manifest-builder.js';

// Contract: qwen4_exp (Flash-Next) experts use Doppler's qwen3_5 expert format:
// fused gate/up, SiLU experts, renormalized top-k routing and a sigmoid-gated
// shared expert. Any other format drops the shared expert at runtime.
const flashNext = JSON.parse(await readFile(new URL('../../artifacts/qwen38-flash-next-20261013/source/config.json', import.meta.url), 'utf8'));
assert.equal(resolveMoEExpertFormat(flashNext, 'transformer', null, null), 'qwen3_5');
assert.equal(resolveMoEExpertFormat({ model_type: 'qwen4_exp_text' }, 'transformer', null, null), 'qwen3_5');
const moe = resolveManifestMoEConfig({ tensors: [] }, { modelId: 'flash-next' }, flashNext, 'transformer');
assert.equal(moe.expertFormat, 'qwen3_5');
assert.equal(moe.numExperts, 512);
assert.equal(moe.numExpertsPerToken, 10);
assert.equal(moe.expertIntermediateSize, 640);
assert.equal(moe.sharedExpertIntermediateSize, 640);
// The 35B family is unchanged; other MoE families keep their formats.
assert.equal(resolveMoEExpertFormat({ model_type: 'qwen3_5_moe' }, 'transformer', null, null), 'qwen3_5');
assert.equal(resolveMoEExpertFormat({ model_type: 'mixtral' }, 'transformer', null, null), 'mixtral');
console.log('qwen4-exp-moe-format.test: ok');
