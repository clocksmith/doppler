import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { resolvePreset, detectPreset } from '../../src/config/loader.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

// === Gemma 4 preset resolves and inherits from Gemma 3 ===

{
  const resolved = resolvePreset('gemma4');
  assert.equal(resolved.id, 'gemma4');
  assert.equal(resolved.modelType, 'mixtral');

  // Attention: slidingWindow overridden from gemma3's 512 to 1024
  assert.equal(resolved.inference.attention.slidingWindow, 1024);
  // Inherited from gemma3
  assert.equal(resolved.inference.attention.queryPreAttnScalar, 256);
  assert.equal(resolved.inference.attention.queryKeyNorm, true);

  // Normalization inherited from gemma3
  assert.equal(resolved.inference.normalization.rmsNormWeightOffset, true);
  assert.equal(resolved.inference.normalization.postAttentionNorm, true);
  assert.equal(resolved.inference.normalization.preFeedforwardNorm, true);
  assert.equal(resolved.inference.normalization.postFeedforwardNorm, true);

  // FFN activation inherited from gemma3
  assert.equal(resolved.inference.ffn.activation, 'gelu');

  // Output inherited from gemma3
  assert.equal(resolved.inference.output.scaleEmbeddings, true);

  // RoPE: overridden values
  assert.equal(resolved.inference.rope.ropeTheta, 1000000);
  assert.equal(resolved.inference.rope.ropeLocalTheta, 10000);
  assert.equal(resolved.inference.rope.ropeScalingType, 'yarn');
  assert.equal(resolved.inference.rope.ropeScalingFactor, 8.0);
  assert.equal(resolved.inference.rope.yarnBetaFast, 4.0);
  assert.equal(resolved.inference.rope.yarnBetaSlow, 1.0);
  assert.equal(resolved.inference.rope.yarnOriginalMaxPos, 32768);

  // Chat template inherited from gemma3
  assert.equal(resolved.inference.chatTemplate.type, 'gemma');
  assert.equal(resolved.inference.chatTemplate.enabled, true);
}

// === MoE family facts are present and complete ===

{
  const resolved = resolvePreset('gemma4');
  const moe = resolved.inference.moe;
  assert.ok(moe, 'gemma4 preset must include inference.moe');

  assert.equal(moe.kernelProfileId, 'mixtral-moe-v1');
  assert.equal(moe.numExperts, 8);
  assert.equal(moe.topK, 2);
  assert.equal(moe.numSharedExperts, 0);
  assert.equal(moe.routerDtype, 'f32');
  assert.equal(moe.tensorPattern, 'mixtral');
  assert.deepEqual(moe.supportedActivationDtypes, ['f16', 'f32']);
  assert.equal(moe.preferredActivationDtype, 'f32');
}

// === kernelProfileId has a backing profile asset ===

{
  const resolved = resolvePreset('gemma4');
  const profileId = resolved.inference.moe.kernelProfileId;
  const profilePath = path.join(
    repoRoot,
    'src/config/kernels/moe',
    `${profileId.replace(/-moe-v1$/, '')}.paths.json`
  );
  const profileContent = JSON.parse(await fs.readFile(profilePath, 'utf8'));
  assert.equal(profileContent.id, profileId,
    `MoE kernel profile file must have id "${profileId}"`);
  assert.ok(profileContent.router, 'profile must define router rules');
  assert.ok(profileContent.dequant, 'profile must define dequant rules');
}

// === Kernel paths reference valid registry entries ===

{
  const resolved = resolvePreset('gemma4');
  const kernelPaths = resolved.inference.kernelPaths;
  assert.ok(kernelPaths, 'gemma4 must define kernelPaths');
  assert.ok(kernelPaths.q4k, 'gemma4 must define kernelPaths.q4k');

  const registryPath = path.join(repoRoot, 'src/config/presets/kernel-paths/registry.json');
  const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
  const registryIds = new Set(registry.entries.map((e) => e.id));

  for (const [dtype, pathMap] of Object.entries(kernelPaths)) {
    for (const [variant, pathId] of Object.entries(pathMap)) {
      assert.ok(
        registryIds.has(pathId),
        `kernelPaths.${dtype}.${variant} = "${pathId}" must exist in kernel-path registry`
      );
    }
  }
}

// === Tensor patterns define Mixtral-style expert FFN ===

{
  const resolved = resolvePreset('gemma4');
  const ffnPatterns = resolved.tensorPatterns?.ffn;
  assert.ok(ffnPatterns, 'gemma4 must define tensorPatterns.ffn');
  assert.ok(ffnPatterns.gate, 'must define gate pattern');
  assert.ok(ffnPatterns.up, 'must define up pattern');
  assert.ok(ffnPatterns.down, 'must define down pattern');

  for (const [name, patterns] of Object.entries(ffnPatterns)) {
    assert.ok(Array.isArray(patterns) && patterns.length > 0,
      `tensorPatterns.ffn.${name} must be a non-empty array`);
    for (const pattern of patterns) {
      assert.ok(pattern.includes('{expert}'),
        `tensorPatterns.ffn.${name} pattern "${pattern}" must include {expert} placeholder`);
      assert.ok(pattern.includes('{layer}'),
        `tensorPatterns.ffn.${name} pattern "${pattern}" must include {layer} placeholder`);
    }
  }
}

// === Detection patterns are present ===

{
  const resolved = resolvePreset('gemma4');
  assert.ok(resolved.detection.architecturePatterns.includes('gemma4'));
  assert.ok(resolved.detection.architecturePatterns.includes('Gemma4ForCausalLM'));
  assert.ok(resolved.detection.modelTypePatterns.includes('gemma4'));
  assert.ok(resolved.detection.modelTypePatterns.includes('gemma4_moe'));
}

// === detectPreset finds gemma4 from model_type ===

{
  const detected = detectPreset({ model_type: 'gemma4' }, 'Gemma4ForCausalLM');
  assert.equal(detected, 'gemma4');
}

{
  const detected = detectPreset({ model_type: 'gemma4_moe' }, 'Gemma4ForCausalLM');
  assert.equal(detected, 'gemma4');
}

console.log('gemma4-preset-contract.test: ok');
