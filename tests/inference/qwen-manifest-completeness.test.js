import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const { hasExecutionV0 } = await import(
  '../../src/inference/pipelines/text/execution-v0.js'
);
const { inferLinearNormMode } = await import(
  '../../src/inference/pipelines/text/linear-attention.js'
);

async function loadJson(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
}

const f16Manifest = await loadJson('../../models/local/qwen-3-5-0-8b-f16/manifest.json');
const q4kManifest = await loadJson('../../models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json');
const convConfigs = await Promise.all([
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json'),
]);

function hasTensor(manifest, tensorName) {
  return Object.prototype.hasOwnProperty.call(manifest.tensors ?? {}, tensorName);
}

function getLinearProjectionLayout(manifest) {
  return {
    headVDim: manifest.architecture.linearValueHeadDim,
    valueDim: manifest.architecture.linearNumValueHeads * manifest.architecture.linearValueHeadDim,
  };
}

// --- Manifest: intentional null fields ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    assert.equal(manifest.inference.schema, null);
    assert.equal(manifest.inference.defaultKernelPath, null);
    assert.equal(manifest.inference.execution, null);
  }
}

// --- Manifest: execution: null means legacy dispatch, not execution-v0 ---

{
  assert.equal(hasExecutionV0(f16Manifest.inference), false);
  assert.equal(hasExecutionV0(q4kManifest.inference), false);
}

// --- Manifest: sessionDefaults shape (decodeLoop only, no kvcache/compute) ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    const sd = manifest.inference.sessionDefaults;
    assert.ok(sd != null);
    assert.ok(sd.decodeLoop != null);
    assert.equal(sd.kvcache, undefined);
    assert.equal(sd.compute, undefined);
    assert.equal(sd.execution, undefined);
  }
}

// --- Manifest: Qwen hybrid layer pattern ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    const lp = manifest.inference.layerPattern;
    assert.equal(lp.type, 'custom');
    assert.equal(lp.layerTypes.length, manifest.architecture.numLayers);

    const linearCount = lp.layerTypes.filter((t) => t === 'linear_attention').length;
    const fullCount = lp.layerTypes.filter((t) => t === 'full_attention').length;
    assert.equal(linearCount, 18);
    assert.equal(fullCount, 6);

    for (let i = 0; i < lp.layerTypes.length; i++) {
      const expected = (i + 1) % 4 === 0 ? 'full_attention' : 'linear_attention';
      assert.equal(lp.layerTypes[i], expected, `${manifest.modelId} layer ${i}`);
    }
  }
}

// --- Manifest: linear attention architecture fields present ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    const arch = manifest.architecture;
    assert.equal(arch.linearNumKeyHeads, 16);
    assert.equal(arch.linearNumValueHeads, 16);
    assert.equal(arch.linearKeyHeadDim, 128);
    assert.equal(arch.linearValueHeadDim, 128);
    assert.equal(arch.linearConvKernelDim, 4);
  }
}

// --- Qwen linear-attention layers use linear-attn norm weights, not self-attn q/k norm ---

{
  assert.equal(f16Manifest.architecture.linearNormMode, undefined);
  assert.equal(q4kManifest.architecture.linearNormMode, 'shared');

  for (const manifest of [f16Manifest, q4kManifest]) {
    const projectionLayout = getLinearProjectionLayout(manifest);
    for (let layerIdx = 0; layerIdx < manifest.inference.layerPattern.layerTypes.length; layerIdx++) {
      const layerType = manifest.inference.layerPattern.layerTypes[layerIdx];
      const prefix = `model.language_model.layers.${layerIdx}`;
      if (layerType === 'linear_attention') {
        assert.equal(hasTensor(manifest, `${prefix}.linear_attn.in_proj_z.weight`), true);
        assert.equal(hasTensor(manifest, `${prefix}.linear_attn.in_proj_a.weight`), true);
        assert.equal(hasTensor(manifest, `${prefix}.linear_attn.in_proj_b.weight`), true);
        assert.equal(hasTensor(manifest, `${prefix}.linear_attn.norm.weight`), true);
        assert.equal(hasTensor(manifest, `${prefix}.self_attn.q_norm.weight`), false);
        assert.equal(hasTensor(manifest, `${prefix}.self_attn.k_norm.weight`), false);
        assert.equal(
          inferLinearNormMode(manifest.tensors[`${prefix}.linear_attn.norm.weight`], projectionLayout),
          'shared',
          `${manifest.modelId} layer ${layerIdx} linear-attention norm should resolve to shared`
        );
        continue;
      }

      if (layerType === 'full_attention') {
        assert.equal(hasTensor(manifest, `${prefix}.self_attn.q_norm.weight`), true);
        assert.equal(hasTensor(manifest, `${prefix}.self_attn.k_norm.weight`), true);
      }
    }
  }
}

// --- Manifest: rmsNormWeightOffset = true (converter override, not preset) ---

{
  assert.equal(f16Manifest.inference.normalization.rmsNormWeightOffset, true);
  assert.equal(q4kManifest.inference.normalization.rmsNormWeightOffset, true);
}

// --- Manifest: mRoPE fields ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    const rope = manifest.inference.rope;
    assert.deepEqual(rope.mropeSection, [11, 11, 10]);
    assert.equal(rope.mropeInterleaved, true);
    assert.equal(rope.partialRotaryFactor, 0.25);

    const sectionSum = rope.mropeSection.reduce((a, b) => a + b, 0);
    const resolvedRotaryDim = manifest.architecture.headDim * rope.partialRotaryFactor;
    assert.equal(sectionSum * 2, resolvedRotaryDim);
  }
}

// --- Conversion configs: all set defaultKernelPath: null ---

for (const config of convConfigs) {
  assert.equal(config.inference.defaultKernelPath, null);
}

// --- Conversion configs: none set execution (absent = null in manifest) ---

for (const config of convConfigs) {
  assert.equal(config.inference.execution, undefined);
}

// --- Conversion configs: all use qwen3 preset ---

for (const config of convConfigs) {
  assert.equal(config.presets.model, 'qwen3');
}

// --- Conversion configs: sessionDefaults.decodeLoop present in all ---

for (const config of convConfigs) {
  assert.ok(config.inference.sessionDefaults.decodeLoop != null);
  assert.equal(config.inference.sessionDefaults.decodeLoop.batchSize, 4);
  assert.equal(config.inference.sessionDefaults.decodeLoop.disableCommandBatching, true);
}

console.log('qwen-manifest-completeness.test: ok');
