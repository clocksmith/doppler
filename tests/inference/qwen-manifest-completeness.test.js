import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';

const { inferLinearNormMode } = await import(
  '../../src/inference/pipelines/text/linear-attention.js'
);

async function loadJson(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
}

const f16ManifestPath = new URL('../../models/local/qwen-3-5-0-8b-f16/manifest.json', import.meta.url);
const q4kManifestPath = new URL('../../models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json', import.meta.url);
const hasExactLocalManifests = existsSync(f16ManifestPath) && existsSync(q4kManifestPath);
const f16Manifest = hasExactLocalManifests
  ? JSON.parse(await readFile(f16ManifestPath, 'utf8'))
  : null;
const q4kManifest = hasExactLocalManifests
  ? JSON.parse(await readFile(q4kManifestPath, 'utf8'))
  : null;
const convConfigs = await Promise.all([
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json'),
]);

const EXPECTED_QWEN_LAYER_TYPES = Array.from(
  { length: 24 },
  (_, index) => ((index + 1) % 4 === 0 ? 'full_attention' : 'linear_attention')
);
const EXPECTED_QWEN_DECODE_LOOP = Object.freeze({
  batchSize: 4,
  stopCheckMode: 'batch',
  readbackInterval: 1,
  ringTokens: 1,
  ringStop: 1,
  ringStaging: 1,
  disableCommandBatching: true,
});
const EXPECTED_QWEN_COMPUTE_DEFAULTS = Object.freeze({
  activationDtype: 'f16',
  mathDtype: 'f16',
  accumDtype: 'f16',
  outputDtype: 'f16',
});

function assertQwenDecodeLoop(decodeLoop, label) {
  assert.ok(decodeLoop && typeof decodeLoop === 'object', `${label} decodeLoop must be present`);
  assert.deepEqual(decodeLoop, EXPECTED_QWEN_DECODE_LOOP, `${label} decodeLoop`);
}

function assertQwenComputeDefaults(computeDefaults, label) {
  assert.ok(computeDefaults && typeof computeDefaults === 'object', `${label} compute defaults must be present`);
  assert.deepEqual(computeDefaults, EXPECTED_QWEN_COMPUTE_DEFAULTS, `${label} compute defaults`);
}

function assertQwenConversionConfig(config) {
  assert.equal(config.inference.normalization.rmsNormWeightOffset, true);
  assert.equal(config.inference.rope.mropeInterleaved, true);
  assert.equal(config.inference.output.tieWordEmbeddings, true);
  assert.equal(config.inference.layerPattern.type, 'custom');
  assert.equal(config.inference.layerPattern.globalPattern, null);
  assert.equal(config.inference.layerPattern.period, null);
  assert.equal(config.inference.layerPattern.offset, null);
  assert.deepEqual(config.inference.layerPattern.layerTypes, EXPECTED_QWEN_LAYER_TYPES);
  assertQwenDecodeLoop(config.sessionDefaults?.decodeLoop, config.output?.modelBaseId ?? 'qwen');
  assertQwenComputeDefaults(
    config.sessionDefaults?.compute?.defaults,
    config.output?.modelBaseId ?? 'qwen'
  );
  assert.ok(config.execution && typeof config.execution === 'object', 'qwen execution config must be present');
  assert.equal(config.execution.inlineKernelPath, false, 'qwen execution.inlineKernelPath');
}

function hasTensor(manifest, tensorName) {
  return Object.prototype.hasOwnProperty.call(manifest.tensors ?? {}, tensorName);
}

function getLinearProjectionLayout(manifest) {
  return {
    headVDim: manifest.architecture.linearValueHeadDim,
    valueDim: manifest.architecture.linearNumValueHeads * manifest.architecture.linearValueHeadDim,
  };
}

if (!hasExactLocalManifests) {
  for (const config of convConfigs) {
    // V1 configs have explicit inference and no legacy family-indirection field
    assert.ok(config.inference?.attention, 'v1 config must have explicit inference.attention');
    assertQwenConversionConfig(config);
  }
  console.log('qwen-manifest-completeness.test: skipped (missing exact local manifests)');
  process.exit(0);
}

// --- Manifest: intentional null fields ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    assert.equal(manifest.inference.defaultKernelPath, null);
  }
}

// --- Manifest: execution v1 is present, but Qwen disables inline kernel-path lowering ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    assert.equal(manifest.inference.schema, 'doppler.execution/v1');
    assert.ok(manifest.inference.execution && typeof manifest.inference.execution === 'object');
    assert.equal(manifest.inference.execution.inlineKernelPath, false);
  }
}

// --- Manifest: sessionDefaults keep explicit compute/kvcache + Qwen decode loop ---

{
  for (const manifest of [f16Manifest, q4kManifest]) {
    const sd = manifest.inference.sessionDefaults;
    assert.ok(sd != null);
    assertQwenDecodeLoop(sd.decodeLoop, `${manifest.modelId}.inference.sessionDefaults`);
    assert.equal(sd.kvcache?.kvDtype, 'f16');
    assertQwenComputeDefaults(sd.compute?.defaults, `${manifest.modelId}.inference.sessionDefaults`);
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

// --- Manifest: rmsNormWeightOffset = true (from the explicit conversion config) ---

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

// --- Conversion configs: Qwen keeps an explicit execution graph but disables inline kernel-path lowering ---

for (const config of convConfigs) {
  assert.ok(config.execution && typeof config.execution === 'object');
  assert.equal(config.execution.inlineKernelPath, false);
}

// --- Conversion configs: explicit qwen3 family config fields are authored directly ---

for (const config of convConfigs) {
  assertQwenConversionConfig(config);
}

// --- Conversion configs: top-level sessionDefaults keep the Qwen decode loop contract ---

for (const config of convConfigs) {
  assertQwenDecodeLoop(config.sessionDefaults?.decodeLoop, config.output?.modelBaseId ?? 'qwen');
}

console.log('qwen-manifest-completeness.test: ok');
