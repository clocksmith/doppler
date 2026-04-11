import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

const { inferLinearNormMode } = await import(
  '../../src/inference/pipelines/text/linear-attention.js'
);

async function loadJson(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
}

const q4kManifestPath = new URL('../../models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json', import.meta.url);
const q4k2bManifestPath = new URL('../../models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json', import.meta.url);
const f16ManifestPath = new URL('../../models/local/qwen-3-5-0-8b-f16/manifest.json', import.meta.url);
const hasLocalQ4KManifest = existsSync(q4kManifestPath);
const hasOptionalLocalQ4K2BManifest = existsSync(q4k2bManifestPath);
const hasOptionalLocalF16Manifest = existsSync(f16ManifestPath);
const q4kManifest = hasLocalQ4KManifest
  ? JSON.parse(await readFile(q4kManifestPath, 'utf8'))
  : null;
const q4k2bManifest = hasOptionalLocalQ4K2BManifest
  ? JSON.parse(await readFile(q4k2bManifestPath, 'utf8'))
  : null;
const f16Manifest = hasOptionalLocalF16Manifest
  ? JSON.parse(await readFile(f16ManifestPath, 'utf8'))
  : null;
const convConfigs = await Promise.all([
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json'),
]);
const availableLocalManifests = [q4kManifest, q4k2bManifest, f16Manifest].filter(Boolean);

const EXPECTED_QWEN_LAYER_TYPES = Array.from(
  { length: 24 },
  (_, index) => ((index + 1) % 4 === 0 ? 'full_attention' : 'linear_attention')
);
const EXPECTED_QWEN_COMPUTE_DEFAULTS = Object.freeze({
  'qwen-3-5-0-8b-q4k-ehaf16': {
    activationDtype: 'f32',
    mathDtype: 'f32',
    accumDtype: 'f32',
    outputDtype: 'f32',
  },
  'qwen-3-5-2b-q4k-ehaf16': {
    activationDtype: 'f16',
    mathDtype: 'f16',
    accumDtype: 'f16',
    outputDtype: 'f16',
  },
});
const EXPECTED_QWEN_PER_LAYER_INPUTS = Object.freeze({
  materialization: 'auto',
  rowCache: {
    mode: 'lru',
    maxRows: 256,
    maxBytes: 134217728,
    decodedDtype: 'f32',
  },
  prefetch: {
    mode: 'next_token',
    rowsAhead: 1,
  },
  gpuUpload: {
    mode: 'per_step_slices',
    stagingRows: 2,
  },
  hotCache: {
    mode: 'prepared_tokens',
    maxTokens: 1024,
    maxBytes: 134217728,
    outputDtype: 'f32',
  },
});
const EXPECTED_QWEN_EOS_TOKEN_ID = 248044;
const EXPECTED_QWEN_DECODE_LOOPS = Object.freeze({
  'qwen-3-5-0-8b-q4k-ehaf16': {
    batchSize: 32,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    readbackMode: 'sequential',
    submitLatencyThresholdMs: null,
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  },
  'qwen-3-5-2b-q4k-ehaf16': {
    batchSize: 8,
    stopCheckMode: 'batch',
    readbackInterval: 32,
    readbackMode: 'sequential',
    submitLatencyThresholdMs: null,
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  },
});

function getExpectedQwenDecodeLoop(label) {
  const modelLabel = String(label ?? '');
  const matchedKey = Object.keys(EXPECTED_QWEN_DECODE_LOOPS).find((key) => modelLabel.includes(key));
  assert.ok(matchedKey, `missing expected qwen decode loop for ${modelLabel}`);
  return EXPECTED_QWEN_DECODE_LOOPS[matchedKey];
}

function assertQwenDecodeLoop(decodeLoop, label) {
  assert.ok(decodeLoop && typeof decodeLoop === 'object', `${label} decodeLoop must be present`);
  assert.deepEqual(decodeLoop, getExpectedQwenDecodeLoop(label), `${label} decodeLoop`);
}

function assertQwenComputeDefaults(computeDefaults, label) {
  const modelLabel = String(label ?? '');
  const matchedKey = Object.keys(EXPECTED_QWEN_COMPUTE_DEFAULTS).find((key) => modelLabel.includes(key));
  assert.ok(matchedKey, `missing expected qwen compute defaults for ${modelLabel}`);
  assert.ok(computeDefaults && typeof computeDefaults === 'object', `${label} compute defaults must be present`);
  assert.deepEqual(computeDefaults, EXPECTED_QWEN_COMPUTE_DEFAULTS[matchedKey], `${label} compute defaults`);
}

function assertQwenPerLayerInputs(perLayerInputs, label) {
  assert.ok(perLayerInputs && typeof perLayerInputs === 'object', `${label} perLayerInputs must be present`);
  assert.deepEqual(perLayerInputs, EXPECTED_QWEN_PER_LAYER_INPUTS, `${label} perLayerInputs`);
}

function assertQwenConversionConfig(config) {
  assert.equal(config.inference.normalization.rmsNormWeightOffset, true);
  assert.equal(config.inference.attention.valueNorm, false);
  assert.equal(config.inference.rope.mropeInterleaved, true);
  assert.equal(config.inference.ffn.useDoubleWideMlp, false);
  assert.equal(config.inference.rope.ropeLocalPartialRotaryFactor, null);
  assert.equal(config.inference.rope.ropeFrequencyBaseDim, null);
  assert.equal(config.inference.rope.ropeLocalFrequencyBaseDim, null);
  assert.equal(config.inference.output.tieWordEmbeddings, true);
  assert.equal(config.inference.layerPattern.type, 'custom');
  assert.equal(config.inference.layerPattern.globalPattern, null);
  assert.equal(config.inference.layerPattern.period, null);
  assert.equal(config.inference.layerPattern.offset, null);
  assert.deepEqual(config.inference.layerPattern.layerTypes, EXPECTED_QWEN_LAYER_TYPES);
  assertQwenDecodeLoop(config.session?.decodeLoop, config.output?.modelBaseId ?? 'qwen');
  assertQwenComputeDefaults(
    config.session?.compute?.defaults,
    config.output?.modelBaseId ?? 'qwen'
  );
  assert.ok(config.execution && typeof config.execution === 'object', 'qwen execution config must be present');
  assert.equal(config.execution.inlineKernelPath, true, 'qwen execution.inlineKernelPath');
  assert.doesNotThrow(
    () => validateRequiredInferenceFields(config.inference, config.output?.modelBaseId ?? 'qwen'),
    `${config.output?.modelBaseId ?? 'qwen'} conversion config must satisfy required inference-field validation`
  );
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

function assertF16KvAttentionPrecision(execution, label) {
  const kernels = execution?.kernels ?? {};
  for (const [kernelKey, decl] of Object.entries(kernels)) {
    if (
      typeof decl?.kernel === 'string'
      && decl.kernel.startsWith('attention_')
      && decl.kernel.includes('f16kv')
    ) {
      assert.equal(
        decl.precision?.kvDtype,
        'f16',
        `${label} execution.kernels.${kernelKey} must declare precision.kvDtype="f16"`
      );
    }
  }
}

function assertQwenF16UtilityKernels(execution, label) {
  if (String(label).includes('qwen-3-5-0-8b-q4k-ehaf16')) {
    assert.equal(execution?.kernels?.rmsnorm?.kernel, 'rmsnorm.wgsl', `${label} rmsnorm kernel`);
    assert.equal(execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl', `${label} q4_decode kernel`);
    assert.equal(execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl', `${label} q4_decode_gemv kernel`);
    assert.equal(execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv', `${label} q4_decode_gemv entry`);
    assert.equal(execution?.kernels?.q4_decode_ffn_gemv?.kernel, 'matmul_gemv_subgroup.wgsl', `${label} q4_decode_ffn_gemv kernel`);
    assert.equal(execution?.kernels?.rope?.kernel, 'rope.wgsl', `${label} rope kernel`);
    assert.equal(execution?.kernels?.residual?.kernel, 'residual.wgsl', `${label} residual kernel`);
    assert.equal(execution?.kernels?.silu?.kernel, 'silu.wgsl', `${label} silu kernel`);
    assert.equal(execution?.kernels?.q4_prefill?.kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl', `${label} q4_prefill kernel`);
    assert.equal(execution?.kernels?.sample?.kernel, 'sample.wgsl', `${label} sample kernel`);
    return;
  }
  assert.equal(execution?.kernels?.rmsnorm?.kernel, 'rmsnorm_f16.wgsl', `${label} rmsnorm kernel`);
  assert.deepEqual(execution?.kernels?.q4_decode?.precision, {
    inputDtype: 'f16',
    outputDtype: 'f16',
  }, `${label} q4_decode precision`);
  assert.equal(execution?.kernels?.rope?.kernel, 'rope_f16.wgsl', `${label} rope kernel`);
  assert.equal(execution?.kernels?.residual?.kernel, 'residual_f16.wgsl', `${label} residual kernel`);
  assert.equal(execution?.kernels?.silu?.kernel, 'silu_f16.wgsl', `${label} silu kernel`);
  assert.deepEqual(execution?.kernels?.q4_prefill?.precision, {
    inputDtype: 'f16',
    outputDtype: 'f16',
  }, `${label} q4_prefill precision`);
  assert.equal(execution?.kernels?.sample?.kernel, 'sample_f16.wgsl', `${label} sample kernel`);
}

for (const config of convConfigs) {
  // V1 configs have explicit inference and no legacy family-indirection field
  assert.ok(config.inference?.attention, 'v1 config must have explicit inference.attention');
  assertQwenConversionConfig(config);
}

if (!hasLocalQ4KManifest) {
  console.log('qwen-manifest-completeness.test: ok (conversion-config contract only; local q4k manifest unavailable)');
  process.exit(0);
}

// --- Manifest: intentional null fields ---

{
  for (const manifest of availableLocalManifests) {
    assert.equal(manifest.inference.defaultKernelPath, undefined);
  }
}

{
  assert.equal(q4kManifest.inference.attention.valueNorm, false);
  assert.equal(q4kManifest.inference.ffn.useDoubleWideMlp, false);
  assert.equal(q4kManifest.inference.rope.ropeLocalPartialRotaryFactor, null);
  assert.equal(q4kManifest.inference.rope.ropeFrequencyBaseDim, null);
  assert.equal(q4kManifest.inference.rope.ropeLocalFrequencyBaseDim, null);
  assert.doesNotThrow(
    () => validateRequiredInferenceFields(q4kManifest.inference, q4kManifest.modelId),
    `${q4kManifest.modelId} manifest must satisfy required inference-field validation`
  );
}

// --- Manifest: execution v1 is present with inline kernel-path lowering enabled ---

{
  for (const manifest of availableLocalManifests) {
    assert.equal(manifest.inference.schema, 'doppler.execution/v1');
    assert.ok(manifest.inference.execution && typeof manifest.inference.execution === 'object');
    assert.equal(manifest.inference.execution.inlineKernelPath, true);
    assertF16KvAttentionPrecision(manifest.inference.execution, manifest.modelId);
    assertQwenF16UtilityKernels(manifest.inference.execution, manifest.modelId);
  }
}

// --- Manifest: session keep explicit compute/kvcache + Qwen decode loop ---

{
  for (const manifest of availableLocalManifests) {
    const sd = manifest.inference.session;
    assert.ok(sd != null);
    assertQwenDecodeLoop(sd.decodeLoop, `${manifest.modelId}.inference.session`);
    assert.equal(sd.kvcache?.kvDtype, 'f16');
    assertQwenComputeDefaults(sd.compute?.defaults, `${manifest.modelId}.inference.session`);
    assert.equal(sd.execution, undefined);
  }
  assertQwenPerLayerInputs(q4kManifest.inference.session?.perLayerInputs, `${q4kManifest.modelId}.inference.session`);
}

// --- Manifest: Qwen hybrid layer pattern ---

{
  for (const manifest of availableLocalManifests) {
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
  for (const manifest of availableLocalManifests) {
    const arch = manifest.architecture;
    assert.equal(arch.linearNumKeyHeads, 16);
    assert.equal(arch.linearNumValueHeads, 16);
    assert.equal(arch.linearKeyHeadDim, 128);
    assert.equal(arch.linearValueHeadDim, 128);
    assert.equal(arch.linearConvKernelDim, 4);
  }
}

// --- Manifest: Qwen EOS resolves to <|im_end|>, not pad/endoftext ---

{
  assert.equal(
    q4kManifest.eos_token_id,
    EXPECTED_QWEN_EOS_TOKEN_ID,
    `${q4kManifest.modelId} eos_token_id`
  );
}

// --- Qwen linear-attention layers use linear-attn norm weights, not self-attn q/k norm ---

{
  assert.equal(q4kManifest.architecture.linearNormMode, 'shared');
  if (f16Manifest != null) {
    assert.equal(f16Manifest.architecture.linearNormMode, undefined);
  }

  for (const manifest of availableLocalManifests) {
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
  for (const manifest of availableLocalManifests) {
    assert.equal(manifest.inference.normalization.rmsNormWeightOffset, true);
  }
}

// --- Manifest: mRoPE fields ---

{
  for (const manifest of availableLocalManifests) {
    const rope = manifest.inference.rope;
    assert.deepEqual(rope.mropeSection, [11, 11, 10]);
    assert.equal(rope.mropeInterleaved, true);
    assert.equal(rope.partialRotaryFactor, 0.25);

    const sectionSum = rope.mropeSection.reduce((a, b) => a + b, 0);
    const resolvedRotaryDim = manifest.architecture.headDim * rope.partialRotaryFactor;
    assert.equal(sectionSum * 2, resolvedRotaryDim);
  }
}

// --- Conversion configs: execution-v1 omits legacy defaultKernelPath ---

for (const config of convConfigs) {
  assert.equal(config.inference.defaultKernelPath, undefined);
}

// --- Conversion configs: Qwen keeps an explicit execution graph with fused Q4 decode/prefill as primary path ---

for (const config of convConfigs) {
  assert.ok(config.execution && typeof config.execution === 'object');
  assert.equal(config.execution.inlineKernelPath, true);
  if ((config.output?.modelBaseId ?? '') === 'qwen-3-5-0-8b-q4k-ehaf16') {
    assert.equal(config.execution.kernels.rmsnorm.kernel, 'rmsnorm.wgsl');
    assert.equal(config.execution.kernels.q4_decode.kernel, 'fused_matmul_q4.wgsl');
    assert.equal(config.execution.kernels.q4_decode.entry, 'main_multicol');
    assert.equal(config.execution.kernels.q4_decode_gemv.kernel, 'fused_matmul_q4.wgsl');
    assert.equal(config.execution.kernels.q4_decode_gemv.entry, 'main_gemv');
    assert.equal(config.execution.kernels.q4_decode_ffn_gemv.kernel, 'matmul_gemv_subgroup.wgsl');
    assert.equal(config.execution.kernels.rope.kernel, 'rope.wgsl');
    assert.equal(config.execution.kernels.residual.kernel, 'residual.wgsl');
    assert.equal(config.execution.kernels.silu.kernel, 'silu.wgsl');
    assert.equal(config.execution.kernels.tiled.kernel, 'matmul_f16w_f32a.wgsl');
    assert.equal(config.execution.kernels.q4_prefill.kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl');
    assert.equal(config.execution.kernels.attn_decode.kernel, 'attention_decode_online_f16kv.wgsl');
    assert.equal(config.execution.kernels.attn_stream.kernel, 'attention_streaming_f16kv.wgsl');
    assert.equal(config.execution.kernels.lm_head_gemv.kernel, 'matmul_gemv_subgroup_f16a.wgsl');
    assert.equal(config.execution.kernels.sample.kernel, 'sample.wgsl');
  } else if ((config.output?.modelBaseId ?? '') === 'qwen-3-5-2b-q4k-ehaf16') {
    assert.equal(config.execution.kernels.rmsnorm.kernel, 'rmsnorm_f16.wgsl');
    assert.equal(config.execution.kernels.q4_decode.kernel, 'fused_matmul_q4_multicol_f16a.wgsl');
    assert.equal(config.execution.kernels.rope.kernel, 'rope_f16.wgsl');
    assert.equal(config.execution.kernels.residual.kernel, 'residual_f16.wgsl');
    assert.equal(config.execution.kernels.silu.kernel, 'silu_f16.wgsl');
    assert.equal(config.execution.kernels.tiled.kernel, 'matmul_f16.wgsl');
    assert.equal(config.execution.kernels.q4_prefill.kernel, 'fused_matmul_q4_batched_f16a.wgsl');
    assert.equal(config.execution.kernels.attn_decode.kernel, 'attention_decode_online_f16.wgsl');
    assert.equal(config.execution.kernels.attn_stream.kernel, 'attention_small_f16.wgsl');
    assert.equal(config.execution.kernels.lm_head_gemv.kernel, 'matmul_gemv_subgroup_f16a.wgsl');
    assert.equal(config.execution.kernels.sample.kernel, 'sample_f16.wgsl');
  }
}

// --- Conversion configs: explicit qwen3 family config fields are authored directly ---

for (const config of convConfigs) {
  assertQwenConversionConfig(config);
}

// --- Conversion configs: top-level session keep the Qwen decode loop contract ---

for (const config of convConfigs) {
  assertQwenDecodeLoop(config.session?.decodeLoop, config.output?.modelBaseId ?? 'qwen');
  assertQwenPerLayerInputs(config.session?.perLayerInputs, config.output?.modelBaseId ?? 'qwen');
}

console.log('qwen-manifest-completeness.test: ok');
