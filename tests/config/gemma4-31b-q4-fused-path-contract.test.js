import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';

const MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af32';
const CONFIG_PATH = `src/config/conversion/gemma4/${MODEL_ID}.json`;
const MANIFEST_PATH = `models/local/${MODEL_ID}/manifest.json`;

const PROJECTION_OPS = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']);
const LARGE_WEIGHT_OVERRIDES = ['model.language_model.embed_tokens.weight'];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function stripUndefined(value) {
  if (Array.isArray(value)) {
    return value.map(stripUndefined);
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const output = {};
  for (const key of Object.keys(value).sort()) {
    if (value[key] !== undefined) {
      output[key] = stripUndefined(value[key]);
    }
  }
  return output;
}

function hashJson(value) {
  return `sha256:${crypto.createHash('sha256').update(JSON.stringify(stripUndefined(value))).digest('hex')}`;
}

function phaseStepGroups(execution, phase) {
  const phaseValue = execution?.[phase];
  if (!Array.isArray(phaseValue)) {
    return [];
  }
  if (phase === 'prefill' && phaseValue.every((entry) => !Array.isArray(entry) && Array.isArray(entry?.steps))) {
    return phaseValue.map((entry) => entry.steps);
  }
  return [phaseValue];
}

function assertLargeWeights(largeWeights, label) {
  assert.deepEqual(
    largeWeights?.gpuResidentOverrides,
    LARGE_WEIGHT_OVERRIDES,
    `${label}: embed_tokens must stay GPU-resident`
  );
}

function assertGemma4AttentionContract(attention, label) {
  assert.equal(attention?.queryPreAttnScalar, 1, `${label}: Gemma 4 31B attention must use unit logit scale`);
}

function assertGemma4ArchitectureContract(architecture, label) {
  assert.equal(architecture?.numGlobalKeyValueHeads, 4, `${label}: full-attention layers must use 4 global KV heads`);
  assert.equal(architecture?.globalHeadDim, 512, `${label}: full-attention layers must use head_dim=512`);
}

function assertGemma4RoPEContract(rope, label) {
  assert.equal(rope?.partialRotaryFactor, 0.25, `${label}: full-attention p-RoPE factor`);
  assert.equal(rope?.ropeLocalPartialRotaryFactor, null, `${label}: sliding attention must use full RoPE`);
  assert.equal(rope?.ropeFrequencyBaseDim, 512, `${label}: full-attention proportional RoPE base dim`);
  assert.equal(rope?.ropeLocalFrequencyBaseDim, null, `${label}: sliding attention must use standard RoPE frequencies`);
}

function assertQ4FusedKernelRefs(execution, label) {
  assert.equal(execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl', `${label}: q4 decode kernel`);
  assert.equal(execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv', `${label}: q4 decode entry`);
  assert.deepEqual(
    execution?.kernels?.q4_decode_gemv?.precision,
    { inputDtype: 'f32', outputDtype: 'f32' },
    `${label}: q4 decode precision`
  );
  assert.equal(execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl', `${label}: q4 prefill kernel`);
  assert.equal(execution?.kernels?.q4_widetile?.entry, 'main', `${label}: q4 prefill entry`);
  assert.deepEqual(
    execution?.kernels?.q4_widetile?.precision,
    { inputDtype: 'f32', outputDtype: 'f32' },
    `${label}: q4 prefill precision`
  );
}

function assertProjectionPath(execution, phase, expectedKernelRef, label) {
  const groups = phaseStepGroups(execution, phase);
  assert.ok(groups.length > 0, `${label}: ${phase} steps must exist`);
  for (const steps of groups) {
    for (const step of steps) {
      if (!PROJECTION_OPS.has(step[0])) {
        continue;
      }
      assert.equal(step[1], expectedKernelRef, `${label}: ${phase} ${step[0]} must use ${expectedKernelRef}`);
    }
  }
}

function assertGemma31BExecutionGraph(execution, label) {
  assertQ4FusedKernelRefs(execution, label);
  assertProjectionPath(execution, 'decode', 'q4_decode_gemv', label);
  assertProjectionPath(execution, 'prefill', 'q4_widetile', label);
}

function assertGemma31BLayerScalars(manifest) {
  for (let layerIdx = 0; layerIdx < 60; layerIdx++) {
    assert.ok(
      manifest.tensors?.[`model.language_model.layers.${layerIdx}.layer_scalar`],
      `local manifest: layer ${layerIdx} layer_scalar tensor must be present`
    );
  }
}

const conversionConfig = readJson(CONFIG_PATH);

assertLargeWeights(conversionConfig.largeWeights, 'conversion config');
assertGemma4AttentionContract(conversionConfig.inference?.attention, 'conversion config');
assertGemma4RoPEContract(conversionConfig.inference?.rope, 'conversion config');
assertGemma31BExecutionGraph(conversionConfig.execution, 'conversion config');
assert.equal(
  conversionConfig.manifest?.artifactIdentity?.manifestVariantId,
  `${MODEL_ID}-mv-q4-fused-v1`,
  'conversion config: manifest variant must identify the fused-Q4 graph'
);

if (fs.existsSync(MANIFEST_PATH)) {
  const manifest = readJson(MANIFEST_PATH);
  assertGemma4ArchitectureContract(manifest.architecture, 'local manifest');
  assertLargeWeights(manifest.inference?.largeWeights, 'local manifest');
  assertGemma4AttentionContract(manifest.inference?.attention, 'local manifest');
  assertGemma4RoPEContract(manifest.inference?.rope, 'local manifest');
  assertGemma31BExecutionGraph(manifest.inference?.execution, 'local manifest');
  assertGemma31BLayerScalars(manifest);
  assert.equal(
    manifest.artifactIdentity?.manifestVariantId,
    conversionConfig.manifest?.artifactIdentity?.manifestVariantId,
    'local manifest: manifest variant must mirror conversion config'
  );
  assert.equal(
    manifest.artifactIdentity?.conversionConfigDigest,
    hashJson(conversionConfig),
    'local manifest: conversionConfigDigest must mirror conversion config'
  );
} else {
  console.log(`gemma4-31b-q4-fused-path-contract.test: skipped local manifest (${MANIFEST_PATH} missing)`);
}

console.log('gemma4-31b-q4-fused-path-contract.test: ok');
