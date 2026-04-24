import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';

const MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af32';
const CONFIG_PATH = `src/config/conversion/gemma4/${MODEL_ID}.json`;
const MANIFEST_PATH = `models/local/${MODEL_ID}/manifest.json`;

const ATTENTION_PROJECTION_OPS = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj']);
const LARGE_WEIGHT_OVERRIDES = ['model.language_model.embed_tokens.weight'];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function hashJson(value) {
  return `sha256:${crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex')}`;
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

function assertQ4FusedKernelRefs(execution, label) {
  assert.equal(execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl', `${label}: q4 decode kernel`);
  assert.equal(execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv', `${label}: q4 decode entry`);
  assert.equal(execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl', `${label}: q4 prefill kernel`);
  assert.equal(execution?.kernels?.q4_widetile?.entry, 'main', `${label}: q4 prefill entry`);
}

function assertAttentionProjectionPath(execution, phase, expectedKernelRef, label) {
  const groups = phaseStepGroups(execution, phase);
  assert.ok(groups.length > 0, `${label}: ${phase} steps must exist`);
  for (const steps of groups) {
    for (const step of steps) {
      if (!ATTENTION_PROJECTION_OPS.has(step[0])) {
        continue;
      }
      assert.equal(step[1], expectedKernelRef, `${label}: ${phase} ${step[0]} must use ${expectedKernelRef}`);
    }
  }
}

function assertGemma31BExecutionGraph(execution, label) {
  assertQ4FusedKernelRefs(execution, label);
  assertAttentionProjectionPath(execution, 'decode', 'q4_decode_gemv', label);
  assertAttentionProjectionPath(execution, 'prefill', 'q4_widetile', label);
}

const conversionConfig = readJson(CONFIG_PATH);

assertLargeWeights(conversionConfig.largeWeights, 'conversion config');
assertGemma31BExecutionGraph(conversionConfig.execution, 'conversion config');
assert.equal(
  conversionConfig.manifest?.artifactIdentity?.manifestVariantId,
  `${MODEL_ID}-mv-q4-fused-v1`,
  'conversion config: manifest variant must identify the fused-Q4 graph'
);

if (fs.existsSync(MANIFEST_PATH)) {
  const manifest = readJson(MANIFEST_PATH);
  assertLargeWeights(manifest.inference?.largeWeights, 'local manifest');
  assertGemma31BExecutionGraph(manifest.inference?.execution, 'local manifest');
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
