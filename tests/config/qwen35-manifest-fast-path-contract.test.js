import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KNOWN_MODELS } from '../../src/models/qwen3.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function assertDecodeLoop(actual, expected, label) {
  assert.deepEqual(
    actual,
    expected,
    `${label}: decode loop must be manifest/config owned`
  );
}

function assertAttentionF16KvPrecision(execution, label) {
  for (const kernelKey of ['attn_decode', 'attn_stream']) {
    assert.deepEqual(
      execution?.kernels?.[kernelKey]?.precision,
      { kvDtype: 'f16' },
      `${label}: ${kernelKey} must explicitly declare f16 KV precision`
    );
  }
}

function assertQwen35SmallFastGraph(execution, label) {
  assert.equal(execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv');
  assert.equal(execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl');
  assert.equal(execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
  assert.equal(execution?.kernels?.lm_head_gemv?.kernel, 'matmul_gemv_subgroup_f16a.wgsl');
  assert.equal(execution?.kernels?.lm_head_q4?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(execution?.kernels?.lm_head_q4?.entry, 'main_gemv');
  assert.deepEqual(execution?.kernels?.lm_head_q4?.constants, {
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
  });

  const decodeProjection = execution?.decode?.find((step) => step[0] === 'gate_proj');
  assert.equal(decodeProjection?.[1], 'q4_decode_gemv', `${label}: decode FFN must use fused-Q4 GEMV`);

  const decodeHead = execution?.postLayer?.find((step) => step[0] === 'lm_head');
  assert.equal(decodeHead?.[1], 'lm_head_q4', `${label}: decode LM head must use quantized tied-head Q4`);

  const prefillProjection = execution?.prefill?.find((step) => step[0] === 'gate_proj');
  assert.equal(prefillProjection?.[1], 'q4_widetile', `${label}: prefill FFN must use WideTile Q4`);

  const prefillAttention = execution?.prefill?.find((step) => step[0] === 'attention');
  assert.equal(prefillAttention?.[1], 'attn_head256', `${label}: prefill attention must use head256`);
}

function assertQwen35TwoBTunedGraph(execution, label) {
  assert.equal(execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(execution?.kernels?.q4_decode?.entry, 'main_multicol');
  assert.equal(execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv');
  assert.equal(execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl');
  assert.equal(execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
  assert.equal(execution?.kernels?.lm_head_gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
  assert.equal(execution?.kernels?.lm_head_q4?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(execution?.kernels?.lm_head_q4?.entry, 'main_gemv');
  assert.deepEqual(execution?.kernels?.lm_head_q4?.constants, {
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
  });

  const decodeProjection = execution?.decode?.find((step) => step[0] === 'gate_proj');
  assert.equal(decodeProjection?.[1], 'q4_decode_gemv', `${label}: decode FFN must use fixed fused-Q4 GEMV`);

  const decodeHead = execution?.postLayer?.find((step) => step[0] === 'lm_head');
  assert.equal(decodeHead?.[1], 'lm_head_q4', `${label}: decode LM head must use quantized tied-head Q4`);

  const prefillProjection = execution?.prefill?.find((step) => step[0] === 'gate_proj');
  assert.equal(prefillProjection?.[1], 'q4_widetile', `${label}: prefill FFN must use WideTile Q4`);

  const prefillAttention = execution?.prefill?.find((step) => step[0] === 'attention');
  assert.equal(prefillAttention?.[1], 'attn_head256', `${label}: prefill attention must use head256`);
}

const qwen08Config = await readJson('src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json');
const qwen2Config = await readJson('src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json');
const qwen08Manifest = await readJson('models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json');
const qwen2Manifest = await readJson('models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json');

const modelProfiles = new Map(KNOWN_MODELS.map((entry) => [entry.modelId, entry.defaultRuntimeProfile]));
assert.equal(modelProfiles.get('qwen-3-5-0-8b-q4k-ehaf16'), 'profiles/throughput');
assert.equal(modelProfiles.get('qwen-3-5-2b-q4k-ehaf16'), 'profiles/throughput');

const qwen08ConfigDecodeLoop = Object.freeze({
  batchSize: 4,
  stopCheckMode: 'batch',
  readbackInterval: 8,
  readbackMode: 'sequential',
  submitLatencyThresholdMs: null,
  ringTokens: 1,
  ringStop: 1,
  ringStaging: 1,
  disableCommandBatching: false,
});
const qwen08ManifestDecodeLoop = Object.freeze({
  batchSize: 4,
  stopCheckMode: 'batch',
  readbackInterval: 32,
  readbackMode: 'sequential',
  submitLatencyThresholdMs: null,
  ringTokens: 1,
  ringStop: 1,
  ringStaging: 1,
  disableCommandBatching: false,
});
const qwen2DecodeLoop = Object.freeze({
  batchSize: 12,
  stopCheckMode: 'batch',
  readbackInterval: 32,
  readbackMode: 'sequential',
  submitLatencyThresholdMs: null,
  ringTokens: 1,
  ringStop: 1,
  ringStaging: 1,
  disableCommandBatching: false,
});

assertDecodeLoop(qwen08Config.session?.decodeLoop, qwen08ConfigDecodeLoop, 'Qwen 3.5 0.8B conversion config');
assertDecodeLoop(qwen08Manifest.inference?.session?.decodeLoop, qwen08ManifestDecodeLoop, 'Qwen 3.5 0.8B local manifest');
assertDecodeLoop(qwen2Config.session?.decodeLoop, qwen2DecodeLoop, 'Qwen 3.5 2B conversion config');
assertDecodeLoop(qwen2Manifest.inference?.session?.decodeLoop, qwen2DecodeLoop, 'Qwen 3.5 2B local manifest');

assert.equal(qwen08Config.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(qwen08Manifest.inference?.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(qwen2Config.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(qwen2Manifest.inference?.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(qwen08Config.quantization?.lmHead, 'q4k');
assert.equal(qwen08Manifest.quantizationInfo?.lmHead, 'q4k');
assert.equal(qwen2Config.quantization?.lmHead, 'q4k');
assert.equal(qwen2Manifest.quantizationInfo?.lmHead, 'q4k');

assertAttentionF16KvPrecision(qwen08Config.execution, 'Qwen 3.5 0.8B conversion config');
assertAttentionF16KvPrecision(qwen08Manifest.inference?.execution, 'Qwen 3.5 0.8B local manifest');
assertAttentionF16KvPrecision(qwen2Config.execution, 'Qwen 3.5 2B conversion config');
assertAttentionF16KvPrecision(qwen2Manifest.inference?.execution, 'Qwen 3.5 2B local manifest');
assertQwen35SmallFastGraph(qwen08Config.execution, 'Qwen 3.5 0.8B conversion config');
assertQwen35SmallFastGraph(qwen08Manifest.inference?.execution, 'Qwen 3.5 0.8B local manifest');
assertQwen35TwoBTunedGraph(qwen2Config.execution, 'Qwen 3.5 2B conversion config');
assertQwen35TwoBTunedGraph(qwen2Manifest.inference?.execution, 'Qwen 3.5 2B local manifest');

console.log('qwen35-manifest-fast-path-contract.test: ok');
