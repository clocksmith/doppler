import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {
  inferConversionConfigModelId,
  resolveMaterializedManifestFromConversionConfig,
} from '../../src/tooling/conversion-config-materializer.js';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const gemmaManifestPath = path.join('models/local/gemma-3-270m-it-q4k-ehf16-af32', 'manifest.json');
if (!fs.existsSync(gemmaManifestPath)) {
  console.log('conversion-config-materializer.test: skipped gemma fixture (local model missing)');
} else {
  const conversionConfig = readJson('src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json');
  const manifest = readJson(gemmaManifestPath);

  assert.equal(
    inferConversionConfigModelId(
      'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json',
      conversionConfig
    ),
    'gemma-3-270m-it-q4k-ehf16-af32'
  );

  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);
  assert.equal(materialized.modelId, manifest.modelId);
  assert.equal(materialized.modelType, 'transformer');
  assert.equal(materialized.inference?.schema, 'doppler.execution/v1');
  assert.equal(materialized.inference?.defaultKernelPath, undefined);
  assert.equal(materialized.inference?.attention?.slidingWindow, 512);
  assert.equal(materialized.inference?.rope?.ropeLocalTheta, 10000);
  assert.equal(materialized.inference?.layerPattern?.type, 'every_n');
  assert.equal(materialized.inference?.layerPattern?.period, 6);
  assert.deepEqual(materialized.inference?.session?.decodeLoop, {
    batchSize: 1,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    readbackMode: 'sequential',
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  });
  assert.ok(Array.isArray(materialized.inference?.execution?.decode));
  assert.ok(materialized.inference.execution.decode.length > 0);
}

const qwenManifestPath = path.join('models/local/qwen-3-5-0-8b-q4k-ehaf16', 'manifest.json');
if (!fs.existsSync(qwenManifestPath)) {
  console.log('conversion-config-materializer.test: skipped qwen fixture (local model missing)');
} else {
  const conversionConfig = readJson('src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json');
  const manifest = readJson(qwenManifestPath);
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);

  assert.equal(
    inferConversionConfigModelId(
      'src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json',
      conversionConfig
    ),
    'qwen-3-5-0-8b-q4k-ehaf16'
  );
  assert.equal(materialized.modelId, manifest.modelId);
  assert.equal(materialized.modelType, 'transformer');
  assert.equal(materialized.inference?.schema, 'doppler.execution/v1');
  assert.equal(materialized.inference?.execution?.inlineKernelPath, true);
  // Qwen 3.5 0.8B decodeLoop values are manifest-owned: bs=4/rbi=32
  // with sequential readback. Runtime profiles must not be the source of truth.
  assert.deepEqual(materialized.inference?.session?.decodeLoop, {
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
  assert.equal(materialized.inference?.session?.compute?.defaults?.activationDtype, 'f32');
  assert.equal(materialized.inference?.execution?.kernels?.rmsnorm?.kernel, 'rmsnorm.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode?.entry, 'main_multicol');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv');
  assert.equal(materialized.inference?.execution?.kernels?.rope?.kernel, 'rope.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.residual?.kernel, 'residual.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.silu?.kernel, 'silu.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.tiled?.kernel, 'matmul_f16w_f32a.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_prefill?.kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.attn_decode?.kernel, 'attention_decode_online_f16kv.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.attn_stream?.kernel, 'attention_streaming_f16kv.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_gemv?.kernel, 'matmul_gemv_subgroup_f16a.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_q4?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_q4?.entry, 'main_gemv');
  assert.deepEqual(materialized.inference?.execution?.kernels?.lm_head_q4?.constants, {
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
  });
  assert.equal(materialized.inference?.execution?.kernels?.sample?.kernel, 'sample.wgsl');
  const decodeHead = materialized.inference?.execution?.postLayer?.find((step) => step[0] === 'lm_head');
  assert.equal(decodeHead?.[1], 'lm_head_q4');
  const decodeGateProj = materialized.inference?.execution?.decode?.find((step) => step[0] === 'gate_proj');
  assert.equal(decodeGateProj?.[1], 'q4_decode_gemv');
  const prefillGateProj = materialized.inference?.execution?.prefill?.find((step) => step[0] === 'gate_proj');
  assert.equal(prefillGateProj?.[1], 'q4_widetile');
}

const qwen2ManifestPath = path.join('models/local/qwen-3-5-2b-q4k-ehaf16', 'manifest.json');
if (!fs.existsSync(qwen2ManifestPath)) {
  console.log('conversion-config-materializer.test: skipped qwen 2b fixture (local model missing)');
} else {
  const conversionConfig = readJson('src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json');
  const manifest = readJson(qwen2ManifestPath);
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);

  assert.equal(materialized.modelId, manifest.modelId);
  assert.equal(materialized.modelType, 'transformer');
  assert.equal(materialized.inference?.schema, 'doppler.execution/v1');
  assert.equal(materialized.inference?.execution?.inlineKernelPath, true);
  assert.deepEqual(materialized.inference?.session?.decodeLoop, {
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
  assert.equal(materialized.inference?.session?.compute?.defaults?.activationDtype, 'f32');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode?.entry, 'main_multicol');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode_gemv?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode_gemv?.entry, 'main_gemv');
  assert.equal(materialized.inference?.execution?.kernels?.q4_widetile?.kernel, 'fused_matmul_q4_widetile.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_q4?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.lm_head_q4?.entry, 'main_gemv');
  assert.deepEqual(materialized.inference?.execution?.kernels?.lm_head_q4?.constants, {
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
  });
  const decodeHead = materialized.inference?.execution?.postLayer?.find((step) => step[0] === 'lm_head');
  assert.equal(decodeHead?.[1], 'lm_head_q4');
  const decodeGateProj = materialized.inference?.execution?.decode?.find((step) => step[0] === 'gate_proj');
  assert.equal(decodeGateProj?.[1], 'q4_decode_gemv');
  const prefillGateProj = materialized.inference?.execution?.prefill?.find((step) => step[0] === 'gate_proj');
  assert.equal(prefillGateProj?.[1], 'q4_widetile');
}

const gemma1bManifestPath = path.join('models/local/gemma-3-1b-it-q4k-ehf16-af32', 'manifest.json');
if (!fs.existsSync(gemma1bManifestPath)) {
  console.log('conversion-config-materializer.test: skipped gemma 1b fixture (local model missing)');
} else {
  const conversionConfig = readJson('src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json');
  const manifest = readJson(gemma1bManifestPath);
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);

  assert.equal(materialized.modelId, manifest.modelId);
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode, undefined);
  assert.equal(materialized.inference?.attention?.slidingWindow, 512);
  assert.equal(materialized.inference?.rope?.ropeLocalTheta, 10000);
  assert.equal(materialized.inference?.layerPattern?.type, 'every_n');
  assert.equal(materialized.inference?.layerPattern?.period, 6);
  assert.equal(materialized.inference?.execution?.kernels?.gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_prefill, undefined);
  assert.equal(materialized.inference?.execution?.kernels?.tiled?.kernel, 'matmul_f16w_f32a_tiled.wgsl');
  const qProjDecode = materialized.inference?.execution?.decode?.find((step) => step[0] === 'q_proj');
  assert.ok(qProjDecode, 'gemma 1b materialized decode q_proj step missing');
  assert.equal(qProjDecode[1], 'gemv');
  const qProjPrefill = materialized.inference?.execution?.prefill?.find((step) => step[0] === 'q_proj');
  assert.ok(qProjPrefill, 'gemma 1b materialized prefill q_proj step missing');
  assert.equal(qProjPrefill[1], 'tiled');
  const attnPrefill = materialized.inference?.execution?.prefill?.find((step) => step[0] === 'attention');
  assert.ok(attnPrefill, 'gemma 1b materialized prefill attention step missing');
  assert.equal(attnPrefill[1], 'attn_head256');
}

const embeddingManifestPath = path.join('models/local/google-embeddinggemma-300m-q4k-ehf16-af32', 'manifest.json');
if (!fs.existsSync(embeddingManifestPath)) {
  console.log('conversion-config-materializer.test: skipped embedding fixture (local model missing)');
} else {
  const conversionConfig = readJson('src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json');
  const manifest = readJson(embeddingManifestPath);
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);

  assert.equal(
    inferConversionConfigModelId(
      'src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json',
      conversionConfig
    ),
    'google-embeddinggemma-300m-q4k-ehf16-af32'
  );
  assert.equal(materialized.modelId, manifest.modelId);
  assert.equal(materialized.modelType, 'embedding');
  assert.equal(materialized.inference?.schema, 'doppler.execution/v1');
  assert.deepEqual(materialized.inference?.session?.compute?.defaults, {
    activationDtype: 'f32',
    mathDtype: 'f32',
    accumDtype: 'f32',
    outputDtype: 'f32',
  });
  assert.deepEqual(materialized.inference?.session?.kvcache, {
    kvDtype: 'f32',
    layout: 'contiguous',
    pageSize: 256,
    tiering: { mode: 'off' },
  });
  assert.deepEqual(materialized.inference?.session?.decodeLoop, {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    readbackMode: 'sequential',
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  });
  assert.deepEqual(materialized.inference?.output?.embeddingPostprocessor, {
    poolingMode: 'mean',
    includePrompt: true,
    projections: [
      {
        weightTensor: 'embedding_postprocessor.projections.0.weight',
        biasTensor: null,
        inputSize: 768,
        outputSize: 3072,
        activation: 'identity',
      },
      {
        weightTensor: 'embedding_postprocessor.projections.1.weight',
        biasTensor: null,
        inputSize: 3072,
        outputSize: 768,
        activation: 'identity',
      },
    ],
    normalize: 'l2',
  });
}

console.log('conversion-config-materializer.test: ok');
