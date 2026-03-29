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
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
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
  assert.deepEqual(materialized.inference?.session?.decodeLoop, {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 32,
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  });
  assert.equal(materialized.inference?.execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.q4_prefill?.kernel, 'fused_matmul_q4_batched.wgsl');
  assert.equal(materialized.inference?.execution?.kernels?.attn_stream?.kernel, 'attention_streaming_f16kv.wgsl');
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
  assert.equal(attnPrefill[1], 'attn_small');
}

console.log('conversion-config-materializer.test: ok');
