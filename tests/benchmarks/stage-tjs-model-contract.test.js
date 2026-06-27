import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  buildDownloadArgs,
  buildTextGenerationRequiredSnapshotFiles,
  loadCatalogTransformersjsRepoDtypes,
  resolveCatalogDefaultDtype,
  resolveRequestedDtype,
  validateStagedSnapshot,
} from '../../tools/stage-tjs-model.js';

{
  const repoDtypeMap = loadCatalogTransformersjsRepoDtypes();
  assert.equal(resolveCatalogDefaultDtype('onnx-community/gemma-4-E2B-it-ONNX', repoDtypeMap), 'q4f16');
  assert.equal(resolveRequestedDtype(null, 'onnx-community/gemma-4-E2B-it-ONNX', repoDtypeMap), 'q4f16');
  assert.equal(resolveRequestedDtype('fp16', 'onnx-community/gemma-4-E2B-it-ONNX', repoDtypeMap), 'fp16');
}

{
  assert.deepEqual(
    buildDownloadArgs('hf', 'org/model', '/tmp/model', 'main', true, [
      'config.json',
      'onnx/model*q4f16*',
    ]),
    [
      'download',
      'org/model',
      '--repo-type',
      'model',
      '--local-dir',
      '/tmp/model',
      '--revision',
      'main',
      '--force-download',
      '--include',
      'config.json',
      '--include',
      'onnx/model*q4f16*',
    ]
  );
}

{
  const required = buildTextGenerationRequiredSnapshotFiles('q4f16');
  assert.deepEqual(required.commonFiles, [
    'config.json',
    'tokenizer_config.json',
  ]);
  assert.deepEqual(required.requiredFiles, [
    'config.json',
    'tokenizer_config.json',
    'onnx/embed_tokens_q4f16.onnx',
    'onnx/decoder_model_merged_q4f16.onnx',
  ]);
  assert.deepEqual(required.requiredOneOf, [
    ['tokenizer.json', 'tokenizer.model'],
  ]);
  assert.deepEqual(required.requiredDataPrefixes, [
    'onnx/embed_tokens_q4f16.onnx_data',
    'onnx/decoder_model_merged_q4f16.onnx_data',
  ]);
  assert.deepEqual(required.requiredLayouts.map((layout) => layout.id), [
    'split-decoder-embed',
    'monolithic-model',
  ]);
  assert.deepEqual(required.requiredLayouts[0].requiredFiles, [
    'onnx/embed_tokens_q4f16.onnx',
    'onnx/decoder_model_merged_q4f16.onnx',
  ]);
  assert.deepEqual(required.requiredLayouts[0].requiredDataPrefixes, [
    'onnx/embed_tokens_q4f16.onnx_data',
    'onnx/decoder_model_merged_q4f16.onnx_data',
  ]);
  assert.deepEqual(required.requiredLayouts[1].requiredFiles, [
    'onnx/model_q4f16.onnx',
  ]);
  assert.deepEqual(required.requiredLayouts[1].requiredDataPrefixes, [
    'onnx/model_q4f16.onnx_data',
  ]);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'stage-tjs-model-complete-'));
  await fs.mkdir(path.join(tempDir, 'onnx'), { recursive: true });
  await fs.writeFile(path.join(tempDir, 'config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer_config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'embed_tokens_q4f16.onnx'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'embed_tokens_q4f16.onnx_data'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'decoder_model_merged_q4f16.onnx'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'decoder_model_merged_q4f16.onnx_data_1'), '', 'utf8');
  validateStagedSnapshot(tempDir, 'text-generation', 'q4f16');
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'stage-tjs-model-monolithic-'));
  await fs.mkdir(path.join(tempDir, 'onnx'), { recursive: true });
  await fs.writeFile(path.join(tempDir, 'config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer_config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer.model'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'model_q4f16.onnx'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'model_q4f16.onnx_data'), '', 'utf8');
  validateStagedSnapshot(tempDir, 'text-generation', 'q4f16');
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'stage-tjs-model-missing-'));
  await fs.mkdir(path.join(tempDir, 'onnx'), { recursive: true });
  await fs.writeFile(path.join(tempDir, 'config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer_config.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'tokenizer.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'embed_tokens_q4f16.onnx'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'embed_tokens_q4f16.onnx_data'), '', 'utf8');
  await fs.writeFile(path.join(tempDir, 'onnx', 'decoder_model_merged_q4f16.onnx'), '', 'utf8');
  assert.throws(
    () => validateStagedSnapshot(tempDir, 'text-generation', 'q4f16'),
    /one complete ONNX layout/
  );
}

console.log('stage-tjs-model-contract.test: ok');
