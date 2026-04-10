import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
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
  const required = buildTextGenerationRequiredSnapshotFiles('q4f16');
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
    /decoder_model_merged_q4f16\.onnx_data\*/
  );
}

console.log('stage-tjs-model-contract.test: ok');
