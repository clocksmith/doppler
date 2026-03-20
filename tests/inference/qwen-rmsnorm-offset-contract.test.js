import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';

async function loadJson(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
}

// V1 configs must have explicit rmsNormWeightOffset
const conversionConfigs = await Promise.all([
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
  loadJson('../../src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json'),
]);

for (const config of conversionConfigs) {
  assert.equal(
    config.inference?.normalization?.rmsNormWeightOffset,
    true,
    `${config.output?.modelBaseId ?? 'unknown'} v1 config must keep offset RMSNorm weight semantics`
  );
}

const localManifestPaths = [
  '../../models/local/qwen-3-5-0-8b-q4k-ehaf16-af32/manifest.json',
  '../../models/local/qwen-3-5-0-8b-wf16-ef16-hf16-f16/manifest.json',
  '../../models/local/qwen-3-5-0-8b-wq4k-ef16-hf16-f16/manifest.json',
  '../../models/local/qwen-3-5-2b-wq4k-ef16-hf16-f16/manifest.json',
];

const existingManifestPaths = localManifestPaths.filter((relativePath) => (
  existsSync(new URL(relativePath, import.meta.url))
));

for (const manifestPath of existingManifestPaths) {
  const manifest = await loadJson(manifestPath);
  assert.equal(
    manifest.inference?.normalization?.rmsNormWeightOffset,
    true,
    `${manifest.modelId} must use offset RMSNorm weight semantics`
  );
}

console.log('qwen-rmsnorm-offset-contract.test: ok');
