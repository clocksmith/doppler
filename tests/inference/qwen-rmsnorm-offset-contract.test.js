import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';

const { resolvePreset } = await import('../../src/config/loader.js');

async function loadJson(path) {
  return JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
}

const qwen35Preset = resolvePreset('qwen3_5');
assert.equal(
  qwen35Preset.inference?.normalization?.rmsNormWeightOffset,
  true,
  'qwen3_5 preset must keep offset RMSNorm weight semantics'
);

const conversionConfigs = await Promise.all([
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json'),
  loadJson('../../tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json'),
]);

for (const config of conversionConfigs) {
  assert.equal(config.presets.model, 'qwen3_5');
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
