// RMSNorm weight semantics for the Qwen family. Hybrid linear-attention
// HF checkpoints (Qwen 3.5 and later) store offsets and need (1 + w).
// Native GGUF weights already fold that addition; source-preserving conversion
// must use them directly. Dense Qwen3 checkpoints also store direct weights.
// Configs must declare the source convention explicitly and stamp it unchanged.
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

const CONFIG_DIR = 'src/config/conversion/qwen3';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

const configs = readdirSync(CONFIG_DIR)
  .filter((name) => name.endsWith('.json'))
  .map((name) => ({ name, config: readJson(path.join(CONFIG_DIR, name)) }));
assert.ok(configs.length > 0, `${CONFIG_DIR} must contain conversion configs`);

let hybrid = 0;
let dense = 0;
let nativeGguf = 0;
let manifestsChecked = 0;
for (const { name, config } of configs) {
  const offset = config.inference?.normalization?.rmsNormWeightOffset;
  assert.equal(typeof offset, 'boolean', `${name}: rmsNormWeightOffset must be an explicit boolean`);

  const layerTypes = config.inference?.layerPattern?.layerTypes ?? [];
  const isHybrid = Array.isArray(layerTypes) && layerTypes.includes('linear_attention');
  const preservesNativeGguf = config.quantization?.preserveSourceTensors === true
    && config.quantization?.sourceQuantizationFormat === 'gguf';
  const expectsOffset = isHybrid && !preservesNativeGguf;
  assert.equal(
    offset,
    expectsOffset,
    `${name}: ${preservesNativeGguf ? 'native GGUF' : isHybrid ? 'hybrid HF' : 'dense'} Qwen source uses ${expectsOffset ? 'offset' : 'direct'} RMSNorm weights`
  );
  if (preservesNativeGguf) nativeGguf += 1;
  if (isHybrid) hybrid += 1;
  else dense += 1;

  const modelId = config.output?.modelBaseId;
  const manifestPath = modelId ? path.join('models/local', modelId, 'manifest.json') : null;
  if (manifestPath && existsSync(manifestPath)) {
    const manifest = readJson(manifestPath);
    assert.equal(
      manifest.inference?.normalization?.rmsNormWeightOffset,
      offset,
      `${modelId}: converted manifest must carry the conversion config's RMSNorm weight semantics`
    );
    manifestsChecked += 1;
  }
}
assert.ok(hybrid > 0 && dense > 0, 'both Qwen RMSNorm conventions must be represented');

console.log(`qwen-rmsnorm-offset-contract.test: ok (${configs.length} configs, ${nativeGguf} native GGUF, ${manifestsChecked} local manifests)`);
