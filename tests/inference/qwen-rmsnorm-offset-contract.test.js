// RMSNorm weight semantics for the Qwen family. Hybrid linear-attention
// checkpoints (Qwen 3.5 and later) store RMSNorm weights as offsets from one
// and need (1 + w) scaling; dense Qwen3 checkpoints (base, embedding,
// reranker) store w directly. Every Qwen conversion config must state the flag
// explicitly, agree with its layer structure, and converted local manifests
// must carry the config's value.
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
let manifestsChecked = 0;
for (const { name, config } of configs) {
  const offset = config.inference?.normalization?.rmsNormWeightOffset;
  assert.equal(typeof offset, 'boolean', `${name}: rmsNormWeightOffset must be an explicit boolean`);

  const layerTypes = config.inference?.layerPattern?.layerTypes ?? [];
  const isHybrid = Array.isArray(layerTypes) && layerTypes.includes('linear_attention');
  assert.equal(
    offset,
    isHybrid,
    `${name}: ${isHybrid ? 'hybrid linear-attention' : 'dense'} Qwen checkpoints use ${isHybrid ? 'offset' : 'direct'} RMSNorm weights`
  );
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

console.log(`qwen-rmsnorm-offset-contract.test: ok (${configs.length} configs, ${manifestsChecked} local manifests)`);
