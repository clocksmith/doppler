import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createModelIR } from '../../src/config/model-ir.js';
import { createTargetPlan } from '../../src/config/target-plan.js';
import { buildPackV2, hashPackV2, loadPackV2, validatePackV2, writePackV2 } from '../../src/tooling/pack-v2.js';

const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-pack-v2-test-'));

const ir = createModelIR({
  modelId: 'unit-pack-model',
  architecture: 'qwen3',
  hiddenSize: 2048,
  numLayers: 12,
  vocabSize: 32000,
});

const targetPlan = createTargetPlan({
  targetId: 'webgpu-f16-subgroups',
  modelId: 'unit-pack-model',
  capabilityPredicate: { requiresF16: true, requiresSubgroups: true, minBufferSize: 0 },
  dtypes: { activation: 'f16-subgroups', kv: 'f16', weight: 'q4k' },
  kernelClosure: [
    { id: 'matmul', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` },
  ],
});

const pack = buildPackV2({
  modelId: 'unit-pack-model',
  modelIR: ir,
  targetPlans: [targetPlan],
  wgslModules: [
    { id: 'matmul', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` },
  ],
  artifacts: [
    { role: 'manifest', path: 'manifest.json', hash: `sha256:${'a'.repeat(64)}`, sizeBytes: 128 },
  ],
});

const validation = validatePackV2(pack);
assert.equal(validation.ok, true);

const hash = hashPackV2(pack);
assert.match(hash, /^sha256:[0-9a-f]{64}$/);

// Test on-disk persistence and re-loading
const outPath = path.join(tmpRoot, 'test.pack.json');
await writePackV2(outPath, pack);

const loaded = await loadPackV2(outPath);
assert.equal(loaded.modelId, 'unit-pack-model');
assert.equal(loaded.targetPlans.length, 1);
assert.equal(loaded.targetPlans[0].targetId, 'webgpu-f16-subgroups');

console.log('✔ pack-v2.test.js passed');
