import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validateOrderedLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/partitions.js';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-split-'));
const manifest = { modelId: 'four-stream-model', architecture: { numLayers: 12, hiddenSize: 8, vocabSize: 128 },
  inference: { hyperConnections: { streamCount: 4, streamWidth: 8 } } };
const bytes = JSON.stringify(manifest);
const run = promisify(execFile);
const manifestPath = path.join(directory, 'manifest.json');
const requestPath = path.join(directory, 'request.json');
const outputPath = path.join(directory, 'plan.json');
const tool = fileURLToPath(new URL('../../tools/compile-layer-partition-plan.js', import.meta.url));
try {
  await fs.writeFile(manifestPath, bytes);
  await fs.writeFile(requestPath, JSON.stringify({ splitLayers: [2, 7], activationDtype: 'f32', activationPrecision: 'f32' }));
  await run(process.execPath, [tool, '--manifest', manifestPath, '--config', requestPath, '--out', outputPath]);
  const result = JSON.parse(await fs.readFile(outputPath, 'utf8'));
  assert.equal(result.model.identity, 'sha256:' + createHash('sha256').update(bytes).digest('hex'));
  assert.equal(result.qualification, null);
  assert.equal(result.planId, hashLayerPartitionPlan(result.plan));
  assert.deepEqual(result.plan.partitions.map(row => row.layerRange), [[0, 1], [2, 6], [7, 11]]);
  assert.equal(result.plan.partitions[0].outputContract.hiddenSize, 32);
  validateOrderedLayerPartitionPlan(manifest, result.plan);
  await fs.unlink(outputPath);
  await fs.writeFile(requestPath, JSON.stringify({ splitLayers: [2, 7], activationDtype: 'f32', activationPrecision: 'f32', hiddenSize: 16 }));
  await assert.rejects(run(process.execPath, [tool, '--manifest', manifestPath, '--config', requestPath, '--out', outputPath]),
    /Partition request requires/);
  await assert.rejects(fs.stat(outputPath), { code: 'ENOENT' });
} finally { await fs.rm(directory, { recursive: true, force: true }); }
console.log('compile-layer-partition-plan: source geometry, identity and rejected overrides passed');
