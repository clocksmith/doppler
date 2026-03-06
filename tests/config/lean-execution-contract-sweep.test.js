import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-lean-execution-contract-sweep-'));

try {
  const transformerDir = path.join(root, 'transformer-model');
  const diffusionDir = path.join(root, 'diffusion-model');
  mkdirSync(transformerDir, { recursive: true });
  mkdirSync(diffusionDir, { recursive: true });

  writeFileSync(path.join(transformerDir, 'manifest.json'), JSON.stringify({
    modelId: 'unit-transformer',
    modelType: 'transformer',
    architecture: {
      headDim: 64,
      maxSeqLen: 2048,
    },
    inference: {
      sessionDefaults: {
        kvcache: {
          layout: 'paged',
        },
        decodeLoop: {
          batchSize: 4,
        },
      },
      execution: {
        steps: [
          { id: 'decode_attention', phase: 'decode', op: 'attention' },
        ],
      },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(diffusionDir, 'manifest.json'), JSON.stringify({
    modelId: 'unit-diffusion',
    modelType: 'diffusion',
  }, null, 2), 'utf8');

  const result = spawnSync(
    process.execPath,
    ['tools/lean-execution-contract-sweep.js', '--root', root, '--json', '--no-check'],
    {
      cwd: process.cwd(),
      encoding: 'utf8',
    }
  );

  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(result.stdout);
  assert.equal(summary.schemaVersion, 1);
  assert.equal(summary.ok, true);
  assert.equal(summary.totals.manifests, 2);
  assert.equal(summary.totals.passed, 1);
  assert.equal(summary.totals.skipped, 1);
  assert.equal(summary.results.some((entry) => entry.modelId === 'unit-transformer' && entry.status === 'pass'), true);
  assert.equal(summary.results.some((entry) => entry.modelId === 'unit-diffusion' && entry.status === 'skipped'), true);
} finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('lean-execution-contract-sweep.test: ok');
