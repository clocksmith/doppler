import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-training-report-id-pub-'));
try {
  const outPath = path.join(tempDir, 'report-ids.json');
  const result = spawnSync(process.execPath, [
    'tools/publish-training-report-ids.js',
    '--registry',
    'tools/configs/training-workloads/registry.json',
    '--out',
    outPath,
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);

  const publication = JSON.parse(readFileSync(outPath, 'utf8'));
  assert.equal(publication.schemaVersion, 1);
  assert.equal(publication.artifactType, 'training_report_id_index');
  assert.equal(Array.isArray(publication.claims), true);
  assert.equal(publication.claims.length >= 4, true);
  assert.equal(typeof publication.claims[0].reportId, 'string');
  assert.equal(publication.claims[0].reportId.length > 0, true);
  assert.equal(typeof publication.claims[0].claimBoundary, 'string');
  assert.equal(publication.claims[0].claimBoundary.length > 0, true);

  const badRegistryPath = path.join(tempDir, 'bad-registry.json');
  writeFileSync(badRegistryPath, JSON.stringify({
    schemaVersion: 1,
    workloads: [
      {
        id: 'toy-workload',
        path: 'tools/configs/training-workloads/lora-toy-tiny.json',
        sha256: 'abc123',
        baselineReportId: 'trn_toy_abc123',
      },
    ],
  }, null, 2), 'utf8');
  const badResult = spawnSync(process.execPath, [
    'tools/publish-training-report-ids.js',
    '--registry',
    badRegistryPath,
    '--out',
    outPath,
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.notEqual(badResult.status, 0);
  assert.match(badResult.stderr, /must include id, path, sha256, baselineReportId, claimBoundary/);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('training-report-id-publication.test: ok');
