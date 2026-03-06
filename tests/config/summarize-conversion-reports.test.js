import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-convert-report-summary-'));

try {
  const modelADir = path.join(root, 'gemma3-a');
  const modelBDir = path.join(root, 'gemma3-b');
  mkdirSync(modelADir, { recursive: true });
  mkdirSync(modelBDir, { recursive: true });

  writeFileSync(path.join(modelADir, '2026-03-06T12-00-00.000Z.json'), JSON.stringify({
    schemaVersion: 1,
    suite: 'convert',
    command: 'convert',
    modelId: 'gemma3-a',
    timestamp: '2026-03-06T12:00:00.000Z',
    source: 'doppler',
    result: {
      presetId: 'gemma3',
      modelType: 'transformer',
      outputDir: 'models/local/gemma3-a',
      shardCount: 1,
      tensorCount: 10,
      totalSize: 1024,
    },
    manifest: null,
    executionContractArtifact: {
      ok: true,
      checks: [{ id: 'a', ok: true }],
      errors: [],
      session: { layout: 'paged' },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(modelBDir, '2026-03-06T13-00-00.000Z.json'), JSON.stringify({
    schemaVersion: 1,
    suite: 'convert',
    command: 'convert',
    modelId: 'gemma3-b',
    timestamp: '2026-03-06T13:00:00.000Z',
    source: 'doppler',
    result: {
      presetId: 'gemma3',
      modelType: 'transformer',
      outputDir: 'models/local/gemma3-b',
      shardCount: 2,
      tensorCount: 12,
      totalSize: 2048,
    },
    manifest: null,
    executionContractArtifact: {
      ok: false,
      checks: [{ id: 'b', ok: false }],
      errors: ['bad contract'],
      session: { layout: 'bdpa' },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(root, 'ignore.json'), JSON.stringify({ schemaVersion: 1, suite: 'training' }), 'utf8');

  const result = spawnSync(
    process.execPath,
    ['tools/summarize-conversion-reports.js', '--root', root, '--json'],
    {
      cwd: process.cwd(),
      encoding: 'utf8',
    }
  );

  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(result.stdout);
  assert.equal(summary.schemaVersion, 1);
  assert.equal(summary.totalReports, 2);
  assert.equal(summary.contractPass, 1);
  assert.equal(summary.contractFail, 1);
  assert.equal(summary.recent[0].modelId, 'gemma3-b');
  assert.equal(summary.recent[0].layout, 'bdpa');
}
finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('summarize-conversion-reports.test: ok');
