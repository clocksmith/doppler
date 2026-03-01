import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { saveReport } from '../../src/storage/reports.js';

const tempReportsRoot = await mkdtemp(path.join(tmpdir(), 'doppler-reports-'));
const previousReportsDir = process.env.DOPPLER_REPORTS_DIR;
process.env.DOPPLER_REPORTS_DIR = tempReportsRoot;

try {
  const timestamp = '2026-03-01T00:00:00.000Z';
  const info = await saveReport(
    'training-smoke',
    { suite: 'training', passed: 2, failed: 0 },
    { timestamp }
  );
  assert.equal(info.backend, 'node-fs');
  assert.ok(typeof info.path === 'string' && info.path.length > 0);

  const filePath = path.isAbsolute(info.path)
    ? info.path
    : path.resolve(process.cwd(), info.path);
  const json = JSON.parse(await readFile(filePath, 'utf8'));
  assert.equal(json.suite, 'training');
  assert.equal(json.passed, 2);
  assert.equal(json.failed, 0);
} finally {
  if (previousReportsDir === undefined) {
    delete process.env.DOPPLER_REPORTS_DIR;
  } else {
    process.env.DOPPLER_REPORTS_DIR = previousReportsDir;
  }
  await rm(tempReportsRoot, { recursive: true, force: true });
}

console.log('reports-node-fallback.test: ok');
