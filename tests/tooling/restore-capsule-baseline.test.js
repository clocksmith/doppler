import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

// Corrupt retained input must fail before package installation or network work.
const root = await fs.mkdtemp(path.join(tmpdir(), 'doppler-baseline-restore-test-'));
const tool = path.resolve('tools/restore-capsule-baseline.js');
try {
  const bundle = path.join(root, 'bundle');
  await fs.mkdir(bundle);
  const bytes = Buffer.from('retained package');
  await fs.writeFile(path.join(bundle, 'runtime.tgz'), bytes);
  const recipe = { schema: 'doppler.capsule-baseline-reproduction/v1', runtimeArchive: 'runtime.tgz',
    runtimeReceipt: 'receipt.json', sources: [], files: [{ path: 'runtime.tgz', sizeBytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex') }] };
  const configPath = path.join(bundle, 'reproduction.json');
  await fs.writeFile(configPath, JSON.stringify(recipe));
  await fs.writeFile(path.join(bundle, 'runtime.tgz'), Buffer.alloc(bytes.length));
  const output = path.join(root, 'corrupt');
  const rejected = spawnSync(process.execPath, [tool, bundle, output], { encoding: 'utf8' });
  assert.equal(rejected.status, 1);
  const report = JSON.parse(await fs.readFile(path.join(output, 'restoration.json'), 'utf8'));
  assert.equal(report.passed, false);
  assert.match(report.error.message, /SHA-256 mismatch/);
  assert.deepEqual(report.commands, []);
  assert.deepEqual(report.downloads, []);
  const retained = await fs.readFile(path.join(output, 'restoration.json'));
  const retry = spawnSync(process.execPath, [tool, bundle, output], { encoding: 'utf8' });
  assert.equal(retry.status, 1);
  assert.deepEqual(await fs.readFile(path.join(output, 'restoration.json')), retained,
    'An existing attempt and its failure receipt cannot be overwritten.');
  recipe.files[0].path = '../outside';
  await fs.writeFile(configPath, JSON.stringify(recipe));
  const traversalOutput = path.join(root, 'traversal');
  const traversal = spawnSync(process.execPath, [tool, bundle, traversalOutput], { encoding: 'utf8' });
  assert.equal(traversal.status, 1);
  const failure = JSON.parse(await fs.readFile(path.join(traversalOutput, 'restoration.json'), 'utf8'));
  assert.match(failure.error.message, /inside their root/);
  assert.deepEqual(failure.commands, []);
} finally {
  await fs.rm(root, { recursive: true, force: true });
}
console.log('restore-capsule-baseline.test: passed (input integrity and immutable failure custody)');
