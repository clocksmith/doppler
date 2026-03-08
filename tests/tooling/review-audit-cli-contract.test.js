import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

{
  const result = spawnSync(process.execPath, [
    'tools/review-audit.js',
    'summary',
    '--scope',
    'tools',
    'stray',
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unexpected positional argument: stray/);
}

{
  const result = spawnSync(process.execPath, [
    'tools/review-audit.js',
    'event',
    '--scope',
    'tools',
    '--path',
    'tools/doppler-cli.js',
    '--owner',
    'A',
    '--agent',
    'review-test',
    '--action',
    'review_started',
    '--status',
    'in_progress',
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Owner mismatch/);
}

{
  const result = spawnSync(process.execPath, [
    'tools/review-audit.js',
    'event',
    '--scope', 'src',
    '--path', 'src/config/merge.js',
    '--owner', 'A',
    '--agent', 'test',
    '--action', 'review_started',
    '--at', 'not-a-timestamp',
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /--at must be a valid ISO 8601 timestamp/);
}

console.log('review-audit-cli-contract.test: ok');
