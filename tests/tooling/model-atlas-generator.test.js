import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'atlas-test-'));
const output = path.join(directory, 'index.html');
const generate = (...args) => execFileSync(process.execPath, ['tools/build-model-atlas.js', '--output', output, ...args], { encoding: 'utf8', stdio: 'pipe' });
try {
  generate();
  const first = fs.readFileSync(output, 'utf8');
  generate('--check');
  generate();
  assert.equal(fs.readFileSync(output, 'utf8'), first);
  assert.match(first, /id="qwen-3-5-2b-q4k-ehaf16"/);
  assert.match(first, /No KV allocation/);
  assert.doesNotMatch(first, /d4da-analytics|\/Users\//);
  fs.appendFileSync(output, 'stale');
  assert.throws(() => generate('--check'), /Atlas is stale/);
  console.log('model-atlas-generator: reproducibility, model anchors and stale-output rejection passed');
} finally {
  fs.rmSync(directory, { recursive: true, force: true });
}
