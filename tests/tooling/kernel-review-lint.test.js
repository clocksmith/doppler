import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('../../', import.meta.url));
const lint = path.join(root, 'skills/doppler-kernel-reviewer/scripts/lint-kernel.js');
const directory = await mkdtemp(path.join(tmpdir(), 'doppler-kernel-review-'));
const run = file => spawnSync(process.execPath, [lint, file], { cwd: directory, encoding: 'utf8' });
try {
  const checked = run(path.join(root, 'src/config/choice-scoring.js'));
  assert.equal(checked.status, 0, checked.stderr);
  const fixture = path.join(directory, 'unchecked.js');
  for (const source of ['/** @param {number} x */\nfunction f(x) {}', '/**\n * @param {number} x\n */\nfunction f(x) {}']) {
    await writeFile(fixture, source);
    const result = run(fixture);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /declared tsconfig.source-strict.json root/);
  }
  await writeFile(fixture, '// Ordinary implementation comment\nfunction f(x) {}');
  assert.equal(run(fixture).status, 0);
} finally {
  await rm(directory, { recursive: true, force: true });
}
console.log('kernel-review-lint.test: passed');
