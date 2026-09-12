// Check the actual stdout descendants without giving the quine a source file.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';

const GENERATIONS = 8;
const TIMEOUT_MS = 5000;
const MAX_OUTPUT_BYTES = 65536;
const original = readFileSync(new URL('../quines/00-minimal.js', import.meta.url));
const checks = [];
let current = original;

for (let generation = 1; generation <= GENERATIONS; generation += 1) {
  const result = spawnSync(process.execPath, ['--input-type=module'], {
    input: current,
    timeout: TIMEOUT_MS,
    maxBuffer: MAX_OUTPUT_BYTES,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr?.toString());
  assert.equal(result.stderr.length, 0, result.stderr.toString());
  assert.deepEqual(result.stdout, original, `Generation ${generation} changed the source bytes`);
  checks.push({ generation, passed: true });
  current = result.stdout;
}

writeFileSync(new URL('../verification/minimal-results.json', import.meta.url),
  `${JSON.stringify({
    schema: 'minimal-quine-verification/v1',
    environment: { node: process.version, platform: process.platform, arch: process.arch },
    scope: 'JavaScript stdout reproduction; each returned program runs through stdin in a fresh process.',
    passed: true,
    source: {
      path: 'quines/00-minimal.js',
      bytes: original.length,
      sha256: createHash('sha256').update(original).digest('hex'),
    },
    checks,
  }, null, 2)}\n`);

console.log(`PASS minimal quine: ${GENERATIONS} byte-identical stdout generations`);
