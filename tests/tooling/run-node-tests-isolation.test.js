import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-run-node-tests-'));
try {
  writeFileSync(
    path.join(tempDir, 'a-first.test.js'),
    "globalThis.__dopplerIsolationLeak = 1;\nconsole.log('a-first.test: ok');\n",
    'utf8'
  );

  writeFileSync(path.join(tempDir, 'owned-worker.js'), [
    "import { parentPort } from 'node:worker_threads';",
    "parentPort.on('message', value => parentPort.postMessage(value + '-ok'));",
    "parentPort.postMessage('ready');",
  ].join('\n'));
  writeFileSync(path.join(tempDir, 'c-owned-worker.test.js'), [
    "import assert from 'node:assert/strict';",
    "import { Worker } from 'node:worker_threads';",
    "const worker = new Worker(new URL('./owned-worker.js', import.meta.url));",
    'try {',
    '  const response = await new Promise((resolve, reject) => {',
    "    worker.once('error', reject);",
    "    worker.once('exit', code => reject(new Error('Owned worker exited before its probe: ' + code)));",
    "    worker.on('message', value => {",
    "      if (value === 'ready') setTimeout(() => worker.postMessage('probe'), 25);",
    "      else resolve(value);",
    '    });',
    '  });',
    "  assert.equal(response, 'probe-ok');",
    '} finally { await worker.terminate(); }',
    "console.log('c-owned-worker.test: ok');",
  ].join('\n'));
  writeFileSync(
    path.join(tempDir, 'b-second.test.js'),
    [
      "if (globalThis.__dopplerIsolationLeak !== undefined) {",
      "  throw new Error('test process state leaked across files');",
      '}',
      "console.log('b-second.test: ok');",
      '',
    ].join('\n'),
    'utf8'
  );

  const result = spawnSync(process.execPath, [
    'tools/run-node-tests.js',
    tempDir,
    '--force-exit',
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /\[node-tests\] ok: 3 files/);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('run-node-tests-isolation.test: ok');
