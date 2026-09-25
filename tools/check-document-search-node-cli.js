#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const fixture = JSON.parse(await fs.readFile(config.fixturePath, 'utf8'));
const queries = fixture.queries.slice(0, 2);
const ownership = spawnSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import { createNodeDocumentSearch } from './node.js';
  const borrowed = {};
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: borrowed } });
  await assert.rejects(createNodeDocumentSearch({ storageDir: process.argv[1] }), /unowned WebGPU process/);
  assert.equal(navigator.gpu, borrowed);
`, config.storageDir], { cwd: config.applicationDir, encoding: 'utf8' });
assert.equal(ownership.status, 0, ownership.stderr);
const child = spawn(process.execPath, ['node.js', 'interactive', config.storageDir], {
  cwd: config.applicationDir, stdio: ['pipe', 'pipe', 'pipe'],
});
let stdout = '';
let stderr = '';
let sent = false;
child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
child.stdout.on('data', value => { stdout += value; });
child.stderr.on('data', value => {
  stderr += value;
  if (!sent && stderr.includes('Models ready')) {
    sent = true;
    child.stdin.end(queries.map(query => query.text).join('\n') + '\n');
  }
});
const exit = await new Promise((resolve, reject) => {
  child.once('error', reject);
  child.once('close', (code, signal) => resolve({ code, signal }));
});
const rows = stdout.split('\n').filter(line => line.startsWith('{"query":')).map(line => JSON.parse(line));
let failure = null;
try {
  assert.equal(exit.code, 0, stderr);
  assert.equal(exit.signal, null);
  assert.equal(rows.length, queries.length, stdout);
  for (const [index, row] of rows.entries()) {
    assert.equal(row.query, queries[index].text);
    assert.equal(row.results[0].id, queries[index].expectedTopId);
  }
} catch (error) { failure = error.message; }
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
await fs.writeFile(config.outputPath, JSON.stringify({ schema: 'doppler.node-document-search-cli/v1', passed: failure === null,
  failure, config, exit, borrowedProviderRejected: true, results: rows, buildReceiptSha256: hash(await fs.readFile(path.join(config.applicationDir, 'build-receipt.json'))),
  probeSha256: hash(await fs.readFile(fileURLToPath(import.meta.url))), stdoutSha256: hash(stdout), stderrSha256: hash(stderr),
  physicalExecution: failure === null, externalAdoption: false }, null, 2) + '\n');
await fs.writeFile(config.outputPath + '.stdout.log', stdout);
await fs.writeFile(config.outputPath + '.stderr.log', stderr);
console.log(JSON.stringify({ passed: failure === null, failure, outputPath: config.outputPath }));
if (failure !== null) process.exitCode = 1;
