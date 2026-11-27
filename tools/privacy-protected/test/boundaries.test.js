// Source boundaries of the protected route: the evaluator side (WebGPU and CPU backends, the kernels, the compiler)
// never imports or contains secret-key code, the evaluator answers only the pinned message kinds, and the tree stays
// JavaScript only.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { findJsOnlyViolations, readTrackedFiles } from '../../check-js-only.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// Every module a file imports, transitively (relative imports only).
function importGraph(entry) {
  const seen = new Set();
  const walk = (file) => {
    if (seen.has(file)) return;
    seen.add(file);
    const src = readFileSync(file, 'utf8');
    for (const m of src.matchAll(/(?:import|export)\s[^'"]*?from\s+'(\.[^']+)'|import\('(\.[^']+)'\)/g)) {
      walk(resolve(dirname(file), m[1] ?? m[2]));
    }
  };
  walk(entry);
  return [...seen];
}

const SECRET = /(?:secret.?key|decryptor|encryptor|keygen|key.?generator|decrypt\(|encrypt\(|\bsk\b)/i;
const SECRET_MODULES = ['ckks/keys.js', 'ckks/key-workers.js', 'bridge.js', 'requester.js'];

test('the evaluator (both backends) and the compiler never reach secret-key code', () => {
  for (const entry of ['webgpu/evaluator.js', 'webgpu/test-vectors.js', 'compiler.js']) {
    const graph = importGraph(join(ROOT, entry)).filter((f) => f.startsWith(ROOT));
    for (const bad of SECRET_MODULES) assert.ok(!graph.includes(join(ROOT, bad)), `${entry} imports ${bad}`);
  }
  const files = [...readdirSync(join(ROOT, 'webgpu')).filter((f) => f.endsWith('.js') && f !== 'test-vectors.js').map((f) => join(ROOT, 'webgpu', f)),
    ...readdirSync(join(ROOT, 'webgpu/kernels')).map((f) => join(ROOT, 'webgpu/kernels', f)), join(ROOT, 'ckks/rns-exec.js'), join(ROOT, 'compiler.js')];
  for (const f of files) {
    const m = SECRET.exec(readFileSync(f, 'utf8'));
    assert.equal(m, null, `${f} contains ${m?.[0]}`);
  }
});

test('the evaluator accepts only the pinned message kinds, in order', () => {
  const src = readFileSync(join(ROOT, 'webgpu/evaluator.js'), 'utf8');
  const got = [...src.matchAll(/recv\(('[a-z-]+'(, )?)+\)/g)].map((m) => m[0]);
  assert.deepEqual(got, ["recv('hello')", "recv('key', 'keys-held')", "recv('key')", "recv('prefill', 'decode', 'bye')", "recv('hop-ready')",
    "recv('prefill', 'decode', 'bye')", "recv('hop-hello')", "recv('hop', 'bye')"]);
});

test('the requester answers no evaluator request: it receives only manifest, ready and result', () => {
  const src = readFileSync(join(ROOT, 'requester.js'), 'utf8');
  const kinds = new Set([...src.matchAll(/recv\(([^)]*)\)/g)].flatMap((m) => [...m[1].matchAll(/'([a-z-]+)'/g)].map((k) => k[1])));
  assert.deepEqual([...kinds].sort(), ['manifest', 'ready', 'result']);
});

test('the protected route is JavaScript only (no Go, Python or shell)', () => {
  const routeFiles = readTrackedFiles(resolve(ROOT, '..', '..')).filter(file => file.path.startsWith('tools/privacy-protected/'));
  assert.deepEqual(findJsOnlyViolations(routeFiles), []);
});
