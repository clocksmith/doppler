import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { buildRuntimeClosure } from '../../tools/check-capsule-runtime-closure.js';
import { sha256Hex } from '../../src/utils/sha256.js';

const root = await fs.mkdtemp(path.join(tmpdir(), 'doppler-runtime-graph-'));
const policy = { schema: 'doppler.runtime-closure-policy/v1', entrypoint: 'entry.js',
  forbiddenPatterns: ['/forbidden/'], dynamicImports: [] };
try {
  await fs.mkdir(path.join(root, 'forbidden'));
  await fs.writeFile(path.join(root, 'forbidden/private.js'), 'export const privateValue = 1;');
  await fs.writeFile(path.join(root, 'entry.js'), "// import './missing-comment.js';\nawait import('./forbidden/private.js');");
  const forbidden = await buildRuntimeClosure(root, policy);
  assert.equal(forbidden.passed, false, 'a literal dynamic import cannot evade ownership checks');
  assert.deepEqual(forbidden.forbiddenFiles, ['forbidden/private.js']);
  await fs.writeFile(path.join(root, 'entry.js'), "const url = new URL('./model.json', import.meta.url);\nimport './cycle.js';");
  await fs.writeFile(path.join(root, 'cycle.js'), "import './entry.js';");
  await fs.writeFile(path.join(root, 'model.json'), '{"name":"test"}');
  const resources = await buildRuntimeClosure(root, policy);
  assert.deepEqual(resources.files.map(file => file.path), ['cycle.js', 'entry.js', 'model.json']);
  const source = 'await import(provider.module);';
  await fs.writeFile(path.join(root, 'entry.js'), source);
  await assert.rejects(buildRuntimeClosure(root, policy), /reviewed closure rule/);
  const rule = { path: 'entry.js', expression: 'provider.module', sourceHash: `sha256:${sha256Hex(source)}`,
    specifier: 'node:fs', environment: 'node', reason: 'Synthetic reviewed external dependency.' };
  const approved = { ...policy, dynamicImports: [rule] };
  assert.equal((await buildRuntimeClosure(root, approved)).dynamicImports[0].specifier, 'node:fs');
  await fs.writeFile(path.join(root, 'entry.js'), `${source}\n// changed source`);
  await assert.rejects(buildRuntimeClosure(root, approved), /reviewed closure rule/,
    'same expression with changed binding source requires review again');
  await fs.writeFile(path.join(root, 'entry.js'), 'export const value = 1;');
  await assert.rejects(buildRuntimeClosure(root, approved), /stale dynamic import rules/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('runtime-closure-graph.test: forbidden dynamic imports, cycles, resources and unresolved imports passed');
