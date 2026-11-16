import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { buildPackageSourceClosure } from '../package-source-closure.js';

const name = 'qwen3-6-35b-a3b-q4_0-pooled-q8-common-diagnostic';
const oldPath = `src/config/runtime/profiles/${name}.json`;
const root = new URL('../../', import.meta.url);
const bytes = await fs.readFile(new URL(`profiles/${name}.json`, import.meta.url));
assert.equal(createHash('sha256').update(bytes).digest('hex'),
  'de27a1fd23f676039b06111b7abd1085b36dd682dcb7baa49fedfe882decf297',
  'Relocation preserves the complete diagnostic identity and kernel pins');
const profile = JSON.parse(bytes);
assert.equal(profile.model, 'qwen-3-6-35b-a3b-text-source-q4_0-af32-pooled-q8-common');
assert.equal(profile.extends, 'profiles/default');
await assert.rejects(fs.access(new URL(oldPath, root)), { code: 'ENOENT' });
const registry = await fs.readFile(new URL('src/config/kernels/registry.json', root), 'utf8');
assert(!registry.includes(`runtime/profiles/${name}`));
const policy = JSON.parse(await fs.readFile(new URL('tools/policies/package-resource-policy.json', root)));
assert(!policy.runtimeResources.includes(oldPath));
const pkg = JSON.parse(await fs.readFile(new URL('package.json', root)));
const closure = await buildPackageSourceClosure(pkg);
assert.deepEqual(closure.issues, []);
assert(!closure.runtimeFiles.has(oldPath));
console.log('peer diagnostic profile boundary: passed');
