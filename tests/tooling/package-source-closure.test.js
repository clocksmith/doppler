import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { buildPackageSourceClosure } from '../../tools/package-source-closure.js';

const packageJson = JSON.parse(await fs.readFile(new URL('../../package.json', import.meta.url)));
const closure = await buildPackageSourceClosure(packageJson);
assert.deepEqual(closure.issues, []);
assert(closure.runtimeFiles.has('src/tooling/program-bundle-host.js'),
  'Forge reads its host module through a package-root file literal, not a module import');
assert(!closure.ignoredSourceFiles.has('tooling/program-bundle-host.js'));

// Publication exclusions cannot hide an imported asset or a profile ancestor.
const missingAncestor = structuredClone(packageJson);
missingAncestor.files.push('!src/config/runtime/profiles/throughput.json');
const brokenProfile = await buildPackageSourceClosure(missingAncestor);
assert(brokenProfile.issues.some((issue) => issue.includes('extends an unpublished runtime profile: profiles/throughput')));
const missingAsset = structuredClone(packageJson);
missingAsset.files.push('!src/config/kernels/registry.json');
const brokenRuntime = await buildPackageSourceClosure(missingAsset);
assert(brokenRuntime.issues.some((issue) => issue.includes('required runtime resource is excluded')));
console.log('package-source-closure.test: ok');
