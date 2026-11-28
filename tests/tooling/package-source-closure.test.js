import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { loadShaderSource } from '../../src/gpu/kernels/shader-cache.js';
import { buildPackageSourceClosure, renderReferenceNpmIgnore } from '../../tools/package-source-closure.js';

const packageJson = JSON.parse(await fs.readFile(new URL('../../package.json', import.meta.url)));
const closure = await buildPackageSourceClosure(packageJson);
assert.deepEqual(closure.issues, []);
assert(closure.runtimeFiles.has('src/tooling/program-bundle-host.js'),
  'Forge reads its host module through a package-root file literal, not a module import');
assert(!closure.ignoredSourceFiles.has('tooling/program-bundle-host.js'));
assert(closure.ignoredReferenceFiles.has('channel-rotation.js'),
  'An unreferenced research oracle must stay outside the installed kernel suite');
assert.match(renderReferenceNpmIgnore(closure), /channel-rotation\.js/u);
const kernelDir = 'src/gpu/kernels/training/';
const kernelUrl = new URL('../../' + kernelDir, import.meta.url);
for (const name of (await fs.readdir(kernelUrl)).filter(file => /^tabgpt_.*\.wgsl$/u.test(file))) {
  const resource = kernelDir + name;
  assert(closure.packageResourceFiles.has(resource), `unpublished training shader: ${resource}`);
  assert.equal(await loadShaderSource(resource), await fs.readFile(new URL(name, kernelUrl), 'utf8'));
}
const resourcePolicy = JSON.parse(await fs.readFile(new URL('../../tools/policies/package-resource-policy.json', import.meta.url)));
const missingPublishedAncestor = structuredClone(resourcePolicy);
missingPublishedAncestor.runtimeResources = missingPublishedAncestor.runtimeResources.filter(file => !file.endsWith('/throughput.json'));
assert((await buildPackageSourceClosure(packageJson, missingPublishedAncestor)).issues
  .some(issue => issue.includes('extends an unpublished runtime profile: profiles/throughput')));
await assert.rejects(buildPackageSourceClosure(packageJson, {
  ...resourcePolicy, runtimeResources: [...resourcePolicy.runtimeResources, 'src/config/runtime/../../package.json'],
}), /Invalid published runtime resource/u);

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
