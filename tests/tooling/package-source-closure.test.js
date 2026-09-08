import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { buildPackageSourceClosure } from '../../tools/package-source-closure.js';

const packageJson = JSON.parse(await fs.readFile(new URL('../../package.json', import.meta.url)));
const closure = await buildPackageSourceClosure(packageJson);
assert.deepEqual(closure.issues, []);
assert(closure.runtimeFiles.has('src/tooling/program-bundle-host.js'),
  'Forge reads its host module through a package-root file literal, not a module import');
assert(!closure.ignoredSourceFiles.has('tooling/program-bundle-host.js'));
console.log('package-source-closure.test: ok');
