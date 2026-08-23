import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as runtime from '../../src/pack-runtime.js';

assert.equal(typeof runtime.openPack, 'function');
assert.equal(typeof runtime.createDopplerRuntime, 'function');
assert.equal(typeof runtime.createFetchPackArtifactStore, 'function');
assert.equal('load' in runtime, false);
assert.equal('open' in runtime, false);
assert.equal('createDopplerProvider' in runtime, false);
assert.throws(() => runtime.openPack({}, {}), /explicit ports/);

const packageJson = JSON.parse(await fs.readFile('package.json', 'utf8'));
assert.equal(packageJson.main, 'src/pack-runtime.js');
assert.equal(packageJson.types, 'src/pack-runtime.d.ts');
assert.equal(packageJson.exports['.'].import, './src/pack-runtime.js');
assert.equal(packageJson.exports['.'].types, './src/pack-runtime.d.ts');
assert.equal(packageJson.exports['./runtime'].import, './src/pack-runtime.js');
assert.equal(packageJson.exports['./compat'].import, './src/index.js');

console.log('✔ pack-runtime-entrypoint.test.js passed');
