import assert from 'node:assert/strict';
import * as root from 'doppler-gpu';
import * as host from 'doppler-gpu/host';

assert(import.meta.resolve('doppler-gpu/host').endsWith('/capsule-host.browser.js'));
for (const name of ['createCapsuleStreamAccumulator', 'capsuleOperationSnapshots']) {
  assert.equal(typeof host[name], 'function', `browser host must export ${name}`);
  assert.equal(host[name], root[name]);
}
console.log('Installed browser-conditioned host stream exports passed.');
