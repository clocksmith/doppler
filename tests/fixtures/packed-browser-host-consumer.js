import assert from 'node:assert/strict';
import * as root from 'dopplerx';
import * as host from 'dopplerx/host';

assert(import.meta.resolve('dopplerx/host').endsWith('/capsule-host.browser.js'));
for (const name of ['createCapsuleStreamAccumulator', 'capsuleOperationSnapshots']) {
  assert.equal(typeof host[name], 'function', `browser host must export ${name}`);
  assert.equal(host[name], root[name]);
}
console.log('Installed browser-conditioned host stream exports passed.');
