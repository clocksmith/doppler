import assert from 'node:assert/strict';
import { closeInstalledCapsule } from '../../tools/close-installed-capsule.js';
const events = [];
const failed = await closeInstalledCapsule({
  closeSession: async () => { events.push('session'); throw new Error('session close failed'); },
  destroyDevice: () => { events.push('device'); throw new Error('device destroy failed'); },
  releaseProvider: async () => { events.push('provider'); },
});
assert.deepEqual(events, ['session', 'device', 'provider']);
assert.deepEqual(failed, { passed: false, errors: ['session close failed', 'device destroy failed'] });
assert.deepEqual(await closeInstalledCapsule({ closeSession() {}, destroyDevice() {}, releaseProvider() {} }), { passed: true, errors: [] });
console.log('close-installed-capsule.test: ok');
