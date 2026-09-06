import assert from 'node:assert/strict';
import { createElectronReleaseIpcHandler } from '../../src/client/electron/ipc-contract.js';

const calls = [];
const coordinator = Object.fromEntries(['load', 'resolveCurrent', 'installCandidate', 'activateCandidate',
  'rejectCandidate', 'rollback', 'applyRevocationSnapshot'].map(name => [name, async (...args) => {
  calls.push({ name, args });
  return name;
}]));
assert.throws(() => createElectronReleaseIpcHandler(coordinator), /authorizeRequest/);
const sender = {};
const handler = createElectronReleaseIpcHandler(coordinator, {
  authorizeRequest: (event, request) => event === sender && request.action === 'status',
});
await assert.rejects(handler({}, { action: 'status' }), error => error.code === 'DOPPLER_ELECTRON_UNAUTHORIZED');
await assert.rejects(handler(sender, { action: 'rollback', customerAuthorizationDigest: 'invented' }), /not authorized/);
assert.deepEqual(calls, [], 'unauthorized requests cannot even read coordinator state');
assert.equal(await handler(sender, { action: 'status' }), 'load');
await assert.rejects(handler(sender, { action: 'status', extra: true }), /unsupported/);
let resume;
const authorized = createElectronReleaseIpcHandler(coordinator, {
  async authorizeRequest(event, request) {
    assert.equal(event, sender);
    request.customerAuthorizationDigest = 'callback-mutation';
    await new Promise(resolve => { resume = resolve; });
    return true;
  },
});
const input = { action: 'rollback', customerAuthorizationDigest: 'original' };
const pending = authorized(sender, input);
input.action = 'status';
input.customerAuthorizationDigest = 'caller-mutation';
resume();
assert.equal(await pending, 'rollback');
assert.deepEqual(calls.at(-1), { name: 'rollback', args: ['original'] }, 'dispatch uses the detached authorized request');
for (const result of [undefined, null, 1, 'true']) {
  const denied = createElectronReleaseIpcHandler(coordinator, { authorizeRequest: async () => result });
  await assert.rejects(denied(sender, { action: 'status' }), /not authorized/);
}
const failed = createElectronReleaseIpcHandler(coordinator, { authorizeRequest: async () => { throw new Error('application verifier failed'); } });
await assert.rejects(failed(sender, { action: 'status' }), /application verifier failed/);
assert.equal(calls.length, 2);
console.log('electron-ipc-authority.test: passed (synthetic IPC events, actual request handler)');
