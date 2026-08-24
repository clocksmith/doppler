import assert from 'node:assert/strict';

import { buildRequest } from '../../src/cli/doppler-cli.js';
import {
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from '../../src/tooling/command-api.js';

const requestInput = {
  command: 'release',
  action: 'decide',
  manifestPath: 'doppler-release.json',
  outputDirectory: '.doppler-release/evidence',
  packTrustedSignersPath: 'trust/pack.json',
  fleetTrustedSignersPath: 'trust/fleet.json',
  fleetReceiptPaths: ['receipts/windows.json', 'receipts/macos.json'],
  signingPrivateKeyPath: 'keys/private.json',
  signingPublicKeyPath: 'keys/public.json',
  signingAuthority: 'doppler-release-authority',
};

const request = normalizeToolingCommandRequest(requestInput);
assert.equal(request.command, 'release');
assert.equal(request.action, 'decide');
assert.deepEqual(request.fleetReceiptPaths, requestInput.fleetReceiptPaths);
assert.throws(
  () => ensureCommandSupportedOnSurface(requestInput, 'browser'),
  /Node-only/u
);
assert.throws(
  () => normalizeToolingCommandRequest({ ...requestInput, runtimeProfile: 'profiles/production' }),
  /does not accept runtimeProfile/u
);
assert.throws(
  () => normalizeToolingCommandRequest({ ...requestInput, action: 'qualify' }),
  /requires targetId and deviceIdentityPath/u
);

const parsed = {
  command: 'release',
  action: null,
  flags: {
    manifest: 'doppler-release.json',
    out: '.doppler-release/evidence',
    'pack-trusted-signers': 'trust/pack.json',
    'fleet-trusted-signers': 'trust/fleet.json',
    'fleet-receipts': 'receipts/windows.json,receipts/macos.json',
    'signing-private-key': 'keys/private.json',
    'signing-public-key': 'keys/public.json',
    'signing-authority': 'doppler-release-authority',
  },
};
const cli = await buildRequest(parsed);
assert.equal(cli.request.action, 'decide');
assert.equal(cli.surface, 'auto');
assert.deepEqual(cli.request.fleetReceiptPaths, requestInput.fleetReceiptPaths);

console.log('command-api-release.test: ok');
