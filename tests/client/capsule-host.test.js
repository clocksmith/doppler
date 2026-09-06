import assert from 'node:assert/strict';
import { openCapsule } from 'doppler-gpu/host';
import { openCapsule as nodeOpenCapsule } from '../../src/client/doppler-api.js';
import { openCapsule as browserOpenCapsule } from '../../src/client/doppler-api.browser.js';
import { openCapsule as browserHostOpenCapsule } from '../../src/client/capsule-host.browser.js';

assert.equal(openCapsule, nodeOpenCapsule);
assert.equal(browserHostOpenCapsule, browserOpenCapsule);
console.log('capsule-host.test: ok (host composition delegates to the existing verified runtime)');
