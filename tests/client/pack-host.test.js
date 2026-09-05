import assert from 'node:assert/strict';
import { openPack } from 'doppler-gpu/host';
import { openPack as nodeOpenPack } from '../../src/client/doppler-api.js';
import { openPack as browserOpenPack } from '../../src/client/doppler-api.browser.js';
import { openPack as browserHostOpenPack } from '../../src/client/pack-host.browser.js';

assert.equal(openPack, nodeOpenPack);
assert.equal(browserHostOpenPack, browserOpenPack);
console.log('pack-host.test: ok (host composition delegates to the existing verified runtime)');
