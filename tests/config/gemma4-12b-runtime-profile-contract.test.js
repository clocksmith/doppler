import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const profile = JSON.parse(readFileSync(
  new URL('../../src/config/runtime/profiles/gemma4-12b-rdrr-safe-b4.json', import.meta.url),
  'utf8'
));

assert.equal(profile.id, 'profiles/gemma4-12b-rdrr-safe-b4');
assert.equal(profile.intent, 'investigate');
assert.equal(profile.stability, 'experimental');
assert.equal(profile.extends, 'profiles/default');

const decodeLoop = profile.runtime?.inference?.session?.decodeLoop;
assert.equal(decodeLoop?.batchSize, 4);
assert.equal(decodeLoop?.readbackInterval, 4);
assert.equal(decodeLoop?.readbackMode, 'overlapped');
assert.equal(decodeLoop?.ringTokens, 4);
assert.equal(decodeLoop?.ringStop, 1);
assert.equal(decodeLoop?.ringStaging, 2);
assert.equal(decodeLoop?.disableCommandBatching, false);

assert.equal(profile.runtime?.inference?.session?.kvcache?.maxSeqLen, 1024);
assert.equal(profile.inference, undefined);
