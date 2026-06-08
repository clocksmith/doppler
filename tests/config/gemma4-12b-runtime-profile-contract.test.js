import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readProfile(name) {
  return JSON.parse(readFileSync(
    new URL(`../../src/config/runtime/profiles/${name}.json`, import.meta.url),
    'utf8'
  ));
}

function assertSafeSingleDecodeLoop(profile, label) {
  const session = profile.runtime?.inference?.session;
  const decodeLoop = session?.decodeLoop;
  assert.equal(profile.runtime?.inference?.largeWeights?.lmHeadChunkRows, 16384, `${label}: LM head chunk rows`);
  assert.equal(session?.prefillTokenChunkSize, 64, `${label}: prefill token chunk size`);
  assert.equal(decodeLoop?.batchSize, 1, `${label}: batch size`);
  assert.equal(decodeLoop?.readbackInterval, 1, `${label}: readback interval`);
  assert.equal(decodeLoop?.readbackMode, 'sequential', `${label}: readback mode`);
  assert.equal(decodeLoop?.ringTokens, 1, `${label}: token ring slots`);
  assert.equal(decodeLoop?.ringStop, 1, `${label}: stop ring slots`);
  assert.equal(decodeLoop?.ringStaging, 2, `${label}: staging ring slots`);
  assert.equal(decodeLoop?.disableCommandBatching, true, `${label}: command batching`);
  assert.equal(session?.kvcache?.maxSeqLen, 1024, `${label}: KV max sequence length`);
  assert.equal(profile.inference, undefined, `${label}: no root inference block`);
}

const safeSingle = readProfile('gemma4-12b-rdrr-safe-single');
assert.equal(safeSingle.id, 'profiles/gemma4-12b-rdrr-safe-single');
assert.equal(safeSingle.intent, 'investigate');
assert.equal(safeSingle.stability, 'experimental');
assert.equal(safeSingle.extends, 'profiles/default');
assertSafeSingleDecodeLoop(safeSingle, 'safe single profile');

const deprecatedB4 = readProfile('gemma4-12b-rdrr-safe-b4');
assert.equal(deprecatedB4.id, 'profiles/gemma4-12b-rdrr-safe-b4');
assert.equal(deprecatedB4.intent, 'investigate');
assert.equal(deprecatedB4.stability, 'deprecated');
assert.equal(deprecatedB4.extends, 'profiles/default');
assert.equal(deprecatedB4.replacementId, 'profiles/gemma4-12b-rdrr-safe-single');
assertSafeSingleDecodeLoop(deprecatedB4, 'deprecated b4 profile');
