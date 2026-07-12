import assert from 'node:assert/strict';
import fs from 'node:fs';

const path = 'src/config/runtime/profiles/qwen-3-5-9b-metal-correctness.json';
const profile = JSON.parse(fs.readFileSync(path, 'utf8'));
const inference = profile.runtime?.inference;
const batching = inference?.batching;
const decodeLoop = inference?.session?.decodeLoop;

assert.equal(profile.id, 'profiles/qwen-3-5-9b-metal-correctness');
assert.equal(profile.intent, 'investigate');
assert.equal(profile.extends, 'profiles/production');
assert.equal(batching?.batchSize, 1);
assert.equal(batching?.maxTokens, 16);
assert.equal(batching?.readbackInterval, 1);
assert.equal(inference?.kvcache?.maxSeqLen, 256);
assert.equal(inference?.session?.kvcache?.maxSeqLen, 256);
assert.equal(decodeLoop?.batchSize, 1);
assert.equal(decodeLoop?.readbackInterval, 1);
assert.equal(decodeLoop?.disableCommandBatching, false);
assert.equal(inference?.kernelPath, undefined);
assert.equal(profile.runtime?.shared?.profiler, undefined);

console.log('qwen35-9b-metal-correctness-profile.test: ok');
