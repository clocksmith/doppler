import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const profile = JSON.parse(readFileSync('src/config/runtime/experiments/verify/lfm2-verify.json', 'utf8'));
const inference = profile.runtime?.inference ?? {};

assert.equal(profile.id, 'experiments/verify/lfm2-verify');
assert.equal(profile.model, 'lfm2-5-1-2b-instruct-q4k-ehf16-af32');
assert.equal(inference.generation?.maxTokens, 12);
assert.equal(inference.batching?.maxTokens, undefined);
assert.match(
  inference.prompt?.messages?.[0]?.content ?? '',
  /Answer in one short sentence\./
);

console.log('lfm2-runtime-profile-contract.test: ok');
