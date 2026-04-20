// Pending-feature test. Asserts the shape of a public logits API
// (advanced.prefillWithLogits / advanced.decodeStepLogits) that does not
// yet exist on the Doppler client surface. Excluded from the default
// test lane by tools/run-node-tests.js via the *.pending.test.js
// suffix. See tools/policies/pending-tests-policy.json for owner and
// expiry.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const nodeApiSource = readFileSync(new URL('../../src/client/doppler-api.js', import.meta.url), 'utf8');
const browserApiSource = readFileSync(new URL('../../src/client/doppler-api.browser.js', import.meta.url), 'utf8');
const apiTypes = readFileSync(new URL('../../src/client/doppler-api.d.ts', import.meta.url), 'utf8');

assert.match(nodeApiSource, /advanced:\s*\{[\s\S]*prefillWithLogits\(prompt, options = \{\}\)[\s\S]*decodeStepLogits\(currentIds, options = \{\}\)/);
assert.match(browserApiSource, /advanced:\s*\{[\s\S]*prefillWithLogits\(prompt, options = \{\}\)[\s\S]*decodeStepLogits\(currentIds, options = \{\}\)/);
assert.match(apiTypes, /prefillWithLogits\(prompt: string \| ChatMessage\[] \| \{ messages: ChatMessage\[] \}, options\?: GenerateOptions\): Promise<PrefillResult>;/);
assert.match(apiTypes, /decodeStepLogits\(currentIds: number\[], options\?: GenerateOptions\): Promise<LogitsStepResult>;/);

console.log('doppler-public-logits-api-contract.test: ok');
