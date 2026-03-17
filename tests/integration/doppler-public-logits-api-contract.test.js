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
