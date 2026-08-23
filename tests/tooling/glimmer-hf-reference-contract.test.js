import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const policy = JSON.parse(await fs.readFile('src/config/forge/reference/glimmer-30b-text.json', 'utf8'));
const source = await fs.readFile('tools/glimmer-hf-text-reference.py', 'utf8');

assert.equal(policy.schema, 'doppler.pinned-transformers-text-reference/v1');
assert.equal(policy.revision, 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68');
assert.equal(policy.transformersCommit, 'c7e57f79348480f73d3ef0ad8c47f807ef1378c8');
assert.equal(policy.generation.maxNewTokens, 128);
assert.equal(policy.generation.sampling, 'greedy-argmax-f32-logits');
assert.equal(policy.generation.useChatTemplate, false);
assert.equal(policy.execution.attentionImplementation, 'eager');
assert.equal(policy.author.kind, 'ai');
assert.match(source, /local_files_only=True/);
assert.match(source, /trust_remote_code=False/);
assert.match(source, /loadedTextParameters/);
assert.match(source, /preservedAuxiliaryParameters/);
assert.doesNotMatch(source, /\.to\(["']cuda["']\)/);

console.log('glimmer-hf-reference-contract.test: ok');
