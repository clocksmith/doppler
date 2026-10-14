import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const promptCapsulePath = new URL('../../tools/data/gemma4-e2b-blog-prompts-512.json', import.meta.url);
const promptCapsule = JSON.parse(await fs.readFile(promptCapsulePath, 'utf8'));

assert.ok(Array.isArray(promptCapsule), 'prompt capsule must be an array');

const ids = new Set();
for (const [index, prompt] of promptCapsule.entries()) {
  assert.equal(typeof prompt?.id, 'string', `prompt[${index}].id must be a string`);
  assert.notEqual(prompt.id.trim(), '', `prompt[${index}].id must be non-empty`);
  assert.equal(typeof prompt?.text, 'string', `prompt[${index}].text must be a string`);
  assert.ok(prompt.text.length > 0, `prompt[${index}].text must be non-empty`);
  assert.equal(ids.has(prompt.id), false, `duplicate prompt id: ${prompt.id}`);
  ids.add(prompt.id);
}

console.log('gemma4-e2b-blog-prompt-capsule-contract.test: ok');
