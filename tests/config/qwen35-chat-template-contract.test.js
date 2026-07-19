import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { formatChatMessages } from '../../src/inference/pipelines/text/chat-format.js';

const config = JSON.parse(await readFile(
  new URL('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', import.meta.url),
  'utf8'
));

assert.equal(config.inference.chatTemplate.type, 'qwen');
assert.equal(config.inference.chatTemplate.thinking, false);

const rendered = formatChatMessages([
  { role: 'system', content: 'Return JSON only.' },
  { role: 'user', content: 'Sentence: Ada visited London.' },
], config.inference.chatTemplate.type);

assert.match(rendered, /<\|im_start\|>assistant\n<think>\n\n<\/think>\n\n$/);

console.log('qwen35-chat-template-contract.test: ok');
