import assert from 'node:assert/strict';

import { formatChatMessages } from '../../src/inference/pipelines/text/chat-format.js';

const prompt = formatChatMessages([
  {
    role: 'user',
    content: [
      { type: 'image' },
      { type: 'text', text: 'Text Recognition:' },
    ],
  },
], 'glmocr');

assert.equal(
  prompt,
  '[gMASK]<sop><|user|>\n' +
    '<|begin_of_image|><|image|><|end_of_image|>Text Recognition:' +
    '<|assistant|>\n'
);

const noThinkingPrompt = formatChatMessages([
  {
    role: 'user',
    content: [
      { type: 'image' },
      { type: 'text', text: 'Text Recognition:' },
    ],
  },
], 'glmocr', { thinking: false });

assert.equal(
  noThinkingPrompt,
  '[gMASK]<sop><|user|>\n' +
    '<|begin_of_image|><|image|><|end_of_image|>Text Recognition:/nothink' +
    '<|assistant|>\n<think></think>\n'
);

assert.throws(
  () => formatChatMessages([
    { role: 'user', content: [{ type: 'audio' }] },
  ], 'glmocr'),
  /does not support content type "audio"/
);

console.log('glmocr-chat-format-contract.test: ok');
