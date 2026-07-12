import assert from 'node:assert/strict';

import { isSendReady, syncSendButton } from '../../demo/input.js';
import { state } from '../../demo/ui/state.js';

assert.equal(isSendReady({ pipeline: null, prompt: 'Hello' }), false);
assert.equal(isSendReady({ pipeline: {}, prompt: '' }), false);
assert.equal(isSendReady({ pipeline: {}, prompt: '   ' }), false);
assert.equal(isSendReady({ pipeline: {}, prompt: 'Hello' }), true);
assert.equal(isSendReady({ pipeline: {}, prompt: 'Hello', generating: true }), false);
assert.equal(isSendReady({ pipeline: {}, prompt: 'Hello', prefilling: true }), false);

const originalDocument = globalThis.document;
const originalPipeline = state.pipeline;
const originalGenerating = state.generating;
const originalPrefilling = state.prefilling;
const promptInput = { value: 'Hello' };
const sendButton = { disabled: true, title: '' };
globalThis.document = {
  getElementById(id) {
    if (id === 'prompt-input') return promptInput;
    if (id === 'run-btn') return sendButton;
    return null;
  },
};

try {
  state.pipeline = {};
  state.generating = false;
  state.prefilling = false;
  assert.equal(syncSendButton(), true);
  assert.equal(sendButton.disabled, false);
  assert.equal(sendButton.title, 'Send message');

  promptInput.value = '';
  assert.equal(syncSendButton(), false);
  assert.equal(sendButton.disabled, true);
  assert.equal(sendButton.title, 'Enter a message to send');

  promptInput.value = 'Hello';
  state.pipeline = null;
  assert.equal(syncSendButton(), false);
  assert.equal(sendButton.title, 'Load a model to send');
} finally {
  state.pipeline = originalPipeline;
  state.generating = originalGenerating;
  state.prefilling = originalPrefilling;
  globalThis.document = originalDocument;
}

console.log('input.test: ok');
