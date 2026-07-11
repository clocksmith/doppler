import assert from 'node:assert/strict';

import {
  MAX_CONTEXT_HISTORY_TURNS,
  appendConversationTurn,
  countConversationTurns,
  createConversationRequest,
} from '../../demo/conversation.js';

const history = [];
for (let index = 1; index <= 10; index += 1) {
  history.push({ role: 'user', content: `question ${index}` });
  history.push({ role: 'assistant', content: `answer ${index}` });
}

{
  const request = createConversationRequest(history, 'question 11', { templateType: 'gemma4' });
  assert.equal(request.priorTurnCount, 10);
  assert.equal(request.promptInput.messages.length, 21);
  assert.equal(request.promptInput.messages[0].content, 'question 1');
  assert.equal(request.promptInput.messages.at(-1).content, 'question 11');
}

{
  const request = createConversationRequest(history, 'Translate this', {
    templateType: 'translategemma',
  });
  const userMessages = request.promptInput.messages.filter((message) => message.role === 'user');
  assert.equal(userMessages.length, 11);
  assert.deepEqual(userMessages.at(-1).content, [{
    type: 'text',
    text: 'Translate this',
    source_lang_code: 'en',
    target_lang_code: 'es',
  }]);
}

{
  const request = createConversationRequest(history, 'question 11');
  const updated = appendConversationTurn(history, request, 'answer 11');
  assert.equal(countConversationTurns(updated), 11);
  assert.equal(updated.at(-2).content, 'question 11');
  assert.equal(updated.at(-1).content, 'answer 11');
}

{
  let conversation = [];
  for (let index = 1; index <= 20; index += 1) {
    const request = createConversationRequest(conversation, `q${index}`);
    conversation = appendConversationTurn(conversation, request, `a${index}`);
  }
  assert.equal(countConversationTurns(conversation), 20);
  assert.equal(conversation[0].content, 'q1');

  const request = createConversationRequest(conversation, 'q21');
  assert.equal(request.priorTurnCount, MAX_CONTEXT_HISTORY_TURNS);
  assert.equal(request.promptInput.messages[0].content, 'q5');
  assert.equal(request.messages[0].content, 'q1');
}

console.log('conversation.test: ok');
