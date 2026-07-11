import assert from 'node:assert/strict';

import {
  MAX_STORED_HISTORY_TURNS,
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
  const request = createConversationRequest(history, 'question 11', {
    historyEnabled: true,
    turnLimit: 4,
    templateType: 'gemma4',
  });
  assert.equal(request.priorTurnCount, 4);
  assert.equal(request.promptInput.messages.length, 9);
  assert.equal(request.promptInput.messages[0].content, 'question 7');
  assert.equal(request.promptInput.messages.at(-1).content, 'question 11');
}

{
  const request = createConversationRequest(history, 'Translate this', {
    historyEnabled: true,
    turnLimit: 4,
    templateType: 'translategemma',
  });
  const userMessages = request.promptInput.messages.filter((message) => message.role === 'user');
  assert.equal(userMessages.length, 5);
  assert.deepEqual(userMessages.at(-1).content, [{
    type: 'text',
    text: 'Translate this',
    source_lang_code: 'en',
    target_lang_code: 'es',
  }]);
}

{
  const request = createConversationRequest(history, 'Fresh prompt', {
    historyEnabled: false,
    turnLimit: 8,
  });
  assert.equal(request.priorTurnCount, 0);
  assert.deepEqual(request.messages, [{ role: 'user', content: 'Fresh prompt' }]);
  assert.deepEqual(
    appendConversationTurn(history, request, 'Fresh answer'),
    history,
    'paused history must neither send nor delete saved turns'
  );
}

{
  let bounded = [];
  for (let index = 1; index <= 20; index += 1) {
    const request = createConversationRequest(bounded, `q${index}`, {
      historyEnabled: true,
      turnLimit: MAX_STORED_HISTORY_TURNS,
    });
    bounded = appendConversationTurn(bounded, request, `a${index}`);
  }
  assert.equal(countConversationTurns(bounded), MAX_STORED_HISTORY_TURNS);
  assert.equal(bounded[0].content, 'q5');
}

console.log('conversation.test: ok');
