import assert from 'node:assert/strict';

const {
  createTranslateTextRequest,
} = await import('../../demo/ui/translate/request.js');
const { formatChatMessages } = await import('../../src/inference/pipelines/text/chat-format.js');

{
  const request = createTranslateTextRequest('Hello world.', 'en', 'es_XX');
  assert.equal(request.messages.length, 1);
  assert.equal(request.messages[0].role, 'user');
  assert.equal(request.messages[0].content.length, 1);
  assert.equal(request.messages[0].content[0].type, 'text');
  assert.equal(request.messages[0].content[0].source_lang_code, 'en');
  assert.equal(request.messages[0].content[0].target_lang_code, 'es_XX');
  assert.equal(request.messages[0].content[0].text, 'Hello world.');
}

{
  const request = createTranslateTextRequest('Hello world.', 'en', 'fr');
  const prompt = formatChatMessages(request.messages, 'translategemma');
  assert.match(prompt, /^<start_of_turn>user\n/);
  assert.match(prompt, /professional English \(en\) to French \(fr\) translator/);
  assert.match(prompt, /Produce only the French translation/);
  assert.match(prompt, /Hello world\./);
  assert.match(prompt, /<end_of_turn>\n<start_of_turn>model\n$/);
}

{
  assert.throws(
    () => formatChatMessages([
      {
        role: 'user',
        content: [
          {
            type: 'text',
            source_lang_code: 'xx',
            target_lang_code: 'fr',
            text: 'Hello world.',
          },
        ],
      },
    ], 'translategemma'),
    /unsupported source_lang_code/i
  );
}

{
  assert.throws(
    () => formatChatMessages([], 'translategemma'),
    /requires at least one message/i
  );
}

console.log('translate-request-shape.test: ok');
