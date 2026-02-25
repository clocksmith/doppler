import assert from 'node:assert/strict';

const {
  createTranslateTextRequest,
  buildTranslatePromptFromRequest,
} = await import('../../demo/app/translate/request.js');

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
  const request = createTranslateTextRequest('Hello world.', 'en', 'es_XX');
  const prompt = buildTranslatePromptFromRequest(request, (code) => {
    if (code === 'en') return 'English';
    if (code === 'es_XX') return 'Spanish';
    return code;
  });
  assert.match(prompt, /professional English \(en\) to Spanish \(es-XX\) translator/);
  assert.match(prompt, /Produce only the Spanish translation/);
  assert.match(prompt, /Hello world\./);
}

{
  assert.throws(
    () => buildTranslatePromptFromRequest({ messages: [] }),
    /exactly one user message/
  );
}

console.log('translate-request-shape.test: ok');
