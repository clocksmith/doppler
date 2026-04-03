import assert from 'node:assert/strict';

const {
  formatChatMessages,
} = await import('../../src/inference/pipelines/text/chat-format.js');
const {
  applyChatTemplate,
} = await import('../../src/inference/pipelines/text/init.js');

{
  const prompt = applyChatTemplate('What color is the sky?', 'gemma4');
  assert.equal(
    prompt,
    '<bos><|turn>user\nWhat color is the sky?<turn|>\n<|turn>model\n'
  );
}

{
  const prompt = formatChatMessages([
    { role: 'system', content: 'You are concise.' },
    { role: 'user', content: 'What color is the sky?' },
    { role: 'assistant', content: 'Blue.' },
    { role: 'user', content: 'Answer in two words.' },
  ], 'gemma4');

  assert.equal(
    prompt,
    '<bos><|turn>system\nYou are concise.<turn|>\n'
      + '<|turn>user\nWhat color is the sky?<turn|>\n'
      + '<|turn>model\nBlue.<turn|>\n'
      + '<|turn>user\nAnswer in two words.<turn|>\n'
      + '<|turn>model\n'
  );
}

{
  assert.throws(
    () => formatChatMessages([
      { role: 'tool', content: 'not supported' },
    ], 'gemma4'),
    /expects message role "system", "user", "assistant"/i
  );
}

console.log('gemma4-chat-template.test: ok');
