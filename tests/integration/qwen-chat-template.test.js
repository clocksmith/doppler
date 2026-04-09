import assert from 'node:assert/strict';

const {
  formatChatMessages,
} = await import('../../src/inference/pipelines/text/chat-format.js');
const {
  applyChatTemplate,
} = await import('../../src/inference/pipelines/text/init.js');

{
  const prompt = applyChatTemplate('What color is the sky?', 'qwen');
  assert.equal(
    prompt,
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n'
      + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
  );
}

{
  const prompt = applyChatTemplate('What color is the sky?', 'qwen', { thinking: true });
  assert.equal(
    prompt,
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n'
      + '<|im_start|>assistant\n<think>\n'
  );
}

{
  const prompt = formatChatMessages([
    { role: 'system', content: 'You are concise.' },
    { role: 'user', content: 'What color is the sky?' },
    { role: 'assistant', content: 'The sky is blue.' },
    { role: 'user', content: 'Answer again in two words.' },
  ], 'qwen');

  assert.equal(
    prompt,
    '<|im_start|>system\nYou are concise.<|im_end|>\n'
      + '<|im_start|>user\nWhat color is the sky?<|im_end|>\n'
      + '<|im_start|>assistant\nThe sky is blue.<|im_end|>\n'
      + '<|im_start|>user\nAnswer again in two words.<|im_end|>\n'
      + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
  );
}

{
  const prompt = formatChatMessages([
    { role: 'user', content: 'What color is the sky?' },
  ], 'qwen', { thinking: true });

  assert.equal(
    prompt,
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n'
      + '<|im_start|>assistant\n<think>\n'
  );
}

{
  assert.throws(
    () => formatChatMessages([
      { role: 'user', content: 'Hello' },
      { role: 'system', content: 'Late system prompt' },
    ], 'qwen'),
    /system message to appear first/i
  );
}

console.log('qwen-chat-template.test: ok');
