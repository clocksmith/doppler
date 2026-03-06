import assert from 'node:assert/strict';

const { resolveEosTokenId } = await import('../../src/converter/tokenizer-utils.js');

{
  const resolved = resolveEosTokenId({
    config: { eos_token_id: 2 },
    tokenizer: null,
    tokenizerJson: null,
  });
  assert.equal(resolved, 2);
}

{
  const resolved = resolveEosTokenId({
    config: {
      language_config: {
        eos_token_id: 100001,
      },
    },
    tokenizer: null,
    tokenizerJson: null,
  });
  assert.equal(resolved, 100001);
}

{
  const resolved = resolveEosTokenId({
    config: null,
    tokenizer: {
      eos_token: '<eos>',
      added_tokens_decoder: {
        '0': { content: '<pad>' },
        '1': { content: '<eos>' },
      },
    },
    tokenizerJson: null,
  });
  assert.equal(resolved, 1);
}

{
  const resolved = resolveEosTokenId({
    config: null,
    tokenizer: {},
    tokenizerJson: {
      special_tokens: {
        eos_token: '</s>',
      },
      added_tokens_decoder: {
        '2': { content: '</s>' },
      },
    },
  });
  assert.equal(resolved, 2);
}

{
  assert.throws(
    () => resolveEosTokenId({
      config: {},
      tokenizer: {},
      tokenizerJson: {},
    }),
    /Missing eos_token_id/
  );
}

console.log('tokenizer-utils.test: ok');
