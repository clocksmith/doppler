import assert from 'node:assert/strict';

import {
  InferencePipeline,
  EmbeddingPipeline,
  createPipeline,
} from '../../src/generation/index.js';
import {
  StructuredJsonHeadPipeline,
  isStructuredJsonHeadModelType,
  createStructuredJsonHeadPipeline,
  DreamStructuredPipeline,
  isDreamStructuredModelType,
  createDreamStructuredPipeline,
} from '../../src/tooling-exports/structured.js';
import { parseModelConfig, parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';
import { loadWeights, initTokenizer, isStopToken } from '../../src/inference/pipelines/text/init.js';
import { initTokenizerFromManifest } from '../../src/inference/pipelines/text/model-load.js';

import { applyChatTemplate } from '../../src/inference/pipelines/text/init.js';
import { formatChatMessages } from '../../src/inference/pipelines/text/chat-format.js';

// =============================================================================
// Public API shape
// =============================================================================

{
  assert.equal(typeof InferencePipeline, 'function', 'InferencePipeline must be a class/constructor');
  assert.equal(typeof EmbeddingPipeline, 'function', 'EmbeddingPipeline must be a class/constructor');
  assert.equal(typeof createPipeline, 'function', 'createPipeline must be a function');
  assert.equal(typeof parseModelConfig, 'function', 'parseModelConfig must be a function');
  assert.equal(typeof parseModelConfigFromManifest, 'function', 'parseModelConfigFromManifest must be a function');
  assert.equal(typeof loadWeights, 'function', 'loadWeights must be a function');
  assert.equal(typeof initTokenizer, 'function', 'initTokenizer must be a function');
  assert.equal(typeof isStopToken, 'function', 'isStopToken must be a function');
  assert.equal(typeof initTokenizerFromManifest, 'function', 'initTokenizerFromManifest must be a function');
  assert.equal(typeof StructuredJsonHeadPipeline, 'function', 'StructuredJsonHeadPipeline must be a class/constructor');
  assert.equal(typeof isStructuredJsonHeadModelType, 'function', 'isStructuredJsonHeadModelType must be a function');
  assert.equal(typeof createStructuredJsonHeadPipeline, 'function', 'createStructuredJsonHeadPipeline must be a function');
  assert.equal(typeof DreamStructuredPipeline, 'function', 'DreamStructuredPipeline must be a class/constructor');
  assert.equal(typeof isDreamStructuredModelType, 'function', 'isDreamStructuredModelType must be a function');
  assert.equal(typeof createDreamStructuredPipeline, 'function', 'createDreamStructuredPipeline must be a function');
}

// =============================================================================
// Class hierarchy contract
// =============================================================================

{
  assert.ok(
    StructuredJsonHeadPipeline.prototype instanceof InferencePipeline,
    'StructuredJsonHeadPipeline must extend InferencePipeline'
  );
  assert.ok(
    DreamStructuredPipeline.prototype instanceof StructuredJsonHeadPipeline,
    'DreamStructuredPipeline must extend StructuredJsonHeadPipeline'
  );
}

// =============================================================================
// isDreamStructuredModelType / isStructuredJsonHeadModelType alias contract
// =============================================================================

{
  assert.strictEqual(
    isDreamStructuredModelType,
    isStructuredJsonHeadModelType,
    'isDreamStructuredModelType must be the same function as isStructuredJsonHeadModelType'
  );
  assert.strictEqual(
    createDreamStructuredPipeline,
    createStructuredJsonHeadPipeline,
    'createDreamStructuredPipeline must be the same function as createStructuredJsonHeadPipeline'
  );
}

// =============================================================================
// isStructuredJsonHeadModelType dispatch contract
// =============================================================================

{
  assert.equal(isStructuredJsonHeadModelType('structured_json_head'), true);
  assert.equal(isStructuredJsonHeadModelType('structured-json-head'), true);
  assert.equal(isStructuredJsonHeadModelType('dream_structured'), true);
  assert.equal(isStructuredJsonHeadModelType('dream_intent_posterior_head'), true);
  assert.equal(isStructuredJsonHeadModelType('gemma3'), false);
  assert.equal(isStructuredJsonHeadModelType(''), false);
  assert.equal(isStructuredJsonHeadModelType(null), false);
  assert.equal(isStructuredJsonHeadModelType(undefined), false);
}

// =============================================================================
// isStopToken contract
// =============================================================================

{
  assert.equal(isStopToken(1, [1, 2], 3), true, 'token in stopTokenIds must be a stop');
  assert.equal(isStopToken(2, [1, 2], 3), true, 'token in stopTokenIds must be a stop');
  assert.equal(isStopToken(3, [1, 2], 3), true, 'token matching eosTokenId must be a stop');
  assert.equal(isStopToken(4, [1, 2], 3), false, 'unmatched token must not be a stop');
  assert.equal(isStopToken(3, [], 3), true, 'eosTokenId alone must be a stop');
  assert.equal(isStopToken(3, [], null), false, 'null eosTokenId must not match');
  assert.equal(isStopToken(3, [3], null), true, 'stopTokenIds match with null eosTokenId');
}

// =============================================================================
// applyChatTemplate prompt formatting dispatch contract
// =============================================================================

{
  const prompt = 'What is the capital of France?';

  // null template → pass-through
  assert.equal(applyChatTemplate(prompt, null), prompt, 'null template must return prompt unchanged');

  // gemma → turn-based format
  const gemmaResult = applyChatTemplate(prompt, 'gemma');
  assert.ok(gemmaResult.includes('<start_of_turn>user'), 'gemma template must include <start_of_turn>user');
  assert.ok(gemmaResult.includes('<end_of_turn>'), 'gemma template must include <end_of_turn>');
  assert.ok(gemmaResult.includes('<start_of_turn>model'), 'gemma template must include <start_of_turn>model');
  assert.ok(gemmaResult.includes(prompt), 'gemma template must include the original prompt');

  // gemma4 → Gemma 4 turn format
  const gemma4Result = applyChatTemplate(prompt, 'gemma4');
  assert.ok(gemma4Result.startsWith('<bos><|turn>user'), 'gemma4 template must start with <bos><|turn>user');
  assert.ok(gemma4Result.includes('<turn|>'), 'gemma4 template must include <turn|>');
  assert.ok(gemma4Result.includes('<|turn>model'), 'gemma4 template must include <|turn>model');
  assert.ok(gemma4Result.includes(prompt), 'gemma4 template must include the original prompt');

  // llama3 → header-based format
  const llama3Result = applyChatTemplate(prompt, 'llama3');
  assert.ok(llama3Result.includes('<|begin_of_text|>'), 'llama3 template must include <|begin_of_text|>');
  assert.ok(llama3Result.includes('<|start_header_id|>user<|end_header_id|>'), 'llama3 template must include user header');
  assert.ok(llama3Result.includes('<|eot_id|>'), 'llama3 template must include <|eot_id|>');
  assert.ok(llama3Result.includes(prompt), 'llama3 template must include the original prompt');

  // gpt-oss → channel-based format
  const gptOssResult = applyChatTemplate(prompt, 'gpt-oss');
  assert.ok(gptOssResult.includes('<|start|>user<|message|>'), 'gpt-oss template must include user channel');
  assert.ok(gptOssResult.includes('<|end|>'), 'gpt-oss template must include <|end|>');
  assert.ok(gptOssResult.includes(prompt), 'gpt-oss template must include the original prompt');

  // chatml → ChatML format
  const chatmlResult = applyChatTemplate(prompt, 'chatml');
  assert.ok(chatmlResult.includes('<|im_start|>user'), 'chatml template must include <|im_start|>user');
  assert.ok(chatmlResult.includes('<|im_end|>'), 'chatml template must include <|im_end|>');
  assert.ok(chatmlResult.includes('<|im_start|>assistant'), 'chatml template must include assistant turn');
  assert.ok(chatmlResult.includes(prompt), 'chatml template must include the original prompt');

  // qwen → Qwen format (extends ChatML with think block)
  const qwenResult = applyChatTemplate(prompt, 'qwen');
  assert.ok(qwenResult.includes('<|im_start|>user'), 'qwen template must include <|im_start|>user');
  assert.ok(qwenResult.includes('<|im_start|>assistant'), 'qwen template must include assistant turn');
  assert.ok(qwenResult.includes('<think>'), 'qwen template must include <think> block');
  assert.ok(qwenResult.includes(prompt), 'qwen template must include the original prompt');

  // translategemma → throws (requires structured messages)
  assert.throws(
    () => applyChatTemplate(prompt, 'translategemma'),
    /TranslateGemma template requires structured messages/,
    'translategemma template must throw for plain prompt'
  );

  // unknown template type → throws
  assert.throws(
    () => applyChatTemplate(prompt, 'unknown-template'),
    /Unrecognized chat template type/,
    'unknown template must throw'
  );
}

// =============================================================================
// applyChatTemplate gemma output format contract (exact structure)
// =============================================================================

{
  const result = applyChatTemplate('Hello', 'gemma');
  assert.equal(
    result,
    '<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n',
    'gemma template must produce exact turn-based format'
  );
}

// =============================================================================
// applyChatTemplate gemma4 output format contract (exact structure)
// =============================================================================

{
  const result = applyChatTemplate('Hello', 'gemma4');
  assert.equal(
    result,
    '<bos><|turn>user\nHello<turn|>\n<|turn>model\n',
    'gemma4 template must produce exact turn-based format'
  );
}

// =============================================================================
// formatChatMessages gemma4 multimodal contract (exact structure)
// =============================================================================

{
  const result = formatChatMessages([
    {
      role: 'user',
      content: [
        { type: 'image' },
        { type: 'text', text: 'Describe the image.' },
      ],
    },
  ], 'gemma4');
  assert.equal(
    result,
    '<bos><|turn>user\n\n\n<|image|>\n\nDescribe the image.<turn|>\n<|turn>model\n',
    'gemma4 multimodal chat format must use a single <|image|> placeholder inside the user turn'
  );
}

{
  assert.throws(
    () => formatChatMessages([{ role: 'user', content: 'Hi' }], 'unknown-template'),
    /Unrecognized chat template type: unknown-template/,
    'formatChatMessages must fail fast for unknown template types'
  );
}

// =============================================================================
// applyChatTemplate qwen output format contract (think block)
// =============================================================================

{
  const result = applyChatTemplate('Hello', 'qwen');
  assert.ok(result.endsWith('<think>\n\n</think>\n\n'), 'qwen template must end with empty think block');
}

console.log('generation-surface-contract.test: ok');
