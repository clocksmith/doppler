import assert from 'node:assert/strict';

import {
  resolveEffectivePrefillTokenChunkSize,
  shouldDisablePrefillCommandBatching,
} from '../../src/inference/pipelines/text/generator-prefill-helpers.js';

{
  const state = {
    modelConfig: {
      sessionSettings: {
        prefillTokenChunkSize: 64,
      },
    },
    runtimeConfig: {
      inference: {
        session: {
          prefillTokenChunkSize: 128,
        },
      },
    },
  };

  assert.equal(
    resolveEffectivePrefillTokenChunkSize(state),
    128,
    'positive runtime prefillTokenChunkSize must override manifest-owned chunk policy'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      kvCache: {
        hasGPUCache: () => true,
      },
    },
    {},
    { offset: 4, length: 70 }
  );

  assert.equal(
    disabled,
    true,
    'multimodal bidirectional prefill with a live KV cache must bypass command batching'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      kvCache: {
        hasGPUCache: () => false,
      },
    },
    {},
    { offset: 4, length: 70 }
  );

  assert.equal(
    disabled,
    false,
    'multimodal prefill without a live KV cache must not disable command batching by default'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      kvCache: {
        hasGPUCache: () => true,
      },
    },
    {},
    null
  );

  assert.equal(
    disabled,
    false,
    'text-only prefill must preserve command batching when no multimodal span is present'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      kvCache: {
        layout: 'bdpa_paged',
        hasGPUCache: () => true,
      },
    },
    {},
    null
  );

  assert.equal(
    disabled,
    true,
    'bdpa_paged prefill must continue to disable command batching regardless of modality'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      runtimeConfig: {
        inference: {
          session: {
            prefillTokenChunkSize: 64,
          },
        },
      },
      kvCache: {
        hasGPUCache: () => true,
      },
    },
    {},
    null
  );

  assert.equal(
    disabled,
    true,
    'token-chunked prefill must bypass command batching so cross-chunk KV writes match the direct path'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      modelConfig: {
        sessionSettings: {
          prefillTokenChunkSize: 64,
        },
      },
      runtimeConfig: {
        inference: {
          session: {
            prefillTokenChunkSize: null,
          },
        },
      },
      kvCache: {
        hasGPUCache: () => true,
      },
    },
    {},
    null
  );

  assert.equal(
    disabled,
    true,
    'manifest-owned token-chunked prefill must bypass command batching even when runtime default is null'
  );
}

{
  const disabled = shouldDisablePrefillCommandBatching(
    {
      kvCache: {
        hasGPUCache: () => true,
      },
    },
    {
      debugLayers: [0, 1],
    },
    null
  );

  assert.equal(
    disabled,
    true,
    'layer-targeted debug prefill must bypass command batching so buffer checks read committed outputs'
  );
}

console.log('multimodal-prefill-command-batching.test: ok');
