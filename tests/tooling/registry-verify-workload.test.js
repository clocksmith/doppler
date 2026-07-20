import assert from 'node:assert/strict';

import {
  buildRegistryVerifyRequest,
  parseArgs,
  resolveRegistryVerifyRuntimeConfig,
  resolveRegistryVerifyRuntimeProfile,
  resolveRegistryVerifyWorkload,
} from '../../tools/run-registry-verify.js';

const hf = {
  repoId: 'clocksmith/rdrr',
  revision: 'abc123',
  path: 'models/example',
};

{
  assert.equal(resolveRegistryVerifyWorkload({
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    modes: ['embedding'],
  }), 'embedding');
  assert.equal(resolveRegistryVerifyRuntimeProfile({
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    modes: ['embedding'],
  }), null);
}

{
  assert.equal(resolveRegistryVerifyWorkload({
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modes: ['text', 'vision'],
  }), 'inference');
  assert.equal(resolveRegistryVerifyRuntimeProfile({
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modes: ['text', 'vision'],
  }), null);
}

{
  assert.equal(resolveRegistryVerifyWorkload({
    modelId: 'qwen-3-reranker-0-6b-f16-af32',
    modes: ['rerank'],
  }), 'rerank');
  assert.equal(resolveRegistryVerifyRuntimeProfile({
    modelId: 'qwen-3-reranker-0-6b-f16-af32',
    modes: ['rerank'],
    verify: {
      runtimeProfile: 'profiles/rerank-stability',
    },
  }), 'profiles/rerank-stability');
}

{
  const runtimeConfig = {
    inference: {
      prompt: 'A compact support prompt.',
    },
  };
  const request = buildRegistryVerifyRequest({
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    modes: ['embedding'],
    hf,
    verify: {
      runtimeConfig,
    },
  });

  assert.equal(request.workload, 'embedding');
  assert.equal(request.runtimeProfile, undefined);
  assert.deepEqual(request.runtimeConfig, runtimeConfig);
  assert.equal(request.modelId, 'google-embeddinggemma-300m-q4k-ehf16-af32');
  assert.equal(
    request.modelUrl,
    'https://huggingface.co/clocksmith/rdrr/resolve/abc123/models/example'
  );
}

{
  const runtimeConfig = resolveRegistryVerifyRuntimeConfig({
    modelId: 'qwen-3-reranker-0-6b-f16-af32',
    verify: {
      runtimeConfig: {
        inference: {
          rerank: {
            query: 'Which API exposes browser GPU compute?',
            documents: [
              'WebGPU exposes GPU compute.',
              'WebSocket streams network messages.',
            ],
          },
        },
      },
    },
  });
  assert.deepEqual(runtimeConfig?.inference?.rerank?.documents, [
    'WebGPU exposes GPU compute.',
    'WebSocket streams network messages.',
  ]);
}

{
  assert.throws(
    () => buildRegistryVerifyRequest({
      modelId: 'qwen-3-reranker-0-6b-f16-af32',
      modes: ['rerank'],
      hf,
    }),
    /rerank registry verify requires verify\.runtimeConfig\.inference\.rerank/
  );
}

{
  const args = parseArgs(['--help']);
  assert.equal(args.help, true);
  assert.equal(args.model, '');
}

console.log('registry-verify-workload.test: ok');
