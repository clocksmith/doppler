import assert from 'node:assert/strict';

import {
  buildRegistryVerifyRequest,
  parseArgs,
  resolveRegistryVerifyRuntimeProfile,
  resolveRegistryVerifyWorkload,
} from '../../tools/run-registry-verify.js';

const hf = {
  repoId: 'Clocksmith/rdrr',
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
  }), 'profiles/vector-stability');
}

{
  assert.equal(resolveRegistryVerifyWorkload({
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modes: ['text', 'vision'],
  }), 'inference');
  assert.equal(resolveRegistryVerifyRuntimeProfile({
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modes: ['text', 'vision'],
  }), 'profiles/verbose-trace');
}

{
  const request = buildRegistryVerifyRequest({
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    modes: ['embedding'],
    hf,
  });

  assert.equal(request.workload, 'embedding');
  assert.equal(request.runtimeProfile, 'profiles/vector-stability');
  assert.equal(request.modelId, 'google-embeddinggemma-300m-q4k-ehf16-af32');
  assert.equal(
    request.modelUrl,
    'https://huggingface.co/Clocksmith/rdrr/resolve/abc123/models/example'
  );
}

{
  const args = parseArgs(['--help']);
  assert.equal(args.help, true);
  assert.equal(args.model, '');
}

console.log('registry-verify-workload.test: ok');
