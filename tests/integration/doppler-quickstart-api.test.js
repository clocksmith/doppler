import assert from 'node:assert/strict';

import { doppler } from '../../src/index.js';
import { resolveLoadProgressHandlers } from '../../src/client/doppler-api.js';
import { getLogLevel, setLogLevel } from '../../src/debug/config.js';
import {
  buildQuickstartModelBaseUrl,
  listQuickstartModels,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';
import { resolveManifestArtifactSource } from '../../src/client/runtime/model-source.js';

assert.equal(typeof doppler, 'function');
assert.equal(typeof doppler.load, 'function');
assert.equal(typeof doppler.text, 'function');
assert.equal(typeof doppler.chat, 'function');
assert.equal(typeof doppler.chatText, 'function');
assert.equal(typeof doppler.evict, 'function');

{
  const models = await listQuickstartModels();
  assert.ok(models.some((entry) => entry.modelId === 'gemma-3-270m-it-q4k-ehf16-af32'));
}

{
  const resolved = await resolveQuickstartModel('gemma3-270m');
  assert.equal(resolved.modelId, 'gemma-3-270m-it-q4k-ehf16-af32');
  assert.ok(resolved.aliases.includes('google/gemma-3-270m-it'));
  assert.equal(
    buildQuickstartModelBaseUrl(resolved),
    `https://huggingface.co/${resolved.hf.repoId}/resolve/${resolved.hf.revision}/${resolved.hf.path}`
  );
}

{
  await assert.rejects(
    () => doppler.load('gemma-3-1b'),
    /Unknown quickstart model "gemma-3-1b"/
  );
}

{
  const storageManifest = {
    modelId: 'weights',
    artifactIdentity: {
      weightPackId: 'toy-wp',
      shardSetHash: 'sha256:toy-shards',
    },
    shards: [
      { index: 0, filename: 'shard_00000.bin', size: 4, hash: 'abcd', offset: 0 },
    ],
  };
  const storageManifestText = JSON.stringify(storageManifest);
  const digestBuffer = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(storageManifestText));
  const digest = Array.from(new Uint8Array(digestBuffer))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (String(url) === 'https://host/models/weights/manifest.json') {
      return {
        ok: true,
        text: async () => storageManifestText,
      };
    }
    return { ok: false, status: 404, text: async () => '' };
  };
  try {
    const resolved = await resolveManifestArtifactSource({
      modelId: 'variant',
      baseUrl: 'https://host/models/variant',
      manifest: null,
      trace: [],
    }, {
      text: '{}',
      manifest: {
        modelId: 'variant',
        artifactIdentity: {
          weightPackId: 'toy-wp',
        },
        weightsRef: {
          weightPackId: 'toy-wp',
          artifactRoot: '../weights',
          manifestDigest: `sha256:${digest}`,
          shardSetHash: 'sha256:toy-shards',
        },
      },
    });
    assert.equal(resolved.baseUrl, 'https://host/models/weights');
    assert.equal(resolved.storageBaseUrl, 'https://host/models/weights');
    assert.equal(resolved.manifest.modelId, 'variant');
    assert.equal(resolved.storageManifest.modelId, 'weights');
  } finally {
    globalThis.fetch = originalFetch;
  }
}

{
  const calls = [];
  const callback = (event) => calls.push(event);
  const resolved = resolveLoadProgressHandlers({ onProgress: callback });
  assert.equal(resolved.userProgress, callback);
  assert.equal(resolved.pipelineProgress, callback);
}

{
  const originalConsoleLog = console.log;
  const originalLogLevel = getLogLevel();
  const output = [];
  console.log = (...args) => output.push(args.join(' '));
  try {
    setLogLevel('info');
    const resolved = resolveLoadProgressHandlers({});
    assert.equal(typeof resolved.userProgress, 'function');
    assert.equal(resolved.pipelineProgress, null);
    resolved.userProgress({ phase: 'resolve', percent: 5, message: 'Resolving model' });
  } finally {
    setLogLevel(originalLogLevel);
    console.log = originalConsoleLog;
  }
  assert.ok(output.length >= 1);
  assert.ok(output.some((line) => /\[doppler\] Resolving model$/.test(line)));
}

console.log('doppler-quickstart-api.test: ok');
