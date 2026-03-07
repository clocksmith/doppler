import assert from 'node:assert/strict';

import { loadLoRAWeights } from '../../src/adapters/lora-loader.js';

const originalFetch = globalThis.fetch;

function createManifest(overrides = {}) {
  return {
    id: 'test_adapter',
    name: 'Test Adapter',
    baseModel: 'gemma-3-1b-it-wq4k-ef16-hf16',
    rank: 1,
    alpha: 1,
    targetModules: ['q_proj'],
    tensors: [
      {
        name: 'layers.0.q_proj.lora_a',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
      {
        name: 'layers.0.q_proj.lora_b',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
    ],
    ...overrides,
  };
}

try {
  globalThis.fetch = async (url) => {
    if (String(url) === 'https://example.test/adapter.json') {
      return new Response(JSON.stringify(createManifest({
        checksum: '0'.repeat(64),
        checksumAlgorithm: 'sha256',
      })), {
        status: 200,
        headers: {
          'content-type': 'application/json',
        },
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };

  await assert.rejects(
    () => loadLoRAWeights('https://example.test/adapter.json'),
    /LoRA checksum mismatch/
  );

  globalThis.fetch = async (url) => {
    if (String(url) === 'https://example.test/adapter.json') {
      return new Response(JSON.stringify(createManifest({
        checksum: '0'.repeat(64),
        checksumAlgorithm: 'blake3',
      })), {
        status: 200,
        headers: {
          'content-type': 'application/json',
        },
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };

  await assert.rejects(
    () => loadLoRAWeights('https://example.test/adapter.json'),
    /Unsupported LoRA checksum algorithm/
  );
} finally {
  globalThis.fetch = originalFetch;
}

console.log('lora-loader-contract.test: ok');
