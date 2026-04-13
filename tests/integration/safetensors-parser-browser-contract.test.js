import assert from 'node:assert/strict';

import { parseSafetensorsSharded } from '../../src/experimental/browser/safetensors-parser-browser.js';

function createSafetensorsBytes(tensorMap) {
  const headerObject = {};
  let currentOffset = 0;
  for (const [name, spec] of Object.entries(tensorMap)) {
    const size = spec.size ?? 2;
    headerObject[name] = {
      dtype: spec.dtype ?? 'F16',
      shape: spec.shape ?? [1],
      data_offsets: [currentOffset, currentOffset + size],
    };
    currentOffset += size;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(headerObject));
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const combined = new Uint8Array(8 + headerBytes.byteLength + currentOffset);
  combined.set(new Uint8Array(prefix), 0);
  combined.set(headerBytes, 8);
  return combined.buffer;
}

function createTensorSource(name, tensorMap) {
  const bytes = createSafetensorsBytes(tensorMap);
  return {
    name,
    size: bytes.byteLength,
    async readRange(offset, length) {
      const start = Math.max(0, offset);
      const end = Math.min(start + length, bytes.byteLength);
      return bytes.slice(start, end);
    },
  };
}

await assert.rejects(
  () => parseSafetensorsSharded(
    [
      createTensorSource('model-00001-of-00002.safetensors', {
        'model.embed_tokens.weight': {},
      }),
    ],
    {
      metadata: {},
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00002.safetensors',
        'lm_head.weight': 'model-00002-of-00002.safetensors',
      },
    }
  ),
  /missing indexed shard files: model-00002-of-00002\.safetensors/
);

await assert.rejects(
  () => parseSafetensorsSharded(
    [
      createTensorSource('model-00001-of-00001.safetensors', {
        'model.embed_tokens.weight': {},
      }),
      createTensorSource('model-extra.safetensors', {
        'lm_head.weight': {},
      }),
    ],
    {
      metadata: {},
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00001.safetensors',
      },
    }
  ),
  /received shard files not referenced by index JSON: model-extra\.safetensors/
);

console.log('safetensors-parser-browser-contract.test: ok');
