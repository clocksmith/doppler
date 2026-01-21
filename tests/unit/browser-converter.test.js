import { describe, it, expect, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../src/storage/backends/opfs-store.js', async () => {
  const { createMemoryStore } = await import('../../src/storage/backends/memory-store.js');
  return {
    createOpfsStore: () => createMemoryStore({ maxBytes: 256 * 1024 * 1024 }),
  };
});

vi.mock('../../src/storage/quota.js', () => ({
  isOPFSAvailable: () => true,
  isIndexedDBAvailable: () => false,
  isStorageAPIAvailable: () => true,
  requestPersistence: async () => ({ granted: false, reason: 'mock' }),
  checkSpaceAvailable: async () => ({ hasSpace: true, info: { available: 1024 * 1024 * 1024 }, shortfall: 0 }),
  QuotaExceededError: class QuotaExceededError extends Error {
    constructor(required, available) {
      super(`Insufficient storage: need ${required}, have ${available}`);
      this.name = 'QuotaExceededError';
      this.required = required;
      this.available = available;
      this.shortfall = required - available;
    }
  },
}));

import { convertModel } from '../../src/browser/browser-converter.js';
import { createConverterConfig } from '../../src/config/index.js';
import { openModelStore, loadManifestFromStore, verifyIntegrity, cleanup } from '../../src/storage/shard-manager.js';
import { parseManifest } from '../../src/storage/rdrr-format.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const tokenizerPath = join(__dirname, '..', 'fixtures', 'mini-model', 'tokenizer.json');
const tokenizerJson = readFileSync(tokenizerPath, 'utf8');
const ggufPath = join(__dirname, '..', 'fixtures', 'sample.gguf');
const ggufBytes = readFileSync(ggufPath);

function createFileLike(name, bytes) {
  const buffer = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return {
    name,
    size: buffer.byteLength,
    slice: (start, end) => new Blob([buffer.slice(start, end)]),
    arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
    text: async () => new TextDecoder().decode(buffer),
  };
}

function createSafetensorsBuffer(tensors, metadata = {}) {
  let offset = 0;
  const header = { __metadata__: metadata };

  for (const tensor of tensors) {
    const size = tensor.data.byteLength;
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + size],
    };
    offset += size;
  }

  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const totalSize = 8 + headerBytes.length + offset;
  const buffer = new Uint8Array(totalSize);
  const view = new DataView(buffer.buffer);
  view.setUint32(0, headerBytes.length, true);
  view.setUint32(4, 0, true);
  buffer.set(headerBytes, 8);

  let dataOffset = 8 + headerBytes.length;
  let cursor = 0;
  for (const tensor of tensors) {
    buffer.set(tensor.data, dataOffset + cursor);
    cursor += tensor.data.byteLength;
  }

  return buffer;
}

function createF16Data(elements, seed = 1) {
  const bytes = new Uint8Array(elements * 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = (seed + i) & 0xff;
  }
  return bytes;
}

describe('browser converter', () => {
  afterEach(async () => {
    await cleanup();
  });

  it('converts a small safetensors model and verifies integrity', async () => {
    const tensors = [
      {
        name: 'model.embed_tokens.weight',
        dtype: 'F16',
        shape: [4, 4],
        data: createF16Data(16, 3),
      },
      {
        name: 'model.layers.0.self_attn.q_proj.weight',
        dtype: 'F16',
        shape: [4, 4],
        data: createF16Data(16, 7),
      },
    ];

    const safetensorsBuffer = createSafetensorsBuffer(tensors, {
      format: 'pt',
      model_type: 'qwen2',
    });

    const configJson = JSON.stringify({
      model_type: 'qwen2',
      num_hidden_layers: 2,
      hidden_size: 32,
      intermediate_size: 64,
      num_attention_heads: 4,
      vocab_size: 128,
      max_position_embeddings: 256,
    });

    const files = [
      createFileLike('model.safetensors', safetensorsBuffer),
      createFileLike('config.json', new TextEncoder().encode(configJson)),
      createFileLike('tokenizer.json', new TextEncoder().encode(tokenizerJson)),
    ];

    const converterConfig = createConverterConfig({
      manifest: { hashAlgorithm: 'blake3' },
      sharding: { shardSizeBytes: 1024 },
    });

    const modelId = await convertModel(files, { converterConfig });
    expect(modelId).toBeTruthy();

    await openModelStore(modelId);
    const manifestText = await loadManifestFromStore();
    expect(manifestText).toBeTruthy();

    const manifest = parseManifest(manifestText);
    expect(manifest.tokenizer?.file).toBe('tokenizer.json');
    expect(manifest.hashAlgorithm).toBe('blake3');

    const integrity = await verifyIntegrity();
    expect(integrity.valid).toBe(true);
  });

  it('converts a small gguf model with preset override', async () => {
    const files = [
      createFileLike('model.gguf', ggufBytes),
      createFileLike('tokenizer.json', new TextEncoder().encode(tokenizerJson)),
    ];

    const converterConfig = createConverterConfig({
      manifest: { hashAlgorithm: 'blake3' },
      presets: { model: 'llama3' },
    });

    const modelId = await convertModel(files, { converterConfig });
    expect(modelId).toBeTruthy();

    await openModelStore(modelId);
    const manifestText = await loadManifestFromStore();
    expect(manifestText).toBeTruthy();

    const manifest = parseManifest(manifestText);
    expect(manifest.inference?.presetId).toBe('llama3');
  });

  it('rejects tokenizer.model in browser conversion', async () => {
    const files = [
      createFileLike('model.gguf', ggufBytes),
      createFileLike('tokenizer.model', new Uint8Array([1, 2, 3])),
    ];

    await expect(convertModel(files)).rejects.toThrow('tokenizer.model is not supported');
  });

  it('rejects Q4_K_M column layout in browser conversion', async () => {
    const tensors = [
      {
        name: 'model.embed_tokens.weight',
        dtype: 'F16',
        shape: [4, 4],
        data: createF16Data(16, 3),
      },
      {
        name: 'model.layers.0.self_attn.q_proj.weight',
        dtype: 'F16',
        shape: [4, 4],
        data: createF16Data(16, 7),
      },
    ];

    const safetensorsBuffer = createSafetensorsBuffer(tensors, {
      format: 'pt',
      model_type: 'qwen2',
    });

    const configJson = JSON.stringify({
      model_type: 'qwen2',
      num_hidden_layers: 2,
      hidden_size: 32,
      intermediate_size: 64,
      num_attention_heads: 4,
      vocab_size: 128,
      max_position_embeddings: 256,
    });

    const files = [
      createFileLike('model.safetensors', safetensorsBuffer),
      createFileLike('config.json', new TextEncoder().encode(configJson)),
      createFileLike('tokenizer.json', new TextEncoder().encode(tokenizerJson)),
    ];

    const converterConfig = createConverterConfig({
      quantization: { weights: 'q4k', q4kLayout: 'col' },
      sharding: { shardSizeBytes: 1024 },
    });

    await expect(convertModel(files, { converterConfig })).rejects.toThrow('Column-wise Q4_K_M');
  });
});
