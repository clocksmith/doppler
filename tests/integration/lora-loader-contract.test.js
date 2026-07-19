import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

import { loadLoRAFromManifest, loadLoRAWeights } from '../../src/experimental/adapters/lora-loader.js';

const originalFetch = globalThis.fetch;

function createManifest(overrides = {}) {
  return {
    id: 'test_adapter',
    name: 'Test Adapter',
    baseModel: 'gemma-3-1b-it-q4k-ehf16-af32',
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

function createSafetensors(tensors) {
  let offset = 0;
  const header = {};
  const chunks = [];
  for (const tensor of tensors) {
    const values = Float32Array.from(tensor.values);
    const data = Buffer.from(values.buffer);
    header[tensor.name] = {
      dtype: 'F32',
      shape: tensor.shape,
      data_offsets: [offset, offset + data.byteLength],
    };
    offset += data.byteLength;
    chunks.push(data);
  }
  const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.byteLength), 0);
  return Buffer.concat([prefix, headerBytes, ...chunks]);
}

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
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

await assert.rejects(
  () => loadLoRAFromManifest(createManifest({
    tensors: [
      {
        name: 'layers.0.q_proj.not_lora',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
    ],
  })),
  /Unrecognized LoRA tensor name/
);

await assert.rejects(
  () => loadLoRAFromManifest(createManifest({
    tensors: [
      {
        name: 'layers.0.q_proj.lora_a',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
    ],
  })),
  /is incomplete; both lora_a and lora_b tensors are required/
);

await assert.rejects(
  () => loadLoRAFromManifest(createManifest({
    tensors: [
      {
        name: 'layers.0.k_proj.lora_a',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
      {
        name: 'layers.0.k_proj.lora_b',
        shape: [1, 1],
        dtype: 'f32',
        data: [1],
      },
    ],
  })),
  /contains module k_proj outside targetModules/
);

await assert.rejects(
  () => loadLoRAFromManifest(createManifest({
    targetModules: ['q_proj', 'k_proj'],
  })),
  /declares k_proj, but no complete tensors were loaded/
);

{
  const weights = createSafetensors([
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight',
      shape: [1, 2],
      values: [1, 2],
    },
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight',
      shape: [3, 1],
      values: [3, 4, 5],
    },
  ]);
  const adapter = await loadLoRAFromManifest(createManifest({
    tensors: undefined,
    weightsPath: 'adapter.safetensors',
    checksum: sha256Hex(weights),
  }), {
    async readFile(filePath) {
      assert.equal(filePath, 'adapter.safetensors');
      return weights;
    },
  });
  const qProj = adapter.layers.get(0).q_proj;
  assert.deepEqual(Array.from(qProj.a), [1, 2]);
  assert.deepEqual(Array.from(qProj.b), [3, 4, 5]);
  assert.equal(qProj.scale, 1);

  const unchecked = await loadLoRAWeights(createManifest({
    tensors: undefined,
    weightsPath: 'adapter.safetensors',
    checksum: '0'.repeat(64),
  }), {
    skipVerify: true,
    async readFile(filePath) {
      assert.equal(filePath, 'adapter.safetensors');
      return weights;
    },
  });
  assert.equal(unchecked.checksumValid, undefined);
}

{
  const weights = createSafetensors([
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight',
      shape: [1, 2],
      values: [1, 2],
    },
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight',
      shape: [3, 1],
      values: [3, 4, 5],
    },
  ]);
  const manifest = createManifest({
    tensors: undefined,
    weightsPath: 'adapter.safetensors',
    checksum: sha256Hex(weights),
  });
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    assert.equal(url, 'https://example.test/adapters/runtime-adapter-manifest.json');
    return new Response(JSON.stringify(manifest), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    let opfsReads = 0;
    const result = await loadLoRAWeights(
      'https://example.test/adapters/runtime-adapter-manifest.json',
      {
        async readOPFS() {
          opfsReads += 1;
          throw new DOMException('missing', 'NotFoundError');
        },
        async fetchUrl(url) {
          assert.equal(url, 'https://example.test/adapters/adapter.safetensors');
          return weights;
        },
      },
    );
    assert.equal(opfsReads, 0);
    assert.deepEqual(Array.from(result.adapter.layers.get(0).q_proj.a), [1, 2]);
    assert.deepEqual(Array.from(result.adapter.layers.get(0).q_proj.b), [3, 4, 5]);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

console.log('lora-loader-contract.test: ok');
