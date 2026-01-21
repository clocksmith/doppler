import { describe, expect, it, vi } from 'vitest';

import {
  ShardPacker,
  sortTensorsByGroup,
  estimateShardCount,
} from '../../src/converter/shard-packer.js';
import { computeHash, createStreamingHasher } from '../../src/storage/shard-manager.js';
import { SHARD_SIZE } from '../../src/config/schema/index.js';


class MockShardIO {
  shards = new Map();
  writeCount = 0;

  async writeShard(index, data) {
    this.shards.set(index, new Uint8Array(data));
    this.writeCount++;
    return this.computeHash(data);
  }

  async computeHash(data) {
    // Simple hash for testing - just use length and first/last bytes
    const first = data.length > 0 ? data[0] : 0;
    const last = data.length > 0 ? data[data.length - 1] : 0;
    return `mock-${data.length}-${first}-${last}`;
  }

  getShard(index) {
    return this.shards.get(index);
  }

  getTotalBytes() {
    let total = 0;
    for (const shard of this.shards.values()) {
      total += shard.length;
    }
    return total;
  }
}

class MockStreamingShardIO {
  shards = new Map();

  async writeShard(index, data) {
    this.shards.set(index, new Uint8Array(data));
    return this.computeHash(data);
  }

  async computeHash(data) {
    const first = data.length > 0 ? data[0] : 0;
    const last = data.length > 0 ? data[data.length - 1] : 0;
    return `mock-${data.length}-${first}-${last}`;
  }

  async createShardWriter(index) {
    const chunks = [];
    let total = 0;
    return {
      write: async (chunk) => {
        const bytes = new Uint8Array(chunk);
        chunks.push(bytes);
        total += bytes.length;
      },
      close: async () => {
        const combined = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        this.shards.set(index, combined);
      },
      abort: async () => {
        chunks.length = 0;
      },
    };
  }

  createHasher() {
    let total = 0;
    let first = 0;
    let last = 0;
    let hasData = false;
    return {
      update: (chunk) => {
        const bytes = new Uint8Array(chunk);
        if (bytes.length === 0) return;
        if (!hasData) {
          first = bytes[0];
          hasData = true;
        }
        last = bytes[bytes.length - 1];
        total += bytes.length;
      },
      finalize: async () => new Uint8Array([
        total & 0xff,
        (total >> 8) & 0xff,
        first,
        last,
      ]),
    };
  }

  getShard(index) {
    return this.shards.get(index);
  }
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

class StreamingHasherIO {
  shards = new Map();
  lastShardHash = null;

  async writeShard(index, data) {
    this.shards.set(index, new Uint8Array(data));
    return this.computeHash(data);
  }

  async computeHash(data) {
    const bytes = new Uint8Array(data);
    let total = bytes.length;
    let last = bytes.length > 0 ? bytes[bytes.length - 1] : 0;
    const hash = new Uint8Array([total & 0xff, last & 0xff]);
    const hex = bytesToHex(hash);
    this.lastShardHash = hex;
    return hex;
  }

  async createShardWriter(index) {
    const chunks = [];
    let total = 0;
    return {
      write: async (chunk) => {
        const bytes = new Uint8Array(chunk);
        if (bytes.length === 0) return;
        chunks.push(bytes);
        total += bytes.length;
      },
      close: async () => {
        const combined = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        this.shards.set(index, combined);
      },
      abort: async () => {
        chunks.length = 0;
      },
    };
  }

  createHasher() {
    let total = 0;
    let last = 0;
    return {
      update: (chunk) => {
        const bytes = new Uint8Array(chunk);
        if (bytes.length === 0) return;
        total += bytes.length;
        last = bytes[bytes.length - 1];
      },
      finalize: async () => new Uint8Array([total & 0xff, last & 0xff]),
    };
  }
}

class HashingShardIO {
  shards = new Map();
  #algorithm;

  constructor(algorithm) {
    this.#algorithm = algorithm;
  }

  async createShardWriter(index) {
    const chunks = [];
    let total = 0;
    return {
      write: async (chunk) => {
        const bytes = new Uint8Array(chunk);
        if (bytes.length === 0) return;
        chunks.push(bytes);
        total += bytes.length;
      },
      close: async () => {
        const combined = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        this.shards.set(index, combined);
      },
      abort: async () => {
        chunks.length = 0;
      },
    };
  }

  createHasher() {
    return createStreamingHasher(this.#algorithm);
  }

  async computeHash(data) {
    return computeHash(data, this.#algorithm);
  }
}

const DEFAULT_PACKER_OPTIONS = {
  shardSize: 1024,
  hashAlgorithm: 'sha256',
  modelType: 'transformer',
};

function createPacker(io, overrides = {}) {
  return new ShardPacker(io, { ...DEFAULT_PACKER_OPTIONS, ...overrides });
}


function createTensor(name, size, shape = [size]) {
  return {
    name,
    shape,
    dtype: 'F16',
    size,
    getData: async () => {
      const data = new Uint8Array(size);
      // Fill with a pattern based on name for verification
      const seed = name.charCodeAt(0);
      for (let i = 0; i < size; i++) {
        data[i] = (seed + i) % 256;
      }
      return data;
    },
  };
}

function createStreamingTensor(name, size, shape = [size], chunkSize = 32) {
  const data = new Uint8Array(size);
  const seed = name.charCodeAt(0);
  for (let i = 0; i < size; i++) {
    data[i] = (seed + i) % 256;
  }

  return {
    name,
    shape,
    dtype: 'F16',
    size,
    getData: async () => data,
    getChunks: async function* () {
      for (let offset = 0; offset < size; offset += chunkSize) {
        const end = Math.min(offset + chunkSize, size);
        yield data.subarray(offset, end);
      }
    },
  };
}

describe('ShardPacker', () => {
  describe('pack', () => {
    it('packs small tensors into single shard', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('model.embed', 100),
        createTensor('model.layers.0.weight', 200),
        createTensor('model.head', 150),
      ];

      const result = await packer.pack(tensors);

      expect(result.tensorCount).toBe(3);
      expect(result.totalSize).toBe(450);
      expect(result.shards.length).toBe(1);
      expect(io.shards.size).toBe(1);
    });

    it('splits large tensor across multiple shards', async () => {
      const io = new MockShardIO();
      const shardSize = 100;
      const packer = createPacker(io, { shardSize });

      // Tensor larger than one shard
      const tensors = [createTensor('large.tensor', 250)];

      const result = await packer.pack(tensors);

      expect(result.shards.length).toBe(3); // 100 + 100 + 50
      expect(result.totalSize).toBe(250);

      // Check it's recorded as multi-span
      const loc = result.tensors['large.tensor'];
      expect('spans' in loc).toBe(true);
      expect(loc.spans.length).toBe(3);
      expect(loc.spans[0].size).toBe(100);
      expect(loc.spans[1].size).toBe(100);
      expect(loc.spans[2].size).toBe(50);
    });

    it('streams tensor chunks when available', async () => {
      const io = new MockStreamingShardIO();
      const shardSize = 64;
      const packer = createPacker(io, { shardSize });

      const tensors = [createStreamingTensor('stream.tensor', 150, [150], 40)];
      const result = await packer.pack(tensors);

      expect(result.shards.length).toBe(3);
      expect(io.getShard(0)?.length).toBe(64);
      expect(io.getShard(1)?.length).toBe(64);
      expect(io.getShard(2)?.length).toBe(22);
    });

    it('records single-shard tensor location correctly', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('tensor.a', 100),
        createTensor('tensor.b', 200),
      ];

      const result = await packer.pack(tensors);

      const locA = result.tensors['tensor.a'];
      expect('shard' in locA).toBe(true);
      expect(locA.shard).toBe(0);
      expect(locA.offset).toBe(0);
      expect(locA.size).toBe(100);

      const locB = result.tensors['tensor.b'];
      expect(locB.shard).toBe(0);
      expect(locB.offset).toBe(100);
      expect(locB.size).toBe(200);
    });

    it('records multi-span tensor location correctly', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io, { shardSize: 64 });

      const tensors = [createTensor('spanning.tensor', 150)];

      const result = await packer.pack(tensors);

      const loc = result.tensors['spanning.tensor'];
      expect('spans' in loc).toBe(true);
      expect(loc.size).toBe(150);
      expect(loc.dtype).toBe('F16');

      // Verify spans add up
      const totalFromSpans = loc.spans.reduce((sum, s) => sum + s.size, 0);
      expect(totalFromSpans).toBe(150);
    });

    it('classifies tensors into component groups', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('model.embed_tokens.weight', 100),
        createTensor('model.layers.0.self_attn.q_proj.weight', 100),
        createTensor('model.layers.1.mlp.gate_proj.weight', 100),
        createTensor('lm_head.weight', 100),
      ];

      const result = await packer.pack(tensors);

      expect(result.groups['embed']).toBeDefined();
      expect(result.groups['layer.0']).toBeDefined();
      expect(result.groups['layer.1']).toBeDefined();
      expect(result.groups['head']).toBeDefined();

      expect(result.groups['embed'].tensors).toContain('model.embed_tokens.weight');
      expect(result.groups['layer.0'].tensors).toContain('model.layers.0.self_attn.q_proj.weight');
      expect(result.groups['head'].tensors).toContain('lm_head.weight');
      expect(result.groups['embed'].hash).toBeTruthy();
    });

    it('calls progress callback for each tensor', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('tensor.a', 100),
        createTensor('tensor.b', 100),
        createTensor('tensor.c', 100),
      ];

      const progressCalls = [];

      await packer.pack(tensors, {
        onProgress: (current, total, name) => {
          progressCalls.push({ current, total, name });
        },
      });

      expect(progressCalls.length).toBe(3);
      expect(progressCalls[0]).toEqual({ current: 1, total: 3, name: 'tensor.a' });
      expect(progressCalls[1]).toEqual({ current: 2, total: 3, name: 'tensor.b' });
      expect(progressCalls[2]).toEqual({ current: 3, total: 3, name: 'tensor.c' });
    });

    it('throws on abort signal', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const controller = new AbortController();
      controller.abort();

      const tensors = [createTensor('tensor.a', 100)];

      await expect(packer.pack(tensors, { signal: controller.signal }))
        .rejects.toThrow('Aborted');
    });

    it('flushes final partial shard', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io, { shardSize: 1000 });

      // Total 500 bytes, less than shard size
      const tensors = [
        createTensor('tensor.a', 200),
        createTensor('tensor.b', 300),
      ];

      const result = await packer.pack(tensors);

      expect(result.shards.length).toBe(1);
      expect(io.shards.size).toBe(1);
      expect(io.getShard(0)?.length).toBe(500);
    });

    it('handles empty tensor list', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const result = await packer.pack([]);

      expect(result.tensorCount).toBe(0);
      expect(result.totalSize).toBe(0);
      expect(result.shards.length).toBe(0);
      expect(Object.keys(result.tensors).length).toBe(0);
    });

    it('requires shard size in constructor', async () => {
      const io = new MockShardIO();
      expect(() => new ShardPacker(io)).toThrow('Missing shard size for shard packer');
    });

    it('computes group hashes from streaming data', async () => {
      const io = new StreamingHasherIO();
      const packer = createPacker(io, { shardSize: 256 });

      const tensorA = createStreamingTensor('layers.0.weight', 16, [16], 8);
      const tensorB = createStreamingTensor('layers.0.bias', 8, [8], 8);

      const result = await packer.pack([tensorA, tensorB]);
      const group = result.groups['layer.0'];

      const combined = new Uint8Array(24);
      combined.set(await tensorA.getData(), 0);
      combined.set(await tensorB.getData(), 16);
      const expected = bytesToHex(new Uint8Array([combined.length & 0xff, combined[combined.length - 1] & 0xff]));

      expect(group.hash).toBe(expected);
    });

    it('records shard hash from streaming hasher', async () => {
      const io = new StreamingHasherIO();
      const packer = createPacker(io, { shardSize: 256 });

      const tensor = createStreamingTensor('layers.1.weight', 20, [20], 10);
      const result = await packer.pack([tensor]);

      expect(result.shards.length).toBe(1);
      expect(result.shards[0].hash).toBe(io.lastShardHash);
    });

    it('matches group hash with computeHash', async () => {
      const io = new HashingShardIO('blake3');
      const packer = createPacker(io, { shardSize: 256, hashAlgorithm: 'blake3' });

      const tensorA = createStreamingTensor('model.layers.0.weight', 16, [16], 8);
      const tensorB = createStreamingTensor('model.layers.0.bias', 8, [8], 8);

      const result = await packer.pack([tensorA, tensorB]);
      const group = result.groups['layer.0'];

      const combined = new Uint8Array(24);
      combined.set(await tensorA.getData(), 0);
      combined.set(await tensorB.getData(), 16);
      const expected = await computeHash(combined, 'blake3');

      expect(group.hash).toBe(expected);
    });

    it('matches shard hashes with computeHash', async () => {
      const io = new HashingShardIO('blake3');
      const packer = createPacker(io, { shardSize: 32, hashAlgorithm: 'blake3' });

      const tensor = createStreamingTensor('model.layers.1.weight', 64, [64], 16);
      const result = await packer.pack([tensor]);

      for (const shard of result.shards) {
        const data = io.shards.get(shard.index);
        const expected = await computeHash(data, 'blake3');
        expect(shard.hash).toBe(expected);
      }
    });

    it('groups MoE expert tensors', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io, { shardSize: 256 });

      const tensors = [
        createTensor('layers.2.experts.3.w1.weight', 4, [4]),
        createTensor('layers.2.shared_expert.w1.weight', 4, [4]),
        createTensor('layers.2.block_sparse_moe.gate.weight', 4, [4]),
      ];

      const result = await packer.pack(tensors);
      expect(Object.keys(result.groups)).toEqual(
        expect.arrayContaining(['layer.2.expert.3', 'layer.2.shared_expert', 'layer.2.shared'])
      );
    });
  });

  describe('buildGroups', () => {
    it('creates embed/layer/head groups', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('model.embed_tokens.weight', 100),
        createTensor('model.layers.0.weight', 100),
        createTensor('lm_head.weight', 100),
      ];

      const result = await packer.pack(tensors);

      expect(result.groups['embed'].type).toBe('embed');
      expect(result.groups['layer.0'].type).toBe('layer');
      expect(result.groups['head'].type).toBe('head');
    });

    it('extracts layer indices from group IDs', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      const tensors = [
        createTensor('model.layers.5.weight', 100),
        createTensor('model.layers.10.weight', 100),
      ];

      const result = await packer.pack(tensors);

      expect(result.groups['layer.5'].layerIndex).toBe(5);
      expect(result.groups['layer.10'].layerIndex).toBe(10);
    });

    it('extracts expert indices for MoE', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io, { modelType: 'mixtral' });

      const tensors = [
        createTensor('model.layers.0.block_sparse_moe.experts.0.w1.weight', 100),
        createTensor('model.layers.0.block_sparse_moe.experts.3.w1.weight', 100),
      ];

      const result = await packer.pack(tensors);

      expect(result.groups['layer.0.expert.0']).toBeDefined();
      expect(result.groups['layer.0.expert.0'].expertIndex).toBe(0);
      expect(result.groups['layer.0.expert.3'].expertIndex).toBe(3);
    });

    it('tracks which shards each group spans', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io, { shardSize: 100 });

      const tensors = [
        createTensor('model.embed_tokens.weight', 250), // Spans 3 shards
      ];

      const result = await packer.pack(tensors);

      expect(result.groups['embed'].shards).toEqual([0, 1, 2]);
    });
  });

  describe('reset', () => {
    it('clears state for reuse', async () => {
      const io = new MockShardIO();
      const packer = createPacker(io);

      // First pack
      await packer.pack([createTensor('first', 100)]);

      // Reset
      packer.reset();

      // Second pack should start fresh
      const result = await packer.pack([createTensor('second', 200)]);

      expect(result.tensorCount).toBe(1);
      expect(Object.keys(result.tensors)).toEqual(['second']);
    });
  });
});

describe('sortTensorsByGroup', () => {
  it('orders embed before layers before head', () => {
    const tensors = [
      { name: 'lm_head.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.layers.0.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.embed_tokens.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
    ];

    const sorted = sortTensorsByGroup(tensors);

    expect(sorted[0].name).toBe('model.embed_tokens.weight');
    expect(sorted[1].name).toBe('model.layers.0.weight');
    expect(sorted[2].name).toBe('lm_head.weight');
  });

  it('orders layers numerically', () => {
    const tensors = [
      { name: 'model.layers.10.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.layers.2.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.layers.1.weight', shape: [100], dtype: 'F16', size: 100, offset: 0 },
    ];

    const sorted = sortTensorsByGroup(tensors);

    expect(sorted[0].name).toBe('model.layers.1.weight');
    expect(sorted[1].name).toBe('model.layers.2.weight');
    expect(sorted[2].name).toBe('model.layers.10.weight');
  });

  it('keeps tensors in same group together', () => {
    const tensors = [
      { name: 'model.layers.0.q_proj', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.layers.1.k_proj', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'model.layers.0.v_proj', shape: [100], dtype: 'F16', size: 100, offset: 0 },
    ];

    const sorted = sortTensorsByGroup(tensors);

    // layer.0 tensors should come before layer.1
    const layer0Indices = sorted
      .map((t, i) => t.name.includes('layers.0') ? i : -1)
      .filter(i => i >= 0);
    const layer1Indices = sorted
      .map((t, i) => t.name.includes('layers.1') ? i : -1)
      .filter(i => i >= 0);

    // All layer.0 indices should be less than layer.1 indices
    expect(Math.max(...layer0Indices)).toBeLessThan(Math.min(...layer1Indices));
  });
});

describe('estimateShardCount', () => {
  it('calculates correct shard count for tensor set', () => {
    const tensors = [
      { name: 'a', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'b', shape: [100], dtype: 'F16', size: 200, offset: 0 },
      { name: 'c', shape: [100], dtype: 'F16', size: 300, offset: 0 },
    ];

    // Total: 600 bytes, default shard size 64MB
    const count = estimateShardCount(tensors, SHARD_SIZE);
    expect(count).toBe(1); // All fit in one shard
  });

  it('handles custom shard size', () => {
    const tensors = [
      { name: 'a', shape: [100], dtype: 'F16', size: 500, offset: 0 },
    ];

    const count = estimateShardCount(tensors, 100);
    expect(count).toBe(5); // 500 / 100 = 5 shards
  });

  it('rounds up partial shards', () => {
    const tensors = [
      { name: 'a', shape: [100], dtype: 'F16', size: 150, offset: 0 },
    ];

    const count = estimateShardCount(tensors, 100);
    expect(count).toBe(2); // 150 / 100 = 1.5 -> 2
  });

  it('returns zero for empty tensor list', () => {
    const count = estimateShardCount([], SHARD_SIZE);
    expect(count).toBe(0);
  });
});
