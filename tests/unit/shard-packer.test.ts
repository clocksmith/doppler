import { describe, expect, it, vi } from 'vitest';

import {
  ShardPacker,
  sortTensorsByGroup,
  estimateShardCount,
  type ShardIO,
  type PackerTensorInput,
  type TensorLocationSingle,
  type TensorLocationMulti,
} from '../../src/converter/shard-packer.js';
import type { TensorInfoSchema } from '../../src/config/schema/index.js';

/**
 * In-memory mock implementation of ShardIO for testing.
 */
class MockShardIO implements ShardIO {
  shards = new Map<number, Uint8Array>();
  writeCount = 0;

  async writeShard(index: number, data: Uint8Array): Promise<string> {
    this.shards.set(index, new Uint8Array(data));
    this.writeCount++;
    return this.computeHash(data);
  }

  async computeHash(data: Uint8Array): Promise<string> {
    // Simple hash for testing - just use length and first/last bytes
    const first = data.length > 0 ? data[0] : 0;
    const last = data.length > 0 ? data[data.length - 1] : 0;
    return `mock-${data.length}-${first}-${last}`;
  }

  getShard(index: number): Uint8Array | undefined {
    return this.shards.get(index);
  }

  getTotalBytes(): number {
    let total = 0;
    for (const shard of this.shards.values()) {
      total += shard.length;
    }
    return total;
  }
}

/**
 * Create a test tensor with specified size filled with a pattern.
 */
function createTensor(name: string, size: number, shape: number[] = [size]): PackerTensorInput {
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

describe('ShardPacker', () => {
  describe('pack', () => {
    it('packs small tensors into single shard', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1024 });

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
      const packer = new ShardPacker(io, { shardSize });

      // Tensor larger than one shard
      const tensors = [createTensor('large.tensor', 250)];

      const result = await packer.pack(tensors);

      expect(result.shards.length).toBe(3); // 100 + 100 + 50
      expect(result.totalSize).toBe(250);

      // Check it's recorded as multi-span
      const loc = result.tensors['large.tensor'] as TensorLocationMulti;
      expect('spans' in loc).toBe(true);
      expect(loc.spans.length).toBe(3);
      expect(loc.spans[0].size).toBe(100);
      expect(loc.spans[1].size).toBe(100);
      expect(loc.spans[2].size).toBe(50);
    });

    it('records single-shard tensor location correctly', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1024 });

      const tensors = [
        createTensor('tensor.a', 100),
        createTensor('tensor.b', 200),
      ];

      const result = await packer.pack(tensors);

      const locA = result.tensors['tensor.a'] as TensorLocationSingle;
      expect('shard' in locA).toBe(true);
      expect(locA.shard).toBe(0);
      expect(locA.offset).toBe(0);
      expect(locA.size).toBe(100);

      const locB = result.tensors['tensor.b'] as TensorLocationSingle;
      expect(locB.shard).toBe(0);
      expect(locB.offset).toBe(100);
      expect(locB.size).toBe(200);
    });

    it('records multi-span tensor location correctly', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 64 });

      const tensors = [createTensor('spanning.tensor', 150)];

      const result = await packer.pack(tensors);

      const loc = result.tensors['spanning.tensor'] as TensorLocationMulti;
      expect('spans' in loc).toBe(true);
      expect(loc.size).toBe(150);
      expect(loc.dtype).toBe('F16');

      // Verify spans add up
      const totalFromSpans = loc.spans.reduce((sum, s) => sum + s.size, 0);
      expect(totalFromSpans).toBe(150);
    });

    it('classifies tensors into component groups', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1024 });

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
    });

    it('calls progress callback for each tensor', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1024 });

      const tensors = [
        createTensor('tensor.a', 100),
        createTensor('tensor.b', 100),
        createTensor('tensor.c', 100),
      ];

      const progressCalls: Array<{ current: number; total: number; name: string }> = [];

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
      const packer = new ShardPacker(io, { shardSize: 1024 });

      const controller = new AbortController();
      controller.abort();

      const tensors = [createTensor('tensor.a', 100)];

      await expect(packer.pack(tensors, { signal: controller.signal }))
        .rejects.toThrow('Aborted');
    });

    it('flushes final partial shard', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1000 });

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
      const packer = new ShardPacker(io, { shardSize: 1024 });

      const result = await packer.pack([]);

      expect(result.tensorCount).toBe(0);
      expect(result.totalSize).toBe(0);
      expect(result.shards.length).toBe(0);
      expect(Object.keys(result.tensors).length).toBe(0);
    });

    it('uses default shard size when not specified', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io); // No options

      // Create tensor smaller than default 64MB
      const tensors = [createTensor('small', 1000)];

      const result = await packer.pack(tensors);

      expect(result.shards.length).toBe(1);
    });
  });

  describe('buildGroups', () => {
    it('creates embed/layer/head groups', async () => {
      const io = new MockShardIO();
      const packer = new ShardPacker(io, { shardSize: 1024 });

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
      const packer = new ShardPacker(io, { shardSize: 1024 });

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
      const packer = new ShardPacker(io, { shardSize: 1024, modelType: 'mixtral' });

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
      const packer = new ShardPacker(io, { shardSize: 100 });

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
      const packer = new ShardPacker(io, { shardSize: 1024 });

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
    const tensors: TensorInfoSchema[] = [
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
    const tensors: TensorInfoSchema[] = [
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
    const tensors: TensorInfoSchema[] = [
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
    const tensors: TensorInfoSchema[] = [
      { name: 'a', shape: [100], dtype: 'F16', size: 100, offset: 0 },
      { name: 'b', shape: [100], dtype: 'F16', size: 200, offset: 0 },
      { name: 'c', shape: [100], dtype: 'F16', size: 300, offset: 0 },
    ];

    // Total: 600 bytes, default shard size 64MB
    const count = estimateShardCount(tensors);
    expect(count).toBe(1); // All fit in one shard
  });

  it('handles custom shard size', () => {
    const tensors: TensorInfoSchema[] = [
      { name: 'a', shape: [100], dtype: 'F16', size: 500, offset: 0 },
    ];

    const count = estimateShardCount(tensors, 100);
    expect(count).toBe(5); // 500 / 100 = 5 shards
  });

  it('rounds up partial shards', () => {
    const tensors: TensorInfoSchema[] = [
      { name: 'a', shape: [100], dtype: 'F16', size: 150, offset: 0 },
    ];

    const count = estimateShardCount(tensors, 100);
    expect(count).toBe(2); // 150 / 100 = 1.5 -> 2
  });

  it('returns zero for empty tensor list', () => {
    const count = estimateShardCount([]);
    expect(count).toBe(0);
  });
});
