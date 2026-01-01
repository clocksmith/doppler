import { describe, expect, it } from 'vitest';

import { Tokenizer } from '../../src/inference/tokenizer.js';
import { MultiModelNetwork } from '../../src/inference/multi-model-network.js';
import { MultiPipelinePool } from '../../src/inference/multi-pipeline-pool.js';
import type { InferencePipeline, KVCacheSnapshot } from '../../src/inference/pipeline.js';
import type { MultiModelLoader } from '../../src/loader/multi-model-loader.js';
import type { NetworkGenome } from '../../src/inference/network-evolution.js';
import type { LoRAAdapter } from '../../src/inference/pipeline/lora.js';

class FakePipeline {
  adapterName = 'base';
  unloaded = false;

  setLoRAAdapter(adapter: LoRAAdapter | null): void {
    this.adapterName = adapter?.name || 'base';
  }

  async *generate(prompt: string): AsyncGenerator<string> {
    yield `${this.adapterName}:${prompt}`;
  }

  async *generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string): AsyncGenerator<string> {
    yield `${this.adapterName}:${prefix.tokens.join(',')}:${prompt}`;
  }

  async prefillKVOnly(prompt: string): Promise<KVCacheSnapshot> {
    return {
      cache: {} as KVCacheSnapshot['cache'],
      seqLen: prompt.length,
      tokens: [prompt.length],
    };
  }

  async unload(): Promise<void> {
    this.unloaded = true;
  }
}

class FakeLoader {
  adapters = new Map<string, LoRAAdapter>();

  getAdapter(name: string): LoRAAdapter | null {
    return this.adapters.get(name) || null;
  }

  async createSharedPipeline(): Promise<FakePipeline> {
    return new FakePipeline();
  }
}

const createAdapter = (name: string): LoRAAdapter => ({
  name,
  rank: 1,
  alpha: 1,
  layers: new Map(),
});

const collect = async (generator: AsyncGenerator<string>): Promise<string> => {
  const chunks: string[] = [];
  for await (const chunk of generator) {
    chunks.push(chunk);
  }
  return chunks.join('');
};

describe('golden path', () => {
  it('tokenizes, runs a prompt, and unloads', async () => {
    const tokenizer = new Tokenizer();
    await tokenizer.initialize({
      tokenizer: {
        vocab: { h: 1, i: 2, ' ': 3 },
        merges: [],
        addBosToken: false,
        addEosToken: false,
        unkToken: 0,
      },
      config: { vocab_size: 4 },
    });

    const tokens = tokenizer.encode('hi');
    expect(tokens).toEqual([1, 2]);
    expect(tokenizer.decode(tokens)).toBe('hi');

    const pipeline = new FakePipeline();
    const output = await collect(pipeline.generate('hello'));
    expect(output).toBe('base:hello');

    await pipeline.unload();
    expect(pipeline.unloaded).toBe(true);
  });

  it('runs a multi-model mesh route', async () => {
    const loader = new FakeLoader();
    loader.adapters.set('alpha', createAdapter('alpha'));
    loader.adapters.set('beta', createAdapter('beta'));

    const pool = new MultiPipelinePool(loader as unknown as MultiModelLoader);
    const basePipeline = new FakePipeline();
    const network = new MultiModelNetwork(
      basePipeline as unknown as InferencePipeline,
      loader as unknown as MultiModelLoader,
      pool
    );

    network.registerExpert({ id: 'alpha', adapterName: 'alpha' });
    network.registerExpert({ id: 'beta', adapterName: 'beta' });

    const genome: NetworkGenome = {
      topology: { type: 'mesh' },
      nodes: [{ id: 'alpha' }, { id: 'beta' }],
      edges: [],
      combiner: { type: 'weighted', weights: [0, 1] },
    };

    const result = await network.executeGenome(genome, 'ping');
    expect(result).toBe('beta:ping');

    await pool.unloadAll();
  });
});
