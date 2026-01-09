import { describe, expect, it } from 'vitest';

import { Tokenizer } from '../../src/inference/tokenizer.js';
import { MultiModelNetwork } from '../../src/inference/multi-model-network.js';
import { MultiPipelinePool } from '../../src/inference/multi-pipeline-pool.js';

class FakePipeline {
  adapterName = 'base';
  unloaded = false;

  setLoRAAdapter(adapter) {
    this.adapterName = adapter?.name || 'base';
  }

  async *generate(prompt) {
    yield `${this.adapterName}:${prompt}`;
  }

  async *generateWithPrefixKV(prefix, prompt) {
    yield `${this.adapterName}:${prefix.tokens.join(',')}:${prompt}`;
  }

  async prefillKVOnly(prompt) {
    return {
      cache: {},
      seqLen: prompt.length,
      tokens: [prompt.length],
    };
  }

  async unload() {
    this.unloaded = true;
  }
}

class FakeLoader {
  adapters = new Map();

  getAdapter(name) {
    return this.adapters.get(name) || null;
  }

  async createSharedPipeline() {
    return new FakePipeline();
  }
}

const createAdapter = (name) => ({
  name,
  rank: 1,
  alpha: 1,
  layers: new Map(),
});

const collect = async (generator) => {
  const chunks = [];
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
        bosToken: 101,
        eosToken: 102,
        padToken: 103,
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

    const pool = new MultiPipelinePool(loader);
    const basePipeline = new FakePipeline();
    const network = new MultiModelNetwork(
      basePipeline,
      loader,
      pool
    );

    network.registerExpert({ id: 'alpha', adapterName: 'alpha' });
    network.registerExpert({ id: 'beta', adapterName: 'beta' });

    const genome = {
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
