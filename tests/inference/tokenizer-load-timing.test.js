import assert from 'node:assert/strict';

const { Tokenizer } = await import('../../src/inference/tokenizer.js');

const manifest = {
  modelId: 'tokenizer-load-timing-test',
  eos_token_id: 1,
  tokenizer: {
    type: 'bundled',
    file: 'tokenizer.json',
    eosToken: 1,
    unkToken: 2,
    addBosToken: false,
    addEosToken: false,
  },
};

const tokenizerJson = {
  type: 'bpe',
  vocab: {
    hello: 0,
    '<eos>': 1,
    '<unk>': 2,
  },
  merges: [],
  specialTokens: {
    eos: '<eos>',
    unk: '<unk>',
  },
};

let loadCalls = 0;
const tokenizer = new Tokenizer();
await tokenizer.initialize(manifest, {
  loadTokenizerJson: async () => {
    loadCalls += 1;
    return JSON.stringify(tokenizerJson);
  },
});

assert.equal(loadCalls, 1);
assert.deepEqual(tokenizer.encode('hello'), [0]);

const timing = tokenizer.getLoadTiming();
assert.equal(timing?.schemaVersion, 1);
assert.equal(timing?.source, 'doppler-tokenizer');
assert.equal(timing?.modelId, 'tokenizer-load-timing-test');
assert.equal(timing?.status, 'complete');
assert.equal(timing?.tokenizerType, 'bundled');
assert.equal(timing?.tokenizerFile, 'tokenizer.json');
assert.equal(timing?.backend, 'bundled');
assert.equal(timing?.assetSource, 'custom-loader');
assert.equal(timing?.cacheHit, false);
assert.equal(timing?.error, null);
assert.ok(Number.isFinite(timing?.totalMs));
assert.ok(Number.isFinite(timing?.phasesMs.configResolution));
assert.ok(Number.isFinite(timing?.phasesMs.cacheLookup));
assert.ok(Number.isFinite(timing?.phasesMs.backendCreate));
assert.ok(Number.isFinite(timing?.phasesMs.assetLoad));
assert.ok(Number.isFinite(timing?.phasesMs.assetParse));
assert.ok(Number.isFinite(timing?.phasesMs.backendLoad));
assert.ok(Number.isFinite(timing?.phasesMs.cacheStore));

timing.phasesMs.backendLoad = 999;
assert.notEqual(tokenizer.getLoadTiming()?.phasesMs.backendLoad, 999);

const cachedTokenizer = new Tokenizer();
await cachedTokenizer.initialize(manifest, {
  loadTokenizerJson: async () => {
    throw new Error('cache hit should not load tokenizer json');
  },
});

const cacheTiming = cachedTokenizer.getLoadTiming();
assert.equal(cacheTiming?.status, 'complete');
assert.equal(cacheTiming?.cacheHit, true);
assert.equal(cacheTiming?.assetSource, 'cache');
assert.equal(cacheTiming?.phasesMs.assetLoad, null);
assert.equal(cacheTiming?.phasesMs.backendLoad, null);

console.log('tokenizer-load-timing.test: ok');
