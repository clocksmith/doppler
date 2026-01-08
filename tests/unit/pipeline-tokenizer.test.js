import { describe, expect, it, beforeAll, afterAll, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { BPETokenizer } from '../../src/inference/tokenizers/bpe.js';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, '..', 'fixtures');

function loadFixture(filename) {
  const filePath = join(fixturesDir, filename);
  const content = readFileSync(filePath, 'utf-8');
  return JSON.parse(content);
}

describe('inference/tokenizers', () => {
  let miniModelTokenizer;
  let bpeTokenizer;
  let unigramTokenizer;

  beforeAll(() => {
    miniModelTokenizer = loadFixture('mini-model/tokenizer.json');
    bpeTokenizer = loadFixture('tokenizer-bpe.json');
    unigramTokenizer = loadFixture('tokenizer-unigram.json');
  });

  describe('BPETokenizer', () => {
    describe('loading', () => {
      it('loads vocabulary from JSON fixture', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: miniModelTokenizer.addBosToken,
          addEosToken: miniModelTokenizer.addEosToken,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

        expect(tokenizer.getVocabSize()).toBe(32);
      });

      it('loads larger BPE vocabulary', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: bpeTokenizer.addBosToken,
          addEosToken: bpeTokenizer.addEosToken,
          specialTokens: bpeTokenizer.specialTokens,
        });
        tokenizer.load(bpeTokenizer.vocab, bpeTokenizer.merges);

        expect(tokenizer.getVocabSize()).toBe(200);
      });

      it('stores special tokens correctly', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: miniModelTokenizer.addBosToken,
          addEosToken: miniModelTokenizer.addEosToken,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

        expect(tokenizer.specialTokens.bos).toBe(1);
        expect(tokenizer.specialTokens.eos).toBe(2);
        expect(tokenizer.specialTokens.pad).toBe(0);
        expect(tokenizer.specialTokens.unk).toBe(3);
      });
    });

    describe('encode', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BPETokenizer({
          addBosToken: miniModelTokenizer.addBosToken,
          addEosToken: miniModelTokenizer.addEosToken,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);
      });

      it('encodes known tokens', () => {
        const ids = tokenizer.encode('the');
        expect(ids).toContain(4);
      });

      it('adds BOS token when configured', () => {
        const ids = tokenizer.encode('the');
        expect(ids[0]).toBe(1);
      });

      it('does not add EOS token when disabled', () => {
        const ids = tokenizer.encode('the');
        expect(ids[ids.length - 1]).not.toBe(2);
      });

      it('encodes multiple words', () => {
        const ids = tokenizer.encode('the and');
        expect(ids).toContain(4);
        expect(ids).toContain(8);
      });

      it('returns UNK for unknown tokens', () => {
        const ids = tokenizer.encode('xyz');
        expect(ids).toContain(3);
      });

      it('handles empty string', () => {
        const ids = tokenizer.encode('');
        expect(ids.length).toBe(1);
        expect(ids[0]).toBe(1);
      });
    });

    describe('decode', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BPETokenizer({
          addBosToken: miniModelTokenizer.addBosToken,
          addEosToken: miniModelTokenizer.addEosToken,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);
      });

      it('decodes token IDs to text', () => {
        const text = tokenizer.decode([4]);
        expect(text).toBe('the');
      });

      it('decodes multiple tokens', () => {
        const text = tokenizer.decode([4, 8]);
        expect(text).toBe('theand');
      });

      it('skips special tokens by default', () => {
        const text = tokenizer.decode([1, 4, 2]);
        expect(text).toBe('the');
        expect(text).not.toContain('<bos>');
        expect(text).not.toContain('<eos>');
      });

      it('includes special tokens when skipSpecialTokens is false', () => {
        const text = tokenizer.decode([1, 4, 2], false);
        expect(text).toContain('<bos>');
        expect(text).toContain('<eos>');
      });

      it('trims whitespace by default', () => {
        const text = tokenizer.decode([4]);
        expect(text).toBe(text.trim());
      });

      it('preserves whitespace when trim is false', () => {
        const text = tokenizer.decode([4], true, false);
        expect(text).toBe('the');
      });
    });

    describe('special token handling', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BPETokenizer({
          addBosToken: miniModelTokenizer.addBosToken,
          addEosToken: miniModelTokenizer.addEosToken,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);
      });

      it('identifies special tokens', () => {
        expect(tokenizer.isSpecialToken(0)).toBe(true);
        expect(tokenizer.isSpecialToken(1)).toBe(true);
        expect(tokenizer.isSpecialToken(2)).toBe(true);
        expect(tokenizer.isSpecialToken(3)).toBe(true);
      });

      it('identifies non-special tokens', () => {
        expect(tokenizer.isSpecialToken(4)).toBe(false);
        expect(tokenizer.isSpecialToken(10)).toBe(false);
      });
    });

    describe('addBosToken / addEosToken config', () => {
      it('respects addBosToken=false', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: false,
          addEosToken: false,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

        const ids = tokenizer.encode('the');
        expect(ids[0]).not.toBe(1);
      });

      it('respects addEosToken=true', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: false,
          addEosToken: true,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

        const ids = tokenizer.encode('the');
        expect(ids[ids.length - 1]).toBe(2);
      });

      it('adds both BOS and EOS when configured', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: true,
          addEosToken: true,
          specialTokens: miniModelTokenizer.specialTokens,
        });
        tokenizer.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

        const ids = tokenizer.encode('the');
        expect(ids[0]).toBe(1);
        expect(ids[ids.length - 1]).toBe(2);
      });
    });

    describe('roundtrip', () => {
      it('encodes and decodes back to original text', () => {
        const tokenizer = new BPETokenizer({
          addBosToken: false,
          addEosToken: false,
          specialTokens: bpeTokenizer.specialTokens,
        });
        tokenizer.load(bpeTokenizer.vocab, bpeTokenizer.merges);

        const original = 'the';
        const ids = tokenizer.encode(original);
        const decoded = tokenizer.decode(ids);
        expect(decoded).toBe(original);
      });
    });
  });

  describe('BundledTokenizer', () => {
    describe('loading', () => {
      it('loads BPE tokenizer from bundled format', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);

        expect(tokenizer.getVocabSize()).toBe(32);
      });

      it('loads larger BPE tokenizer', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        expect(tokenizer.getVocabSize()).toBe(200);
      });

      it('loads Unigram tokenizer from bundled format', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(unigramTokenizer);

        expect(tokenizer.getVocabSize()).toBe(120);
      });

      it('stores special tokens from bundled format', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);

        expect(tokenizer.specialTokens.bos).toBe(1);
        expect(tokenizer.specialTokens.eos).toBe(2);
        expect(tokenizer.specialTokens.pad).toBe(0);
        expect(tokenizer.specialTokens.unk).toBe(3);
      });
    });

    describe('BPE encode', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);
      });

      it('encodes known tokens', () => {
        const ids = tokenizer.encode('the');
        expect(ids).toContain(1);
      });

      it('adds BOS token when configured', () => {
        const ids = tokenizer.encode('the');
        expect(ids[0]).toBe(1);
      });

      it('encodes multiple words', () => {
        const ids = tokenizer.encode('good time');
        expect(ids.length).toBeGreaterThan(1);
      });

      it('handles punctuation', () => {
        const ids = tokenizer.encode('hello.');
        expect(ids).toContain(149);
      });

      it('handles numbers', () => {
        const ids = tokenizer.encode('123');
        expect(ids.length).toBeGreaterThan(1);
      });

      it('handles empty string', () => {
        const ids = tokenizer.encode('');
        expect(ids.length).toBe(1);
        expect(ids[0]).toBe(1);
      });
    });

    describe('Unigram encode', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BundledTokenizer();
        tokenizer.load(unigramTokenizer);
      });

      it('encodes text using Viterbi algorithm', () => {
        const ids = tokenizer.encode('the');
        expect(ids.length).toBeGreaterThan(0);
      });

      it('adds BOS token when configured', () => {
        const ids = tokenizer.encode('the');
        expect(ids[0]).toBe(1);
      });

      it('uses scores for optimal tokenization', () => {
        const ids = tokenizer.encode('and');
        expect(ids.length).toBeGreaterThan(0);
      });

      it('handles empty string', () => {
        const ids = tokenizer.encode('');
        expect(ids.length).toBe(1);
        expect(ids[0]).toBe(1);
      });
    });

    describe('decode', () => {
      let bpeTokenizerInstance;
      let unigramTokenizerInstance;

      beforeEach(() => {
        bpeTokenizerInstance = new BundledTokenizer();
        bpeTokenizerInstance.load(bpeTokenizer);

        unigramTokenizerInstance = new BundledTokenizer();
        unigramTokenizerInstance.load(unigramTokenizer);
      });

      it('decodes BPE token IDs to text', () => {
        const text = bpeTokenizerInstance.decode([4]);
        expect(text).toBe('the');
      });

      it('decodes Unigram token IDs to text', () => {
        const text = unigramTokenizerInstance.decode([4]);
        expect(text).toBe('the');
      });

      it('skips special tokens by default', () => {
        const text = bpeTokenizerInstance.decode([1, 4, 2]);
        expect(text).toBe('the');
      });

      it('includes special tokens when skipSpecialTokens is false', () => {
        const text = bpeTokenizerInstance.decode([1, 4, 2], false);
        expect(text).toContain('<bos>');
        expect(text).toContain('<eos>');
      });

      it('trims whitespace by default', () => {
        const text = bpeTokenizerInstance.decode([4]);
        expect(text).toBe(text.trim());
      });

      it('preserves whitespace when trim is false', () => {
        const text = bpeTokenizerInstance.decode([4], true, false);
        expect(text).toBe('the');
      });

      it('handles space prefix characters', () => {
        const text = bpeTokenizerInstance.decode([4]);
        expect(text).not.toContain('\u2581');
        expect(text).not.toContain('\u0120');
      });
    });

    describe('special token handling', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);
      });

      it('identifies special tokens', () => {
        expect(tokenizer.isSpecialToken(0)).toBe(true);
        expect(tokenizer.isSpecialToken(1)).toBe(true);
        expect(tokenizer.isSpecialToken(2)).toBe(true);
        expect(tokenizer.isSpecialToken(3)).toBe(true);
      });

      it('identifies non-special tokens', () => {
        expect(tokenizer.isSpecialToken(4)).toBe(false);
        expect(tokenizer.isSpecialToken(10)).toBe(false);
      });
    });

    describe('vocabulary lookup', () => {
      let tokenizer;

      beforeEach(() => {
        tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);
      });

      it('returns correct vocab size', () => {
        expect(tokenizer.getVocabSize()).toBe(32);
      });

      it('encodes known vocabulary tokens correctly', () => {
        const ids = tokenizer.encode('the');
        expect(ids).toContain(4);
      });

      it('decodes known token IDs correctly', () => {
        const text = tokenizer.decode([4]);
        expect(text).toBe('the');
      });
    });

    describe('unknown token fallback', () => {
      it('falls back to UNK for unknown characters in BPE', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);

        const ids = tokenizer.encode('\u4e2d');
        expect(ids.length).toBeGreaterThan(1);
      });

      it('uses byte fallback for unknown characters in Unigram', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(unigramTokenizer);

        const ids = tokenizer.encode('\u4e2d');
        expect(ids.length).toBeGreaterThan(1);
      });
    });

    describe('addBosToken / addEosToken config', () => {
      it('respects addBosToken from bundled config', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);

        const ids = tokenizer.encode('the');
        expect(ids[0]).toBe(1);
      });

      it('respects addEosToken from bundled config', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(miniModelTokenizer);

        const ids = tokenizer.encode('the');
        expect(ids[ids.length - 1]).not.toBe(2);
      });

      it('handles config with addEosToken=true', () => {
        const configWithEos = {
          ...miniModelTokenizer,
          addEosToken: true,
        };
        const tokenizer = new BundledTokenizer();
        tokenizer.load(configWithEos);

        const ids = tokenizer.encode('the');
        expect(ids[ids.length - 1]).toBe(2);
      });
    });

    describe('whitespace handling', () => {
      it('handles whitespace in input', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        const ids = tokenizer.encode('the cat');
        expect(ids.length).toBeGreaterThan(2);
      });

      it('converts space prefix back to spaces on decode', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        const ids = tokenizer.encode('hello world');
        const text = tokenizer.decode(ids);
        expect(text).not.toContain('\u2581');
      });

      it('handles multiple consecutive spaces', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        const ids = tokenizer.encode('the  cat');
        expect(ids.length).toBeGreaterThan(2);
      });

      it('handles leading whitespace', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        const ids = tokenizer.encode(' the');
        expect(ids.length).toBeGreaterThan(1);
      });

      it('handles trailing whitespace', () => {
        const tokenizer = new BundledTokenizer();
        tokenizer.load(bpeTokenizer);

        const ids = tokenizer.encode('the ');
        expect(ids.length).toBeGreaterThan(1);
      });
    });

    describe('roundtrip', () => {
      it('BPE roundtrip preserves content', () => {
        const configNoBos = {
          ...bpeTokenizer,
          addBosToken: false,
          addEosToken: false,
        };
        const tokenizer = new BundledTokenizer();
        tokenizer.load(configNoBos);

        const original = 'the';
        const ids = tokenizer.encode(original);
        const decoded = tokenizer.decode(ids);
        expect(decoded).toBe(original);
      });

      it('Unigram roundtrip preserves content', () => {
        const configNoBos = {
          ...unigramTokenizer,
          addBosToken: false,
          addEosToken: false,
        };
        const tokenizer = new BundledTokenizer();
        tokenizer.load(configNoBos);

        const original = 'the';
        const ids = tokenizer.encode(original);
        const decoded = tokenizer.decode(ids);
        expect(decoded).toBe(original);
      });
    });

    describe('error handling', () => {
      it('throws when encoding before loading', () => {
        const tokenizer = new BundledTokenizer();
        expect(() => tokenizer.encode('test')).toThrow('not loaded');
      });

      it('throws when decoding before loading', () => {
        const tokenizer = new BundledTokenizer();
        expect(() => tokenizer.decode([1, 2, 3])).toThrow('not loaded');
      });
    });
  });

  describe('HuggingFace format loading', () => {
    it('loads HuggingFace BPE format', () => {
      const hfFormat = {
        model: {
          type: 'BPE',
          vocab: {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'the': 4,
            'a': 5,
          },
          merges: ['t h', 'th e'],
        },
        added_tokens: [
          { id: 0, content: '<pad>', special: true },
          { id: 1, content: '<bos>', special: true },
          { id: 2, content: '<eos>', special: true },
          { id: 3, content: '<unk>', special: true },
        ],
        add_bos_token: true,
        add_eos_token: false,
      };

      const tokenizer = new BundledTokenizer();
      tokenizer.load(hfFormat);

      expect(tokenizer.getVocabSize()).toBe(6);
      expect(tokenizer.specialTokens.bos).toBe(1);
      expect(tokenizer.specialTokens.eos).toBe(2);
    });

    it('loads HuggingFace Unigram format', () => {
      const hfFormat = {
        model: {
          type: 'Unigram',
          vocab: [
            ['<pad>', 0],
            ['<bos>', 0],
            ['<eos>', 0],
            ['<unk>', 0],
            ['the', -2.5],
            ['a', -2.8],
          ],
        },
        added_tokens: [
          { id: 0, content: '<pad>', special: true },
          { id: 1, content: '<bos>', special: true },
          { id: 2, content: '<eos>', special: true },
          { id: 3, content: '<unk>', special: true },
        ],
        add_bos_token: true,
        add_eos_token: false,
      };

      const tokenizer = new BundledTokenizer();
      tokenizer.load(hfFormat);

      expect(tokenizer.getVocabSize()).toBe(6);
    });

    it('extracts special tokens from added_tokens', () => {
      const hfFormat = {
        model: {
          type: 'BPE',
          vocab: {
            '<s>': 0,
            '</s>': 1,
            'hello': 2,
          },
          merges: [],
        },
        added_tokens: [
          { id: 0, content: '<s>', special: true },
          { id: 1, content: '</s>', special: true },
        ],
      };

      const tokenizer = new BundledTokenizer();
      tokenizer.load(hfFormat);

      expect(tokenizer.specialTokens.bos).toBe(0);
      expect(tokenizer.specialTokens.eos).toBe(1);
    });
  });

  describe('cross-tokenizer consistency', () => {
    it('both BPE implementations decode same token IDs to same text', () => {
      const bpe = new BPETokenizer({
        addBosToken: false,
        addEosToken: false,
        specialTokens: miniModelTokenizer.specialTokens,
      });
      bpe.load(miniModelTokenizer.vocab, miniModelTokenizer.merges);

      const bundled = new BundledTokenizer();
      bundled.load({ ...miniModelTokenizer, addBosToken: false, addEosToken: false });

      const ids = [4, 8];
      const bpeDecoded = bpe.decode(ids);
      const bundledDecoded = bundled.decode(ids);

      expect(bpeDecoded).toBe(bundledDecoded);
    });
  });
});
