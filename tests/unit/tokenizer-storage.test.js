import { describe, it, expect, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturePath = join(__dirname, '..', 'fixtures', 'mini-model', 'tokenizer.json');
const fixtureJson = JSON.parse(readFileSync(fixturePath, 'utf8'));

vi.mock('../../src/storage/shard-manager.js', () => ({
  loadTokenizerFromStore: vi.fn(async () => JSON.stringify(fixtureJson)),
}));

import { Tokenizer } from '../../src/inference/tokenizer.js';

describe('Tokenizer storage loading', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('loads bundled tokenizer from storage when no baseUrl', async () => {
    const tokenizer = new Tokenizer();
    await tokenizer.initialize({
      modelId: 'test-model',
      tokenizer: {
        type: 'bundled',
        file: 'tokenizer.json',
      },
    });

    const ids = tokenizer.encode('the');
    expect(ids.length).toBeGreaterThan(0);
    expect(tokenizer.getVocabSize()).toBeGreaterThan(0);
  });
});
