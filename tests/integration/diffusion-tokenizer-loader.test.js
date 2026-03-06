import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { loadDiffusionTokenizers } from '../../src/inference/pipelines/diffusion/text-encoder.js';

installNodeFileFetchShim();

const fixtureDir = mkdtempSync(path.join(os.tmpdir(), 'doppler-diffusion-tokenizer-'));
mkdirSync(path.join(fixtureDir, 'tokenizer'), { recursive: true });

try {
  writeFileSync(
    path.join(fixtureDir, 'tokenizer', 'tokenizer.json'),
    JSON.stringify({
      model: {
        type: 'Unigram',
        vocab: [
          ['<pad>', 0.0],
          ['<bos>', 0.0],
          ['<eos>', 0.0],
          ['hello', 0.0],
        ],
        unk_id: 0,
      },
      special_tokens_map: {
        bos_token: '<bos>',
        eos_token: '<eos>',
        pad_token: '<pad>',
      },
    }),
    'utf8'
  );

  const tokenizers = await loadDiffusionTokenizers(
    {
      tokenizers: {
        text_encoder: {
          type: 'bundled',
          tokenizerFile: 'tokenizer/tokenizer.json',
        },
      },
    },
    {
      baseUrl: `file://${fixtureDir}`,
    }
  );

  assert.ok(tokenizers.text_encoder);
  assert.deepEqual(tokenizers.text_encoder.encode('hello'), [1, 3]);
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('diffusion-tokenizer-loader.test: ok');
