import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';

import { resolveBundledTokenizerRefreshPatch } from '../../tools/refresh-converted-manifest.js';

const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'refresh-tokenizer-'));

try {
  await writeFile(
    path.join(tmpDir, 'tokenizer.json'),
    JSON.stringify({
      add_eos_token: true,
      post_processor: {
        type: 'TemplateProcessing',
        single: [
          { SpecialToken: { id: '<bos>', type_id: 0 } },
          { Sequence: { id: 'A', type_id: 0 } },
        ],
        special_tokens: {
          '<bos>': { id: '<bos>', ids: [1], tokens: ['<bos>'] },
        },
      },
      model: {
        vocab: {
          a: 0,
        },
      },
    }),
    'utf8'
  );

  const patch = await resolveBundledTokenizerRefreshPatch(tmpDir, {
    tokenizer: {
      type: 'bundled',
      file: 'tokenizer.json',
    },
  });

  assert.deepEqual(patch, {
    addBosToken: true,
    addEosToken: true,
  });
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}

console.log('refresh-converted-manifest-tokenizer.test: ok');
