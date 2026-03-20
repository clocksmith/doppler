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
      add_bos_token: false,
      add_eos_token: true,
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
    addBosToken: false,
    addEosToken: true,
  });
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}

console.log('refresh-converted-manifest-tokenizer.test: ok');
