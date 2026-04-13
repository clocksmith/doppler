import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const {
  buildDistillPrompt,
  loadDistillDatasetFromJsonl,
  resolveDistillDataScope,
} = await import('../../src/experimental/training/suite.js');

const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-distill-suite-data-'));

try {
  assert.equal(
    buildDistillPrompt({
      direction: 'en->es',
      source: 'Hello world',
    }),
    'Translate from English to Spanish:\nHello world\nTranslation:'
  );

  const datasetPath = path.join(tmpDir, 'dataset.jsonl');
  await writeFile(
    datasetPath,
    [
      JSON.stringify({
        source: 'Hello world',
        target_pos: 'Hola mundo',
        src_lang: 'en',
        tgt_lang: 'es',
        pair: 'en->es',
      }),
      JSON.stringify({
        source: 'Hola mundo',
        target_pos: 'Hello world',
        src_lang: 'es',
        tgt_lang: 'en',
        pair: 'es->en',
      }),
      JSON.stringify({
        source: 'Bonjour',
        target_pos: 'Hola',
        src_lang: 'fr',
        tgt_lang: 'es',
        pair: 'fr->es',
      }),
    ].join('\n'),
    'utf8'
  );

  const scopedDataset = await loadDistillDatasetFromJsonl(
    datasetPath,
    resolveDistillDataScope({
      distillSourceLangs: ['en'],
      distillTargetLangs: ['es'],
      distillPairAllowlist: ['en->es'],
    })
  );
  assert.equal(scopedDataset.rowCount, 3);
  assert.equal(scopedDataset.sampleCount, 1);
  assert.deepEqual(scopedDataset.directionCounts, { 'en->es': 1 });
  assert.deepEqual(scopedDataset.dataScope, {
    sourceLangs: ['en'],
    targetLangs: ['es'],
    pairAllowlist: ['en->es'],
    strictPairContract: false,
  });

  const strictDatasetPath = path.join(tmpDir, 'strict.jsonl');
  await writeFile(
    strictDatasetPath,
    JSON.stringify({
      source: 'Hello world',
      target_pos: 'Hola mundo',
      src_lang: 'en',
      tgt_lang: 'es',
      pair: 'en->fr',
    }),
    'utf8'
  );

  await assert.rejects(
    () => loadDistillDatasetFromJsonl(
      strictDatasetPath,
      resolveDistillDataScope({
        strictPairContract: true,
      })
    ),
    /pair "en->fr" does not match src\/tgt "en-es"/
  );
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}

console.log('distill-suite-data.test: ok');
