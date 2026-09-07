import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRetainedReleaseCheckpoint } from '../../tools/retained-release-checkpoint.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-retained-checkpoint-'));
const digest = character => `sha256:${character.repeat(64)}`;
try {
  const source = path.join(root, 'source.json'), output = path.join(root, 'attempt');
  const original = JSON.stringify({ [digest('a')]: { sequence: 2, digest: digest('b') } });
  await fs.writeFile(source, original); await fs.mkdir(output);
  const store = await createRetainedReleaseCheckpoint(source, output, digest('a'));
  await assert.rejects(store.persist({ sequence: 1, digest: digest('b') }), /rollback/);
  await assert.rejects(store.persist({ sequence: 2, digest: digest('c') }), /equivocation/);
  await store.persist({ sequence: 2, digest: digest('b') });
  await Promise.all([store.persist({ sequence: 3, digest: digest('c') }), store.persist({ sequence: 4, digest: digest('d') })]);
  assert.equal(JSON.parse(await fs.readFile(store.filename, 'utf8'))[digest('a')].sequence, 4);
  await store.verifySourceUnchanged();
  assert.equal(await fs.readFile(source, 'utf8'), original);
  await assert.rejects(createRetainedReleaseCheckpoint(source, output, digest('a')), /EEXIST/);
  await fs.writeFile(source, '{}');
  await assert.rejects(store.verifySourceUnchanged(), /changed/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('retained-release-checkpoint.test: ok');
