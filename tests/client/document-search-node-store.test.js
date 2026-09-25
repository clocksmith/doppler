import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createNodeDocumentStore, acquireNodeStoreLease } from '../../examples/document-search/node-store.js';
import { writeDocumentSnapshot } from '../../examples/document-search/document-store.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-node-store-'));
const encoder = new TextEncoder();
try {
  const release = await acquireNodeStoreLease(root);
  await assert.rejects(acquireNodeStoreLease(root), /Storage is locked/);
  const store = await createNodeDocumentStore(path.join(root, 'documents'));
  await store.writeFile('snapshot', encoder.encode('accepted'));
  const writer = await store.createWriteStream('snapshot');
  const writing = writer.write(encoder.encode('uncommitted'));
  await writer.abort();
  await writing;
  assert.equal(String(await store.readFile('snapshot')), 'accepted');
  await assert.rejects(writer.close(), /aborted/);
  await assert.rejects(store.readFile('../escape'), /Invalid storage path/);

  const originalOpen = fs.open;
  fs.open = async (...args) => {
    const handle = await originalOpen(...args);
    if (String(args[0]).endsWith('.pending')) handle.writeFile = async () => {
      throw Object.assign(new Error('Injected storage exhaustion'), { code: 'ENOSPC' });
    };
    return handle;
  };
  try { await assert.rejects(store.writeFile('snapshot', encoder.encode('failed')), { code: 'ENOSPC' }); }
  finally { fs.open = originalOpen; }
  assert.equal(String(await store.readFile('snapshot')), 'accepted');
  assert.deepEqual(await fs.readdir(path.join(root, 'documents')), ['snapshot']);
  const signal = new AbortController().signal;
  await writeDocumentSnapshot(store, [], {}, signal);
  const snapshot = await store.readFile('document-snapshot.json');
  const rename = fs.rename;
  fs.rename = async () => { throw Object.assign(new Error('Interrupted commit'), { code: 'EIO' }); };
  try { await assert.rejects(writeDocumentSnapshot(store, [], { revision: 2 }, signal), { code: 'EIO' }); }
  finally { fs.rename = rename; }
  assert.deepEqual(await store.readFile('document-snapshot.json'), snapshot);

  const moduleUrl = new URL('../../examples/document-search/node-store.js', import.meta.url).href;
  const child = spawnSync(process.execPath, ['--input-type=module', '-e',
    `import {createNodeDocumentStore} from ${JSON.stringify(moduleUrl)};
     const store=await createNodeDocumentStore(process.argv[1]);
     const writer=await store.createWriteStream('snapshot');
     await writer.write(new TextEncoder().encode('interrupted'));
     process.exit(17);`, path.join(root, 'documents')], { encoding: 'utf8' });
  assert.equal(child.status, 17, child.stderr);
  assert.equal(String(await store.readFile('snapshot')), 'accepted', 'Process interruption must preserve the committed snapshot');
  await store.writeFile('snapshot', encoder.encode('next revision'));
  assert.equal(String(await store.readFile('snapshot')), 'next revision');
  await release();
  await release();
  await (await acquireNodeStoreLease(root))();
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('document-search-node-store.test: ok');
