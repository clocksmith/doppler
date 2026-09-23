const encode = value => new TextEncoder().encode(JSON.stringify(value));
export async function digestBytes(bytes) {
  return 'sha256:' + Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
}
async function readJson(store, name) {
  try {
    const bytes = await store.readFile(name);
    return bytes == null ? null : JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (error) {
    if (error.name === 'NotFoundError') return null;
    throw error;
  }
}

export async function readDocumentSnapshot(store) {
  const snapshot = await readJson(store, 'document-snapshot.json');
  if (snapshot) {
    if (snapshot.schema !== 'doppler.document-snapshot/v1'
      || await digestBytes(encode(snapshot.data)) !== snapshot.digest) {
      throw new Error('Retained document snapshot integrity failed.');
    }
    return snapshot.data;
  }
  // Read old installations without rewriting their evidence. A legacy index
  // already contains the document text; never combine it with a different save.
  const retained = await readJson(store, 'index.json');
  if (retained) {
    if (await digestBytes(encode(retained.index)) !== retained.digest) throw new Error('Retained index integrity failed.');
    const documents = retained.index.documents.map(({ vector, contentHash, ...document }) => document);
    return { documents, index: retained.index };
  }
  return { documents: await readJson(store, 'documents.json'), index: null };
}

// The application store must provide atomic createWritable-style replacement,
// not in-place SyncAccessHandle writes. Documents and vectors share ONE commit.
export async function writeDocumentSnapshot(store, documents, index, signal) {
  const data = { documents, index };
  const bytes = encode({ schema: 'doppler.document-snapshot/v1', data, digest: await digestBytes(encode(data)) });
  signal.throwIfAborted();
  if (typeof store.createWriteStream !== 'function') throw new Error('Atomic document snapshot storage is required.');
  const writer = await store.createWriteStream('document-snapshot.json');
  let committing = false;
  let abortPromise = null;
  const abort = () => {
    if (!committing) {
      abortPromise ??= writer.abort();
      // Awaited below; attach immediately to avoid an unhandled rejection.
      abortPromise.catch(() => {});
    }
  };
  signal.addEventListener('abort', abort, { once: true });
  try {
    signal.throwIfAborted();
    await writer.write(bytes);
    signal.throwIfAborted();
    // close() is the commit point. Cancellation before it aborts replacement;
    // cancellation after it cannot turn a committed generation into a partial pair.
    committing = true;
    await writer.close();
  } catch (error) {
    try { await (abortPromise ?? writer.abort()); }
    catch (cleanupError) { throw new AggregateError([error, cleanupError], 'Snapshot publication and cleanup failed.'); }
    throw error;
  } finally {
    signal.removeEventListener('abort', abort);
  }
}
