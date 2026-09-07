import assert from 'node:assert/strict';
import { createDocumentModelInstallation } from '../../examples/document-search/installation.js';

// Synthetic storage and execution ports exercise application persistence failures.
// Physical inference is qualified separately through installed packages.
const bytes = new Uint8Array([1, 3, 7]);
const hash = 'sha256:' + Buffer.from(await crypto.subtle.digest('SHA-256', bytes)).toString('hex');
const artifact = { artifactId: 'weight', hash, sizeBytes: bytes.byteLength };
const files = new Map();
let writesFail = false;
let downloads = 0;
let closed = 0;
let accept = true;
const store = {
  async readFile(key) { if (!files.has(key)) throw new DOMException('Missing', 'NotFoundError'); return files.get(key).slice(); },
  async writeFile(key, value) {
    if (writesFail) {
      if (!files.has(key)) files.set(key, new Uint8Array());
      throw new DOMException('Full', 'QuotaExceededError');
    }
    files.set(key, value.slice());
  },
  async deleteFile(key) { return files.delete(key); },
};
const record = { capsule: { artifacts: [artifact] }, options: {} };
const installation = createDocumentModelInstallation({ store, authorizeRecord: async () => accept,
  withLock: task => task(), fetchArtifact: async () => { downloads++; return bytes.slice(); },
  openCapsule: async (capsule, options) => {
    for (const entry of capsule.artifacts) await options.artifactStore.readArtifact(entry, options);
    return { close: async () => { closed++; } };
  } });
await assert.rejects(installation.openRetained(), /No completed/);
accept = false;
await assert.rejects(installation.install(record), /not application-authorized/);
assert.equal(downloads, 0);
accept = true;
writesFail = true;
await assert.rejects(installation.install(record), { name: 'QuotaExceededError' });
assert(!files.has('installation.json'));
assert(!files.has('artifacts/' + hash.slice(7)), 'failed writes remove their newly created artifact entry');
writesFail = false;
const controller = new AbortController(); controller.abort();
await assert.rejects(installation.install(record, { signal: controller.signal }), { name: 'AbortError' });
assert(!files.has('installation.json'));
await (await installation.install(record)).session.close();
const completedDownloads = downloads;
await (await installation.openRetained()).session.close();
assert.equal(downloads, completedDownloads, 'retained open performs no download');
files.set('artifacts/' + hash.slice(7), new Uint8Array([0, 3, 7]));
await assert.rejects(installation.openRetained(), /integrity failed/);
assert.equal(downloads, completedDownloads, 'damaged cache cannot silently refetch');
assert.equal(await installation.removeDamagedArtifact(artifact), true);
await assert.rejects(installation.openRetained(), /incomplete/);
await (await installation.install(record)).session.close();
assert.equal(await installation.removeDamagedArtifact(artifact), false, 'repair preserves valid bytes');
const priorRecord = files.get('installation.json').slice();
writesFail = true;
await assert.rejects(installation.install(record), { name: 'QuotaExceededError' });
assert.deepEqual(files.get('installation.json'), priorRecord);
assert.equal(closed, 4, 'failed metadata commit closes the acquired session');
writesFail = false;
files.set('artifacts/' + hash.slice(7), new Uint8Array([0, 3, 7]));
const repaired = await installation.install(record, { repairDamagedArtifacts: true });
assert(repaired.observations.some(event => event.action === 'removed-damaged-artifact'));
await repaired.session.close();
assert.equal(downloads, completedDownloads + 2, 'explicit repair fetches the damaged artifact');
files.set('installation.json', new TextEncoder().encode('{broken'));
await assert.rejects(installation.openRetained(), SyntaxError);
const writeStarted = Promise.withResolvers();
const finishWrite = Promise.withResolvers();
const cancellationDuringWrite = new AbortController();
let locked = false;
const pendingInstallation = createDocumentModelInstallation({ authorizeRecord: () => true,
  withLock: async task => { locked = true; try { return await task(); } finally { locked = false; } },
  fetchArtifact: async () => bytes.slice(),
  store: { readFile: async () => { throw new DOMException('Missing', 'NotFoundError'); },
    writeFile: async () => { writeStarted.resolve(); await finishWrite.promise; }, deleteFile: async () => true },
  openCapsule: async (capsule, options) => {
    options.artifactStore.readArtifact(artifact, options).catch(() => {});
    await writeStarted.promise;
    cancellationDuringWrite.abort(new Error('Cancelled during write'));
    throw cancellationDuringWrite.signal.reason;
  } });
const rejectedWrite = assert.rejects(pendingInstallation.install(record, { signal: cancellationDuringWrite.signal }), /Cancelled during write/);
await writeStarted.promise;
await new Promise(resolve => setImmediate(resolve));
assert.equal(locked, true, 'failed opening holds its lock while a storage write is outstanding');
finishWrite.resolve();
await rejectedWrite;
assert.equal(locked, false);
console.log('document-model-installation.test: passed (synthetic storage, no inference claim)');
