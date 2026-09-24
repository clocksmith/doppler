import assert from 'node:assert/strict';

// Faults are applied at native submission/storage boundaries. Model operations,
// the installed application, and the retained corpus remain unchanged.
export async function checkSearchLifecycle(page, fixture, storage) {
  const result = await page.evaluate(async ({ fixture, storage }) => {
    const app = globalThis.documentSearch;
    const controller = app.controller;
    const sessions = controller.getSessions();
    const originalIndex = JSON.stringify(controller.getIndex());
    const report = {};
    const beforeReuse = globalThis.documentSearchGpuMetrics.submitCalls;
    await app.indexDocuments(fixture.documents);
    report.unchangedDocumentSubmissions = globalThis.documentSearchGpuMetrics.submitCalls - beforeReuse;
    report.unchangedIndex = JSON.stringify(controller.getIndex()) === originalIndex;

    async function cancelledAfterSubmission(kind, task) {
      const submit = GPUQueue.prototype.submit;
      let submitted = false;
      GPUQueue.prototype.submit = function(commands) {
        const value = submit.call(this, commands);
        if (!submitted) {
          submitted = true;
          queueMicrotask(() => kind === 'index' ? controller.cancelIndexing() : controller.cancelSearch());
        }
        return value;
      };
      try {
        await task();
        return { submitted, rejected: false };
      } catch (error) { return { submitted, rejected: true, name: error.name, message: error.message }; }
      finally { GPUQueue.prototype.submit = submit; }
    }
    report.cancelledIndex = await cancelledAfterSubmission('index', () => app.indexDocuments([
      ...fixture.documents, { ...fixture.documents[0], id: 'cancelled-new-document', text: 'Different text for a cancelled index.' },
    ]));
    report.cancelledIndex.preserved = JSON.stringify(controller.getIndex()) === originalIndex;
    report.cancelledQuery = await cancelledAfterSubmission('query', () => app.search(fixture.queries[0].text));
    const first = app.search(fixture.queries[0].text);
    const second = app.search(fixture.queries[1].text);
    const [superseded, latest] = await Promise.all([first, second]);
    report.superseded = superseded.superseded === true;
    report.latestTopId = latest.results[0]?.document.id;

    const root = await navigator.storage.getDirectory();
    const directory = await (await root.getDirectoryHandle(storage.opfsRootDir)).getDirectoryHandle('documents');
    const snapshot = await directory.getFileHandle('document-snapshot.json');
    const priorBytes = await (await snapshot.getFile()).text();
    const create = FileSystemFileHandle.prototype.createWritable;
    const write = FileSystemWritableFileStream.prototype.write;
    const targets = new WeakSet();
    let interrupted = false;
    FileSystemFileHandle.prototype.createWritable = async function(options) {
      const stream = await create.call(this, options);
      if (this.name === 'document-snapshot.json') targets.add(stream);
      return stream;
    };
    FileSystemWritableFileStream.prototype.write = async function(data) {
      const value = await write.call(this, data);
      if (targets.has(this)) { interrupted = true; throw new DOMException('Injected interrupted snapshot save', 'AbortError'); }
      return value;
    };
    try {
      await app.indexDocuments(fixture.documents.map((document, index) => index ? document : { ...document, title: 'Interrupted title change' }));
      report.interruptedSave = { rejected: false };
    } catch (error) { report.interruptedSave = { rejected: true, name: error.name }; }
    finally {
      FileSystemFileHandle.prototype.createWritable = create;
      FileSystemWritableFileStream.prototype.write = write;
    }
    report.interruptedSave.reachedWrite = interrupted;
    report.interruptedSave.retainedBytesUnchanged = await (await snapshot.getFile()).text() === priorBytes;
    report.interruptedSave.indexUnchanged = JSON.stringify(controller.getIndex()) === originalIndex;
    report.reusedSessions = Object.entries(sessions).every(([role, session]) => controller.getSessions()[role] === session);
    const recovered = await app.search(fixture.queries[0].text);
    report.recoveredTopId = recovered.results[0]?.document.id;
    return report;
  }, { fixture, storage });
  assert.equal(result.unchangedDocumentSubmissions, 0);
  assert.equal(result.unchangedIndex, true);
  for (const cancelled of [result.cancelledIndex, result.cancelledQuery]) {
    assert.equal(cancelled.submitted, true); assert.equal(cancelled.rejected, true); assert.equal(cancelled.name, 'AbortError');
  }
  assert.equal(result.cancelledIndex.preserved, true);
  assert.equal(result.superseded, true);
  assert.equal(result.latestTopId, fixture.queries[1].expectedTopId);
  assert.equal(result.recoveredTopId, fixture.queries[0].expectedTopId);
  assert.equal(result.reusedSessions, true);
  assert.deepEqual(result.interruptedSave, { rejected: true, name: 'AbortError', reachedWrite: true,
    retainedBytesUnchanged: true, indexUnchanged: true });
  return result;
}
