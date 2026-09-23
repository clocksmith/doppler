function defaultGetCapsuleIdentity(capsule) {
  if (!capsule || typeof capsule !== 'object') {
    throw new Error('Capsule object is required to extract identity.');
  }
  const { schema, semanticRoot, envelopeDigest } = capsule;
  if (!schema || !semanticRoot || !envelopeDigest) {
    throw new Error('Capsule is missing schema, semanticRoot, or envelopeDigest.');
  }
  return { schema, semanticRoot, envelopeDigest };
}

import { createDocumentModelInstallation } from './installation.js';
import { createDocumentSearch } from './search.js';
import { readDocumentSnapshot, writeDocumentSnapshot } from './document-store.js';

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return 'sha256:' + Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

export function createDocumentSearchController({
  config,
  storeFor,
  openCapsule,
  fetchCapsuleArtifactStore = null,
  fetchArtifact = null,
  withLock = null,
  getCapsuleIdentity = defaultGetCapsuleIdentity,
  authorizeRecord = null,
  fetch = globalThis.fetch,
  observer = null,
  onProgress = null,
  onStateChange = null,
}) {
  if (!config || !Array.isArray(config.models)) {
    throw new Error('Configuration with models array is required.');
  }
  if (typeof storeFor !== 'function') {
    throw new Error('storeFor factory function is required.');
  }
  if (typeof openCapsule !== 'function') {
    throw new Error('openCapsule function is required.');
  }
  if (config.storage?.useSyncAccessHandle === true) throw new Error('Document publication requires atomic asynchronous storage, not SyncAccessHandle.');

  const installations = {};
  let sessions = {};
  let search = null;
  let index = null;
  let indexInvalidated = false;
  const observations = [];

  let documentsStore = null;
  let initPromise = null;
  let loadingController = null;
  let searchController = null;
  let latestQueryId = 0;
  let isDisposed = false;
  let isClosing = false;
  let closePromise = null;
  const activeOperations = new Set();
  const cleanupErrors = [];
  let operationTail = Promise.resolve();

  function changed() { onStateChange?.(); }
  function assertUsable() {
    if (isDisposed || isClosing) throw new DOMException('Controller disposed or closing.', 'AbortError');
  }
  function track(kind, options, task) {
    assertUsable();
    if (initPromise) throw new Error('Wait for model initialization.');
    const controller = new AbortController();
    const onAbort = () => controller.abort(options.signal.reason);
    if (options.signal?.aborted) onAbort();
    else options.signal?.addEventListener('abort', onAbort, { once: true });
    const operation = { kind, controller, promise: null };
    activeOperations.add(operation);
    operation.promise = operationTail.catch(() => {}).then(async () => {
      controller.signal.throwIfAborted();
      assertUsable();
      const result = await task({ ...options, signal: controller.signal });
      // Once atomic snapshot close begins, publication owns its completion.
      // Do not report a committed save as cancelled by a later signal.
      if (!(kind === 'index' && result.committed === true)) controller.signal.throwIfAborted();
      assertUsable();
      return result;
    }).finally(() => {
      options.signal?.removeEventListener('abort', onAbort);
      activeOperations.delete(operation);
      changed();
    });
    operationTail = operation.promise;
    changed();
    return operation.promise;
  }

  async function getDocumentsStore() {
    if (!documentsStore) {
      documentsStore = await storeFor('documents');
    }
    return documentsStore;
  }

  async function getInstallation(model) {
    if (!installations[model.role]) {
      const store = await storeFor(model.storageId);
      let modelFetchArtifact;
      if (typeof fetchArtifact === 'function') {
        modelFetchArtifact = (artifact, opts) => fetchArtifact(model, artifact, opts);
      } else if (typeof fetchCapsuleArtifactStore === 'function') {
        const source = fetchCapsuleArtifactStore(model.capsuleUrl);
        modelFetchArtifact = (artifact, opts) => source.readArtifact(artifact, opts);
      } else {
        throw new Error('fetchCapsuleArtifactStore or fetchArtifact is required.');
      }

      const modelLock = task => {
        if (typeof withLock === 'function') {
          if (withLock.length >= 2) return withLock(model.storageId, task);
          return withLock(task);
        }
        return task();
      };

      const modelAuthorizeRecord = record => {
        if (typeof authorizeRecord === 'function') {
          return authorizeRecord(model, record);
        }
        const idFn = getCapsuleIdentity ?? defaultGetCapsuleIdentity;
        return JSON.stringify(idFn(record.capsule)) === JSON.stringify(model.identity)
          && JSON.stringify(record.options.trustedSigners) === JSON.stringify(model.options.trustedSigners)
          && JSON.stringify(record.options.acceptedTargetPlanDigests) === JSON.stringify(model.options.acceptedTargetPlanDigests)
          && JSON.stringify(record.options.releaseTrustedSigners) === JSON.stringify(model.options.releaseTrustedSigners);
      };

      installations[model.role] = createDocumentModelInstallation({
        store,
        openCapsule,
        fetchArtifact: modelFetchArtifact,
        withLock: modelLock,
        authorizeRecord: modelAuthorizeRecord,
      });
    }
    return installations[model.role];
  }

  async function closeSessions() {
    const errors = [];
    for (const session of Object.values(sessions)) {
      try {
        await session.close();
      } catch (error) {
        errors.push(error);
      }
    }
    sessions = {};
    search = null;
    if (errors.length > 1) throw new AggregateError(errors, 'Model cleanup failed.');
    if (errors.length === 1) throw errors[0];
  }

  async function loadRetainedIndex() {
    const docStore = await getDocumentsStore();
    const retained = await readDocumentSnapshot(docStore);
    if (retained.index) {
      index = retained.index;
      try {
        search.assertIndex(index);
      } catch (err) {
        if (err.code === 'DOCUMENT_SEARCH_INDEX_INCOMPATIBLE') {
          index = null;
          indexInvalidated = true;
        } else {
          throw err;
        }
      }
    }
  }

  function initialize({ install = false, repairDamagedArtifacts = false, signal = null } = {}) {
    assertUsable();
    if (initPromise) return initPromise;
    if (activeOperations.size) throw new Error('Wait for active document operations before opening models.');
    if (search && !install) return Promise.resolve({ models: Object.keys(sessions), retainedDocuments: index?.documents.length ?? 0, indexInvalidated });

    const promise = Promise.resolve().then(async () => {
      const controller = new AbortController();
      loadingController = controller;

      let onAbort;
      if (signal) {
        if (signal.aborted) {
          controller.abort(signal.reason);
        }
        onAbort = () => controller.abort(signal.reason);
        signal.addEventListener('abort', onAbort, { once: true });
      }

      try {
        assertUsable();
        controller.signal.throwIfAborted();
        await closeSessions();
        index = null;
        indexInvalidated = false;

        for (const model of config.models) {
          controller.signal.throwIfAborted();
          const controls = {
            signal: controller.signal,
            repairDamagedArtifacts,
            observer: {
              observe: event => {
                const entry = {
                  model: model.role,
                  ...event,
                  at: performance.now(),
                  heap: globalThis.performance?.memory?.usedJSHeapSize ?? null,
                };
                observations.push(entry);
                observer?.observe?.(entry);
              },
            },
            onLoadProgress: event => {
              onProgress?.({ type: 'load-progress', model: model.role, ...event });
            },
          };

          const installation = await getInstallation(model);
          let opened;
          if (install) {
            controller.signal.throwIfAborted();
            onProgress?.({ type: 'phase', phase: 'downloading', model: model.role });
            const response = await fetch(model.capsuleUrl, { signal: controller.signal });
            if (!response.ok) throw new Error(`Capsule metadata acquisition failed: ${response.status}`);
            const capsule = await response.json();
            const options = structuredClone(model.options);
            if (options.releasePolicy && model.retainedLocalUse) {
              options.releasePolicy.retainedLocalUse = {
                ...model.retainedLocalUse,
                acceptedAtUtc: model.retainedLocalUse.acceptedAtUtc ?? new Date().toISOString(),
              };
            }
            controller.signal.throwIfAborted();
            onProgress?.({ type: 'phase', phase: 'verifying', model: model.role });
            opened = await installation.install({ capsule, options }, controls);
          } else {
            controller.signal.throwIfAborted();
            onProgress?.({ type: 'phase', phase: 'verifying', model: model.role });
            opened = await installation.openRetained(controls);
          }

          sessions[model.role] = opened.session;
          controller.signal.throwIfAborted();
          assertUsable();
          observations.push({ model: model.role, acquisition: opened.observations });
          onProgress?.({ type: 'phase', phase: 'ready', model: model.role });
        }

        controller.signal.throwIfAborted();

        const embedding = config.models.find(m => m.role === 'embedding');
        const reranker = config.models.find(m => m.role === 'reranker');
        search = createDocumentSearch({
          ...config.search,
          embedding: sessions.embedding,
          reranker: sessions.reranker,
          embeddingIdentity: embedding.identity,
          embeddingApplication: embedding.application,
          rerankerApplication: reranker.application,
        });

        await loadRetainedIndex();
        controller.signal.throwIfAborted();
        assertUsable();

        return {
          models: Object.keys(sessions),
          retainedDocuments: index?.documents?.length ?? 0,
          indexInvalidated,
        };
      } catch (error) {
        try {
          await closeSessions();
        } catch (cleanupError) {
          cleanupErrors.push(cleanupError);
          throw new AggregateError([error, cleanupError], error.message);
        }
        throw error;
      } finally {
        if (signal && onAbort) signal.removeEventListener('abort', onAbort);
        if (loadingController === controller) {
          loadingController = null;
        }
        initPromise = null;
        changed();
      }
    });

    initPromise = promise;
    changed();
    return promise;
  }

  function cancelLoading(reason = new DOMException('Loading cancelled by application', 'AbortError')) {
    loadingController?.abort(reason);
  }

  function cancelSearch(reason = new DOMException('Search cancelled by application', 'AbortError')) {
    searchController?.abort(reason);
    for (const operation of activeOperations) if (operation.kind === 'search') operation.controller.abort(reason);
  }
  function cancelIndexing(reason = new DOMException('Indexing cancelled by application', 'AbortError')) {
    for (const operation of activeOperations) if (operation.kind === 'index') operation.controller.abort(reason);
  }

  async function searchDocuments(query, options, queryId) {
    if (isDisposed) throw new Error('Controller is disposed.');
    if (!search || !index) throw new Error('Save a document index first.');
    if (typeof query !== 'string' || !query.trim()) throw new Error('Search query is required.');

    const controller = new AbortController();
    searchController = controller;

    let onAbort;
    if (options.signal) {
      if (options.signal.aborted) throw options.signal.reason;
      onAbort = () => controller.abort(options.signal.reason);
      options.signal.addEventListener('abort', onAbort, { once: true });
    }

    try {
      controller.signal.throwIfAborted();
      const result = await search.search(index, query, { signal: controller.signal });
      controller.signal.throwIfAborted();
      assertUsable();
      if (queryId !== latestQueryId) {
        return { query, count: 0, results: [], superseded: true };
      }
      return result;
    } finally {
      if (options.signal && onAbort) options.signal.removeEventListener('abort', onAbort);
      if (searchController === controller) {
        searchController = null;
      }
    }
  }

  async function indexDocuments(documents, options = {}) {
    if (isDisposed) throw new Error('Controller is disposed.');
    if (!search) throw new Error('Open installed models first.');
    if (!Array.isArray(documents)) throw new Error('Documents array is required.');

    const records = [];
    for (const doc of documents) {
      options.signal?.throwIfAborted();
      if (!['text/plain', 'text/markdown'].includes(doc.mediaType)) {
        throw new Error('Only text and Markdown documents are supported.');
      }
      if (typeof doc.text !== 'string' || !doc.text.trim()) throw new Error('Document text is required.');
      if (typeof doc.id !== 'string' || !doc.id.trim()) throw new Error('Document id is required.');
      if (typeof doc.title !== 'string') throw new Error('Document title is required.');
      const textBytes = new TextEncoder().encode(doc.text);
      const sourceDigest = await sha256Hex(textBytes);
      records.push({ ...doc, sourceDigest });
    }

    const newIndex = await search.indexDocuments(records, index, { signal: options.signal });
    const docStore = await getDocumentsStore();
    await writeDocumentSnapshot(docStore, records, newIndex, options.signal);
    assertUsable();

    index = newIndex;
    indexInvalidated = false;

    return { documents: records.length, index: newIndex, committed: true };
  }

  async function rebuild(options = {}) {
    if (isDisposed) throw new Error('Controller is disposed.');
    if (!search) throw new Error('Open installed models first.');

    const docStore = await getDocumentsStore();
    const retained = await readDocumentSnapshot(docStore);
    if (!retained.documents) {
      throw new Error('No retained documents to rebuild.');
    }
    const input = retained.documents;
    for (const record of input) {
      options.signal?.throwIfAborted();
      const textBytes = new TextEncoder().encode(record.text);
      if (await sha256Hex(textBytes) !== record.sourceDigest) {
        throw new Error('Retained document integrity failed.');
      }
    }

    const newIndex = await search.indexDocuments(input, null, { signal: options.signal });
    await writeDocumentSnapshot(docStore, input, newIndex, options.signal);
    assertUsable();

    index = newIndex;
    indexInvalidated = false;

    return { documents: input.length, index: newIndex, committed: true };
  }

  function close() {
    if (closePromise) return closePromise;
    isClosing = true;
    const reason = new DOMException('Controller disposed or closed', 'AbortError');
    cancelLoading(reason);
    for (const operation of activeOperations) operation.controller.abort(reason);
    closePromise = (async () => {
      const settled = await Promise.allSettled([initPromise, ...Array.from(activeOperations, operation => operation.promise)]);
      for (const result of settled) {
        if (result.status === 'rejected' && result.reason?.name !== 'AbortError') cleanupErrors.push(result.reason);
      }
      try { await closeSessions(); } catch (error) { cleanupErrors.push(error); }
      index = null;
      if (cleanupErrors.length) throw new AggregateError(cleanupErrors.splice(0), 'Model cleanup failed.');
    })().finally(() => {
      isClosing = false;
      if (!isDisposed) closePromise = null;
      changed();
    });
    changed();
    return closePromise;
  }
  function dispose() { isDisposed = true; return close(); }

  return {
    initialize,
    install: (options = {}) => initialize({ ...options, install: true }),
    openRetained: (options = {}) => initialize({ ...options, install: false }),
    repair: (options = {}) => initialize({ ...options, install: true, repairDamagedArtifacts: true }),
    cancelLoading,
    cancelSearch,
    cancelIndexing,
    cancel: () => { cancelLoading(); cancelSearch(); cancelIndexing(); },
    search: (query, options = {}) => {
      const queryId = ++latestQueryId;
      return track('search', options, controls => searchDocuments(query, controls, queryId));
    },
    indexDocuments: (documents, options = {}) => track('index', options, controls => indexDocuments(documents, controls)),
    rebuild: (options = {}) => track('index', options, rebuild),
    close,
    dispose,
    getState() {
      return {
        isDisposed,
        isInitializing: initPromise !== null,
        isSearching: [...activeOperations].some(operation => operation.kind === 'search'),
        isIndexing: [...activeOperations].some(operation => operation.kind === 'index'),
        isClosing,
        hasSessions: Object.keys(sessions).length > 0,
        sessionRoles: Object.keys(sessions),
        hasIndex: index !== null,
        indexInvalidated,
        documentCount: index?.documents?.length ?? 0,
      };
    },
    getIndex() { return index; },
    getSessions() { return { ...sessions }; },
    getObservations() { return [...observations]; },
  };
}
