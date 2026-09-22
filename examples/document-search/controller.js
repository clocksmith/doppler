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

const encodeJson = value => new TextEncoder().encode(JSON.stringify(value));

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return 'sha256:' + Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

async function readTextOrNull(store, filename) {
  try {
    if (typeof store.readText === 'function') {
      return await store.readText(filename);
    }
    const bytes = await store.readFile(filename);
    if (!bytes) return null;
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch (err) {
    if (err?.name === 'NotFoundError') return null;
    throw err;
  }
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
    const retained = await readTextOrNull(docStore, 'index.json');
    if (retained !== null && retained !== undefined) {
      const record = JSON.parse(retained);
      const expectedDigest = await sha256Hex(encodeJson(record.index));
      if (expectedDigest !== record.digest) {
        throw new Error('Retained index integrity failed.');
      }
      index = record.index;
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
    if (isDisposed) throw new Error('Controller is disposed.');
    if (initPromise) return initPromise;

    const promise = (async () => {
      const controller = new AbortController();
      loadingController = controller;

      let onAbort;
      if (signal) {
        if (signal.aborted) {
          controller.abort(signal.reason);
          throw signal.reason;
        }
        onAbort = () => controller.abort(signal.reason);
        signal.addEventListener('abort', onAbort, { once: true });
      }

      try {
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

        return {
          models: Object.keys(sessions),
          retainedDocuments: index?.documents?.length ?? 0,
          indexInvalidated,
        };
      } catch (error) {
        try {
          await closeSessions();
        } catch (cleanupError) {
          throw new AggregateError([error, cleanupError], error.message);
        }
        throw error;
      } finally {
        if (signal && onAbort) signal.removeEventListener('abort', onAbort);
        if (loadingController === controller) {
          loadingController = null;
        }
        initPromise = null;
      }
    })();

    initPromise = promise;
    return promise;
  }

  function cancelLoading(reason = new DOMException('Loading cancelled by application', 'AbortError')) {
    loadingController?.abort(reason);
  }

  function cancelSearch(reason = new DOMException('Search cancelled by application', 'AbortError')) {
    searchController?.abort(reason);
  }

  async function searchDocuments(query, options = {}) {
    if (isDisposed) throw new Error('Controller is disposed.');
    if (!search || !index) throw new Error('Save a document index first.');
    if (typeof query !== 'string' || !query.trim()) throw new Error('Search query is required.');

    const queryId = ++latestQueryId;
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
    const indexBytes = encodeJson(newIndex);
    const indexDigest = await sha256Hex(indexBytes);

    const docStore = await getDocumentsStore();
    await docStore.writeFile('documents.json', encodeJson(records));
    await docStore.writeFile('index.json', encodeJson({ index: newIndex, digest: indexDigest }));

    index = newIndex;
    indexInvalidated = false;

    return { documents: records.length, index: newIndex };
  }

  async function rebuild(options = {}) {
    if (isDisposed) throw new Error('Controller is disposed.');
    if (!search) throw new Error('Open installed models first.');

    const docStore = await getDocumentsStore();
    const retained = await readTextOrNull(docStore, 'documents.json');
    if (retained === null || retained === undefined) {
      throw new Error('No retained documents to rebuild.');
    }
    const input = JSON.parse(retained);
    for (const record of input) {
      options.signal?.throwIfAborted();
      const textBytes = new TextEncoder().encode(record.text);
      if (await sha256Hex(textBytes) !== record.sourceDigest) {
        throw new Error('Retained document integrity failed.');
      }
    }

    const newIndex = await search.indexDocuments(input, null, { signal: options.signal });
    const indexBytes = encodeJson(newIndex);
    const indexDigest = await sha256Hex(indexBytes);

    await docStore.writeFile('index.json', encodeJson({ index: newIndex, digest: indexDigest }));

    index = newIndex;
    indexInvalidated = false;

    return { documents: input.length, index: newIndex };
  }

  async function dispose() {
    isDisposed = true;
    cancelLoading(new DOMException('Controller disposed', 'AbortError'));
    cancelSearch(new DOMException('Controller disposed', 'AbortError'));
    if (initPromise) {
      await initPromise.catch(() => {});
    }
    await closeSessions().catch(() => {});
  }

  return {
    initialize,
    install: (options = {}) => initialize({ ...options, install: true }),
    openRetained: (options = {}) => initialize({ ...options, install: false }),
    repair: (options = {}) => initialize({ ...options, install: true, repairDamagedArtifacts: true }),
    cancelLoading,
    cancelSearch,
    cancel: () => { cancelLoading(); cancelSearch(); },
    search: searchDocuments,
    indexDocuments,
    rebuild,
    close: closeSessions,
    dispose,
    getState() {
      return {
        isDisposed,
        isInitializing: initPromise !== null,
        isSearching: searchController !== null,
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
