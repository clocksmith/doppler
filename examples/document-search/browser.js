import { openCapsule } from 'doppler-gpu/host';
import { createFetchCapsuleArtifactStore } from 'doppler-gpu';
import { getCapsuleIdentity } from 'doppler-gpu/capsule';
import { createOpfsStore } from 'doppler-gpu/tooling/storage';
import { createDocumentModelInstallation } from './installation.js';
import { createDocumentSearch } from './search.js';

const config = await (await fetch('./models.json')).json();
const state = { sessions: {}, installations: {}, observations: [], index: null, search: null, controller: null, indexInvalidated: false };
const $ = id => document.getElementById(id);
const json = value => new TextEncoder().encode(JSON.stringify(value));
const hash = async bytes => 'sha256:' + Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
const storeFor = async id => {
  const store = createOpfsStore(config.storage);
  await store.openModel(id);
  return store;
};
const documents = await storeFor('documents');
for (const model of config.models) {
  const source = createFetchCapsuleArtifactStore(new URL(model.capsuleUrl, location.href).href);
  state.installations[model.role] = createDocumentModelInstallation({ store: await storeFor(model.storageId), openCapsule,
    fetchArtifact: (artifact, options) => source.readArtifact(artifact, options),
    withLock: task => navigator.locks.request('document-search-' + model.storageId, task),
    authorizeRecord: record => JSON.stringify(getCapsuleIdentity(record.capsule)) === JSON.stringify(model.identity)
      && JSON.stringify(record.options.trustedSigners) === JSON.stringify(model.options.trustedSigners)
      && JSON.stringify(record.options.acceptedTargetPlanDigests) === JSON.stringify(model.options.acceptedTargetPlanDigests)
      && JSON.stringify(record.options.releaseTrustedSigners) === JSON.stringify(model.options.releaseTrustedSigners),
  });
}
async function close() {
  const errors = [];
  for (const session of Object.values(state.sessions)) {
    try { await session.close(); } catch (error) { errors.push(error); }
  }
  state.sessions = {}; state.search = null;
  if (errors.length) throw new AggregateError(errors, 'Model cleanup failed.');
}
async function loadModels(install, repairDamagedArtifacts = false) {
  await close();
  state.index = null; state.indexInvalidated = false;
  try {
    for (const model of config.models) {
      const controls = { signal: state.controller?.signal, repairDamagedArtifacts, observer: { observe: event => state.observations.push({
        model: model.role, ...event, at: performance.now(), heap: performance.memory?.usedJSHeapSize ?? null }) },
      onLoadProgress: event => { $('status').textContent = `${model.role}: ${event.phase}, ${event.loadedBytes} bytes`; } };
      let opened;
      if (install) {
        if (!$('retention').checked) throw new Error('Choose retained local use before installing.');
        const response = await fetch(model.capsuleUrl, { signal: state.controller?.signal });
        if (!response.ok) throw new Error(`Capsule metadata acquisition failed: ${response.status}`);
        const capsule = await response.json();
        const options = structuredClone(model.options);
        options.releasePolicy.retainedLocalUse = { ...model.retainedLocalUse, acceptedAtUtc: new Date().toISOString() };
        opened = await state.installations[model.role].install({ capsule, options }, controls);
      } else opened = await state.installations[model.role].openRetained(controls);
      state.sessions[model.role] = opened.session;
      state.observations.push({ model: model.role, acquisition: opened.observations });
    }
    const embedding = config.models.find(model => model.role === 'embedding');
    const reranker = config.models.find(model => model.role === 'reranker');
    state.search = createDocumentSearch({ ...config.search, embedding: state.sessions.embedding,
      reranker: state.sessions.reranker, embeddingIdentity: embedding.identity,
      embeddingApplication: embedding.application, rerankerApplication: reranker.application });
    const retained = await documents.readText('index.json');
    if (retained !== null) {
      const record = JSON.parse(retained);
      if (await hash(json(record.index)) !== record.digest) throw new Error('Retained index integrity failed.');
      state.index = record.index;
      try { state.search.assertIndex(state.index); }
      catch (error) {
        if (error.code !== 'DOCUMENT_SEARCH_INDEX_INCOMPATIBLE') throw error;
        state.index = null; state.indexInvalidated = true;
      }
    }
    $('index').disabled = false; $('search').disabled = false; $('rebuild').disabled = false;
    return { models: Object.keys(state.sessions), retainedDocuments: state.index?.documents.length ?? 0, indexInvalidated: state.indexInvalidated };
  } catch (error) {
    try { await close(); } catch (cleanupError) { throw new AggregateError([error, cleanupError], error.message); }
    throw error;
  }
}
async function indexDocuments(input) {
  if (!state.search) throw new Error('Open installed models first.');
  const records = [];
  for (const document of input) records.push({ ...document, sourceDigest: await hash(new TextEncoder().encode(document.text)) });
  const index = await state.search.indexDocuments(records, { signal: state.controller?.signal });
  await documents.writeFile('documents.json', json(records));
  await documents.writeFile('index.json', json({ index, digest: await hash(json(index)) }));
  state.index = index; state.indexInvalidated = false;
  return { documents: records.length };
}
async function search(query) {
  if (!state.search || !state.index) throw new Error('Save a document index first.');
  return state.search.search(state.index, query, { signal: state.controller?.signal });
}
async function run(action) {
  if (state.controller) throw new Error('Another application operation is active.');
  state.controller = new AbortController(); $('cancel').disabled = false;
  try { const result = await action(); $('status').textContent = state.indexInvalidated ? 'Embedding release changed. Rebuild the retained index.' : 'Ready for local search.'; return result; }
  catch (error) { $('status').textContent = error.message; throw error; }
  finally { state.controller = null; $('cancel').disabled = true; }
}
$('cancel').onclick = () => state.controller?.abort(new DOMException('Cancelled by application', 'AbortError'));
$('install').onclick = () => run(() => loadModels(true)).catch(() => {});
$('repair').onclick = () => run(() => loadModels(true, true)).catch(() => {});
$('open').onclick = () => run(() => loadModels(false)).catch(() => {});
async function rebuild() {
  const retained = await documents.readText('documents.json');
  if (retained === null) throw new Error('No retained documents to rebuild.');
  const input = JSON.parse(retained);
  for (const record of input) if (await hash(new TextEncoder().encode(record.text)) !== record.sourceDigest) {
    throw new Error('Retained document integrity failed.');
  }
  return indexDocuments(input);
}
$('rebuild').onclick = () => run(rebuild).catch(() => {});
$('index').onclick = () => run(async () => {
  const input = [];
  for (const file of $('files').files) {
    if (!/\.(txt|md|markdown)$/i.test(file.name)) throw new Error('Choose a text or Markdown file.');
    const mediaType = /\.(md|markdown)$/i.test(file.name) ? 'text/markdown' : 'text/plain';
    input.push({ id: await hash(new TextEncoder().encode(file.name)), title: file.name, text: await file.text(), mediaType });
  }
  return indexDocuments(input);
}).catch(() => {});
$('search').onclick = () => run(async () => {
  const result = await search($('query').value);
  $('results').replaceChildren(...result.results.map(({ document: record }) => {
    const article = document.createElement('article'); const title = document.createElement('strong'); const text = document.createElement('p');
    title.textContent = record.title; text.textContent = record.text; article.append(title, text); return article;
  }));
}).catch(() => {});
const registration = await navigator.serviceWorker.getRegistration('./')
  ?? await navigator.serviceWorker.register('./service-worker.js', { scope: './' });
await navigator.serviceWorker.ready;
if (!navigator.serviceWorker.controller) await new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true }));
$('status').textContent = 'Application installed. Choose model installation or open retained models.';
// Exposes application operations for retained qualification, using the same UI implementation.
globalThis.documentSearch = { install: () => run(() => loadModels(true)), open: () => run(() => loadModels(false)),
  repair: () => run(() => loadModels(true, true)),
  indexDocuments: input => run(() => indexDocuments(input)), search: query => run(() => search(query)), close,
  rebuildIndex: () => run(rebuild),
  observations: state.observations, registration, ready: true };
