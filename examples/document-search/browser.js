import { openCapsule } from 'doppler-gpu/host';
import { createFetchCapsuleArtifactStore } from 'doppler-gpu';
import { getCapsuleIdentity } from 'doppler-gpu/capsule';
import { createOpfsStore } from 'doppler-gpu/tooling/storage';
import { createDocumentSearchController } from './controller.js';

const config = await (await fetch('./models.json')).json();
const $ = id => document.getElementById(id);

const storeFor = async id => {
  const store = createOpfsStore(config.storage);
  await store.openModel(id);
  return store;
};

const controller = createDocumentSearchController({
  config,
  storeFor,
  openCapsule,
  fetchCapsuleArtifactStore: capsuleUrl => createFetchCapsuleArtifactStore(new URL(capsuleUrl, location.href).href),
  withLock: (storageId, task) => navigator.locks.request('document-search-' + storageId, task),
  getCapsuleIdentity,
  onProgress: event => {
    if (event.type === 'load-progress') {
      const mb = (event.loadedBytes / (1024 * 1024)).toFixed(1);
      $('status').textContent = `${event.model}: ${event.phase}, ${mb} MB`;
    } else if (event.type === 'phase') {
      const phaseNames = {
        downloading: 'Downloading weights…',
        verifying: 'Verifying integrity…',
        ready: 'Ready',
      };
      const text = phaseNames[event.phase] ?? event.phase;
      $('status').textContent = `${event.model}: ${text}`;
    }
  },
});

function updateUiState() {
  const state = controller.getState();
  const ready = state.hasSessions;
  const hasIndex = state.hasIndex;

  $('index').disabled = !ready || state.isInitializing || state.isSearching;
  $('rebuild').disabled = !ready || state.isInitializing || state.isSearching;
  $('search').disabled = !ready || !hasIndex || state.isInitializing || state.isSearching;

  $('cancel-load').disabled = !state.isInitializing;
  $('cancel-search').disabled = !state.isSearching;
  if ($('cancel')) $('cancel').disabled = !state.isInitializing && !state.isSearching;
}

async function runAction(action, loading = false) {
  updateUiState();
  try {
    const result = await action();
    const state = controller.getState();
    if (state.indexInvalidated) {
      $('status').textContent = 'Embedding release changed. Rebuild the retained index.';
    } else if (loading) {
      $('status').textContent = `Models ready (${state.sessionRoles.join(', ')}). ${state.documentCount} documents indexed.`;
    }
    return result;
  } catch (error) {
    if (error.name === 'AbortError') {
      $('status').textContent = 'Operation cancelled.';
    } else {
      $('status').textContent = error.message;
    }
    throw error;
  } finally {
    updateUiState();
  }
}

$('cancel-load').onclick = () => controller.cancelLoading();
$('cancel-search').onclick = () => controller.cancelSearch();
if ($('cancel')) $('cancel').onclick = () => { controller.cancelLoading(); controller.cancelSearch(); };

$('install').onclick = () => {
  if (!$('retention').checked) {
    $('status').textContent = 'Choose retained local use before installing.';
    return;
  }
  return runAction(() => controller.install(), true).catch(() => {});
};

$('repair').onclick = () => {
  if (!$('retention').checked) {
    $('status').textContent = 'Choose retained local use before repairing.';
    return;
  }
  return runAction(() => controller.repair(), true).catch(() => {});
};

$('open').onclick = () => runAction(() => controller.openRetained(), true).catch(() => {});

$('rebuild').onclick = () => runAction(async () => {
  $('status').textContent = 'Rebuilding index from retained documents…';
  const result = await controller.rebuild();
  $('status').textContent = `Index rebuilt: ${result.documents} documents indexed.`;
  return result;
}).catch(() => {});

async function hashBytes(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return 'sha256:' + Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

$('index').onclick = () => runAction(async () => {
  const files = $('files').files;
  if (!files || files.length === 0) {
    throw new Error('Choose files to index.');
  }
  const input = [];
  for (const file of files) {
    if (!/\.(txt|md|markdown)$/i.test(file.name)) throw new Error(`Unsupported file type: ${file.name}. Choose a text or Markdown file.`);
    const mediaType = /\.(md|markdown)$/i.test(file.name) ? 'text/markdown' : 'text/plain';
    const text = await file.text();
    const id = await hashBytes(new TextEncoder().encode(file.name));
    input.push({ id, title: file.name, text, mediaType });
  }
  $('status').textContent = `Indexing ${input.length} documents…`;
  const result = await controller.indexDocuments(input);
  $('status').textContent = `Indexed ${result.documents} documents. Ready for local search.`;
  return result;
}).catch(() => {});

async function doSearch() {
  const query = $('query').value;
  if (!query || !query.trim()) return;
  return runAction(async () => {
    $('status').textContent = 'Searching…';
    const result = await controller.search(query.trim());
    if (result.superseded) return;
    $('results').replaceChildren(...result.results.map(({ document: record, score }) => {
      const article = document.createElement('article');
      const title = document.createElement('strong');
      const text = document.createElement('p');
      const meta = document.createElement('small');
      title.textContent = record.title;
      text.textContent = record.text;
      if (typeof score === 'number') {
        meta.textContent = `Score: ${score.toFixed(4)}`;
      }
      article.append(title, text, meta);
      return article;
    }));
    $('status').textContent = `Found ${result.results.length} results.`;
    return result;
  });
}

$('search').onclick = () => doSearch().catch(() => {});
$('search-form')?.addEventListener('submit', event => {
  event.preventDefault();
  doSearch().catch(() => {});
});

const registration = await navigator.serviceWorker.getRegistration('./')
  ?? await navigator.serviceWorker.register('./service-worker.js', { scope: './' });
await navigator.serviceWorker.ready;
if (!navigator.serviceWorker.controller) {
  await new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true }));
}

$('status').textContent = 'Application installed. Choose model installation or open retained models.';

// Expose application operations for test automation and retained qualification
globalThis.documentSearch = {
  install: () => runAction(() => controller.install(), true),
  open: () => runAction(() => controller.openRetained(), true),
  repair: () => runAction(() => controller.repair(), true),
  indexDocuments: input => runAction(() => controller.indexDocuments(input)),
  search: query => runAction(() => controller.search(query)),
  close: () => controller.close(),
  rebuildIndex: () => runAction(() => controller.rebuild()),
  get observations() { return controller.getObservations(); },
  controller,
  registration,
  ready: true,
};
