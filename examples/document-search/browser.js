import { openCapsule } from 'doppler-gpu/host';
import { createFetchCapsuleArtifactStore } from 'doppler-gpu';
import { getCapsuleIdentity } from 'doppler-gpu/capsule';
import { createOpfsStore } from 'doppler-gpu/tooling/storage';
import { createDocumentSearchController } from './controller.js';
import { importDocuments } from './document-import.js';

const config = await (await fetch('./models.json')).json();
const $ = id => document.getElementById(id);
const requirements = await (await fetch('./requirements.json')).json();
let preflight = { complete: false, hardwareErrors: [], acquisitionErrors: [] };
const timings = { installationMs: null, firstQueryMs: null, subsequentQueryMs: [] };
const mib = bytes => (bytes / (1024 * 1024)).toFixed(1) + ' MiB';

async function checkPrerequisites() {
  const rows = requirements.models.map(model => `${model.role}: ${mib(model.downloadBytes)}; GPU features: ${model.requiredFeatures.join(', ') || 'WebGPU'}; maximum buffer size must support ${mib(model.minBufferSize)}.`);
  const minimumStorage = requirements.runtimeBytes + requirements.models.reduce((sum, model) => sum + model.downloadBytes, 0);
  rows.push(`Storage: at least ${mib(minimumStorage)} for model and runtime files, plus application files, documents, indexes, and temporary save space.`);
  const hardwareErrors = [], acquisitionErrors = [];
  if (!navigator.gpu) hardwareErrors.push('WebGPU is unavailable.');
  else {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) hardwareErrors.push('No WebGPU adapter is available.');
    else for (const model of requirements.models) {
      for (const feature of model.requiredFeatures) if (!adapter.features.has(feature)) hardwareErrors.push(`${model.role} needs GPU feature ${feature}.`);
      if (adapter.limits.maxBufferSize < model.minBufferSize) hardwareErrors.push(`${model.role} needs larger GPU buffers.`);
    }
  }
  if (!navigator.storage?.getDirectory || !navigator.locks) hardwareErrors.push('This browser needs private file storage and Web Locks.');
  const estimate = await navigator.storage?.estimate?.();
  if (Number.isFinite(estimate?.quota) && Number.isFinite(estimate?.usage)) {
    rows.push(`Browser storage: ${mib(estimate.quota - estimate.usage)} currently available. Quotas may change.`);
    if (estimate.quota < minimumStorage) acquisitionErrors.push('Browser storage quota is smaller than the model and runtime files.');
  }
  if (requirements.missingSources.length) acquisitionErrors.push(`${requirements.missingSources.length} model files have no published download source. Installation and repair are unavailable until the distributor supplies these exact files.`);
  $('requirements').replaceChildren(...rows.map(text => { const item = document.createElement('li'); item.textContent = text; return item; }));
  preflight = { complete: true, hardwareErrors, acquisitionErrors };
  $('preflight').textContent = [...hardwareErrors, ...acquisitionErrors].join(' ') || 'Required GPU features and declared download sources are present. Model integrity is checked during installation.';
}

function assertPrerequisites(acquire) {
  const errors = [...preflight.hardwareErrors, ...(acquire ? preflight.acquisitionErrors : [])];
  if (!preflight.complete || errors.length) throw new Error(errors.join(' ') || 'Device checks are still running.');
}

function showTimings() {
  const rows = [];
  if (timings.installationMs !== null) rows.push(`Model installation and GPU preparation: ${timings.installationMs.toFixed(0)} ms`);
  if (timings.firstQueryMs !== null) rows.push(`First query with both models loaded: ${timings.firstQueryMs.toFixed(0)} ms`);
  if (timings.subsequentQueryMs.length) rows.push(`Subsequent queries: ${timings.subsequentQueryMs.map(ms => ms.toFixed(0)).join(', ')} ms`);
  $('timings').textContent = rows.join(' · ');
}

async function loadModels(operation, acquire) {
  assertPrerequisites(acquire);
  if (acquire && !$('retention').checked) throw new Error('Choose retained local use before installing or repairing.');
  const started = performance.now();
  const result = await operation();
  timings.firstQueryMs = null; timings.subsequentQueryMs = [];
  if (acquire) timings.installationMs = performance.now() - started;
  showTimings();
  return result;
}

async function searchModels(query) {
  const started = performance.now();
  const result = await controller.search(query);
  if (!result.superseded) {
    const elapsed = performance.now() - started;
    if (timings.firstQueryMs === null) timings.firstQueryMs = elapsed;
    else timings.subsequentQueryMs.push(elapsed);
    showTimings();
  }
  return result;
}

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
  observer: { observe(event) {
    if (event.type === 'capsule-validation-complete') $('status').textContent = `${event.model}: Preparing GPU resources…`;
  } },
  onStateChange: () => updateUiState(),
  onProgress: event => {
    if (event.type === 'load-progress') {
      const labels = { artifact: 'Acquiring model file', metadata: 'Acquiring model description',
        acquiring: 'Acquiring model file', verifying: 'Verifying bytes', verified: 'Size and hash verified',
        reused: 'Verified retained file' };
      $('status').textContent = `${event.model}: ${labels[event.phase] ?? event.phase}, ${mib(event.loadedBytes)}`;
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
  const busy = state.isInitializing || state.isSearching || state.isIndexing || state.isClosing || state.isDisposed;

  $('index').disabled = !ready || busy;
  $('rebuild').disabled = !ready || busy;
  $('search').disabled = !ready || !hasIndex || busy;
  for (const id of ['install', 'open', 'repair']) $(id).disabled = busy || !preflight.complete
    || preflight.hardwareErrors.length > 0 || (id !== 'open' && preflight.acquisitionErrors.length > 0);

  $('cancel-load').disabled = !state.isInitializing;
  $('cancel-search').disabled = !state.isSearching;
  $('cancel-index').disabled = !state.isIndexing;
  if ($('cancel')) $('cancel').disabled = !state.isInitializing && !state.isSearching && !state.isIndexing;
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
$('cancel-index').onclick = () => controller.cancelIndexing();
if ($('cancel')) $('cancel').onclick = () => controller.cancel();
$('close').onclick = () => runAction(() => controller.close()).catch(() => {});

$('install').onclick = () => {
  if (!$('retention').checked) {
    $('status').textContent = 'Choose retained local use before installing.';
    return;
  }
  return runAction(() => loadModels(() => controller.install(), true), true).catch(() => {});
};

$('repair').onclick = () => {
  if (!$('retention').checked) {
    $('status').textContent = 'Choose retained local use before repairing.';
    return;
  }
  return runAction(() => loadModels(() => controller.repair(), true), true).catch(() => {});
};

$('open').onclick = () => runAction(() => loadModels(() => controller.openRetained(), false), true).catch(() => {});

$('rebuild').onclick = () => runAction(async () => {
  $('status').textContent = 'Rebuilding index from retained documents…';
  const result = await controller.rebuild();
  $('status').textContent = `Index rebuilt: ${result.documents} documents indexed.`;
  return result;
}).catch(() => {});

$('index').onclick = () => runAction(async () => {
  const files = $('files').files;
  if (!files || files.length === 0) {
    throw new Error('Choose files to index.');
  }
  const input = await importDocuments(files, controller.getIndex()?.documents);
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
    const result = await searchModels(query.trim());
    if (result.superseded) return;
    $('results').replaceChildren(...result.results.map(({ document: record, rerankScore: score }) => {
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

await checkPrerequisites();
updateUiState();
const registration = await navigator.serviceWorker.getRegistration('./')
  ?? await navigator.serviceWorker.register('./service-worker.js', { scope: './' });
await navigator.serviceWorker.ready;
if (!navigator.serviceWorker.controller) {
  await new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true }));
}

$('status').textContent = 'Application installed. Choose model installation or open retained models.';

// Expose application operations for test automation and retained qualification
globalThis.documentSearch = {
  install: () => runAction(() => loadModels(() => controller.install(), true), true),
  open: () => runAction(() => loadModels(() => controller.openRetained(), false), true),
  repair: () => runAction(() => loadModels(() => controller.repair(), true), true),
  indexDocuments: input => runAction(() => controller.indexDocuments(input)),
  search: query => runAction(() => searchModels(query)),
  close: () => controller.close(),
  rebuildIndex: () => runAction(() => controller.rebuild()),
  get observations() { return controller.getObservations(); },
  get timings() { return structuredClone(timings); },
  get prerequisites() { return structuredClone(preflight); },
  controller,
  registration,
  ready: true,
};
