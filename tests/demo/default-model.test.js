import assert from 'node:assert/strict';
import { dr } from 'doppler-gpu/compat';
import { state } from '../../demo/ui/state.js';
import { DEFAULT_DEMO_MODEL_ID, loadCatalog, loadDefaultStoredModel } from '../../demo/models.js';
import { initInput, setRunHandler } from '../../demo/input.js';

const originals = { load: dr.load, list: dr.listModelDetails, stored: dr.listPersistentModels, fetch: globalThis.fetch };
const prompt = { value: 'Keep this prompt.', addEventListener() {} };
const handlers = {};
const send = { addEventListener: (name, handler) => { handlers[name] = handler; } };
const phase = {};
globalThis.document = {
  getElementById: (id) => ({ 'prompt-input': prompt, 'run-btn': send, 'output-phase': phase })[id] ?? null,
  querySelectorAll: () => [],
};
globalThis.localStorage = { setItem() {}, getItem: () => 'larger-model' };
let cached = [];
let fail = false;
const loads = [];
dr.listModelDetails = async () => [{ modelId: 'larger-model' }, { modelId: DEFAULT_DEMO_MODEL_ID }];
dr.listPersistentModels = async () => cached;
dr.load = async (modelId) => {
  loads.push(modelId);
  if (fail) throw new Error('Download interrupted');
  return { modelId };
};
globalThis.fetch = async () => ({ json: async () => ({ text: [], suggestions: [] }) });
try {
  await loadCatalog();
  cached = [{ modelId: 'larger-model' }];
  assert.equal(await loadDefaultStoredModel(), null, 'Do not preload an unrelated larger model');
  cached.push({ modelId: DEFAULT_DEMO_MODEL_ID });
  await loadDefaultStoredModel();
  assert.deepEqual(loads, [DEFAULT_DEMO_MODEL_ID], 'Preload the cached 270M model regardless of catalog order');
  state.model = null;
  state.modelId = null;
  await initInput();
  let runs = 0;
  setRunHandler(() => { runs++; });
  fail = true;
  await handlers.click();
  assert.equal(runs, 0);
  assert.equal(prompt.value, 'Keep this prompt.');
  assert.equal(state.modelBusy, false);
  assert.equal(send.disabled, false, 'A failed load can be retried');
  assert.match(phase.textContent, /Download interrupted/);
  fail = false;
  await handlers.click();
  assert.equal(runs, 1, 'Sending loads the default and then runs without another click');
  assert.equal(state.modelId, DEFAULT_DEMO_MODEL_ID);
  await handlers.click();
  assert.equal(runs, 2);
  assert.equal(loads.length, 3, 'Subsequent sends reuse the loaded model');
  assert.equal(send.textContent, 'Send');
  console.log('default-model.test: ok');
} finally {
  dr.load = originals.load;
  dr.listModelDetails = originals.list;
  dr.listPersistentModels = originals.stored;
  globalThis.fetch = originals.fetch;
  delete globalThis.document;
  delete globalThis.localStorage;
}
