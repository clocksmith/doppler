import assert from 'node:assert/strict';
import { dr } from 'doppler-gpu/compat';
import { boot } from '../../demo/boot.js';
import { state } from '../../demo/ui/state.js';
import { DEFAULT_DEMO_MODEL_ID } from '../../demo/models.js';

const originals = { load: dr.load, list: dr.listModelDetails, stored: dr.listPersistentModels,
  document: globalThis.document, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
const elements = new Map(['boot-overlay', 'app', 'model-picker', 'output-phase'].map(id => [id, {
  hidden: id === 'app', dataset: {}, classList: { toggle() {} }, setAttribute() {},
}]));
globalThis.document = { getElementById: id => elements.get(id) ?? null, querySelectorAll: () => [] };
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {} } });
const chat = { modelId: DEFAULT_DEMO_MODEL_ID, modes: ['text'], classification: { tasks: ['generation'] } };
let cached = true, loads = 0;
dr.listModelDetails = async () => [chat];
dr.listPersistentModels = async () => cached ? [chat] : [];
dr.load = async () => { loads++; throw new Error('Embedding exceeds device binding limit'); };
try {
  await boot();
  assert.equal(loads, 1);
  assert.equal(state.phase, 'ready');
  assert.equal(elements.get('boot-overlay').hidden, true);
  assert.equal(elements.get('app').hidden, false);
  assert.equal(elements.get('model-picker').open, true);
  assert.match(elements.get('output-phase').textContent, /Choose another model or retry.*binding limit/);
  assert.equal(state.modelBusy, false);
  cached = false;
  await boot();
  assert.equal(loads, 1, 'Fresh startup does not load or switch to a recorded example model');
  assert.equal(state.bootError, null);
  console.log('boot-recovery.test: ok');
} finally {
  dr.load = originals.load; dr.listModelDetails = originals.list; dr.listPersistentModels = originals.stored;
  globalThis.document = originals.document;
  if (originals.navigator) Object.defineProperty(globalThis, 'navigator', originals.navigator);
  else delete globalThis.navigator;
}
