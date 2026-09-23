import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { createServer } from '../../examples/document-search/server.js';

// Actual UI/controller, synthetic model ports. Never counted as GPU acceptance.
const server = createServer();
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ serviceWorkers: 'block' });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const roles = ['embedding', 'reranker'];
const identity = role => ({ schema: 'fixture', semanticRoot: role, envelopeDigest: role });
const models = roles.map(role => ({ role, storageId: role, capsuleUrl: './capsules/' + role + '/capsule-v3.json',
  identity: identity(role), options: { trustedSigners: {}, acceptedTargetPlanDigests: [], releaseTrustedSigners: {} } }));
const config = { models, search: { dimension: 2, candidateCount: 2, queryPrefix: '', documentPrefix: '' }, storage: { useSyncAccessHandle: false } };
const respond = (route, body, contentType = 'application/javascript') => route.fulfill({ status: 200, contentType, body });
try {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'serviceWorker', { value: { controller: {},
      ready: Promise.resolve(), getRegistration: async () => ({}), register: async () => ({}) } });
  });
  await page.route('**/models.json', route => respond(route, JSON.stringify(config), 'application/json'));
  for (const role of roles) await page.route('**/capsules/' + role + '/capsule-v3.json', route => respond(route,
    JSON.stringify({ ...identity(role), artifacts: [] }), 'application/json'));
  await page.route('**/runtime/src/capsule.js', route => respond(route,
    'export const getCapsuleIdentity = ({schema,semanticRoot,envelopeDigest}) => ({schema,semanticRoot,envelopeDigest});'));
  await page.route('**/runtime/src/capsule-runtime.js', route => respond(route,
    'export const createFetchCapsuleArtifactStore = () => ({ readArtifact: async () => new Uint8Array() });'));
  await page.route('**/runtime/src/client/capsule-host.browser.js', route => respond(route, `
    const delay = signal => new Promise((resolve, reject) => {
      signal?.throwIfAborted();
      const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 200);
      const abort = () => { clearTimeout(timer); reject(signal.reason); };
      signal?.addEventListener('abort', abort, { once: true });
    });
    export async function openCapsule(capsule, options) {
      await delay(options.signal);
      globalThis.syntheticOpenCount = (globalThis.syntheticOpenCount ?? 0) + 1;
      return {
        async embed(request) { await delay(request.options.signal); return { embedding: [1, 0] }; },
        async rerank(request) { await delay(request.options.signal); return { evidence: { scores: [{ index: 0, score: 1 }] } }; },
        async close() {},
      };
    }`));
  await page.route('**/runtime/src/tooling-exports/storage.js', route => respond(route, `
    const stores = new Map();
    export function createOpfsStore() {
      let files;
      return {
        async openModel(id) { if (!stores.has(id)) stores.set(id, new Map()); files = stores.get(id); },
        async readFile(name) { return files.get(name) ?? null; },
        async writeFile(name, bytes) { files.set(name, bytes.slice()); },
        async deleteFile(name) { files.delete(name); },
        async createWriteStream(name) { let bytes; return {
          async write(value) { bytes = value.slice(); }, async close() { files.set(name, bytes); }, async abort() {},
        }; },
      };
    }`));
  await page.goto('http://127.0.0.1:' + server.address().port + '/index.html');
  await page.waitForFunction(() => globalThis.documentSearch?.ready);
  await page.check('#retention');
  await page.click('#install');
  await page.waitForFunction(() => !document.querySelector('#cancel-load').disabled);
  await page.click('#cancel-load');
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Operation cancelled.');
  await page.click('#install');
  await page.waitForFunction(() => globalThis.documentSearch.controller.getState().hasSessions && !globalThis.documentSearch.controller.getState().isInitializing);
  await page.setInputFiles('#files', { name: 'same.txt', mimeType: 'text/plain', buffer: Buffer.from('Old complete document') });
  await page.click('#index');
  await page.waitForFunction(() => globalThis.documentSearch.controller.getState().hasIndex);
  const opens = await page.evaluate(() => globalThis.syntheticOpenCount);
  await page.fill('#query', 'query');
  await page.click('#search');
  await page.waitForFunction(() => !document.querySelector('#cancel-search').disabled);
  await page.click('#cancel-search');
  await page.waitForFunction(() => !globalThis.documentSearch.controller.getState().isSearching);
  await page.click('#search');
  await page.waitForFunction(() => document.querySelectorAll('#results article').length === 1);
  assert.equal(await page.evaluate(() => globalThis.syntheticOpenCount), opens);
  await page.setInputFiles('#files', { name: 'same.txt', mimeType: 'text/plain', buffer: Buffer.from('Replacement text') });
  await page.click('#index');
  await page.waitForFunction(() => !document.querySelector('#cancel-index').disabled);
  await page.click('#cancel-index');
  await page.waitForFunction(() => !globalThis.documentSearch.controller.getState().isIndexing);
  assert.equal(await page.evaluate(() => globalThis.documentSearch.controller.getIndex().documents[0].text), 'Old complete document');
  await page.click('#close');
  await page.waitForFunction(() => !globalThis.documentSearch.controller.getState().hasSessions);
  assert.deepEqual(errors, []);
} finally {
  await context.close(); await browser.close();
  await new Promise(resolve => server.close(resolve));
}
console.log('document-search-ui: real loading/query/index cancellation controls and reuse passed; synthetic models');
