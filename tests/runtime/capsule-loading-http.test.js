import assert from 'node:assert/strict';
import http from 'node:http';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createFetchCapsuleArtifactStore } from '../../src/client/runtime/fetch-capsule-artifact-store.js';
import { openCapsule as browserOpenCapsule } from '../../src/client/doppler-api.browser.js';
import { openCapsule as nodeOpenCapsule } from '../../src/client/doppler-api.js';
import { fetchCapsuleMetadata } from '../../src/client/runtime/capsule-acquisition.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture();
const trustedSigners = { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY };
const requests = [];
let stallPath = null;
let onStall = null;
let corrupt = false;
let oversize = false;
const server = http.createServer((req, res) => {
  requests.push(req.url);
  res.writeHead(200);
  if (req.url === stallPath) {
    res.write(new Uint8Array([1]));
    onStall(res);
    return;
  }
  if (req.url === '/capsule.json') { res.end(JSON.stringify(fixture.capsule)); return; }
  const artifact = fixture.capsule.artifacts.find(row => `/${row.path}` === req.url);
  if (!artifact) { res.destroy(); return; }
  const bytes = fixture.artifactBytes.get(artifact.artifactId);
  res.end(oversize ? new Uint8Array(bytes.length + 1) : corrupt ? new Uint8Array(bytes.length) : bytes);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/capsule.json`;
let programs = 0;
let closes = 0;
const device = { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }),
  getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) };
const createRuntime = (overrides = {}) => createDopplerRuntime({ device, trustedSigners,
  artifactStore: createFetchCapsuleArtifactStore(url),
  async programFactory() { programs += 1; return { async close() { closes += 1; } }; }, ...overrides });
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
try {
  const invalid = createRuntime({ device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 0 }) } });
  await assert.rejects(invalid.openCapsule(fixture.capsule), /capability predicates/i);
  assert.equal(requests.length, 0, 'rejected hardware never contacts the artifact server');

  for (const scenario of ['metadata-cancel', 'node-metadata-cancel', 'metadata-timeout', 'artifact-cancel', 'artifact-timeout']) {
    const controller = new AbortController();
    const received = Promise.withResolvers();
    const disconnected = Promise.withResolvers();
    const metadata = scenario.includes('metadata');
    const timeout = scenario.endsWith('timeout');
    stallPath = metadata ? '/capsule.json' : '/weights.bin';
    onStall = response => { response.on('close', disconnected.resolve); received.resolve(); };
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const options = { signal: controller.signal, loadTimeoutMs: timeout ? 200 : 5000 };
    const opening = metadata ? (scenario === 'node-metadata-cancel' ? nodeOpenCapsule(url, options) : browserOpenCapsule(url, options))
      : createRuntime().openCapsule(fixture.capsule, options);
    const rejected = assert.rejects(opening, error => error.name === (timeout ? 'TimeoutError' : 'AbortError'));
    const watchdog = setTimeout(() => { controller.abort(); received.reject(new Error('HTTP acquisition did not reach its stall.')); }, 5000);
    try {
      await received.promise;
      if (!timeout) controller.abort();
      await rejected;
      await disconnected.promise;
    } finally { clearTimeout(watchdog); }
    assert.equal(programs, 0, 'cancelled acquisition cannot create a model');
  }
  stallPath = null;
  corrupt = true;
  await assert.rejects(createRuntime().openCapsule(fixture.capsule), /hash or size mismatch/);
  corrupt = false;
  oversize = true;
  await assert.rejects(createRuntime().openCapsule(fixture.capsule), /byte limit/);
  oversize = false;
  await assert.rejects(fetchCapsuleMetadata(url, { maxMetadataBytes: 4 }), /byte limit/);
  const progress = [];
  const observations = [];
  const session = await createRuntime({ observer: { observe: event => observations.push(event) } }).openCapsule(fixture.capsule,
    { onLoadProgress: event => progress.push(event), loadTimeoutMs: 5000 });
  assert.equal(programs, 1);
  for (const artifact of fixture.capsule.artifacts) {
    const events = progress.filter(event => event.artifactId === artifact.artifactId);
    assert.equal(events[0].loadedBytes, 0);
    assert.equal(events.at(-1).loadedBytes, artifact.sizeBytes);
    assert.equal(events.at(-1).totalBytes, artifact.sizeBytes);
  }
  const validation = observations.find(event => event.type === 'capsule-validation-complete');
  const totalBytes = fixture.capsule.artifacts.reduce((total, artifact) => total + artifact.sizeBytes, 0);
  assert.equal(validation.artifactMetrics.hashedBytes, totalBytes);
  assert.equal(validation.artifactMetrics.copiedBytes, totalBytes, 'closure verification does not request full copies');
  await session.close();
  assert.equal(closes, 1);
} finally {
  if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator);
  else delete globalThis.navigator;
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
console.log('capsule-loading-http.test: ok (real HTTP and signed Capsule; synthetic GPU)');
