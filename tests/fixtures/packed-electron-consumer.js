import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { createDocumentSearchRenderer } from './renderer.js';
import { exposeDocumentSearchReleaseBridge } from './preload.js';
import { registerDocumentSearchReleaseMain } from './main.js';
import { runElectronPackContract } from './electron-pack-contract.js';
import { createDocumentSearchReleaseStore, createDocumentSearchCheckpointStore } from './release-storage.js';

const fixture = JSON.parse(await fs.readFile(new URL('./pack-fixture.json', import.meta.url), 'utf8'));
const bytes = new Map(fixture.artifacts.map(([id, values]) => [id, Uint8Array.from(values)]));
const artifactStore = {
  async hashArtifact(artifact) {
    const value = bytes.get(artifact.artifactId);
    return { hash: `sha256:${createHash('sha256').update(value).digest('hex')}`, sizeBytes: value.byteLength };
  },
  async readArtifact(artifact) { return bytes.get(artifact.artifactId).slice(); },
};
await runElectronPackContract({
  fixture: { pack: fixture.pack, artifactStore },
  trustedSigners: fixture.trustedSigners,
  createRenderer: createDocumentSearchRenderer,
});

let handler;
let bridge;
registerDocumentSearchReleaseMain({
  stateStore: { async load() { return null; }, async compareAndSwap() { return false; } },
  verifyReleaseDecision: async () => false,
  verifyRevocationSnapshot: async () => false,
  authorizeRequest: (_event, request) => ['status', 'resolve-current'].includes(request.action),
  ipcMain: { handle(_channel, value) { handler = value; } },
});
exposeDocumentSearchReleaseBridge(
  { exposeInMainWorld(_name, value) { bridge = value; } },
  { invoke(_channel, request) { return handler({}, request); } },
);
assert.equal((await bridge.status()).current, null);
await assert.rejects(bridge.resolveCurrent(), /active Pack/);
const privateDirectory = await fs.mkdtemp(path.join(tmpdir(), 'doppler-installed-release-'));
try {
  const filename = path.join(privateDirectory, 'checkpoint.json');
  const checkpoint = { sequence: 1, digest: `sha256:${'a'.repeat(64)}` };
  const store = createDocumentSearchCheckpointStore(filename);
  assert.equal(await store.load(), null);
  assert.equal(await store.compareAndSwap(0, checkpoint), true);
  assert.deepEqual(await createDocumentSearchCheckpointStore(filename).load(), checkpoint);
  assert.equal(await store.compareAndSwap(0, { ...checkpoint, sequence: 2 }), false);
  const releaseStore = createDocumentSearchReleaseStore(path.join(privateDirectory, 'releases.json'));
  assert.equal(await releaseStore.load(), null);
  await assert.rejects(releaseStore.compareAndSwap(0, checkpoint), /state|release/i);
} finally {
  await fs.rm(privateDirectory, { recursive: true, force: true });
}
console.log('installed Electron Pack contract passed (synthetic device/program, no repository imports)');
