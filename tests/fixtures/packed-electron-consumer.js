import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createDocumentSearchRenderer } from './renderer.js';
import { exposeDocumentSearchReleaseBridge } from './preload.js';
import { registerDocumentSearchReleaseMain } from './main.js';
import { runElectronPackContract } from './electron-pack-contract.js';

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
  ipcMain: { handle(_channel, value) { handler = value; } },
});
exposeDocumentSearchReleaseBridge(
  { exposeInMainWorld(_name, value) { bridge = value; } },
  { invoke(_channel, request) { return handler({}, request); } },
);
assert.equal((await bridge.status()).current, null);
await assert.rejects(bridge.resolveCurrent(), /active Pack/);
console.log('installed Electron Pack contract passed (synthetic device/program, no repository imports)');
