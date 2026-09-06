import assert from 'node:assert/strict';
import { qualifyRerankerElectron, compileElectronReleasePreload, resolveRerankerPackDistribution, resolvePinnedElectronHost } from '../../tools/qualify-reranker-electron.js';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import os from 'node:os';
import path from 'node:path';
import { ELECTRON_RELEASE_IPC_CHANNEL } from '../../src/client/electron/ipc-contract.js';
import { buildRerankerEvaluationPack } from '../../tools/build-reranker-evaluation-pack.js';
import { createRerankReferenceFixture } from '../helpers/rerank-reference-fixture.js';

let bridge;
const requests = [];
const preloadSource = await fs.readFile(new URL('../../examples/electron-document-search/preload.js', import.meta.url), 'utf8');
vm.runInNewContext(compileElectronReleasePreload(preloadSource, ELECTRON_RELEASE_IPC_CHANNEL), {
  exports: {}, require(name) {
    assert.equal(name, 'electron');
    return { contextBridge: { exposeInMainWorld(name, value) { assert.equal(name, 'dopplerRelease'); bridge = value; } },
      ipcRenderer: { async invoke(channel, request) { requests.push({ channel, ...request }); } } };
  },
});
await bridge.resolveCurrent();
await bridge.rollback('application-reference');
assert.equal(requests[0].channel, ELECTRON_RELEASE_IPC_CHANNEL);
assert.equal(requests[0].action, 'resolve-current');
assert.equal(requests[1].customerAuthorizationDigest, 'application-reference');
assert.throws(() => compileElectronReleasePreload(preloadSource.replace('doppler-gpu/electron', 'unknown-module'), ELECTRON_RELEASE_IPC_CHANNEL), /Unsupported installed/);
assert.throws(() => compileElectronReleasePreload('export function other() {}', ELECTRON_RELEASE_IPC_CHANNEL), /installed channel/);
assert.throws(() => compileElectronReleasePreload(preloadSource, undefined), /installed channel/);
const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-pack-distribution-'));
try {
  const hostRoot = path.join(directory, 'host');
  const childRoot = path.join(hostRoot, 'child');
  const electronDir = path.join(hostRoot, 'node_modules/electron');
  await fs.mkdir(childRoot, { recursive: true });
  await assert.rejects(resolvePinnedElectronHost(hostRoot, '43.4.0'), /Pinned Electron is unavailable/);
  await fs.mkdir(electronDir, { recursive: true });
  await fs.writeFile(path.join(electronDir, 'package.json'), JSON.stringify({ name: 'electron', version: '43.4.0', main: 'index.js' }));
  const binary = path.join(electronDir, 'test-binary');
  await fs.writeFile(binary, 'synthetic executable identity; never launched');
  await fs.writeFile(path.join(electronDir, 'index.js'), `module.exports = ${JSON.stringify(binary)};\n`);
  const host = await resolvePinnedElectronHost(hostRoot, '43.4.0');
  assert.equal(host.executablePath, binary);
  assert.equal(host.resolution, 'qualification-checkout-local');
  assert.match(host.executableSha256, /^[a-f0-9]{64}$/);
  await assert.rejects(resolvePinnedElectronHost(hostRoot, '43.5.0'), /Pinned Electron 43.5.0 required/);
  await assert.rejects(resolvePinnedElectronHost(childRoot, '43.4.0'), /must not borrow Electron/);
  const outsideBinary = path.join(directory, 'external-binary');
  await fs.writeFile(outsideBinary, 'outside package');
  await fs.rm(binary);
  await fs.symlink(outsideBinary, binary);
  await assert.rejects(resolvePinnedElectronHost(hostRoot, '43.4.0'), /external binary overrides/);
  await fs.mkdir(path.join(directory, 'distribution'));
  await fs.mkdir(path.join(directory, 'distribution/version one'));
  const manifest = path.join(directory, 'distribution/version one/pack.json');
  await fs.writeFile(manifest, '{}');
  const distribution = await resolveRerankerPackDistribution(manifest, path.join(directory, 'distribution'));
  assert.equal(distribution.urlPath, '/pack/version%20one/pack.json');
  assert.equal((await resolveRerankerPackDistribution(manifest)).urlPath, '/pack/pack.json');
  const outside = path.join(directory, 'outside.json');
  await fs.writeFile(outside, '{}');
  await assert.rejects(resolveRerankerPackDistribution(outside, distribution.root), /inside.*distribution root/);
  await fs.symlink(outside, path.join(distribution.root, 'escaped.json'));
  await assert.rejects(resolveRerankerPackDistribution(path.join(distribution.root, 'escaped.json'), distribution.root), /inside.*distribution root/);
} finally { await fs.rm(directory, { recursive: true, force: true }); }
await assert.rejects(qualifyRerankerElectron({ mode: 'model', packDistributionRoot: '/tmp/distribution' }), /Pack distribution root/);
await assert.rejects(qualifyRerankerElectron({ mode: 'pack', packDistributionRoot: 'relative' }), /Pack distribution root/);
await assert.rejects(qualifyRerankerElectron({ mode: 'pack', releaseCoordinator: {} }), /Release coordinator qualification/);
await assert.rejects(qualifyRerankerElectron({ mode: 'pack', releaseCoordinator: {
  statePath: '/private/release.json', trustedSigners: {}, now: '2026-09-06T00:00:00.000Z', actions: [], allowedRendererActions: ['activate'],
} }), /read-only renderer permissions/);
for (const config of [
  { mode: 'model', releaseCheckpointPath: '/private/checkpoint.json' },
  { mode: 'pack', releaseCheckpointPath: 'relative.json' },
  { mode: 'pack', releaseCheckpointPath: '/private/checkpoint.json' },
  { mode: 'pack', releaseCheckpointPath: '/private/checkpoint.json', openOptions: {
    releaseEvents: [], releaseTrustedSigners: {}, releasePolicy: { checkpoint: { sequence: 0, digest: null } },
  } },
]) await assert.rejects(qualifyRerankerElectron(config), /Durable release qualification/);
await assert.rejects(qualifyRerankerElectron({ mode: 'pack', fault: { kind: 'unknown' } }), /Unsupported qualification fault/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model', fault: { kind: 'device-loss' } }), /Unsupported qualification fault/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model' }), /requires policyPath/);
await assert.rejects(qualifyRerankerElectron({ mode: 'pack', diagnosticCapture: {
  documentIndex: 0, captureConfig: {} } }), /diagnosticCapture requires model mode/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model', diagnosticCapture: {
  documentIndex: -1, captureConfig: {} } }), /non-negative documentIndex/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model', diagnosticCapture: {
  documentIndex: 0, captureConfig: { defaultLevel: 'invalid' } } }), /CapturePolicy/);
const config = { mode: 'pack', policyPath: 'missing-policy', referencePath: 'missing-reference',
  modelDir: 'missing-model', packageRoot: 'missing-package', outputDir: 'must-not-be-created' };
await assert.rejects(qualifyRerankerElectron(config), /retained packageBundlePath/);
await assert.rejects(buildRerankerEvaluationPack({}), /requires qualificationPath/);
await assert.rejects(buildRerankerEvaluationPack({ qualificationPath: 'missing', conversionConfigPath: 'missing',
  licensePath: 'missing', applicationPath: 'missing', outputDir: 'must-not-be-created', authorityId: 'test' }), /explicit fail-closed/);
const qualificationDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-evaluation-surface-'));
try {
  const transcript = createRerankReferenceFixture();
  const qualificationPath = path.join(qualificationDir, 'report.json');
  const outputDir = path.join(qualificationDir, 'candidate');
  for (const surface of [undefined, 'cpu', 'webgpu-unspecified']) {
    await fs.writeFile(qualificationPath, JSON.stringify({ schema: 'doppler.rerankModelQualification.v1', passed: true,
      reference: transcript.reference, observation: transcript.observation, runtime: { surface } }));
    await assert.rejects(buildRerankerEvaluationPack({ qualificationPath, outputDir,
      conversionConfigPath: 'unused', licensePath: 'unused', applicationPath: 'unused', authorityId: 'test',
      revocation: { offlineExpirySeconds: 60, failClosedAfterExpiry: true } }), /explicitly observed WebGPU surface/);
    await assert.rejects(fs.access(outputDir), error => error.code === 'ENOENT', 'reject before creating signing custody');
  }
} finally { await fs.rm(qualificationDir, { recursive: true, force: true }); }
console.log('reranker-evaluation-preflight.test: ok (no hardware launched)');
