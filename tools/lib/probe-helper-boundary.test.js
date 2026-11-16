import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import * as server from './probe-server.js';
import * as peerServer from '../pooled-equal-weights/probe-server.js';
import * as budget from './gpu-memory-budget.js';
import * as peerBudget from '../pooled-equal-weights/gpu-memory-budget.js';

test('snapshot executes committed bytes, accepts internal aliases, freezes dependencies, and detects drift', async t => {
  const { execFileSync } = await import('node:child_process');
  const { freezeResidentInputs, verifyResidentInputs, fileInventory, hashFile } = await import('../pooled-equal-weights/resident-snapshot.js');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'resident-snapshot-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const repo = path.join(root, 'repo'), vendor = path.join(root, 'vendor');
  await fs.mkdir(repo); await fs.mkdir(vendor);
  const git = (directory, args) => execFileSync('git', ['-C', directory, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  for (const [directory, files] of [[repo, ['src/engine.js', 'src/config/policy.json', 'demo/a.js', 'tests/a.js',
    'benchmarks/vendors/pooled-equal-weights/contract-f32kv.json', 'models/catalog.json', 'package.json', 'package-lock.json',
    'tools/pooled-equal-weights/adapter.js', 'tools/lib/hash-file.js', 'tools/lib/probe-server.js', 'tools/lib/gpu-memory-budget.js']], [vendor, ['engine/main.js']]]) {
    for (const file of files) { await fs.mkdir(path.dirname(path.join(directory, file)), { recursive: true }); await fs.writeFile(path.join(directory, file), 'committed'); }
    git(directory, ['init']);
    if (directory === repo) await fs.symlink('config', path.join(repo, 'src/alias'));
    git(directory, ['add', '.']);
    git(directory, ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture']);
  }
  for (const name of ['playwright', 'playwright-core']) {
    await fs.mkdir(path.join(repo, 'node_modules', name), { recursive: true });
    await fs.writeFile(path.join(repo, 'node_modules', name, 'index.js'), 'dependency');
  }
  const canonicalDir = path.join(root, 'canonical'), rdrrDir = path.join(root, 'rdrr'), ggufPath = path.join(root, 'metadata.gguf');
  for (const dir of [canonicalDir, rdrrDir]) { await fs.mkdir(dir); await fs.writeFile(path.join(dir, 'file'), 'artifact'); }
  await fs.writeFile(ggufPath, 'header');
  await fs.writeFile(path.join(repo, 'src/engine.js'), 'uncommitted-WIP');
  const pooledIdentity = path.join(root, 'vendor-identity.json');
  await fs.writeFile(pooledIdentity, JSON.stringify({ vendorChanges: [], engineFiles: (await fileInventory(path.join(vendor, 'engine'))).map(file => ({ ...file, path: `vendor/engine/${file.path}` })) }));
  const config = { dopplerRoot: repo, dopplerCommit: git(repo, ['rev-parse', 'HEAD']), harnessCommit: git(repo, ['rev-parse', 'HEAD']), pooledRoot: vendor,
    pooledSourceIdentity: { path: pooledIdentity, sha256: await hashFile(pooledIdentity) }, canonicalDir, rdrrDir, ggufPath,
    dependencyPaths: ['node_modules/playwright', 'node_modules/playwright-core'], browserExecutable: process.execPath, browserArgs: [] };
  const output = path.join(root, 'output');
  const identity = await freezeResidentInputs(config, JSON.stringify(config), output);
  assert.equal(await fs.readFile(path.join(output, 'runtime/src/engine.js'), 'utf8'), 'committed');
  assert.equal(await fs.readFile(path.join(repo, 'src/engine.js'), 'utf8'), 'uncommitted-WIP');
  assert.equal(identity.runtimeFiles.find(file => file.path === 'src/alias').symlink, 'config');
  assert.ok(identity.runtimeFiles.some(file => file.path === 'tools/lib/probe-server.js'));
  assert.ok(identity.helpers.archiveSha256);
  assert.equal(identity.browser.sha256, await hashFile(process.execPath));
  await verifyResidentInputs(config, identity, output);
  const helper = path.join(output, 'runtime/tools/lib/probe-server.js');
  await fs.chmod(helper, 0o644);
  await fs.writeFile(helper, 'helper drift');
  await assert.rejects(verifyResidentInputs(config, identity, output), /drift/);
  await fs.writeFile(helper, 'committed');
  await fs.chmod(path.join(output, 'runtime/node_modules/playwright/index.js'), 0o644);
  await fs.writeFile(path.join(output, 'runtime/node_modules/playwright/index.js'), 'drift');
  await assert.rejects(verifyResidentInputs(config, identity, output), /drift/);
  await fs.symlink(path.join(repo, 'src/engine.js'), path.join(rdrrDir, 'outside'));
  await assert.rejects(fileInventory(rdrrDir), /escapes frozen root/);
});


test('generic helpers and peer compatibility entries share implementation', async () => {
  assert.equal(server.createProbeServer, peerServer.createProbeServer);
  assert.equal(budget.assessGpuBenchmarkMemory, peerBudget.assessGpuBenchmarkMemory);
  const instance = await server.createProbeServer({ entry: 'probe.js', config: { version: 1 }, routes: [] });
  try {
    const response = await fetch(`${server.probeOrigin(instance)}/probe-config.json`);
    assert.deepEqual(await response.json(), { version: 1 });
    assert.equal((await fetch(`${server.probeOrigin(instance)}/absent`)).status, 404);
  } finally { await server.closeProbeServer(instance); }
});
