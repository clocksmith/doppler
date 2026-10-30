import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateResidentPlan, residentSchedule, runResidentPairs } from './resident-pairs.js';
import { summarizeResidentPairs } from './resident-statistics.js';
import { measureDopplerRequest, measurePooledRequest } from './resident-request.js';
import { createProbeServer, closeProbeServer, probeOrigin } from './probe-server.js';
import { runTextInference } from '../../src/tooling/browser-harness/text-execution.js';

const contract = JSON.parse(await fs.readFile(new URL('../../benchmarks/vendors/pooled-equal-weights/contract-f32kv.json', import.meta.url)));
const method = JSON.parse(await fs.readFile(new URL('./resident-method-f32kv.json', import.meta.url)));
const prompts = ['code', 'explanation', 'arithmetic', 'long-context'].map(id => ({ id, text: id,
  promptIds: [1, 2], expectedText: 'response', expectedTokenIds: Array.from({ length: 64 }, (_, i) => i + 10) }));
function fixture(change = () => {}) {
  const calls = [], records = [];
  let writing = false;
  const sessions = Object.fromEntries(['pooled', 'doppler'].map(engine => [engine, {
    async ready() { calls.push(`ready:${engine}`); },
    async prepare(prompt) { assert.equal(writing, false); calls.push(`prepare:${engine}:${prompt.id}`); },
    async request(prompt) {
      calls.push(`request:${engine}:${prompt.id}`);
      const completeMs = engine === 'doppler' ? 80 : 100;
      const result = { promptIds: prompt.promptIds, tokenIds: [...prompt.expectedTokenIds], text: 'response',
        completeMs, firstTokenMs: completeMs / 2, deliveryMs: Array.from({ length: 64 }, (_, i) => completeMs / 2 + i / 100), stopReason: 'max-tokens' };
      change(result, engine, calls);
      return result;
    },
    async close() { calls.push(`close:${engine}`); },
  }]));
  return { sessions, calls, records, async record(row) {
    writing = true; await Promise.resolve(); records.push(structuredClone(row)); writing = false;
  } };
}

test('both models ready first; 20 AB/BA pairs serialized; warmups retained but excluded from estimator', async () => {
  const f = fixture();
  const rows = await runResidentPairs({ ...f, contract, method, prompts });
  assert.deepEqual(f.calls.slice(0, 2), ['ready:pooled', 'ready:doppler']);
  assert.deepEqual(rows.map(({ sequence, promptId, engine, warmup, pair }) => ({ sequence, promptId, engine, warmup, pair })), residentSchedule(method, prompts));
  assert.equal(rows.length, 48);
  assert.equal(rows.filter(row => row.warmup).length, 8);
  assert.equal(rows.filter(row => !row.warmup && row.sequence % 2 === 0 && row.engine === 'pooled').length, 10);
  assert.equal(f.records.length, 48);
  for (const row of rows.filter(row => row.warmup)) row.completeMs = 100000;
  const summary = summarizeResidentPairs(rows, prompts, contract, method);
  assert.ok(Math.abs(summary.completeRatio - 0.8) < 1e-12);
  assert.ok(summary.interval95.every(value => Math.abs(value - 0.8) < 1e-12));
  assert.equal(summary.pairs, 20);
  assert.equal(summary.performancePass, true);
  assert.deepEqual(f.calls.slice(-2), ['close:pooled', 'close:doppler']);
});

test('seeded fixed-prompt bootstrap retains stalls and applies per-prompt regression gates', async () => {
  const f = fixture((row, engine, calls) => {
    if (engine === 'doppler' && calls.at(-1).endsWith(':long-context')) {
      row.completeMs = 110; row.firstTokenMs = 56; row.deliveryMs = Array(64).fill(56);
    }
  });
  const rows = await runResidentPairs({ ...f, contract, method, prompts });
  rows.find(row => !row.warmup && row.engine === 'doppler').completeMs = 400;
  const one = summarizeResidentPairs(rows, prompts, contract, method);
  const two = summarizeResidentPairs(rows, prompts, contract, method);
  assert.deepEqual(one, two);
  assert.equal(one.outliersRemoved, 0);
  assert.equal(one.checks.perPromptComplete, false);
  assert.equal(one.checks.perPromptFirstToken, false);
  assert.equal(one.performancePass, false);
});

for (const kind of ['mismatch', 'eos', 'throw', 'prepare', 'persist']) test(`retains first ${kind} failure, never retries, always closes both realms`, async () => {
  const f = fixture((row, engine) => {
    if (engine !== 'pooled') return;
    if (kind === 'mismatch') row.tokenIds[1]++;
    if (kind === 'eos') { row.tokenIds = [contract.shared.eosTokenIds[0]]; row.stopReason = 'eos'; }
    if (kind === 'throw') { const error = new Error('request failed'); error.partialRequest = { tokenIds: [17], deliveryMs: [2] }; throw error; }
  });
  if (kind === 'prepare') f.sessions.pooled.prepare = async () => { throw new Error('reset failed'); };
  if (kind === 'persist') f.record = async row => { f.records.push(structuredClone(row)); throw new Error('disk failed'); };
  await assert.rejects(runResidentPairs({ ...f, contract, method, prompts }), error => {
    assert.equal(error.attempts.length, 1);
    assert.equal(f.records.length, 1);
    if (kind !== 'persist') assert.equal(f.records[0].status, 'failed');
    if (kind === 'throw') assert.deepEqual(f.records[0].tokenIds, [17]);
    return true;
  });
  assert.ok(f.calls.filter(call => call.startsWith('request:')).length <= 1);
  assert.deepEqual(f.calls.slice(-2), ['close:pooled', 'close:doppler']);
  assert.throws(() => summarizeResidentPairs(f.records, prompts, contract, method), /Incomplete/);
});

test('Doppler delegates supported inputIds and native text; harness decode diagnostics happen outside clock', async () => {
  let clock = 0, fenced = false, encodeCalls = 0, diagnosticDecodes = 0;
  const pipeline = {
    tokenizer: { encode() { encodeCalls++; clock += 100; return [1, 2]; }, decode() {
      assert.equal(fenced, true); diagnosticDecodes++; clock += 1000; return 'x';
    } },
    async *generate(prompt, options) {
      assert.deepEqual(options.inputIds, [1, 2]);
      assert.equal(typeof prompt, 'string');
      clock += 5;
      for (const id of prompts[0].expectedTokenIds) { clock += 2; yield 'x'; options.onToken(id, 'x'); }
    },
  };
  const device = { queue: { async onSubmittedWorkDone() { fenced = true; clock++; } } };
  const result = await measureDopplerRequest({ pipeline, device, runTextInference, runtimeConfig: {
    inference: { sampling: { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1 } },
  }, prompt: prompts[0], contract, method, now: () => clock });
  assert.equal(result.completeMs, 134);
  assert.equal(result.firstTokenMs, 7);
  assert.equal(result.text, 'x'.repeat(64));
  assert.equal(encodeCalls, 1);
  assert.equal(diagnosticDecodes, 64);
  assert.equal(clock, 64234);
});

test('native Pooled EOS stops naturally; buffered text decoding and final fence stay inside clock', async () => {
  let clock = 0, forwards = 0;
  const result = await measurePooledRequest({ engine: {
    async prefillTokens(ids) { assert.deepEqual(ids, [1]); clock += 5; },
    async forwardTokenIds() { forwards++; clock += 2; return { ids: [contract.shared.eosTokenIds[0]] }; },
  }, device: { queue: { async onSubmittedWorkDone() { clock++; } } },
  tokenizer: { decode() { clock += 3; return ''; } }, prompt: prompts[0], contract, method, now: () => clock });
  assert.equal(forwards, 1); assert.equal(result.stopReason, 'eos');
  assert.equal(result.firstTokenMs, 7); assert.equal(result.completeMs, 11);
});

test('shared server serves immutable roots, proper byte ranges, wasm, and closes', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'resident-server-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.writeFile(path.join(root, 'data.wasm'), '0123456789');
  const server = await createProbeServer({ entry: 'resident-browser.js', config: { kind: 'test' }, routes: [['/data/', root]] });
  try {
    const response = await fetch(`${probeOrigin(server)}/data/data.wasm`, { headers: { Range: 'bytes=2-5' } });
    assert.equal(response.status, 206); assert.equal(await response.text(), '2345');
    assert.equal(response.headers.get('content-type'), 'application/wasm');
    assert.equal((await fetch(`${probeOrigin(server)}/missing`)).status, 404);
    assert.equal((await fetch(`${probeOrigin(server)}/data/data.wasm`, { headers: { Range: 'bytes=99-100' } })).status, 500);
    assert.deepEqual(await (await fetch(`${probeOrigin(server)}/probe-config.json`)).json(), { kind: 'test' });
  } finally { await closeProbeServer(server); }
});

test('plan rejects missing outputs and forced continuation after early EOS', () => {
  for (const mutate of [rows => rows[0].expectedTokenIds.pop(), rows => { rows[0].expectedTokenIds[0] = contract.shared.eosTokenIds[0]; }]) {
    const rows = structuredClone(prompts); mutate(rows);
    assert.throws(() => validateResidentPlan(contract, method, rows), /reference/);
  }
});

test('shared Doppler resource verifier reads actual slots and retains first mismatch', async () => {
  const { verifyDopplerGpu } = await import('./verify-doppler-gpu.js');
  const bytes = new Float32Array([1, -2, 3]).buffer;
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
  const buffer = { size: 256, label: 'actual-buffer', async mapAsync() {} };
  const weights = new Map([['layer_0', { qkvProj: { buffer } }], ['embed', buffer], ['final_norm', buffer], ['lm_head', { buffer }]]);
  const names = ['model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
    'model.language_model.embed_tokens.weight', 'model.language_model.norm.weight', 'model.language_model.lm_head.weight'];
  const canonical = { schema: 'doppler.equal-weight-canonical.v1', completed: true,
    tensors: names.map(name => ({ name, bytes: 12, canonicalBytesSha256: hash })) };
  let reads = 0;
  const read = async (actual, count) => { assert.equal(actual, buffer); assert.equal(count, 12); reads++; return bytes; };
  const result = await verifyDopplerGpu({ generator: { _state: { weights } } }, canonical, read);
  assert.equal(result.checks.length, 4); assert.equal(reads, 4);
  canonical.tensors[1].canonicalBytesSha256 = '0'.repeat(64);
  await assert.rejects(verifyDopplerGpu({ generator: { _state: { weights } } }, canonical, read), error => {
    assert.equal(error.completedChecks.length, 2); assert.equal(error.completedChecks[1].pass, false); return true;
  });
});

test('deadline failure is retained without retry and both session owners close', async () => {
  const { withDeadline } = await import('./run-resident-pairs.js');
  const f = fixture();
  f.sessions.pooled.request = () => withDeadline(new Promise(() => {}), 1, 'test-request');
  await assert.rejects(runResidentPairs({ ...f, contract, method, prompts }), /exceeded frozen timeout/);
  assert.equal(f.records.length, 1); assert.equal(f.records[0].status, 'failed');
  assert.deepEqual(f.calls.slice(-2), ['close:pooled', 'close:doppler']);
});

test('snapshot executes committed bytes, accepts internal aliases, freezes dependencies, and detects drift', async t => {
  const { execFileSync } = await import('node:child_process');
  const { freezeResidentInputs, verifyResidentInputs, fileInventory, hashFile } = await import('./resident-snapshot.js');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'resident-snapshot-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const repo = path.join(root, 'repo'), vendor = path.join(root, 'vendor');
  await fs.mkdir(repo); await fs.mkdir(vendor);
  const git = (directory, args) => execFileSync('git', ['-C', directory, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  for (const [directory, files] of [[repo, ['src/engine.js', 'src/config/policy.json', 'demo/a.js', 'tests/a.js',
    'benchmarks/vendors/pooled-equal-weights/contract-f32kv.json', 'models/catalog.json', 'package.json', 'package-lock.json',
    'tools/pooled-equal-weights/adapter.js']], [vendor, ['engine/main.js']]]) {
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
  assert.equal(identity.browser.sha256, await hashFile(process.execPath));
  await verifyResidentInputs(config, identity, output);
  await fs.chmod(path.join(output, 'runtime/node_modules/playwright/index.js'), 0o644);
  await fs.writeFile(path.join(output, 'runtime/node_modules/playwright/index.js'), 'drift');
  await assert.rejects(verifyResidentInputs(config, identity, output), /drift/);
  await fs.symlink(path.join(repo, 'src/engine.js'), path.join(rdrrDir, 'outside'));
  await assert.rejects(fileInventory(rdrrDir), /escapes frozen root/);
});

test('numerical reuse binds full source, inference overlay, actual receipt hashes, tokens/text and every logit position', async t => {
  const { admitResidentProofs } = await import('./resident-admission.js');
  const { fileInventory, hashFile } = await import('./resident-snapshot.js');
  const { execFileSync } = await import('node:child_process');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'resident-admission-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const git = args => execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  const sourcePaths = ['src/tooling/browser-harness/benchmark-execution.js', 'src/inference/pipelines/text/probes.js',
    'src/inference/pipelines/text/generator.js', 'src/inference/pipelines/text/generator-steps.js', 'src/inference/pipelines/text/generator/decode.js'];
  for (const file of sourcePaths) { await fs.mkdir(path.dirname(path.join(root, file)), { recursive: true }); await fs.writeFile(path.join(root, file), 'source'); }
  const sourceFiles = await fileInventory(root);
  git(['init']); git(['add', '.']); git(['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture']);
  const write = async (label, value) => {
    const file = path.join(root, `${label}.json`); await fs.writeFile(file, JSON.stringify(value));
    return { path: file, sha256: await hashFile(file) };
  };
  const source = await write('source', { commit: git(['rev-parse', 'HEAD']), files: sourceFiles });
  const original = sourceFiles[0].sha256;
  const observation = await write('observation', { completed: true, numericPrototype: null,
    file: sourcePaths[0], originalSha256: original, probeOriginalSha256: original,
    logitObservation: { originalSha256: original, originalStepsSha256: original, originalDecodeSha256: original } });
  const canonicalDir = path.join(root, 'canonical'); await fs.mkdir(canonicalDir);
  await fs.writeFile(path.join(canonicalDir, 'canonical.json'), JSON.stringify({ schema: 'doppler.equal-weight-canonical.v1', completed: true,
    sourceManifestSha256: contract.sourceManifestSha256, tensors: [{ name: 'tensor', canonicalBytesSha256: '0'.repeat(64) }] }));
  const inference = { sampling: { temperature: 0 }, executionPatch: { set: [] } };
  const prompt = prompts[0];
  const adapter = { vendor: 'fixture', architecture: 'mock', device: '0' };
  const doppler = await write('doppler', { ok: true, request: { runtimeConfig: { inference } }, result: { failed: 0, deviceInfo: { adapterInfo: adapter }, env: { browserUserAgent: 'fixture' }, metrics: {
    referenceTranscript: { prompt: { ids: prompt.promptIds }, tokens: { ids: prompt.expectedTokenIds } }, generatedText: prompt.expectedText } } });
  const pooled = await write('pooled', { promptIds: prompt.promptIds, tokenIds: prompt.expectedTokenIds, text: prompt.expectedText,
    adapter: { ...adapter, device: '' }, engineVariant: 'pooled-native-f32', settings: { kvDtype: 'f32', attnFlash: false }, gpuErrors: [], browserErrors: [] });
  const vendorFiles = [{ path: 'engine/main.js', sha256: original }];
  const pooledSource = await write('pooled-source', { vendorChanges: [], engineFiles: [{ path: 'vendor/engine/main.js', sha256: original }] });
  const comparisons = ['doppler-native', 'pooled-native', 'doppler-pooled'];
  const numerical = { schema: 'doppler.pooled-f32kv-numerical-comparison.v1', pass: true, prompts: [{ id: prompt.id, promptIdsEqual: true, tokenIdsEqual: true, tokenCount: 64,
    dopplerSha256: doppler.sha256, pooledSha256: pooled.sha256,
    checks: comparisons.flatMap(comparison => contract.gates.logitPositions.map(step => ({ comparison, step, pass: true, failedValues: 0, maxAbs: 0.0001 }))) }] };
  const binding = { schema: 'doppler.equal-weight-resident-numerical-binding.v1', dopplerSource: source,
    dopplerObservation: observation, numerical: await write('numerical', numerical), prompts: { [prompt.id]: { doppler, pooled, pooledSource } } };
  const config = { canonicalDir, dopplerRoot: root, doppler: { runtimeConfig: { inference } }, contract,
    method: { expectedTensorCount: 1 }, prompts: [prompt], proofBinding: await write('binding', binding) };
  const identity = { runtimeFiles: sourceFiles, vendorFiles, artifacts: { canonical: await fileInventory(canonicalDir) },
    doppler: { srcTree: git(['rev-parse', 'HEAD:src']) } };
  const admitted = await admitResidentProofs(config, identity, path.join(root, 'admitted'));
  assert.equal(admitted.pass, true); assert.ok(admitted.retained.length >= 7);
  const changed = structuredClone(config); changed.doppler.runtimeConfig.inference.executionPatch.set.push({ op: 'q_proj', kernelKey: 'wrong' });
  await assert.rejects(admitResidentProofs(changed, identity, path.join(root, 'wrong-graph')), /inference overlay/);
  numerical.prompts[0].checks.pop();
  binding.numerical = await write('numerical', numerical); config.proofBinding = await write('binding', binding);
  await assert.rejects(admitResidentProofs(config, identity, path.join(root, 'missing-logit')), /full logit positions/);
  numerical.teacherReceiptSha256 = '1'.repeat(64);
  binding.pooledNumerical = await write('pooled-numerical', numerical);
  const subgroup = { schema: 'doppler.subgroup-f32-numerical-comparison.v1', pass: true,
    sourceCommit: git(['rev-parse', 'HEAD']), teacherReceiptSha256: numerical.teacherReceiptSha256,
    cases: [{ id: prompt.id, promptIdsEqual: true, tokensEqual: true, tokens: 64, pass: true,
      rawReceiptSha256: doppler.sha256, pooledReceiptSha256: pooled.sha256,
      checks: ['native', 'pooled'].flatMap(against => contract.gates.logitPositions.map(step => ({ against, step, pass: true, maxAbs: 0.0001 }))) }] };
  binding.numerical = await write('subgroup-numerical', subgroup); config.proofBinding = await write('binding', binding);
  assert.equal((await admitResidentProofs(config, identity, path.join(root, 'subgroup-admitted'))).pass, true);
  subgroup.cases[0].checks[0].maxAbs = 0.0011;
  binding.numerical = await write('subgroup-numerical', subgroup); config.proofBinding = await write('binding', binding);
  await assert.rejects(admitResidentProofs(config, identity, path.join(root, 'subgroup-bad-value')), /Numerical values failed/);
});
