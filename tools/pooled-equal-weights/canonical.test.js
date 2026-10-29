import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { pathToFileURL } from 'node:url';
import { encodeSharedPrompt } from './shared-input.js';
import { preparePooledRounding } from './prepare-pooled.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function run(script, ...args) { return spawnSync(process.execPath, [path.join(here, script), ...args], { encoding: 'utf8' }); }
test('special chat tokens reach native inference as atomic IDs, preserving adjacent text', () => {
  const seen = [], tokenizer = { vocab: { '<|im_start|>': 101, '<|im_end|>': 102 },
    encode(text) { seen.push(text); return [...text].map(c => c.charCodeAt(0)); } };
  const format = { before: '<|im_start|>user\n', after: '<|im_end|>\n', specialTokens: ['<|im_start|>', '<|im_end|>'] };
  assert.deepEqual(encodeSharedPrompt(tokenizer, 'Hi', format), [101, ...[...'user\nHi'].map(c => c.charCodeAt(0)), 102, 10]);
  assert.equal(seen.some(x => x.includes('<|')), false);
  assert.throws(() => encodeSharedPrompt({ ...tokenizer, vocab: {} }, 'Hi', format), /Missing special token/);
});
async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-canonical-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const source = path.join(root, 'source'); await fs.mkdir(source);
  const a = Buffer.from(new Float32Array([1, -2, 3, 4]).buffer);
  const b = Buffer.from(new Uint16Array([0x3c00, 0xc000]).buffer);
  await fs.writeFile(path.join(source, 's0.bin'), a.subarray(0, 8));
  await fs.writeFile(path.join(source, 's1.bin'), Buffer.concat([a.subarray(8), b]));
  const manifest = { hashAlgorithm: 'sha256', shards: [
    { index: 0, filename: 's0.bin', size: 8, hash: sha(a.subarray(0, 8)) },
    { index: 1, filename: 's1.bin', size: 12, hash: sha(Buffer.concat([a.subarray(8), b])) }],
    tensors: {
      'text.matrix': { shape: [2, 2], dtype: 'F32', spans: [
        { shardIndex: 0, offset: 0, size: 8 }, { shardIndex: 1, offset: 0, size: 8 }] },
      'text.vector': { shape: [2], dtype: 'F16', shard: 1, offset: 8, size: 4 } } };
  const bytes = Buffer.from(JSON.stringify(manifest));
  await fs.writeFile(path.join(source, 'manifest.json'), bytes);
  await fs.writeFile(path.join(source, 'tokenizer.json'), '{"fixture":true}');
  const output = path.join(root, 'canonical'), config = path.join(root, 'config.json');
  await fs.writeFile(config, JSON.stringify({ schema: 'doppler.equal-weight-export.v1', sourceModelDir: source,
    outputDir: output, sourceManifestSha256: sha(bytes), tensorPrefix: 'text.' }));
  return { root, source, output, config, manifest, a, manifestBytes: bytes };
}

test('canonical export preserves split F32 tensors, exactly decodes F16, and leaves source bytes unchanged', async t => {
  const f = await fixture(t), result = run('export-canonical.js', f.config);
  assert.equal(result.status, 0, result.stderr);
  const proof = JSON.parse(await fs.readFile(path.join(f.output, 'canonical.json')));
  assert.equal(proof.completed, true); assert.equal(proof.tensors.length, 2);
  assert.equal(proof.tensors[0].canonicalBytesSha256, sha(f.a));
  assert.equal(proof.tensors[1].canonicalBytesSha256, sha(Buffer.from(new Float32Array([1, -2]).buffer)));
  assert.deepEqual(await fs.readFile(path.join(f.source, 'manifest.json')), f.manifestBytes);
  assert.deepEqual(await fs.readFile(path.join(f.source, 's0.bin')), f.a.subarray(0, 8));
  assert.notEqual(run('export-canonical.js', f.config).status, 0, 'must not overwrite an existing output');
});

test('changed source shard fails before a completed canonical artifact can be produced', async t => {
  const f = await fixture(t); await fs.writeFile(path.join(f.source, 's0.bin'), Buffer.alloc(8));
  const result = run('export-canonical.js', f.config);
  assert.notEqual(result.status, 0); assert.match(result.stderr, /Source shard integrity failed/);
  await assert.rejects(fs.stat(f.output), { code: 'ENOENT' });
});

test('converted tensor verification rejects a changed numeric value even when shape and dtype match', async t => {
  const f = await fixture(t); assert.equal(run('export-canonical.js', f.config).status, 0);
  const converted = path.join(f.root, 'converted'); await fs.mkdir(converted);
  const data = Buffer.concat([f.a, Buffer.from(new Float32Array([1, -2]).buffer)]);
  await fs.writeFile(path.join(converted, 'data.bin'), data);
  await fs.writeFile(path.join(converted, 'manifest.json'), JSON.stringify({ shards: [{ index: 0, filename: 'data.bin' }], tensors: {
    'text.matrix': { shape: [2, 2], dtype: 'F32', shard: 0, offset: 0, size: 16 },
    'text.vector': { shape: [2], dtype: 'F32', shard: 0, offset: 16, size: 8 } } }));
  const report = path.join(f.root, 'check.json');
  assert.equal(run('verify-rdrr.js', f.output, converted, report).status, 0);
  data.writeFloatLE(9, 4); await fs.writeFile(path.join(converted, 'data.bin'), data);
  assert.equal(run('verify-rdrr.js', f.output, converted, report).status, 1);
  assert.equal(JSON.parse(await fs.readFile(report)).pass, false);
});

test('diagnostic snapshots exclude local model caches and never alter the source observation owner', async t => {
  const f = await fixture(t), source = path.join(f.root, 'runtime'), output = path.join(f.root, 'snapshot');
  const owner = 'src/tooling/browser-harness/benchmark-execution.js';
  for (const dir of ['src/tooling/browser-harness', 'demo', 'tests', 'tools', 'benchmarks', 'models/local', 'node_modules']) {
    await fs.mkdir(path.join(source, dir), { recursive: true });
  }
  await fs.mkdir(path.join(source, 'src/inference/pipelines/text'), { recursive: true });
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/probes.js'), 'export async function runProbes(stage, buffer, options) {}');
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/generator-steps.js'), '        logitsBuffer = logits.logitsBuffer;');
  const original = '  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);';
  await fs.writeFile(path.join(source, owner), original);
  await fs.writeFile(path.join(source, 'models/local/cache.bin'), 'must-not-copy');
  await fs.writeFile(path.join(source, 'models/catalog.json'), '{}');
  await fs.writeFile(path.join(source, 'package.json'), '{"type":"module"}');
  const config = path.join(f.root, 'snapshot.json');
  await fs.writeFile(config, JSON.stringify({ sourceRoot: source, snapshotDir: output, harnessRelativeDir: 'src/tooling/browser-harness' }));
  const result = run('prepare-doppler-probe.js', config); assert.equal(result.status, 0, result.stderr);
  await assert.rejects(fs.stat(path.join(output, 'models/local')), { code: 'ENOENT' });
  assert.equal(await fs.readFile(path.join(source, owner), 'utf8'), original);
  assert.match(await fs.readFile(path.join(output, owner), 'utf8'), /metrics.canonicalFirstLogits/);
  assert.equal(await fs.readlink(path.join(output, 'node_modules')), path.join(source, 'node_modules'));
});

test('vendor rounding control requires pinned source bytes and records every changed file', async t => {
  const f = await fixture(t), vendor = path.join(f.root, 'vendor');
  await fs.mkdir(path.join(vendor, 'engine/wgsl'), { recursive: true });
  const originals = { 'engine/wgsl/qwen35.js': 'export const WGSL2 = /* wgsl */ `pack2x16float(a); pack2x16float(b); unpack2x16float(c);`;',
    'engine/wgsl/layer_fuse.js': 'pack2x16float(c); pack2x16float(d);' };
  for (const [name, text] of Object.entries(originals)) await fs.writeFile(path.join(vendor, name), text);
  const policy = { mode: 'explicit-nearest-even', originalSha256: Object.fromEntries(Object.entries(originals).map(([name, text]) => [name, sha(text)])) };
  await assert.rejects(preparePooledRounding(vendor, { ...policy, originalSha256: {} }), /source identity mismatch/);
  const changes = await preparePooledRounding(vendor, policy); assert.equal(changes.length, 2);
  assert.equal(changes.every(x => x.originalSha256 !== x.preparedSha256), true);
  assert.match(await fs.readFile(path.join(vendor, 'engine/wgsl/qwen35.js'), 'utf8'), /\$\{F16_WGSL\}/);
  assert.match(await fs.readFile(path.join(vendor, 'engine/wgsl/qwen35.js'), 'utf8'), /unpack2x16float\(c\)/);
  await assert.rejects(preparePooledRounding(vendor, policy), /source identity mismatch/);
});

test('numerical observations capture raw finite logits before sampling masks', async t => {
  const f = await fixture(t), source = path.join(f.root, 'runtime'), output = path.join(f.root, 'snapshot');
  for (const dir of ['src/tooling/browser-harness', 'src/inference/pipelines/text', 'demo', 'tests', 'tools', 'benchmarks', 'models', 'node_modules']) {
    await fs.mkdir(path.join(source, dir), { recursive: true });
  }
  await fs.writeFile(path.join(source, 'package.json'), '{"type":"module"}');
  await fs.writeFile(path.join(source, 'src/tooling/browser-harness/benchmark-execution.js'),
    '    for (let i = 0; i < warmupRuns + timedRuns; i++) {}\n'
    + '    for (let i = 0; i < warmupRuns + timedRuns; i++) {\n'
    + '      const run = await withHarnessPhase(\n        `bench.runTextInference[${i}]`, options, callback);\n}\n'
    + '  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);');
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/probes.js'), 'export async function runProbes(stage, buffer, options) {}');
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/generator.js'), `export class Probe {
  _sampleNextTokenFromLogits(logits, generatedIds, opts) {
    const sampled = Float32Array.from(logits); sampled[1] = -Infinity; return sampled;
  }
}`);
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/generator-steps.js'), '        logitsBuffer = logits.logitsBuffer;');
  await fs.mkdir(path.join(source, 'src/inference/pipelines/text/generator'), { recursive: true });
  await fs.writeFile(path.join(source, 'src/inference/pipelines/text/generator/decode.js'), `
export function recorded(resources, logitsBuffer) {
      resources.register(logitsBuffer, 'decode logits');
}
function extractLastPositionLogits(rawLogits) { return rawLogits; }
export function immediate(rawLogits, numTokens, config, currentIds) {
        const sampledLogits = extractLastPositionLogits(rawLogits, numTokens, config.vocabSize);
        sampledLogits[1] = -Infinity;
        return sampledLogits;
}
`);
  const config = path.join(f.root, 'snapshot.json');
  await fs.writeFile(config, JSON.stringify({ sourceRoot: source, snapshotDir: output,
    harnessRelativeDir: 'src/tooling/browser-harness', captureLogitPositions: [0, 1] }));
  const result = run('prepare-doppler-probe.js', config); assert.equal(result.status, 0, result.stderr);
  const syntax = spawnSync(process.execPath, ['--check', path.join(output, 'src/tooling/browser-harness/benchmark-execution.js')], { encoding: 'utf8' });
  assert.equal(syntax.status, 0, syntax.stderr);
  delete globalThis.__canonicalPromptLength; delete globalThis.__canonicalFullLogitRows;
  try {
    const { Probe } = await import(pathToFileURL(path.join(output, 'src/inference/pipelines/text/generator.js')));
    const sampled = new Probe()._sampleNextTokenFromLogits(new Float32Array([1, 42]), [], {});
    assert.equal(sampled[1], -Infinity);
    assert.deepEqual(globalThis.__canonicalFullLogitRows, [{ index: 0, stage: 'pre-sampling', values: [1, 42] }]);
    globalThis.__canonicalLogitCaptureActive = true;
    const { immediate } = await import(pathToFileURL(path.join(output, 'src/inference/pipelines/text/generator/decode.js')));
    const next = immediate(new Float32Array([2, 43]), 1, { vocabSize: 2 }, [5]);
    assert.equal(next[1], -Infinity);
    assert.deepEqual(globalThis.__canonicalFullLogitRows[1], { index: 1, stage: 'pre-sampling', values: [2, 43] });
  } finally { delete globalThis.__canonicalPromptLength; delete globalThis.__canonicalFullLogitRows; delete globalThis.__canonicalLogitCaptureActive; }
});

test('failed preparation retains its failure without publishing a runnable snapshot', async t => {
  const f = await fixture(t), source = path.join(f.root, 'runtime'), output = path.join(f.root, 'snapshot');
  for (const dir of ['src', 'demo', 'tests', 'tools', 'benchmarks', 'models', 'node_modules']) await fs.mkdir(path.join(source, dir), { recursive: true });
  await fs.writeFile(path.join(source, 'package.json'), '{"type":"module"}');
  const config = path.join(f.root, 'broken-snapshot.json');
  await fs.writeFile(config, JSON.stringify({ sourceRoot: source, snapshotDir: output, harnessRelativeDir: 'src/tooling/browser-harness' }));
  const result = run('prepare-doppler-probe.js', config); assert.notEqual(result.status, 0);
  await assert.rejects(fs.stat(output), { code: 'ENOENT' });
  const incomplete = (await fs.readdir(f.root)).filter(name => name.startsWith('snapshot.preparing-'));
  assert.equal(incomplete.length, 1);
  const failure = JSON.parse(await fs.readFile(path.join(f.root, incomplete[0], 'preparation-failure.json')));
  assert.equal(failure.completed, false); assert.match(failure.error, /ENOENT/);
});
