import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { discoverModelRevision } from '../../tools/discover-model-revision.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

const root = process.cwd();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-discovery-'));
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
async function pin(file) { return { path: file, digest: hashBytesSha256(await fs.readFile(file)) }; }
const spec = await read('reports/model-ir-v2/qwen3.8-27b.spec.json');
const onboarding = await read('src/config/forge/onboarding/qwen3.8-27b.json');
onboarding.lineage = null;
const onboardingPath = path.join(temp, 'onboarding.json');
await fs.writeFile(onboardingPath, JSON.stringify(onboarding));
const files = new Map([
  ['config.json', await fs.readFile(spec.sources.config)],
  ['generation_config.json', await fs.readFile(spec.sources.generation)],
]);
const header = Buffer.from(JSON.stringify((await read(spec.sources.headers)).tensors));
const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length));
files.set('model-00001-of-00018.safetensors', Buffer.concat([prefix, header, Buffer.alloc(32)]));
const previousRevision = spec.sourceIdentity.revision, revision = 'a'.repeat(40);
const siblings = [...files].map(([rfilename, bytes]) => ({ rfilename,
  blobId: createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex') }));
const config = {
  schema: 'doppler.model-revision-discovery/v1', repository: spec.sourceIdentity.repository,
  ref: 'main', previousRevision, onboarding: await pin(onboardingPath),
  remoteSources: [
    { artifactId: 'config', path: 'config.json', format: 'json' },
    { artifactId: 'generation', path: 'generation_config.json', format: 'json' },
    { artifactId: 'headers', path: 'model-00001-of-00018.safetensors', format: 'safetensors-header' },
  ],
  limits: { jsonBytes: 1048576, headerBytes: 1048576, timeoutMs: 1000 },
};
let calls = [], mode = null;
async function fetchImpl(url, options) {
  calls.push(url);
  if (url.includes('/api/models/')) {
    const current = url.includes('/main?');
    return new Response(JSON.stringify({ sha: current ? revision : previousRevision,
      siblings: [...siblings, { rfilename: 'README.md', blobId: (current ? 'b' : 'c').repeat(40) }],
      cardData: { license: 'apache-2.0' } }), { status: 200 });
  }
  assert(url.includes(`/resolve/${revision}/`), 'Acquisition must pin the resolved revision, never main.');
  const filename = url.split(`/resolve/${revision}/`)[1];
  const bytes = files.get(filename); assert(bytes);
  if (options.headers.Range) {
    const [, first, last] = options.headers.Range.match(/bytes=(\d+)-(\d+)/);
    const a = Number(first), b = Number(last);
    if (mode === 'unbounded') return new Response(bytes, { status: 200 });
    return new Response(bytes.subarray(a, b + 1), { status: 206,
      headers: { 'content-range': `bytes ${mode === 'wrong-range' ? 1 : a}-${b}/${bytes.length}` } });
  }
  return new Response(mode === 'tamper' ? Buffer.from('{}') : bytes, { status: 200 });
}
async function run(name, input = config) {
  return discoverModelRevision(input, { sourceRoot: root, outputDir: path.join(temp, name), fetchImpl });
}
try {
  const result = await run('changed');
  assert.equal(result.passed, true);
  assert.equal(result.revisionChanged, true);
  assert.deepEqual(result.changedFiles.map(file => file.path), ['README.md']);
  assert.equal(result.assessment.status, 'recipe-required');
  assert.equal(result.qualified, false);
  assert.equal(result.published, false);
  assert.equal(result.files.find(file => file.artifactId === 'headers').wholeFileVerified, false);
  const model = await read(path.join(temp, 'changed/assessment/model-ir-receipt.json'));
  assert.equal(model.modelIR.sourceIdentity.revision, revision);
  assert(model.modelIR.blockClasses.some(block => block.kind === 'linear-recurrent-attention'));
  assert(model.modelIR.blockClasses.some(block => block.kind === 'full-attention'));
  const before = calls.length;
  await assert.rejects(run('changed'), /EEXIST/);
  assert.equal(calls.length, before, 'Retained attempts may not be overwritten or reacquired.');

  for (const [failure, pattern] of [['unbounded', /response contract/], ['wrong-range', /Content-Range/], ['tamper', /blob mismatch/]]) {
    mode = failure;
    await assert.rejects(run(failure), pattern);
    const receipt = await read(path.join(temp, failure, 'discovery.json'));
    assert.equal(receipt.passed, false); assert.equal(receipt.qualified, false);
    await assert.rejects(fs.stat(path.join(temp, failure, 'assessment/onboarding-result.json')), /ENOENT/);
  }
  mode = null;
  await assert.rejects(run('omit', { ...config, remoteSources: config.remoteSources.filter(source => source.artifactId !== 'generation') }), /must be acquired/);
  await assert.rejects(run('limit', { ...config, limits: { ...config.limits, jsonBytes: 8 } }), /byte limit/);
  await assert.rejects(run('traverse', { ...config, remoteSources: [{ ...config.remoteSources[0], path: '../config.json' }] }), /traverse/);
  await assert.rejects(run('duplicate', { ...config, remoteSources: [config.remoteSources[0], config.remoteSources[0]] }), /Duplicate/);
  await assert.rejects(run('unknown', { ...config, publish: true }), /requires exactly/);
  await assert.rejects(run('unpinned', { ...config, previousRevision: 'main' }), /Immutable/);
  await assert.rejects(run('changed-input', { ...config, onboarding: { ...config.onboarding, digest: `sha256:${'0'.repeat(64)}` } }), /Pinned input changed/);
} finally {
  await fs.rm(temp, { recursive: true, force: true });
}
console.log('model-revision-discovery.test: ok (source discovery and assessment, no physical qualification)');
