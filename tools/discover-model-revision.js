#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { runModelOnboarding } from '../src/tooling/model-onboarding.js';
import { hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { readSafetensorsHeaderLength } from '../src/converter/safetensors-header-evidence.js';

function exact(value, keys, label) {
  assert(value && typeof value === 'object' && !Array.isArray(value), `${label} must be an object.`);
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), `${label} requires exactly ${keys.join(', ')}.`);
}

function relative(value) {
  assert(typeof value === 'string' && value.length > 0 && !value.includes('\\')
    && value.split('/').every(part => part && part !== '.' && part !== '..')
    && !path.isAbsolute(value), 'Source paths must be relative and cannot traverse directories.');
  return value;
}

export async function discoverModelRevision(config, { sourceRoot, outputDir, fetchImpl = fetch }) {
  exact(config, ['schema', 'repository', 'ref', 'previousRevision', 'onboarding', 'remoteSources', 'limits'], 'Discovery config');
  assert.equal(config.schema, 'doppler.model-revision-discovery/v1');
  assert(/^[\w.-]+\/[\w.-]+$/.test(config.repository), 'Explicit upstream repository required.');
  assert(typeof config.ref === 'string' && config.ref.trim(), 'Upstream ref required.');
  assert(/^[a-f0-9]{40}$/.test(config.previousRevision), 'Immutable previous revision required.');
  exact(config.limits, ['jsonBytes', 'headerBytes', 'timeoutMs'], 'Discovery limits');
  for (const value of Object.values(config.limits)) assert(Number.isSafeInteger(value) && value > 0, 'Positive integer limits required.');
  assert(Array.isArray(config.remoteSources) && config.remoteSources.length, 'Explicit remote sources required.');
  const ids = new Set();
  for (const source of config.remoteSources) {
    exact(source, ['artifactId', 'path', 'format'], 'Remote source');
    assert(typeof source.artifactId === 'string' && /^[\w.-]+$/.test(source.artifactId), 'Source artifact ID required.');
    assert(!ids.has(source.artifactId), 'Duplicate remote source.');
    ids.add(source.artifactId); relative(source.path);
    assert(['json', 'safetensors-header'].includes(source.format), 'Unsupported source format.');
  }
  async function pinned(input) {
    exact(input, ['path', 'digest'], 'Pinned input');
    const bytes = await fs.readFile(path.resolve(sourceRoot, input.path));
    assert.equal(hashBytesSha256(bytes), input.digest, `Pinned input changed: ${input.path}`);
    return { bytes, value: JSON.parse(bytes.toString('utf8')) };
  }
  const { value: onboarding } = await pinned(config.onboarding);
  const { value: spec } = await pinned(onboarding.sourceSpec);
  assert.equal(spec.sourceIdentity.repository, config.repository, 'Source repository mismatch.');
  assert.equal(spec.sourceIdentity.revision, config.previousRevision, 'Previous source revision mismatch.');
  // Discovery assesses source semantics. It cannot reuse a signed candidate or approve a new revision.
  assert.equal(onboarding.lineage, null, 'Discovery requires lineage:null; materialize an explicitly reviewed recipe afterward.');
  await fs.mkdir(outputDir);
  const report = { schema: 'doppler.model-revision-discovery-result/v1', passed: false,
    configDigest: hashBytesSha256(Buffer.from(JSON.stringify(config))), config: structuredClone(config),
    startedAtUtc: new Date().toISOString(), requests: [], files: [], qualified: false, published: false };
  async function retain(name, bytes) {
    const file = path.join(outputDir, relative(name));
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, bytes, { flag: 'wx' });
    return { path: file, digest: hashBytesSha256(bytes) };
  }
  async function retainJson(name, value) { return retain(name, Buffer.from(JSON.stringify(value, null, 2) + '\n')); }
  async function request(url, maxBytes, range = null) {
    const response = await fetchImpl(url, { signal: AbortSignal.timeout(config.limits.timeoutMs),
      headers: range ? { Range: `bytes=${range[0]}-${range[1]}`, 'Accept-Encoding': 'identity' } : {} });
    const entry = { url, status: response.status, range, contentRange: response.headers.get('content-range') };
    report.requests.push(entry);
    try {
      assert.equal(response.status, range ? 206 : 200, 'Upstream must honor the requested response contract.');
      if (range) {
        const match = entry.contentRange?.match(/^bytes (\d+)-(\d+)\/(\d+)$/);
        assert(match && Number(match[1]) === range[0] && Number(match[2]) === range[1]
          && Number(match[3]) > range[1], 'Unexpected source Content-Range.');
      }
      const chunks = []; let size = 0;
      for await (const chunk of response.body) {
        size += chunk.length;
        assert(size <= maxBytes, 'Source response exceeds the declared byte limit.');
        chunks.push(chunk);
      }
      const bytes = Buffer.concat(chunks);
      if (range) assert.equal(size, range[1] - range[0] + 1, 'Truncated source range.');
      entry.sizeBytes = size; entry.digest = hashBytesSha256(bytes);
      return bytes;
    } finally {
      if (response.body && !response.body.locked) await response.body.cancel().catch(() => {});
    }
  }
  try {
    const base = `https://huggingface.co/api/models/${config.repository}/revision/`;
    const latestBytes = await request(`${base}${encodeURIComponent(config.ref)}?blobs=true`, config.limits.jsonBytes);
    const latest = JSON.parse(latestBytes.toString('utf8'));
    assert(/^[a-f0-9]{40}$/.test(latest.sha), 'Upstream did not resolve an immutable revision.');
    const previousBytes = await request(`${base}${config.previousRevision}?blobs=true`, config.limits.jsonBytes);
    const previous = JSON.parse(previousBytes.toString('utf8'));
    assert.equal(previous.sha, config.previousRevision, 'Previous metadata revision mismatch.');
    await retain('upstream-current.json', latestBytes); await retain('upstream-previous.json', previousBytes);
    function inventory(metadata) {
      assert(Array.isArray(metadata.siblings), 'Upstream file inventory required.');
      const values = new Map();
      for (const file of metadata.siblings) {
        relative(file.rfilename); assert(!values.has(file.rfilename), 'Duplicate upstream file.');
        assert(/^[a-f0-9]{40}$/.test(file.blobId), 'Upstream Git blob identity required.');
        if (file.lfs) assert(/^[a-f0-9]{64}$/.test(file.lfs.sha256), 'Upstream LFS identity required.');
        values.set(file.rfilename, file);
      }
      return values;
    }
    const currentFiles = inventory(latest), previousFiles = inventory(previous);
    report.resolvedRevision = latest.sha; report.previousRevision = previous.sha;
    report.revisionChanged = latest.sha !== previous.sha;
    report.changedFiles = [...new Set([...currentFiles.keys(), ...previousFiles.keys()])].sort()
      .filter(name => currentFiles.get(name)?.blobId !== previousFiles.get(name)?.blobId)
      .map(name => ({ path: name, previous: previousFiles.get(name) ?? null, current: currentFiles.get(name) ?? null }));
    report.upstreamLicense = latest.cardData?.license ?? null;
    for (const remote of config.remoteSources) {
      const artifact = spec.sourceIdentity.artifacts.find(item => item.artifactId === remote.artifactId);
      assert(artifact && spec.sources[remote.artifactId], 'Remote artifact missing from authored source specification.');
      assert.equal(artifact.path.split('#')[0], remote.path, 'Remote source path differs from authored artifact.');
      const metadata = currentFiles.get(remote.path); assert(metadata, `Missing upstream source: ${remote.path}`);
      const url = `https://huggingface.co/${config.repository}/resolve/${latest.sha}/${remote.path.split('/').map(encodeURIComponent).join('/')}`;
      let bytes;
      if (remote.format === 'safetensors-header') {
        const prefix = await request(url, 8, [0, 7]);
        const length = readSafetensorsHeaderLength(prefix);
        assert(length + 8 <= config.limits.headerBytes, 'SafeTensors header exceeds declared byte limit.');
        bytes = await request(url, length + 8, [0, length + 7]);
        assert(bytes.subarray(0, 8).equals(prefix), 'SafeTensors prefix changed between requests.');
      } else {
        bytes = await request(url, config.limits.jsonBytes);
        JSON.parse(bytes.toString('utf8'));
        if (metadata.lfs) assert.equal(hashBytesSha256(bytes), `sha256:${metadata.lfs.sha256}`, 'Source LFS hash mismatch.');
        else assert.equal(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex'), metadata.blobId, 'Source Git blob mismatch.');
      }
      const acquired = await retain(`sources/${remote.artifactId}/${path.basename(remote.path)}`, bytes);
      spec.sources[remote.artifactId] = { path: acquired.path, format: remote.format, hash: acquired.digest };
      artifact.hash = acquired.digest;
      if (metadata.lfs) artifact.upstreamHash = `sha256:${metadata.lfs.sha256}`;
      else delete artifact.upstreamHash;
      report.files.push({ artifactId: remote.artifactId, format: remote.format, ...acquired,
        upstream: metadata, wholeFileVerified: remote.format === 'json' });
    }
    report.retainedReferenceInputs = [];
    for (const [id, input] of Object.entries(spec.sources)) {
      if (ids.has(id)) continue;
      const artifact = spec.sourceIdentity.artifacts.find(item => item.artifactId === id);
      const upstreamPath = artifact?.path.split('#')[0];
      assert(!currentFiles.has(upstreamPath) && !previousFiles.has(upstreamPath),
        `Checkpoint source ${id} must be acquired for the resolved revision.`);
      // Static semantic/reference evidence keeps its original identity; it is not a new numerical qualification.
      const filename = typeof input === 'string' ? input : input.path;
      const bytes = await fs.readFile(path.resolve(sourceRoot, filename));
      const retained = await retain(`references/${id}/${path.basename(filename)}`, bytes);
      spec.sources[id] = typeof input === 'string' ? retained.path : { ...input, path: retained.path };
      report.retainedReferenceInputs.push({ artifactId: id, ...retained });
    }
    spec.sourceIdentity.revision = latest.sha;
    onboarding.sourceSpec = await retainJson('source-spec.json', spec);
    const vocabulary = await pinned(onboarding.vocabulary);
    onboarding.vocabulary = await retain('vocabulary.json', vocabulary.bytes);
    await retainJson('onboarding.json', onboarding);
    const result = await runModelOnboarding(onboarding, { sourceRoot: outputDir, outputDir: path.join(outputDir, 'assessment') });
    report.assessment = { status: result.status, outputs: result.outputs, manualRequirements: result.manualRequirements };
    report.passed = true;
  } catch (error) {
    report.error = { message: error.message, stack: error.stack };
    throw error;
  } finally {
    report.completedAtUtc = new Date().toISOString();
    await retainJson('discovery.json', report);
  }
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const [configPath, outputDir] = process.argv.slice(2);
  assert(configPath && outputDir && process.argv.length === 4,
    'Usage: node tools/discover-model-revision.js <config.json> <new-output-directory>');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const result = await discoverModelRevision(config, { sourceRoot: path.resolve(import.meta.dirname, '..'), outputDir: path.resolve(outputDir) });
  console.log(JSON.stringify({ passed: result.passed, revision: result.resolvedRevision,
    changedFiles: result.changedFiles.map(file => file.path), assessment: result.assessment.status, qualified: false }));
}
