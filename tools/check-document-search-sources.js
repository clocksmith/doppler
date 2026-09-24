#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { verifyCapsule } from '../src/capsule.js';

const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const publishedPath = artifact => 'document-search/artifacts/sha256/' + artifact.hash.slice(7) + '.bin';

// Byte publication does not promote a model or authorize a new release event.
export async function checkDocumentSearchSources(config) {
  const root = path.resolve(config.applicationDir);
  const sources = await read(path.join(root, 'shard-sources.json'));
  const models = await read(path.join(root, 'models.json'));
  const report = { schema: 'doppler.document-search-source-audit/v1', passed: false,
    generatedAt: new Date().toISOString(), physicalAcceptance: false,
    probeSha256: hash(await fs.readFile(fileURLToPath(import.meta.url))),
    models: [], artifacts: [], missingSources: [] };
  const declared = new Set();
  for (const model of models.models) {
    const capsule = await read(path.join(root, model.capsuleUrl));
    const verified = await verifyCapsule(capsule, { ...model.options, artifactStore: {
      async readArtifact(artifact) {
        assert.ok(!artifact.path.split('/').includes('..') && !path.isAbsolute(artifact.path));
        const relative = 'capsules/' + model.role + '/' + artifact.path;
        if (!/^artifacts\/model\/shard_\d+\.bin$/.test(artifact.path)) return fs.readFile(path.join(root, relative));
        declared.add(relative);
        const source = sources.artifacts[relative];
        assert.equal(source?.hash, artifact.hash, relative + ' hash declaration');
        assert.equal(source.sizeBytes, artifact.sizeBytes, relative + ' size declaration');
        if (source.url === null) report.missingSources.push(relative);
        let url = source.url;
        if (config.revision && !url) {
          assert.match(config.revision, /^[a-f0-9]{40}$/);
          url = 'https://huggingface.co/clocksmith/rdrr/resolve/' + config.revision + '/' + publishedPath(artifact);
        }
        let bytes;
        if (config.localRoot) bytes = await fs.readFile(path.join(config.localRoot, relative));
        else {
          assert.equal(typeof url, 'string', 'Missing immutable source: ' + relative);
          assert.match(url, /^https:\/\/huggingface\.co\/clocksmith\/rdrr\/resolve\/[a-f0-9]{40}\//);
          const response = await fetch(url);
          assert.equal(response.status, 200, relative + ' download status');
          bytes = new Uint8Array(await response.arrayBuffer());
        }
        assert.equal(bytes.byteLength, artifact.sizeBytes, relative + ' downloaded size');
        assert.equal(hash(bytes), artifact.hash, relative + ' downloaded hash');
        report.artifacts.push({ path: relative, hash: artifact.hash, sizeBytes: bytes.byteLength,
          url, source: config.localRoot ? 'local-byte-audit' : 'immutable-public-download' });
        return bytes;
      },
    } });
    assert.deepEqual(verified.identity, model.identity);
    report.models.push({ role: model.role, identity: verified.identity, artifactCount: verified.artifactReceipts.length });
  }
  assert.deepEqual(Object.keys(sources.artifacts).sort(), [...declared].sort(), 'Source map must cover exactly the declared shards');
  if (config.stagingDir) {
    assert.ok(config.localRoot, 'Staging requires audited local bytes');
    await fs.mkdir(config.stagingDir, { recursive: true });
    for (const name of report.missingSources) {
      const destination = path.join(config.stagingDir, publishedPath(sources.artifacts[name]));
      await fs.mkdir(path.dirname(destination), { recursive: true });
      await fs.copyFile(path.join(config.localRoot, name), destination);
      assert.equal(hash(await fs.readFile(destination)), sources.artifacts[name].hash);
    }
    for (const model of models.models) {
      await fs.copyFile(path.join(root, 'capsules', model.role, 'MODEL_LICENSE.txt'),
        path.join(config.stagingDir, 'document-search', model.role + '-MODEL_LICENSE.txt'));
    }
  }
  if (config.revision) {
    assert.ok(!config.localRoot, 'Only verified public downloads may populate acquisition URLs');
    for (const artifact of report.artifacts) sources.artifacts[artifact.path].url = artifact.url;
    await fs.writeFile(path.join(root, 'shard-sources.json'), JSON.stringify(sources, null, 2) + '\n');
  }
  report.passed = true;
  report.publicAcquisitionComplete = !config.localRoot && report.artifacts.every(artifact => artifact.url);
  await fs.writeFile(config.receiptPath, JSON.stringify(report, null, 2) + '\n');
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const report = await checkDocumentSearchSources(await read(process.argv[2]));
  console.log(JSON.stringify({ passed: report.passed, shards: report.artifacts.length,
    missingSources: report.missingSources.length, publicAcquisitionComplete: report.publicAcquisitionComplete }));
}
