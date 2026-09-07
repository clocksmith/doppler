#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { migrateCapsuleV2, getCapsuleIdentity, signCapsuleReleaseEvent, verifyCapsuleReleaseEvents, verifyCapsule } from '../src/capsule.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { normalizeCapsuleLoadingPolicy } from '../src/config/capsule-loading.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function buildDocumentSearchApplication(config) {
  const loading = normalizeCapsuleLoadingPolicy(config.loading ?? {});
  for (const key of ['packageBundlePath', 'outputDir']) if (!path.isAbsolute(config[key] ?? '')) throw new Error(`Absolute ${key} required.`);
  if (config.previousApplicationDir !== null && !path.isAbsolute(config.previousApplicationDir ?? '')) {
    throw new Error('Explicit previousApplicationDir required; use null only for the initial application build.');
  }
  const retainModels = config.models === null;
  if (retainModels && config.previousApplicationDir === null) throw new Error('Retaining signed models requires previousApplicationDir.');
  if (!retainModels && (!Array.isArray(config.models) || config.models.length !== 2
    || new Set(config.models.map(model => model.role)).size !== 2
    || config.models.some(model => !['embedding', 'reranker'].includes(model.role) || !path.isAbsolute(model.capsuleRoot)))) {
    throw new Error('One explicit embedding Capsule and one reranker Capsule required.');
  }
  const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
  const previous = config.previousApplicationDir === null ? null
    : await read(path.join(config.previousApplicationDir, 'models.json'));
  const installed = await read(path.join(config.packageBundlePath, 'receipt.json'));
  if (!installed.passed || hashBytesSha256(await fs.readFile(path.join(config.packageBundlePath, installed.package.filename))) !== `sha256:${installed.package.sha256}`) {
    throw new Error('Passing installed-package evidence and matching archive required.');
  }
  await fs.mkdir(config.outputDir);
  const write = (filename, value) => fs.writeFile(path.join(config.outputDir, filename), JSON.stringify(value, null, 2), { flag: 'wx' });
  const applicationDir = path.join(ROOT, 'examples/document-search');
  for (const name of ['index.html', 'browser.js', 'search.js', 'installation.js', 'service-worker.js']) {
    await fs.copyFile(path.join(applicationDir, name), path.join(config.outputDir, name));
  }
  await fs.cp(path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu/src'), path.join(config.outputDir, 'runtime/src'), { recursive: true });
  const applicationDigest = hashBytesSha256(await fs.readFile(path.join(applicationDir, 'search.js')));
  const models = [];
  if (retainModels) {
    if (computeCanonicalSha256(previous.search) !== computeCanonicalSha256(config.search)
      || computeCanonicalSha256(previous.storage) !== computeCanonicalSha256(config.storage)) {
      throw new Error('Retained model rebuild requires unchanged search and storage semantics.');
    }
    if (previous.models.length !== 2 || new Set(previous.models.map(model => model.role)).size !== 2) {
      throw new Error('Retained application must contain exactly one embedding and reranker model.');
    }
    for (const model of previous.models) {
      if (!['embedding', 'reranker'].includes(model.role) || model.capsuleUrl !== `./capsules/${model.role}/capsule-v3.json`
        || model.application.applicationRevisionDigest !== applicationDigest) {
        throw new Error('Retained model requires the same application program and a local Capsule path.');
      }
      const source = path.join(config.previousApplicationDir, 'capsules', model.role);
      const capsule = await read(path.join(source, 'capsule-v3.json'));
      const verified = await verifyCapsule(capsule, { ...model.options, artifactStore: {
        async readArtifact(artifact) {
          const filename = path.resolve(source, artifact.path), relative = path.relative(source, filename);
          if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Retained artifact escapes its Capsule directory.');
          return fs.readFile(filename);
        },
      } });
      if (computeCanonicalSha256(verified.identity) !== computeCanonicalSha256(model.identity)) throw new Error('Retained Capsule identity changed.');
      await fs.cp(source, path.join(config.outputDir, 'capsules', model.role), { recursive: true });
      models.push({ ...structuredClone(model), options: { ...structuredClone(model.options), ...loading } });
    }
  }
  for (const model of config.models ?? []) {
    const capsule = await read(path.join(model.capsuleRoot, 'distribution/capsule.json'));
    const options = await read(path.join(model.capsuleRoot, 'open-options.json'));
    const signer = { authority: capsule.signature.authority,
      publicKeyJwk: await read(path.join(model.capsuleRoot, 'custody/public-key.json')),
      privateKeyJwk: await read(path.join(model.capsuleRoot, 'custody/private-key.json')) };
    const built = await migrateCapsuleV2(capsule, { trustedSigners: options.trustedSigners, signer });
    const identity = getCapsuleIdentity(built.capsule);
    const reference = { schema: identity.schema, semanticRoot: identity.semanticRoot, envelopeDigest: identity.envelopeDigest };
    const release = structuredClone(built.release);
    release.application = { ...release.application, applicationId: 'doppler-offline-document-search',
      applicationRevision: applicationDigest, applicationRevisionDigest: applicationDigest };
    const issuedAtUtc = new Date().toISOString();
    const expiresAtUtc = new Date(Date.parse(issuedAtUtc) + release.revocation.offlineExpirySeconds * 1000).toISOString();
    const prior = previous?.models.find(prior => prior.role === model.role
      && computeCanonicalSha256(prior.identity) === computeCanonicalSha256(identity));
    let events = [];
    if (prior) {
      await verifyCapsuleReleaseEvents(prior.options.releaseEvents, { capsule: built.capsule,
        trustedSigners: options.trustedSigners,
        policy: { ...prior.options.releasePolicy, now: issuedAtUtc } });
      events = structuredClone(prior.options.releaseEvents);
    }
    let eligible = events.at(-1);
    if (!eligible || computeCanonicalSha256(eligible.release) !== computeCanonicalSha256(release)) {
      eligible = await signCapsuleReleaseEvent({ capsule: reference, sequence: (eligible?.sequence ?? 0) + 1,
        previousEventDigest: eligible?.digest ?? null, issuedAtUtc, expiresAtUtc, action: 'eligible', release,
        migratedFrom: built.migratedFrom, nextSigner: null }, signer);
      events.push(eligible);
    }
    const destination = path.join(config.outputDir, 'capsules', model.role);
    await fs.cp(path.join(model.capsuleRoot, 'distribution'), destination, { recursive: true });
    await fs.writeFile(path.join(destination, 'capsule-v3.json'), JSON.stringify(built.capsule, null, 2), { flag: 'wx' });
    models.push({ role: model.role, capsuleUrl: `./capsules/${model.role}/capsule-v3.json`, identity,
      storageId: model.role + '-' + identity.semanticRoot.slice(7), application: release.application,
      retainedLocalUse: { schema: 'doppler.capsule-retained-local-use/v1', capsule: reference,
        releaseEventDigest: eligible.digest, applicationDigest: computeCanonicalSha256(release.application), acknowledgeUnseenRevocations: true },
      options: { ...options, ...loading, releaseEvents: events, releaseTrustedSigners: options.trustedSigners,
        releasePolicy: { now: issuedAtUtc, minimumSequence: eligible.sequence,
          checkpoint: { sequence: eligible.sequence, digest: eligible.digest } } } });
  }
  await write('models.json', { models, search: config.search, storage: config.storage });
  const assets = [];
  async function inventory(directory, prefix = '') {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      if (!prefix && entry.name === 'capsules') continue;
      const relative = prefix + entry.name;
      if (entry.isDirectory()) await inventory(path.join(directory, entry.name), relative + '/');
      else if (/\.(js|json|wgsl|html|wasm)$/.test(entry.name)) {
        const bytes = await fs.readFile(path.join(directory, entry.name));
        assets.push({ path: relative, sha256: hashBytesSha256(bytes).slice(7), sizeBytes: bytes.byteLength });
      }
    }
  }
  await inventory(config.outputDir);
  assets.sort((a, b) => a.path.localeCompare(b.path));
  const cacheName = 'doppler-document-search-' + computeCanonicalSha256(assets).slice(7);
  await fs.writeFile(path.join(config.outputDir, 'application-assets.js'), `self.DOCUMENT_SEARCH_ASSETS = ${JSON.stringify({ cacheName, assets })};\n`, { flag: 'wx' });
  const receipt = { schema: 'doppler.document-search-build/v1', installedPackage: installed.package,
    config, applicationDigest, cacheName, assets, models: models.map(model => ({ role: model.role, identity: model.identity })),
    physicalExecution: false, externalAdoption: false,
    releaseScope: retainModels ? 'retained signed models and release history; no signing or promotion'
      : 'new internal evaluation streams; existing checkpoints and denials untouched' };
  await write('build-receipt.json', receipt);
  return receipt;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const receipt = await buildDocumentSearchApplication(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  console.log(JSON.stringify({ outputDir: receipt.config.outputDir, cacheName: receipt.cacheName, models: receipt.models }));
}
