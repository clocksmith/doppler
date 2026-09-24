import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
async function inventory(root, prefix = '') {
  const assets = [];
  for (const entry of (await fs.readdir(root, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
    if (entry.isSymbolicLink()) throw new Error(`Installed assets must not be symlinks: ${entry.name}`);
    const filename = path.join(root, entry.name);
    const relative = prefix + entry.name;
    if (entry.isDirectory()) assets.push(...await inventory(filename, relative + '/'));
    else if (entry.isFile()) {
      const bytes = await fs.readFile(filename);
      assets.push({ path: relative, sha256: hash(bytes), sizeBytes: bytes.length });
    }
  }
  return assets;
}

export async function prepareRequirements(root, runtime, { localArtifacts = false } = {}) {
  const models = JSON.parse(await fs.readFile(path.join(root, 'models.json')));
  const sources = localArtifacts ? null : JSON.parse(await fs.readFile(path.join(root, 'shard-sources.json')));
  const requirements = { schema: 'doppler.document-search-requirements/v1', models: [],
    missingSources: [], runtimeBytes: runtime.reduce((sum, asset) => sum + asset.sizeBytes, 0) };
  for (const model of models.models) {
    const capsule = JSON.parse(await fs.readFile(path.join(root, model.capsuleUrl)));
    if (capsule.targetPlans.length !== 1) throw new Error('Starter requirements need one explicitly selected plan per model.');
    const predicate = capsule.targetPlans[0].capabilityPredicate;
    requirements.models.push({ role: model.role, identity: model.identity,
      downloadBytes: capsule.artifacts.reduce((sum, artifact) => sum + artifact.sizeBytes, 0),
      requiredFeatures: [predicate.requiresF16 && 'shader-f16', predicate.requiresSubgroups && 'subgroups'].filter(Boolean),
      minBufferSize: predicate.minBufferSize });
    for (const artifact of capsule.artifacts.filter(artifact => /^artifacts\/model\/shard_\d+\.bin$/.test(artifact.path))) {
      const name = 'capsules/' + model.role + '/' + artifact.path;
      if (localArtifacts) {
        const bytes = await fs.readFile(path.join(root, name));
        if (bytes.length !== artifact.sizeBytes || 'sha256:' + hash(bytes) !== artifact.hash) throw new Error('Local model artifact integrity failed: ' + name);
        continue;
      }
      const source = sources.artifacts[name];
      if (!source || source.hash !== artifact.hash || source.sizeBytes !== artifact.sizeBytes) {
        throw new Error('Acquisition descriptor differs from Capsule: ' + name);
      }
      if (!source.url) requirements.missingSources.push(name);
    }
  }
  await fs.writeFile(path.join(root, 'requirements.json'), JSON.stringify(requirements, null, 2) + '\n');
  return requirements;
}

export async function prepareApplication(root = ROOT) {
  const lockBytes = await fs.readFile(path.join(root, 'package-lock.json'));
  const lock = JSON.parse(lockBytes);
  const dependency = lock.packages['node_modules/doppler-gpu'];
  const spec = JSON.parse(await fs.readFile(path.join(root, 'package.json'))).dependencies['doppler-gpu'];
  if (lock.packages[''].dependencies['doppler-gpu'] !== spec || !spec.startsWith('file:vendor/')) {
    throw new Error('Starter package and lock must identify the same bundled runtime archive.');
  }
  const archive = await fs.readFile(path.join(root, spec.slice(5)));
  if ('sha512-' + createHash('sha512').update(archive).digest('base64') !== dependency.integrity) {
    throw new Error('Runtime archive does not match package-lock integrity.');
  }
  const runtimeRoot = path.join(root, 'node_modules/doppler-gpu');
  const runtime = await inventory(runtimeRoot, 'runtime/');
  if (!runtime.some(asset => asset.path === 'runtime/src/client/capsule-host.browser.js')) {
    throw new Error('Installed browser host is missing. Run npm ci; source fallback is prohibited.');
  }
  const models = JSON.parse(await fs.readFile(path.join(root, 'models.json')));
  const requirements = await prepareRequirements(root, runtime);
  const appNames = (await fs.readdir(root)).filter(name => /\.(js|html|json)$/.test(name)
    && !['application-assets.js', 'build-receipt.json', 'server.js', 'prepare.js'].includes(name));
  const assets = [...runtime, ...await inventory(path.join(root, 'samples'), 'samples/')];
  for (const name of appNames.sort()) {
    const bytes = await fs.readFile(path.join(root, name));
    assets.push({ path: name, sha256: hash(bytes), sizeBytes: bytes.length });
  }
  assets.sort((a, b) => a.path.localeCompare(b.path, 'en'));
  const content = { cacheName: 'doppler-document-search-' + hash(JSON.stringify(assets)), assets };
  const manifest = `self.DOCUMENT_SEARCH_ASSETS = ${JSON.stringify(content, null, 2)};\n`;
  await fs.writeFile(path.join(root, 'application-assets.js'), manifest);
  const receipt = { schema: 'doppler.installed-document-search-build/v1',
    tooling: Object.fromEntries(await Promise.all(['server.js', 'prepare.js'].map(async name => [name, hash(await fs.readFile(path.join(root, name)))]))),
    packageSha256: hash(archive), packageIntegrity: dependency.integrity,
    lockSha256: hash(lockBytes), applicationManifestSha256: hash(manifest),
    models: models.models.map(model => ({ role: model.role, identity: model.identity })),
    runtimeFiles: runtime.length, sourceSubstitution: false,
    unavailableArtifacts: requirements.missingSources,
    physicalAcceptance: false };
  await fs.writeFile(path.join(root, 'build-receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
  return receipt;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) console.log(JSON.stringify(await prepareApplication(), null, 2));
