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
  const models = JSON.parse(await fs.readFile(path.join(root, 'models.json')));
  const sources = JSON.parse(await fs.readFile(path.join(root, 'shard-sources.json')));
  const receipt = { schema: 'doppler.installed-document-search-build/v1',
    packageSha256: hash(archive), packageIntegrity: dependency.integrity,
    lockSha256: hash(lockBytes), applicationManifestSha256: hash(manifest),
    models: models.models.map(model => ({ role: model.role, identity: model.identity })),
    runtimeFiles: runtime.length, sourceSubstitution: false,
    unavailableArtifacts: Object.entries(sources.artifacts).filter(([, value]) => value.url === null).map(([name]) => name),
    physicalAcceptance: false };
  await fs.writeFile(path.join(root, 'build-receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
  return receipt;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) console.log(JSON.stringify(await prepareApplication(), null, 2));
