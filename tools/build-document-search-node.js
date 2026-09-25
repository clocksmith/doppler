#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { buildDocumentSearchApplication } from './build-document-search-app.js';
import { hashBytesSha256 } from '../src/formats/canonical-hash.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function buildDocumentSearchNode(config) {
  const built = await buildDocumentSearchApplication(config);
  const root = config.outputDir;
  const source = path.join(ROOT, 'examples/document-search');
  const read = async name => JSON.parse(await fs.readFile(path.join(root, name), 'utf8'));
  const models = await read('models.json');
  const sources = JSON.parse(await fs.readFile(path.join(source, 'shard-sources.json'), 'utf8'));
  for (const model of models.models) {
    const capsule = await read(model.capsuleUrl);
    for (const artifact of capsule.artifacts.filter(item => item.role === 'weight-shard')) {
      const relative = path.join(path.dirname(model.capsuleUrl), artifact.path);
      const descriptor = sources.artifacts[relative];
      if (!descriptor?.url || descriptor.hash !== artifact.hash || descriptor.sizeBytes !== artifact.sizeBytes) {
        throw new Error(`No exact published source for ${relative}.`);
      }
      await fs.unlink(path.join(root, relative));
    }
  }
  // The existing builder owns migration and release history. Only the Node
  // application files and installed-package loader differ at this boundary.
  for (const name of ['node.js', 'node-store.js', 'shard-sources.json']) await fs.copyFile(path.join(source, name), path.join(root, name));
  await fs.copyFile(path.join(source, 'NODE.md'), path.join(root, 'README.md'));
  for (const name of ['index.html', 'browser.js', 'service-worker.js', 'document-import.js', 'application-assets.js', 'requirements.json', 'runtime']) {
    await fs.rm(path.join(root, name), { recursive: true, force: true });
  }
  await fs.mkdir(path.join(root, 'vendor'));
  await fs.copyFile(path.join(config.packageBundlePath, built.installedPackage.filename), path.join(root, 'vendor', built.installedPackage.filename));
  await fs.writeFile(path.join(root, 'package.json'), JSON.stringify({ name: 'doppler-node-document-search', version: '0.1.0',
    private: true, type: 'module', engines: { node: '>=22' },
    dependencies: { 'doppler-gpu': `file:vendor/${built.installedPackage.filename}`, webgpu: '0.4.0' } }, null, 2) + '\n');
  execFileSync('npm', ['install', '--package-lock-only', '--ignore-scripts', '--omit=optional', '--no-audit', '--no-fund'], { cwd: root, stdio: 'pipe' });
  const assets = [];
  async function inventory(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) await inventory(full);
      else if (entry.name !== 'build-receipt.json') {
        const bytes = await fs.readFile(full);
        assets.push({ path: path.relative(root, full), sha256: hashBytesSha256(bytes).slice(7), sizeBytes: bytes.length });
      }
    }
  }
  await inventory(root);
  assets.sort((a, b) => a.path.localeCompare(b.path));
  const receipt = { schema: 'doppler.document-search-node-build/v1', installedPackage: built.installedPackage,
    assets, runtimeAssets: built.assets.filter(asset => asset.path.startsWith('runtime/'))
      .map(asset => ({ ...asset, path: asset.path.slice('runtime/'.length) })),
    models: built.models, physicalExecution: false, externalAdoption: false,
    provider: { name: 'webgpu', version: '0.4.0', createArgs: ['enable-dawn-features=allow_unsafe_apis'] } };
  await fs.writeFile(path.join(root, 'build-receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
  return receipt;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await buildDocumentSearchNode(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  console.log(JSON.stringify({ models: result.models, assets: result.assets.length }));
}
