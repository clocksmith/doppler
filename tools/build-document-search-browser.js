#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { buildDocumentSearchApplication } from './build-document-search-app.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function buildDocumentSearchBrowser(config) {
  if (!/^\d+\.\d+\.\d+$/.test(config.applicationVersion ?? '')) {
    throw new Error('Explicit numeric applicationVersion required.');
  }
  const built = await buildDocumentSearchApplication(config);
  const root = config.outputDir;
  const source = path.join(ROOT, 'examples/document-search');
  const sources = JSON.parse(await fs.readFile(path.join(source, 'shard-sources.json'), 'utf8'));
  const models = JSON.parse(await fs.readFile(path.join(root, 'models.json'), 'utf8'));
  for (const model of models.models) {
    const capsule = JSON.parse(await fs.readFile(path.join(root, model.capsuleUrl), 'utf8'));
    for (const artifact of capsule.artifacts.filter(item => item.role === 'weight-shard')) {
      const relative = path.join(path.dirname(model.capsuleUrl), artifact.path);
      const descriptor = sources.artifacts[relative];
      if (!descriptor?.url || descriptor.hash !== artifact.hash || descriptor.sizeBytes !== artifact.sizeBytes) {
        throw new Error(`No exact published source for ${relative}.`);
      }
      await fs.unlink(path.join(root, relative));
    }
  }
  for (const name of ['server.js', 'prepare.js', 'shard-sources.json']) {
    await fs.copyFile(path.join(source, name), path.join(root, name));
  }
  const guide = await fs.readFile(path.join(source, 'README.md'), 'utf8');
  await fs.writeFile(path.join(root, 'README.md'), guide
    .replaceAll('../../docs/', 'https://github.com/clocksmith/doppler/blob/main/docs/')
    .replaceAll('](ENGINEERING.md)', '](https://github.com/clocksmith/doppler/blob/main/examples/document-search/ENGINEERING.md)')
    .replaceAll('](../capsule-capabilities/README.md)', '](https://github.com/clocksmith/doppler/blob/main/examples/capsule-capabilities/README.md)'));
  await fs.cp(path.join(source, 'samples'), path.join(root, 'samples'), { recursive: true });
  // The consumer's frozen installation generates these from its vendored archive.
  for (const name of ['runtime', 'application-assets.js', 'requirements.json', 'build-receipt.json']) {
    await fs.rm(path.join(root, name), { recursive: true });
  }
  await fs.mkdir(path.join(root, 'vendor'));
  await fs.copyFile(path.join(config.packageBundlePath, built.installedPackage.filename), path.join(root, 'vendor', built.installedPackage.filename));
  const pkg = JSON.parse(await fs.readFile(path.join(source, 'package.json'), 'utf8'));
  pkg.version = config.applicationVersion;
  pkg.dependencies['doppler-gpu'] = `file:vendor/${built.installedPackage.filename}`;
  await fs.writeFile(path.join(root, 'package.json'), JSON.stringify(pkg, null, 2) + '\n');
  execFileSync('npm', ['install', '--package-lock-only', '--ignore-scripts', '--omit=optional', '--no-audit', '--no-fund'], { cwd: root, stdio: 'pipe' });
  return { ...built, scope: 'Browser archive source; frozen consumer installation and physical qualification remain required.' };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await buildDocumentSearchBrowser(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  console.log(JSON.stringify({ outputDir: result.config.outputDir, package: result.installedPackage, models: result.models, scope: result.scope }));
}
