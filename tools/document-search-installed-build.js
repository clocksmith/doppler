import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { getCapsuleIdentity } from '../src/capsule.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export async function verifyInstalledSearchBuild(root, build) {
  assert.equal(build.schema, 'doppler.installed-document-search-build/v1');
  assert.equal(build.sourceSubstitution, false);
  for (const name of ['server.js', 'prepare.js']) assert.equal(hash(await fs.readFile(path.join(root, name))), build.tooling?.[name], name);
  const lockBytes = await fs.readFile(path.join(root, 'package-lock.json'));
  assert.equal(hash(lockBytes), build.lockSha256, 'Installed lock changed');
  const lock = JSON.parse(lockBytes);
  const spec = JSON.parse(await fs.readFile(path.join(root, 'package.json'))).dependencies['doppler-gpu'];
  assert.match(spec, /^file:vendor\/[^/]+\.tgz$/);
  assert.equal(lock.packages[''].dependencies['doppler-gpu'], spec);
  const archive = await fs.readFile(path.join(root, spec.slice(5)));
  assert.equal(hash(archive), build.packageSha256);
  assert.equal('sha512-' + createHash('sha512').update(archive).digest('base64'), build.packageIntegrity);
  assert.equal(build.packageIntegrity, lock.packages['node_modules/doppler-gpu'].integrity);
  const manifestBytes = await fs.readFile(path.join(root, 'application-assets.js'));
  assert.equal(hash(manifestBytes), build.applicationManifestSha256);
  const manifest = JSON.parse(manifestBytes.toString().replace(/^self\.DOCUMENT_SEARCH_ASSETS = /, '').replace(/;\s*$/, ''));
  for (const asset of manifest.assets) {
    assert.ok(!path.isAbsolute(asset.path) && !asset.path.split('/').includes('..'));
    const filename = path.join(root, asset.path.replace(/^runtime\//, 'node_modules/doppler-gpu/'));
    assert.equal(await fs.realpath(filename), filename, 'Qualification cannot use symlinked source');
    const bytes = await fs.readFile(filename);
    assert.equal(bytes.length, asset.sizeBytes, asset.path);
    assert.equal(hash(bytes), asset.sha256, asset.path);
  }
  const models = JSON.parse(await fs.readFile(path.join(root, 'models.json')));
  for (const model of models.models) {
    assert.deepEqual(getCapsuleIdentity(JSON.parse(await fs.readFile(path.join(root, model.capsuleUrl)))), model.identity);
    assert.deepEqual(build.models.find(row => row.role === model.role)?.identity, model.identity);
  }
  return { packageSha256: build.packageSha256, lockSha256: build.lockSha256, tooling: build.tooling,
    applicationManifestSha256: build.applicationManifestSha256, assetCount: manifest.assets.length, models: build.models };
}
