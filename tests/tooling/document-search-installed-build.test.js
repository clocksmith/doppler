import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { verifyInstalledSearchBuild } from '../../tools/document-search-installed-build.js';
import { createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';
import { getCapsuleIdentity } from '../../src/capsule.js';

// Synthetic archive tests identity rejection; only the installed browser runner
// establishes that npm installed an executable package and actual models ran.
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-installed-search-contract-'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const write = async (filename, value) => {
  await fs.mkdir(path.dirname(path.join(root, filename)), { recursive: true });
  await fs.writeFile(path.join(root, filename), value);
};
try {
  const archive = Buffer.from('synthetic archive');
  const spec = 'file:vendor/runtime.tgz';
  const integrity = 'sha512-' + createHash('sha512').update(archive).digest('base64');
  const lock = JSON.stringify({ packages: { '': { dependencies: { 'doppler-gpu': spec } },
    'node_modules/doppler-gpu': { integrity } } });
  await write('vendor/runtime.tgz', archive);
  await write('package.json', JSON.stringify({ dependencies: { 'doppler-gpu': spec } }));
  await write('package-lock.json', lock);
  await write('server.js', '// fixture server');
  await write('prepare.js', '// fixture preparation');
  await write('node_modules/doppler-gpu/src/host.js', 'installed bytes');
  const { capsule } = await createSignedCapsuleFixture();
  const models = [{ role: 'embedding', identity: getCapsuleIdentity(capsule), capsuleUrl: './capsules/embedding/capsule-v3.json' }];
  await write('capsules/embedding/capsule-v3.json', JSON.stringify(capsule));
  await write('models.json', JSON.stringify({ models }));
  const assets = [{ path: 'runtime/src/host.js', sizeBytes: 15, sha256: hash('installed bytes') },
    { path: 'models.json', sizeBytes: Buffer.byteLength(JSON.stringify({ models })), sha256: hash(JSON.stringify({ models })) }];
  const manifest = 'self.DOCUMENT_SEARCH_ASSETS = ' + JSON.stringify({ assets }) + ';\n';
  await write('application-assets.js', manifest);
  const build = { schema: 'doppler.installed-document-search-build/v1', sourceSubstitution: false,
    packageSha256: hash(archive), packageIntegrity: integrity, lockSha256: hash(lock),
    applicationManifestSha256: hash(manifest), models,
    tooling: { 'server.js': hash('// fixture server'), 'prepare.js': hash('// fixture preparation') } };
  assert.equal((await verifyInstalledSearchBuild(root, build)).assetCount, 2);
  await write('node_modules/doppler-gpu/src/host.js', 'substitute bytes');
  await assert.rejects(verifyInstalledSearchBuild(root, build), /runtime\/src\/host.js/);
  await fs.unlink(path.join(root, 'node_modules/doppler-gpu/src/host.js'));
  await write('checkout.js', 'installed bytes');
  await fs.symlink(path.join(root, 'checkout.js'), path.join(root, 'node_modules/doppler-gpu/src/host.js'));
  await assert.rejects(verifyInstalledSearchBuild(root, build), /symlinked source/);
  await fs.unlink(path.join(root, 'node_modules/doppler-gpu/src/host.js'));
  await write('node_modules/doppler-gpu/src/host.js', 'installed bytes');
  await write('server.js', '// substituted server');
  await assert.rejects(verifyInstalledSearchBuild(root, build), /server.js/);
  await write('server.js', '// fixture server');
  await write('vendor/runtime.tgz', 'different candidate');
  await assert.rejects(verifyInstalledSearchBuild(root, build));
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('document-search-installed-build: synthetic archive, runtime substitution, symlink and server tampering checks passed');
