import fs from 'node:fs/promises';
import os from 'node:os';
import { createReadStream, createWriteStream } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { pipeline } from 'node:stream/promises';
const exec = promisify(execFile);
export const hashJson = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
export async function hashFile(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
export async function fileInventory(root) {
  const files = [];
  async function visit(directory) {
    for (const entry of (await fs.readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
      const file = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const target = await fs.readlink(file), resolved = path.resolve(path.dirname(file), target);
        if (resolved !== path.resolve(root) && !resolved.startsWith(path.resolve(root) + path.sep)) throw new Error(`Snapshot/artifact link escapes frozen root: ${file}`);
        files.push({ path: path.relative(root, file), symlink: target, bytes: Buffer.byteLength(target),
          sha256: createHash('sha256').update(target).digest('hex') });
        continue;
      }
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile()) files.push({ path: path.relative(root, file), bytes: (await fs.stat(file)).size, sha256: await hashFile(file) });
    }
  }
  await visit(root);
  return files;
}
async function archive(repo, revision, destination, members = []) {
  if (!/^[a-f0-9]{40}$/.test(revision.split(':')[0])) throw new Error('Pin full Git commit before snapshot');
  const { stdout } = await exec('git', ['-C', repo, 'rev-parse', `${revision.split(':')[0]}^{commit}`]);
  if (stdout.trim() !== revision.split(':')[0]) throw new Error('Source commit does not resolve exactly');
  const file = destination + '.tar';
  const child = spawn('git', ['-C', repo, 'archive', '--format=tar', revision, ...members], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = ''; child.stderr.on('data', chunk => { stderr += chunk; });
  const exited = new Promise((resolve, reject) => {
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolve() : reject(new Error(`Git archive failed: ${stderr}`)));
  });
  await Promise.all([pipeline(child.stdout, createWriteStream(file, { flags: 'wx' })), exited]);
  await fs.mkdir(destination);
  await exec('tar', ['-xf', file, '-C', destination]);
  return { revision, archiveSha256: await hashFile(file) };
}
export async function freezeResidentInputs(config, configBytes, output) {
  for (const artifact of [config.canonicalDir, config.rdrrDir]) {
    const root = path.resolve(artifact);
    if (output === root || output.startsWith(root + path.sep)) throw new Error('Snapshot output must be outside artifact directories');
  }
  await fs.mkdir(output, { recursive: false });
  const runtime = path.join(output, 'runtime'), vendor = path.join(output, 'vendor');
  const current = (await exec('git', ['-C', config.dopplerRoot, 'rev-parse', 'HEAD'])).stdout.trim();
  if (current !== config.harnessCommit) throw new Error('Harness must pin current committed HEAD; engine commit is pinned separately');
  const doppler = await archive(config.dopplerRoot, config.dopplerCommit, runtime,
    ['src', 'demo', 'tests', 'benchmarks/vendors/pooled-equal-weights/contract-f32kv.json', 'models/catalog.json', 'package.json', 'package-lock.json']);
  const harness = await archive(config.dopplerRoot, `${config.harnessCommit}:tools/pooled-equal-weights`, path.join(output, 'harness'));
  await fs.mkdir(path.join(runtime, 'tools'), { recursive: true });
  await fs.rename(path.join(output, 'harness'), path.join(runtime, 'tools/pooled-equal-weights'));
  await fs.mkdir(vendor);
  const pooledReference = config.pooledSourceIdentity;
  if (!pooledReference?.path || !/^[a-f0-9]{64}$/.test(pooledReference.sha256)
      || await hashFile(pooledReference.path) !== pooledReference.sha256) throw new Error('Pin native Pooled byte-snapshot source identity');
  const pooledSource = JSON.parse(await fs.readFile(pooledReference.path));
  if (pooledSource.vendorChanges?.length) throw new Error('Pooled byte snapshot is not native');
  await fs.cp(path.join(config.pooledRoot, 'engine'), path.join(vendor, 'engine'), { recursive: true, dereference: false });
  const copiedVendor = (await fileInventory(vendor)).map(({ path, sha256 }) => ({ path: `vendor/${path}`, sha256 }));
  const expectedVendor = pooledSource.engineFiles.map(({ path, sha256 }) => ({ path, sha256 }));
  const byPath = (a, b) => a.path.localeCompare(b.path, 'en');
  if (hashJson(copiedVendor.sort(byPath)) !== hashJson(expectedVendor.sort(byPath))) throw new Error('Copied Pooled bytes differ from pinned complete source inventory');
  const pooled = { kind: 'immutable-byte-snapshot', sourceIdentitySha256: pooledReference.sha256, filesSha256: hashJson(copiedVendor) };
  doppler.srcTree = (await exec('git', ['-C', config.dopplerRoot, 'rev-parse', `${config.dopplerCommit}:src`])).stdout.trim();
  if (!Array.isArray(config.dependencyPaths) || !config.dependencyPaths.includes('node_modules/playwright')
      || !config.dependencyPaths.includes('node_modules/playwright-core')) throw new Error('Declare frozen Playwright and browser dependency paths');
  for (const relative of config.dependencyPaths) {
    if (!relative.startsWith('node_modules/') || relative.split('/').includes('..')) throw new Error('Invalid dependency path');
    await fs.cp(path.join(config.dopplerRoot, relative), path.join(runtime, relative), { recursive: true, dereference: true, errorOnExist: true, force: false });
  }
  const runtimeFiles = await fileInventory(runtime), vendorFiles = await fileInventory(vendor);
  const canonical = await fileInventory(config.canonicalDir), rdrr = await fileInventory(config.rdrrDir);
  const frozenConfig = { ...config, outputDir: output };
  const identity = { host: { hostname: os.hostname(), platform: process.platform, architecture: process.arch, release: os.release() }, schema: 'doppler.equal-weight-resident-snapshot.v1', configSha256: createHash('sha256').update(configBytes).digest('hex'),
    doppler, pooled, harness, runtimeFiles, vendorFiles, artifacts: { canonical, rdrr, gguf: await hashFile(config.ggufPath) },
    browser: { path: config.browserExecutable, sha256: await hashFile(config.browserExecutable),
      version: (await exec(config.browserExecutable, ['--version'])).stdout.trim(), args: config.browserArgs },
    node: { path: process.execPath, sha256: await hashFile(process.execPath), version: process.version } };
  await fs.writeFile(path.join(output, 'frozen-config.json'), JSON.stringify(frozenConfig, null, 2));
  await fs.writeFile(path.join(output, 'source-identity.json'), JSON.stringify(identity, null, 2));
  // Frozen source and dependencies are never compiler/probe mutation targets.
  for (const [root, files] of [[runtime, runtimeFiles], [vendor, vendorFiles]]) {
    for (const file of files) {
      if (file.symlink !== undefined) continue;
      const target = path.join(root, file.path), stat = await fs.stat(target);
      await fs.chmod(target, stat.mode & ~0o222);
    }
  }
  return identity;
}
export async function verifyResidentInputs(config, identity, output) {
  for (const [directory, expected] of [[path.join(output, 'runtime'), identity.runtimeFiles],
    [path.join(output, 'vendor'), identity.vendorFiles], [config.canonicalDir, identity.artifacts.canonical], [config.rdrrDir, identity.artifacts.rdrr]]) {
    if (hashJson(await fileInventory(directory)) !== hashJson(expected)) throw new Error(`Source/dependency/artifact drift: ${directory}`);
  }
  if (await hashFile(config.ggufPath) !== identity.artifacts.gguf
      || await hashFile(config.browserExecutable) !== identity.browser.sha256
      || await hashFile(process.execPath) !== identity.node.sha256) throw new Error('Browser/Node/GGUF drift');
}
