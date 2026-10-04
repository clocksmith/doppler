/** Run an existing Reploid acceptance fixture against one installed archive.
 * Only this process's fresh browser contexts see the candidate configuration.
 * The production checkout, vendor tree, running server and dependency pin stay untouched. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const [reploidRoot, consumerRoot, archive, fixture] = process.argv.slice(2).map(value => resolve(value));
assert(reploidRoot && consumerRoot && archive && fixture, 'Reploid root, installed consumer, archive and fixture are required');
assert(fixture.startsWith(resolve(reploidRoot, 'tests/fixtures') + sep), 'Use an existing Reploid acceptance fixture');
assert(!process.env.DOPPLER_DISPATCH_SOURCE_ROOT && !process.env.DOPPLER_DISPATCH_RMSNORM_SOURCE,
  'An archive acceptance run cannot substitute individual source files');
const packageRoot = resolve(consumerRoot, 'node_modules/doppler-gpu');
const packageInfo = JSON.parse(await readFile(resolve(packageRoot, 'package.json')));
const archiveBytes = await readFile(archive), digest = bytes => createHash('sha256').update(bytes).digest('hex');
const installedReceipt = JSON.parse(await readFile(resolve(consumerRoot, '../receipt.json')));
assert.equal(installedReceipt.passed, true, 'Use a consumer installed by the exact-archive package smoke');
assert.equal(installedReceipt.package.sha256, digest(archiveBytes), 'Installed consumer belongs to this archive');
assert.equal(installedReceipt.package.version, packageInfo.version);
const config = JSON.parse(await readFile(resolve(reploidRoot, 'self/config/doppler-package.json')));
Object.assign(config, { version: packageInfo.version, spec: `file:${archive.split(sep).at(-1)}`,
  resolved: `/candidate/${archive.split(sep).at(-1)}`, integrity: `sha512-${createHash('sha512').update(archiveBytes).digest('base64')}` });
const require = createRequire(pathToFileURL(resolve(reploidRoot, 'package.json')));
const { chromium } = require('playwright');
const served = new Map(), errors = [], restore = [];
for (const method of ['launch', 'connect']) {
  const original = chromium[method];
  chromium[method] = async function (...args) {
    const browser = await original.apply(this, args), create = browser.newContext.bind(browser);
    browser.newContext = async function (...contextArgs) {
      const context = await create(...contextArgs);
      await context.route('**/config/doppler-package.json', route => route.fulfill({
        contentType: 'application/json', body: JSON.stringify(config) }));
      await context.route('**/vendor/doppler/**', async route => {
        try {
          const path = new URL(route.request().url()).pathname;
          const prefix = `/vendor/doppler/${packageInfo.version}/`;
          assert(path.startsWith(prefix), `Unexpected package version requested: ${path}`);
          const relative = decodeURIComponent(path.slice(prefix.length)), file = resolve(packageRoot, relative);
          assert(file.startsWith(packageRoot + sep), 'Package request escaped installed root');
          const bytes = await readFile(file); served.set(relative, { path: relative, sha256: digest(bytes), bytes: bytes.length });
          const contentType = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.json') ? 'application/json'
            : file.endsWith('.wasm') ? 'application/wasm' : 'text/plain';
          await route.fulfill({ contentType, body: bytes });
        } catch (error) {
          errors.push(error.message); await route.fulfill({ status: 500, body: error.message });
        }
      });
      return context;
    };
    return browser;
  };
  restore.push(() => { chromium[method] = original; });
}
const priorCwd = process.cwd();
const receipt = { scope: 'Existing Reploid fixture with candidate served solely from its installed package in isolated browser contexts',
  archiveSha256: digest(archiveBytes), archiveIntegrity: config.integrity, version: packageInfo.version,
  fixture: fixture.slice(reploidRoot.length + 1), fixtureSha256: digest(await readFile(fixture)),
  productionPinChanged: false, sourceSubstitution: false, passed: false };
try {
  process.chdir(consumerRoot); // Fixture file reads inspect this installed package, never production node_modules.
  await import(pathToFileURL(fixture));
  assert.deepEqual(errors, []);
  assert(served.has('src/partitions.js'), 'The candidate partition entry must actually load');
  const version = await readFile(resolve(packageRoot, 'src/version.js'), 'utf8');
  assert(version.includes(`'${packageInfo.version}'`), 'Package and runtime versions agree');
  receipt.passed = true;
} catch (error) { receipt.failure = error.stack; throw error; }
finally {
  process.chdir(priorCwd); for (const undo of restore.reverse()) undo();
  receipt.served = [...served.values()].sort((a, b) => a.path.localeCompare(b.path)); receipt.errors = errors;
  await writeFile(process.env.REPLOID_CAPTURE_OUT + '.package.json', JSON.stringify(receipt, null, 2) + '\n');
}
