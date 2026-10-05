/** Run an existing Reploid acceptance fixture against one installed archive.
 * Only this process's fresh browser contexts see the candidate configuration.
 * The production checkout, vendor tree, running server and dependency pin stay untouched. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { observeAttentionCache } from './attention-cache-observer.js';
import { buildQ4KAccumulationDiagnostic } from '../kernels/q4k-accumulation-diagnostic.js';
import { buildReciprocalRootDiagnostic } from '../kernels/rmsnorm-platform-diagnostic.js';
import { buildRecurrentAccumulationDiagnostic } from '../kernels/recurrent-accumulation-diagnostic.js';

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
const attentionCaptures = [];
const arithmeticCandidate = process.env.DOPPLER_Q4K_DIAGNOSTIC ?? null;
const normalizationCandidate = process.env.DOPPLER_RMS_DIAGNOSTIC ?? null;
const recurrentCandidate = process.env.DOPPLER_RECURRENT_DOT_CANDIDATE ?? null;
assert(!recurrentCandidate || ['memory', 'readout'].includes(recurrentCandidate));
assert(!recurrentCandidate || !(arithmeticCandidate || normalizationCandidate),
  'Recurrent candidates freeze upstream projection and normalization');
const recurrentPaths = ['src/gpu/kernels/gated_delta_recurrent.wgsl', 'src/gpu/kernels/gated_delta_fused_decode.wgsl'];
const substitutions = [];
assert(normalizationCandidate === null || normalizationCandidate === 'refined-rsqrt');
assert(!(arithmeticCandidate || normalizationCandidate || recurrentCandidate) || process.env.DOPPLER_TEST_ONLY_ARITHMETIC === '1',
  'Arithmetic substitutions require an explicitly identified diagnostic, never archive acceptance');
for (const method of ['launch', 'connect']) {
  const original = chromium[method];
  chromium[method] = async function (...args) {
    const browser = await original.apply(this, args), create = browser.newContext.bind(browser);
    browser.newContext = async function (...contextArgs) {
      const context = await create(...contextArgs);
      if (process.env.DOPPLER_ATTENTION_CACHE_CAPTURE) {
        await observeAttentionCache(context, packageRoot, attentionCaptures);
      }
      await context.route('**/config/doppler-package.json', route => route.fulfill({
        contentType: 'application/json', body: JSON.stringify(config) }));
      await context.route('**/vendor/doppler/**', async route => {
        try {
          const path = new URL(route.request().url()).pathname;
          const prefix = `/vendor/doppler/${packageInfo.version}/`;
          assert(path.startsWith(prefix), `Unexpected package version requested: ${path}`);
          const relative = decodeURIComponent(path.slice(prefix.length)), file = resolve(packageRoot, relative);
          assert(file.startsWith(packageRoot + sep), 'Package request escaped installed root');
          let bytes = await readFile(file);
          if (arithmeticCandidate && relative === 'src/gpu/kernels/fused_matmul_q4_widetile.wgsl') {
            const originalSha256 = digest(bytes);
            bytes = Buffer.from(buildQ4KAccumulationDiagnostic(bytes.toString('utf8'), arithmeticCandidate));
            if (!substitutions.some(s => s.path === relative)) substitutions.push({
              path: relative, candidate: arithmeticCandidate, originalSha256, diagnosticSha256: digest(bytes),
            });
          }
          if (normalizationCandidate && relative === 'src/gpu/kernels/rmsnorm.wgsl') {
            const originalSha256 = digest(bytes);
            bytes = Buffer.from(buildReciprocalRootDiagnostic(
              buildReciprocalRootDiagnostic(bytes.toString('utf8'), 'main'), 'main_subgroup'));
            if (!substitutions.some(s => s.path === relative)) substitutions.push({
              path: relative, candidate: normalizationCandidate, originalSha256, diagnosticSha256: digest(bytes),
            });
          }
          if (recurrentCandidate && recurrentPaths.includes(relative)) {
            const originalSha256 = digest(bytes);
            bytes = Buffer.from(buildRecurrentAccumulationDiagnostic(bytes.toString('utf8'),
              recurrentCandidate, relative === recurrentPaths[1]));
            if (!substitutions.some(s => s.path === relative)) substitutions.push({
              path: relative, candidate: recurrentCandidate, originalSha256, diagnosticSha256: digest(bytes),
            });
          }
          served.set(relative, { path: relative, sha256: digest(bytes), bytes: bytes.length });
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
if (arithmeticCandidate || normalizationCandidate || recurrentCandidate) {
  receipt.scope = 'Test-only arithmetic substitution over the installed archive; not installed-package acceptance';
  receipt.sourceSubstitution = true;
  receipt.arithmeticCandidate = arithmeticCandidate;
  receipt.normalizationCandidate = normalizationCandidate;
  receipt.recurrentCandidate = recurrentCandidate;
}
try {
  process.chdir(consumerRoot); // Fixture file reads inspect this installed package, never production node_modules.
  await import(pathToFileURL(fixture));
  if (process.env.DOPPLER_ATTENTION_CACHE_CAPTURE) {
    assert(attentionCaptures.some(capture => capture.records.length > 0), 'Attention observation must hit its boundaries');
  }
  assert.deepEqual(errors, []);
  assert(served.has('src/partitions.js'), 'The candidate partition entry must actually load');
  if (arithmeticCandidate || normalizationCandidate) {
    assert.equal(substitutions.length, Number(!!arithmeticCandidate) + Number(!!normalizationCandidate),
      'Every requested diagnostic shader must be fetched');
  }
  if (recurrentCandidate) {
    assert(substitutions.some(s => s.path === recurrentPaths[0]), 'Recurrent diagnostic must actually load');
  }
  const version = await readFile(resolve(packageRoot, 'src/version.js'), 'utf8');
  assert(version.includes(`'${packageInfo.version}'`), 'Package and runtime versions agree');
  receipt.passed = true;
} catch (error) { receipt.failure = error.stack; throw error; }
finally {
  process.chdir(priorCwd); for (const undo of restore.reverse()) undo();
  receipt.served = [...served.values()].sort((a, b) => a.path.localeCompare(b.path)); receipt.errors = errors;
  receipt.substitutions = substitutions;
  await writeFile(process.env.REPLOID_CAPTURE_OUT + '.package.json', JSON.stringify(receipt, null, 2) + '\n');
  if (process.env.DOPPLER_ATTENTION_CACHE_CAPTURE) {
    await writeFile(process.env.DOPPLER_ATTENTION_CACHE_CAPTURE, JSON.stringify({
      scope: 'Read-only full-attention operands from installed package; instrumentation is not release acceptance',
      archiveSha256: digest(archiveBytes), sourceSubstitution: receipt.sourceSubstitution,
      substitutions, captures: attentionCaptures,
    }));
  }
}
