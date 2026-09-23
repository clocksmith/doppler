#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const source = fileURLToPath(new URL('../examples/document-search/', import.meta.url));
const output = process.argv[2] ? path.resolve(process.argv[2]) : await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-installed-starter-'));
await fs.mkdir(output, { recursive: true });
const app = path.join(output, 'application');
await fs.cp(source, app, { recursive: true, filter: filename => !path.relative(source, filename).split(path.sep).includes('node_modules') });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const lock = await fs.readFile(path.join(app, 'package-lock.json'));
const report = { schema: 'doppler.installed-starter-check/v1', passed: false, physicalExecution: false,
  scope: 'Frozen npm installation, served runtime bytes, and offline application shell. No model execution or numerical qualification.',
  probeSha256: hash(await fs.readFile(fileURLToPath(import.meta.url))), nodeVersion: process.version,
  applicationDir: app, requests: [] };
let server;
let context;
try {
  const install = await promisify(execFile)('npm', ['ci', '--omit=optional', '--no-audit', '--no-fund'], { cwd: app });
  report.installOutput = install.stdout;
  assert.deepEqual(await fs.readFile(path.join(app, 'package-lock.json')), lock, 'acceptance must not rewrite its lock');
  report.build = JSON.parse(await fs.readFile(path.join(app, 'build-receipt.json')));
  const manifestBytes = await fs.readFile(path.join(app, 'application-assets.js'));
  assert.equal(hash(manifestBytes), report.build.applicationManifestSha256);
  const manifest = JSON.parse(manifestBytes.toString().replace(/^self\.DOCUMENT_SEARCH_ASSETS = /, '').replace(/;\s*$/, ''));
  assert.ok(manifest.assets.filter(asset => asset.path.startsWith('runtime/')).length > 1000);
  const { createServer } = await import(pathToFileURL(path.join(app, 'server.js')).href);
  server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = 'http://127.0.0.1:' + server.address().port + '/index.html';
  for (const asset of manifest.assets) {
    const response = await fetch(new URL(asset.path, url));
    assert.equal(response.status, 200, asset.path);
    assert.equal(hash(Buffer.from(await response.arrayBuffer())), asset.sha256, asset.path);
  }
  report.assetCount = manifest.assets.length;
  const launch = async offline => {
    context = await chromium.launchPersistentContext(path.join(output, 'profile'), { headless: true });
    report.browserVersion = context.browser().version();
    context.on('request', request => report.requests.push({ offline, url: request.url(), method: request.method(), body: request.postData() }));
    await context.setOffline(offline);
    const page = context.pages()[0];
    page.on('pageerror', error => { report.pageErrors ??= []; report.pageErrors.push(error.message); });
    await page.goto(url);
    await page.waitForFunction(() => globalThis.documentSearch?.ready, { timeout: 60000 });
    return page;
  };
  await launch(false);
  await context.close(); context = null;
  await new Promise(resolve => server.close(resolve)); server = null;
  report.serverStoppedBeforeRestart = true;
  await launch(true);
  report.offlineShellPassed = true;
  assert.deepEqual(report.pageErrors ?? [], []);
  report.passed = true;
} catch (error) {
  report.error = error.stack;
  process.exitCode = 1;
} finally {
  await context?.close();
  if (server) await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(output, 'check.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: report.passed, physicalExecution: false, output, error: report.error }));
}
