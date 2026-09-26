#!/usr/bin/env node
// Diagnostic hashing only. Full installed-application opening is a separate probe.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { createSha256Hasher } from '../src/formats/sha256.js';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
for (const number of [...config.sizes, config.repeats, config.chunkBytes]) assert(Number.isSafeInteger(number) && number > 0);
const report = { schema: 'doppler.artifact-hash-probe/v1', config, node: process.version,
  probeSha256: createHash('sha256').update(await fs.readFile(new URL(import.meta.url))).digest('hex'),
  scope: 'Synthetic hashing diagnostics, no GPU execution or complete-opening performance claim.', rows: [] };
for (const size of config.sizes) {
  const bytes = new Uint8Array(size).fill(37);
  const expected = createHash('sha256').update(bytes).digest('hex');
  for (let repeat = 0; repeat < config.repeats; repeat++) {
    for (const backend of ['javascript', 'node-crypto', 'webcrypto']) {
      const start = performance.now();
      let digest;
      if (backend === 'webcrypto') digest = Buffer.from(await crypto.subtle.digest('SHA-256', bytes)).toString('hex');
      else {
        const hasher = backend === 'javascript' ? createSha256Hasher() : createHash('sha256');
        for (let offset = 0; offset < size; offset += config.chunkBytes) hasher.update(bytes.subarray(offset, offset + config.chunkBytes));
        digest = backend === 'javascript' ? hasher.digestHex() : hasher.digest('hex');
      }
      const elapsedMs = performance.now() - start;
      assert.equal(digest, expected);
      report.rows.push({ surface: 'node', backend, size, repeat, elapsedMs, digest });
    }
  }
}
if (config.blake3Modules) {
  const modules = await Promise.all(config.blake3Modules.map(async filename => ({ filename,
    sha256: createHash('sha256').update(await fs.readFile(filename)).digest('hex'),
    module: await import(pathToFileURL(filename)) })));
  report.blake3Modules = modules.map(({ module, ...identity }) => identity);
  for (const size of config.sizes) {
    const bytes = new Uint8Array(size).fill(37);
    let expected;
    for (let repeat = 0; repeat < config.repeats; repeat++) {
      for (const { module, filename } of repeat % 2 ? [...modules].reverse() : modules) {
        const start = performance.now();
        const digest = Buffer.from(await module.hash(bytes)).toString('hex');
        const elapsedMs = performance.now() - start;
        expected ??= digest;
        assert.equal(digest, expected, 'Existing artifact digest semantics must be unchanged.');
        report.rows.push({ surface: 'node', backend: filename, size, repeat, elapsedMs, digest });
      }
    }
  }
}
let browser, server;
try {
  if (config.browser) {
    server = await createStaticFileServer({ rootDir: process.cwd(), host: '127.0.0.1', port: 0 });
    browser = await chromium.launch({ headless: true });
    report.browserVersion = browser.version();
    const page = await browser.newPage();
    await page.goto(server.baseUrl);
    report.rows.push(...await page.evaluate(async config => {
      const { createSha256Hasher } = await import('/src/formats/sha256.js');
      const rows = [];
      for (const size of config.sizes) {
        const bytes = new Uint8Array(size).fill(37);
        let expected;
        for (let repeat = 0; repeat < config.repeats; repeat++) for (const backend of ['javascript', 'webcrypto']) {
          const start = performance.now();
          let digest;
          if (backend === 'webcrypto') digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
          else {
            const hasher = createSha256Hasher();
            for (let offset = 0; offset < size; offset += config.chunkBytes) hasher.update(bytes.subarray(offset, offset + config.chunkBytes));
            digest = hasher.digestHex();
          }
          const elapsedMs = performance.now() - start;
          expected ??= digest;
          if (digest !== expected) throw new Error('Browser digest mismatch.');
          rows.push({ surface: 'browser', backend, size, repeat, elapsedMs, digest });
        }
      }
      return rows;
    }, config));
    report.browserMemoryScope = 'One bounded input per digest. Web Crypto may copy it; no model concatenation. This probe does not measure physical peak memory.';
  }
} finally {
  await browser?.close();
  await server?.close();
}
await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ outputPath: config.outputPath, rows: report.rows.length }));
