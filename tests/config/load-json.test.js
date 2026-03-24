import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { loadJson } from '../../src/utils/load-json.js';

{
  const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-load-json-'));
  try {
    writeFileSync(path.join(fixtureDir, 'ok.json'), JSON.stringify({ ok: true, n: 7 }), 'utf8');
    writeFileSync(path.join(fixtureDir, 'bad.json'), '{bad-json', 'utf8');

    const baseUrl = pathToFileURL(path.join(fixtureDir, 'base.js')).toString();
    const parsed = await loadJson('./ok.json', baseUrl, 'unused-prefix');
    assert.deepEqual(parsed, { ok: true, n: 7 });

    await assert.rejects(
      () => loadJson('./bad.json', baseUrl, 'unused-prefix'),
      /Unexpected token|JSON/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({
      ok: false,
      status: 500,
      async json() {
        return null;
      },
    });

    await assert.rejects(
      () => loadJson('https://example.com/not-found.json', 'https://example.com/', 'custom-prefix'),
      /custom-prefix: https:\/\/example\.com\/not-found\.json/
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
}

{
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url) => ({
      ok: true,
      async json() {
        return { url: String(url), mode: 'mocked' };
      },
      async text() {
        return JSON.stringify({ url: String(url), mode: 'mocked' });
      },
    });

    const parsed = await loadJson('https://example.com/ok.json', 'https://example.com/');
    assert.equal(parsed.mode, 'mocked');
    assert.equal(parsed.url, 'https://example.com/ok.json');
  } finally {
    globalThis.fetch = originalFetch;
  }
}

console.log('load-json.test: ok');
