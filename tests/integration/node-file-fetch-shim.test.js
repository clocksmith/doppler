import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();
installNodeFileFetchShim();

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-node-file-fetch-'));
  try {
    const samplePath = path.join(tempDir, 'sample.txt');
    writeFileSync(samplePath, 'doppler-node-file-fetch', 'utf8');

    const fileResponse = await fetch(pathToFileURL(samplePath));
    assert.equal(fileResponse.status, 200);
    assert.equal(await fileResponse.text(), 'doppler-node-file-fetch');

    const missingResponse = await fetch(pathToFileURL(path.join(tempDir, 'missing.txt')));
    assert.equal(missingResponse.status, 404);
    assert.match(await missingResponse.text(), /(Not Found|ENOENT|no such file)/i);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const response = await fetch('data:text/plain,non-file-passthrough');
  assert.equal(response.status, 200);
  assert.equal(await response.text(), 'non-file-passthrough');
}

{
  await assert.rejects(
    () => fetch({ url: 'http://[invalid-url' }),
    /(TypeError|URL|Invalid)/
  );
}

console.log('node-file-fetch-shim.test: ok');
