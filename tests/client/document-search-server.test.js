import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createServer } from '../../examples/document-search/server.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-starter-server-'));
const server = createServer({ root });
try {
  await fs.mkdir(path.join(root, 'runtime/src'), { recursive: true });
  await fs.writeFile(path.join(root, 'runtime/src/capsule-runtime.js'), 'must not serve a substituted runtime');
  await fs.writeFile(path.join(root, 'index.html'), 'starter');
  await fs.writeFile(path.join(root, 'shard-sources.json'), JSON.stringify({ artifacts: {} }));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = 'http://127.0.0.1:' + server.address().port;
  assert.equal((await fetch(url + '/runtime/src/capsule-runtime.js')).status, 404);
  assert.equal((await fetch(url + '/capsules/embedding/artifacts/model/shard_00017.bin')).status, 503);
  assert.equal((await fetch(url + '/index.html', { headers: { range: 'bytes=100-1' } })).status, 416);
  assert.equal(await (await fetch(url + '/index.html', { headers: { range: 'bytes=0-2' } })).text(), 'sta');
  assert.equal((await fetch(url + '/runtime/%2e%2e%2fpackage.json')).status, 400);
  await fs.mkdir(path.join(root, 'node_modules/doppler-gpu/src'), { recursive: true });
  await fs.writeFile(path.join(root, 'node_modules/doppler-gpu/src/capsule-runtime.js'), 'installed');
  assert.equal(await (await fetch(url + '/runtime/src/capsule-runtime.js')).text(), 'installed');
} finally {
  await new Promise(resolve => server.close(resolve));
  await fs.rm(root, { recursive: true, force: true });
}
console.log('document-search-server: installed-only runtime, unavailable-artifact and range regressions passed');
