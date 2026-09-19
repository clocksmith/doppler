import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

const root = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-graph-'));
const entry = path.join(root, 'entry.js');
const run = () => spawnSync(process.execPath, ['tools/check-browser-import-graph.js', entry], { encoding: 'utf8' });
try {
  await fs.writeFile(entry, "// import 'node:fs';\nconst prose = \"import './missing.js'\";\nexport const value = 1;");
  assert.equal(run().status, 0, 'comments and inert source strings cannot create browser imports');
  await fs.writeFile(entry, "await import('./private.js');");
  await fs.writeFile(path.join(root, 'private.js'), "import 'node:fs';");
  const forbidden = run();
  assert.notEqual(forbidden.status, 0);
  assert.match(forbidden.stderr, /node:\* specifier/);
  await fs.writeFile(entry, "await import('./missing.js', { with: { type: 'json' } });");
  const missing = run();
  assert.notEqual(missing.status, 0);
  assert.match(missing.stderr, /unresolved dynamic import/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('browser-import-graph.test: dynamic targets and missing dependencies reject');
