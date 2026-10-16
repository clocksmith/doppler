import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
const fixture = mkdtempSync(path.join(tmpdir(), 'doppler-unused-locals-'));
try {
  mkdirSync(path.join(fixture, 'tools'));
  copyFileSync(path.join(root, 'tools/check-unused-locals.js'), path.join(fixture, 'tools/check-unused-locals.js'));
  symlinkSync(path.join(root, 'node_modules'), path.join(fixture, 'node_modules'), 'dir');
  writeFileSync(path.join(fixture, 'package.json'), '{"type":"module"}');
  const config = {
    compilerOptions: {
      allowJs: true, checkJs: true, noUnusedLocals: true, noEmit: true,
      module: 'ESNext', moduleResolution: 'bundler', target: 'ES2022',
      skipLibCheck: true, types: [],
    },
    files: ['subject.js'],
  };
  const writeConfig = () => writeFileSync(path.join(fixture, 'tsconfig.source-unused.json'), JSON.stringify(config));
  writeConfig();
  writeFileSync(path.join(fixture, 'dependency.js'), 'export const value = 1;');
  // A clean sibling declaration must never conceal unused implementation imports.
  writeFileSync(path.join(fixture, 'subject.d.ts'), 'export declare const result: number;');
  function check(source) {
    writeFileSync(path.join(fixture, 'subject.js'), source);
    return spawnSync(process.execPath, [path.join(fixture, 'tools/check-unused-locals.js')], {
      cwd: fixture, encoding: 'utf8', timeout: 30000,
    });
  }
  let result = check("import { value } from './dependency.js'; export const result = 2;");
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /TS6133/);
  result = check("import { value } from './dependency.js'; export const result = value;");
  assert.equal(result.status, 0, result.stderr);
  result = check("import './dependency.js'; export { value as result } from './dependency.js';");
  assert.equal(result.status, 0, result.stderr);
  result = check('const unused = 1; export const result = 2;');
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /TS6133/);
  result = check('export const result = ;');
  assert.equal(result.status, 1, result.stderr);
  config.files = ['missing.js'];
  writeConfig();
  result = check('export const result = 1;');
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /must resolve to its JavaScript implementation/);
  console.log('unused-locals-check.test: ok (6 cases)');
} finally {
  rmSync(fixture, { recursive: true, force: true });
}
