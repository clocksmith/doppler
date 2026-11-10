import assert from 'node:assert/strict';
import path from 'node:path';
import { portableSourcePath } from '../../src/tooling/portable-source-path.js';

const base = path.resolve('/work/doppler');
const home = path.resolve('/home/someone');

assert.equal(portableSourcePath(path.join(base, 'downloads', 'qwen'), { baseDir: base, homeDir: home }), 'downloads/qwen');
assert.equal(portableSourcePath(path.join(home, 'models', 'q.gguf'), { baseDir: base, homeDir: home }), '~/models/q.gguf');
assert.equal(portableSourcePath('/data/models/q', { baseDir: base, homeDir: home }), path.resolve('/data/models/q').split(path.sep).join('/'));
assert.equal(portableSourcePath(base, { baseDir: base, homeDir: home }), base.split(path.sep).join('/'));
assert.equal(portableSourcePath('', { baseDir: base, homeDir: home }), null);
assert.equal(
  portableSourcePath(path.join(base, 'downloads', 'qwen'), { baseDir: base, homeDir: home }),
  portableSourcePath(path.join(base, 'downloads', 'qwen'), { baseDir: base, homeDir: path.resolve('/Users/other') }),
  'the same input names the same source on machines with different home directories'
);
console.log('portable-source-path.test: ok');
