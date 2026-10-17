import assert from 'node:assert/strict';
import { parseMacProcesses } from '../../tools/browser-renderer-memory.js';

const rows = parseMacProcesses(`
  101 1 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir=/tmp/qwen-profile
  102 101 /Applications/Google Chrome.app/Contents/Frameworks/Chrome Helper --type=renderer
  103 101 /Applications/Google Chrome.app/Contents/Frameworks/Chrome Helper --type=gpu-process
  201 1 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir=/tmp/other-profile
  202 201 /Applications/Google Chrome.app/Contents/Frameworks/Chrome Helper --type=renderer
`, '/tmp/qwen-profile');
assert.deepEqual(rows.filter(row => row.root).map(row => row.pid), [101]);
assert.deepEqual(rows.filter(row => row.renderer).map(row => [row.pid, row.parent]), [[102, 101], [202, 201]]);
assert.throws(() => parseMacProcesses('invalid observation', '/tmp/qwen-profile'), /Unrecognized/);
console.log('browser-renderer-memory.test: ok (process parsing only, no physical memory claim)');
