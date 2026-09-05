import assert from 'node:assert/strict';
import path from 'node:path';
import { parsePackageSmokeArgs } from '../../tools/check-packed-package.js';

assert.deepEqual(parsePackageSmokeArgs([]), { retain: null });
assert.deepEqual(parsePackageSmokeArgs(['--retain', 'reports/package-candidate']), {
  retain: path.resolve('reports/package-candidate'),
});
for (const args of [['--retain'], ['--retain', '--help'], ['--force'], ['--retain', 'a', 'b']]) {
  assert.throws(() => parsePackageSmokeArgs(args), /Usage:/);
}
console.log('packed-package-options.test: ok');
