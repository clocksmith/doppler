import assert from 'node:assert/strict';
import path from 'node:path';
import { parsePackageSmokeArgs } from '../../tools/check-packed-package.js';

assert.deepEqual(parsePackageSmokeArgs([]), { retain: null, archive: null });
assert.deepEqual(parsePackageSmokeArgs(['--retain', 'reports/package-candidate']), {
  retain: path.resolve('reports/package-candidate'), archive: null,
});
assert.deepEqual(parsePackageSmokeArgs(['--archive', 'candidate.tgz', '--retain', 'receipt']), {
  retain: path.resolve('receipt'), archive: path.resolve('candidate.tgz'),
});
for (const args of [['--retain'], ['--retain', '--help'], ['--force'], ['--retain', 'a', 'b']]) {
  assert.throws(() => parsePackageSmokeArgs(args), /Usage:/);
}
console.log('packed-package-options.test: ok');
