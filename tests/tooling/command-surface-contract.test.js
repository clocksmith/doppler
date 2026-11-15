import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateCommandAdmission } from '../../tools/check-command-surface-contract.js';
const policy = JSON.parse(fs.readFileSync(new URL('../../tools/policies/command-surface-contract.json', import.meta.url), 'utf8'));
assert.deepEqual(validateCommandAdmission(policy), []);
assert(validateCommandAdmission(policy, (request, surface) => ({ request, surface }))
  .some(error => error.includes('admitted a forbidden command')));
assert(validateCommandAdmission(policy, () => { throw Error('broken normalization'); })
  .some(error => error.includes('unexpected node admission failure')));
assert(validateCommandAdmission({ commands: [{ id: 'missing-probe', surfaces: ['node'] }] })
  .some(error => error.includes('add an admission probe')));
console.log('command-surface-contract: admission failures and missing probes are detected');
