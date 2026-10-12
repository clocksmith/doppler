import assert from 'node:assert/strict';
import { settleFrozenPrefixResources } from '../helpers/frozen-prefix-resources.js';

const primaryFailure = new Error('Device lost during preparation');
const events = [], failures = [];
const count = await settleFrozenPrefixResources({ primaryFailure,
  steps: [['snapshot', () => { throw Error('No live device'); }],
    ['queue', () => { throw Error('Queue unavailable'); }], ['unload', () => events.push('unload')]],
  recordFailure: row => failures.push(row) });
assert.equal(count, 2); assert.deepEqual(events, ['unload']);
assert.deepEqual(failures.map(row => row.phase), ['snapshot', 'queue']);
await assert.rejects(settleFrozenPrefixResources({ primaryFailure: null,
  steps: [['unload', () => { throw Error('Cleanup failed'); }]], recordFailure: () => {} }),
  error => error instanceof AggregateError && error.errors[0].message === 'Cleanup failed');
const later = [];
assert.equal(await settleFrozenPrefixResources({ primaryFailure,
  steps: [['snapshot', () => { throw Error('No device'); }], ['unload', () => later.push('unload')]],
  recordFailure: () => { throw Error('Recorder unavailable'); } }), 2);
assert.deepEqual(later, ['unload']);
assert.equal(await settleFrozenPrefixResources({ primaryFailure: null,
  steps: [['unload', () => {}]], recordFailure: () => {} }), 0);
console.log('frozen prefix resource settlement controls passed');
