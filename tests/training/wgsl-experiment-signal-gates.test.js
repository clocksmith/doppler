import assert from 'node:assert/strict';

import { summarizeGrpoLearningSignal } from '../../tools/run-wgsl-repair-experiment.js';

const zero = summarizeGrpoLearningSignal([
  { samples: [{ advantage: 0 }, { advantage: 0 }] },
  { samples: [{ advantage: 0 }, { advantage: 0 }] },
]);
assert.deepEqual(zero, {
  groupCount: 2,
  sampleCount: 4,
  varyingGroups: 0,
  nonzeroAdvantages: 0,
  hasLearningSignal: false,
});

const varying = summarizeGrpoLearningSignal([
  { samples: [{ advantage: -1 }, { advantage: 1 }] },
  { samples: [{ advantage: 0 }, { advantage: 0 }] },
]);
assert.equal(varying.varyingGroups, 1);
assert.equal(varying.nonzeroAdvantages, 2);
assert.equal(varying.hasLearningSignal, true);

assert.throws(
  () => summarizeGrpoLearningSignal([{ samples: [{ advantage: 0 }] }]),
  /at least two samples/
);

console.log('wgsl-experiment-signal-gates.test: ok');
